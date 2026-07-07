"""
pipeline_agent.py

A hierarchical optimization framework with adaptive meta-controller and governor.
"""

from __future__ import annotations
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Dict, List, Tuple, Protocol

PHASES = [
    "Probing", "Exploring", "Sleuthing", "Sifting", "Figuring", "Reckoning",
    "Analyzing", "Synthesizing", "Crystallizing", "Evaluating",
    "Optimizing", "Fine-tuning", "Honing", "Validating", "Iterating",
]

@dataclass
class PhaseLog:
    round: int
    phase: str
    summary: dict
    duration_s: float

class SurrogateModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def propose(self, base_x: np.ndarray, radius: float, bounds: np.ndarray) -> np.ndarray: ...
    def get_summary(self) -> dict: ...

class QuadraticSurrogate:
    def __init__(self, model_interactions: bool = True):
        self.model_interactions = model_interactions
        self.actual_interactions = model_interactions
        self.coef, self.cov, self.n_params, self.lam = None, None, 0, 0.0
    def _get_feats(self, X, use_int: bool):
        n, d = X.shape
        feats = [np.ones((n, 1)), X, X ** 2]
        if use_int:
            cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
            if cross_idx: feats.append(np.column_stack([X[:, i] * X[:, j] for i, j in cross_idx]))
        return np.hstack(feats)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        needed = 1 + 2 * d + (d * (d - 1) // 2 if self.model_interactions else 0)
        self.actual_interactions = self.model_interactions and (n >= needed * 1.5)
        A = self._get_feats(X, self.actual_interactions)
        self.n_params = A.shape[1]
        self.lam = 0.05 / max((n / self.n_params) - 1.0, 0.05)
        A_aug = np.vstack([A, np.sqrt(self.lam) * np.eye(self.n_params)])
        y_aug = np.concatenate([y, np.zeros(self.n_params)])
        self.coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        sigma2 = np.sum((y - A @ self.coef)**2) / max(n - self.n_params, 1)
        self.cov = sigma2 * np.linalg.inv(A.T @ A + (self.lam + 1e-6) * np.eye(self.n_params))
    def propose(self, base_x: np.ndarray, radius: float, bounds: np.ndarray) -> np.ndarray:
        d = len(base_x)
        b_lin, b_quad = self.coef[1 : 1 + d], self.coef[1 + d : 1 + 2 * d]
        grad = b_lin + 2 * b_quad * base_x
        if self.actual_interactions:
            cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
            b_cross = self.coef[1 + 2 * d :]
            for (i, j), c in zip(cross_idx, b_cross): grad[i] += c * base_x[j]; grad[j] += c * base_x[i]
        gnorm = np.linalg.norm(grad) + 1e-12
        steps = np.linspace(0.1, 0.5, 5) * radius
        proposals = [np.clip(base_x - s * grad / gnorm, bounds[:, 0], bounds[:, 1]) for s in steps]
        best_lcb, best_p = np.inf, proposals[0]
        for p in proposals:
            phi = self._get_feats(p[None, :], self.actual_interactions)[0]
            lcb = phi @ self.coef - 2.0 * np.sqrt(max(0, phi @ self.cov @ phi))
            if lcb < best_lcb: best_lcb, best_p = lcb, p
        return best_p
    def get_summary(self) -> dict: return {"n_params": self.n_params, "ridge_lambda": self.lam, "interactions": self.actual_interactions}

class DiscreteSurrogate:
    def __init__(self, step_size: float = 1.0):
        self.step_size, self.best_x, self.best_y = step_size, None, np.inf
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        idx = np.argmin(y); self.best_x, self.best_y = X[idx], y[idx]
    def propose(self, base_x: np.ndarray, radius: float, bounds: np.ndarray) -> np.ndarray:
        p = base_x.copy(); dim = np.random.randint(len(base_x))
        p[dim] = np.clip(round(p[dim] + np.random.choice([-1, 1]) * self.step_size), bounds[dim, 0], bounds[dim, 1])
        return p
    def get_summary(self) -> dict: return {"model": "DiscreteNeighborhood"}

class PipelineAgent:
    def __init__(
        self, objective_fn: Callable, bounds: np.ndarray, mode: str = "SurrogateGuided",
        surrogate_model: SurrogateModel = None, seed: int = 0, max_rounds: int = 20,
        tol: float = 1e-6, budget: int = 2000
    ):
        self.f, self.bounds, self.d, self.rng = objective_fn, np.asarray(bounds, dtype=float), bounds.shape[0], np.random.default_rng(seed)
        self.mode, self.max_rounds, self.tol, self.budget, self.seed = mode, max_rounds, tol, budget, seed
        self.explore_n, self.sleuth_samples, self.noise_repeats, self.radius_decay = 40, 8, 5, 0.6
        model_int = True
        if mode == "GlobalExploring": self.explore_n, self.sleuth_samples, model_int, self.radius_decay = 80, 4, False, 0.8
        elif mode == "LocalRefining": self.explore_n, self.sleuth_samples, self.radius_decay = 20, 8, 0.4
        elif mode == "NoisyMode": self.noise_repeats = 10
        self.surrogate = surrogate_model or QuadraticSurrogate(model_interactions=model_int)
        self.adaptive_explore_n, self.adaptive_sleuth_samples = self.explore_n, self.sleuth_samples
        self.opt_rounds, self.ft_rounds, self.honing_samples = 3, 20, 9
        self.log, self.ctx = [], {}
        self.full_radius = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0])) / 2.0
        self.radius, self.center = self.full_radius, self.bounds.mean(axis=1)
        self.best_x_ever, self.best_score_ever, self.stagnant_rounds, self.eval_count, self.current_phase = None, np.inf, 0, 0, None
        self.phase_stats = {p: {"improvement": 0.0, "evals": 0} for p in PHASES}
        self.is_deterministic, self.basin_archive = None, []

    def _clip(self, x): return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    def _eval(self, x):
        score = float(self.f(x)); self.eval_count += 1
        if self.current_phase:
            self.phase_stats[self.current_phase]["evals"] += 1
            if self.best_score_ever != np.inf and score < self.best_score_ever:
                self.phase_stats[self.current_phase]["improvement"] += (self.best_score_ever - score)
        if score < self.best_score_ever: self.best_score_ever, self.best_x_ever = score, x.copy()
        return score
    def _probe_determinism(self):
        self.current_phase = "Probing"
        x0 = self.center; a, b = self._eval(x0), self._eval(x0)
        self.is_deterministic = abs(a - b) <= 1e-12 * max(1.0, abs(a)); self.current_phase = None
    def _record(self, r, p, s, t0): self.log.append(PhaseLog(r, p, s, time.time() - t0))
    def _adapt_budget(self):
        def get_util(p): s = self.phase_stats[p]; return s["improvement"] / max(s["evals"], 1)
        tunable = ["Exploring", "Sleuthing", "Optimizing", "Fine-tuning", "Honing"]
        utils = {p: get_util(p) for p in tunable}; mean_u = sum(utils.values()) / len(tunable) + 1e-12
        def scale(val, u, min_v, max_v):
            if u > mean_u * 1.5: return min(max_v, int(val * 1.2) + 1)
            if u < mean_u * 0.5: return max(min_v, int(val * 0.8))
            return val
        self.adaptive_explore_n = scale(self.adaptive_explore_n, utils["Exploring"], 10, 200)
        self.adaptive_sleuth_samples = scale(self.adaptive_sleuth_samples, utils["Sleuthing"], 2, 32)
        self.opt_rounds = scale(self.opt_rounds, utils["Optimizing"], 1, 10)
        self.ft_rounds = scale(self.ft_rounds, utils["Fine-tuning"], 5, 100)
        self.honing_samples = scale(self.honing_samples, utils["Honing"], 3, 21)
    def _should_skip(self, p): return self.phase_stats[p]["evals"] >= 50 and self.phase_stats[p]["improvement"] == 0
    def exploring(self, r):
        t0 = time.time(); self.current_phase = "Exploring"
        pts = self._clip(self.center + self.rng.uniform(-1, 1, (self.adaptive_explore_n, self.d)) * self.radius)
        scores = np.array([self._eval(p) for p in pts]); self.ctx["explore_pts"], self.ctx["explore_scores"] = pts, scores
        self._record(r, "Exploring", {"best": float(scores.min())}, t0); self.current_phase = None
    def sleuthing(self, r):
        t0 = time.time(); self.current_phase = "Sleuthing"
        pts, scores = self.ctx["explore_pts"], self.ctx["explore_scores"]
        idx = np.argsort(scores)[: 5]; leads = pts[idx]; new_pts, new_scores = [], []
        for lead in leads:
            cand = self._clip(lead + self.rng.uniform(-1, 1, (self.adaptive_sleuth_samples, self.d)) * self.radius * 0.2)
            for c in cand: new_pts.append(c); new_scores.append(self._eval(c))
        self.ctx["sleuth_pts"] = np.vstack([pts, np.array(new_pts)])
        self.ctx["sleuth_scores"] = np.concatenate([scores, np.array(new_scores)])
        self._record(r, "Sleuthing", {"best": float(self.ctx["sleuth_scores"].min())}, t0); self.current_phase = None
    def sifting(self, r):
        t0 = time.time(); self.current_phase = "Sifting"
        pts, scores = self.ctx["sleuth_pts"], self.ctx["sleuth_scores"]
        keep_n = min(len(scores), max(30, int(0.3 * len(scores)))); idx = np.argsort(scores)[:keep_n]
        self.ctx["sift_pts"], self.ctx["sift_scores"] = pts[idx], scores[idx]
        self._record(r, "Sifting", {"kept": keep_n}, t0); self.current_phase = None
    def figuring(self, r):
        t0 = time.time(); self.current_phase = "Figuring"; self.surrogate.fit(self.ctx["sift_pts"], self.ctx["sift_scores"])
        self._record(r, "Figuring", self.surrogate.get_summary(), t0); self.current_phase = None
    def reckoning(self, r):
        t0 = time.time(); self.current_phase = "Reckoning"; best_x = self.ctx["sift_pts"][np.argmin(self.ctx["sift_scores"])]
        proposal = self.surrogate.propose(best_x, self.radius, self.bounds); self.ctx["reckoning_proposal"] = proposal
        self.ctx["reckoning_proposal_true_score"] = self._eval(proposal)
        self._record(r, "Reckoning", {"score": self.ctx["reckoning_proposal_true_score"]}, t0); self.current_phase = None
    def analyzing(self, r):
        t0 = time.time(); self.current_phase = "Analyzing"
        self._record(r, "Analyzing", {"mean": float(self.ctx["sift_scores"].mean())}, t0); self.current_phase = None
    def synthesizing(self, r):
        t0 = time.time(); self.current_phase = "Synthesizing"
        self.ctx["synth_pts"] = np.vstack([self.ctx["sift_pts"], self.ctx["reckoning_proposal"][None, :]])
        self.ctx["synth_scores"] = np.concatenate([self.ctx["sift_scores"], [self.ctx["reckoning_proposal_true_score"]]])
        self._record(r, "Synthesizing", {"pool_size": len(self.ctx["synth_scores"])}, t0); self.current_phase = None
    def crystallizing(self, r):
        t0 = time.time(); self.current_phase = "Crystallizing"; i = np.argmin(self.ctx["synth_scores"])
        self.ctx["current_x"], self.ctx["current_score"] = self.ctx["synth_pts"][i], float(self.ctx["synth_scores"][i])
        self._record(r, "Crystallizing", {"score": self.ctx["current_score"]}, t0); self.current_phase = None
    def evaluating(self, r):
        t0 = time.time(); self.current_phase = "Evaluating"; prev = self.ctx.get("best_score_so_far", np.inf)
        self.ctx["evaluation_improvement"] = prev - self.ctx["current_score"]
        self._record(r, "Evaluating", {"improvement": self.ctx["evaluation_improvement"]}, t0); self.current_phase = None
    def optimizing(self, r):
        t0 = time.time(); self.current_phase = "Optimizing"; x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]; step = self.radius * 0.15
        for _ in range(self.opt_rounds):
            for i in range(self.d):
                for sign in (+1, -1):
                    cand = x.copy(); cand[i] = np.clip(x[i] + sign * step, self.bounds[i, 0], self.bounds[i, 1])
                    s = self._eval(cand)
                    if s < best_score: best_score, x = s, cand
            step *= 0.6
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(r, "Optimizing", {"score": best_score}, t0); self.current_phase = None
    def fine_tuning(self, r):
        t0 = time.time(); self.current_phase = "Fine-tuning"; x, best_score = self.ctx["current_x"], self.ctx["current_score"]; step = self.radius * 0.03
        for _ in range(self.ft_rounds):
            cand = self._clip(x + self.rng.normal(0, step, size=self.d))
            s = self._eval(cand)
            if s < best_score: best_score, x = s, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(r, "Fine-tuning", {"score": best_score}, t0); self.current_phase = None
    def honing(self, r):
        t0 = time.time(); self.current_phase = "Honing"; x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]; eps = self.radius * 0.01 + 1e-6
        sens = [abs(self._eval(np.clip(x+eps*np.eye(self.d)[i], self.bounds[:,0], self.bounds[:,1])) - best_score) for i in range(self.d)]
        target_dim, step = np.argmax(sens), self.radius * 0.05
        for s_ in np.linspace(-step, step, self.honing_samples):
            cand = x.copy(); cand[target_dim] = np.clip(x[target_dim] + s_, self.bounds[target_dim, 0], self.bounds[target_dim, 1])
            sc = self._eval(cand)
            if sc < best_score: best_score, x = sc, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(r, "Honing", {"score": best_score}, t0); self.current_phase = None
    def validating(self, r):
        t0 = time.time(); self.current_phase = "Validating"
        repeats = [self._eval(self.ctx["current_x"])] if self.is_deterministic else [self._eval(self.ctx["current_x"]) for _ in range(self.noise_repeats)]
        self.ctx["current_score"] = float(np.mean(repeats))
        self._record(r, "Validating", {"mean": self.ctx["current_score"]}, t0); self.current_phase = None
    def iterating(self, round_, prev_best):
        t0 = time.time(); self.current_phase = "Iterating"; cur = self.ctx["current_score"]
        if cur < self.best_score_ever: self.best_score_ever, self.best_x_ever = cur, self.ctx["current_x"].copy()
        stalled = (prev_best - cur < self.tol) and prev_best != np.inf
        self.stagnant_rounds = self.stagnant_rounds + 1 if stalled else 0
        hard_conv = stalled and (self.radius < self.full_radius * 0.01 or self.stagnant_rounds >= 3)
        hop = stalled and not hard_conv and self.stagnant_rounds >= 1
        if hard_conv: action = "stop"
        elif hop:
            action = "basin_hop"; self.basin_archive.append({"center": self.center.copy(), "best_x": self.ctx["current_x"].copy(), "best_score": cur})
            for _ in range(10):
                cand = self._clip(self.bounds[:, 0] + self.rng.uniform(size=self.d) * (self.bounds[:, 1] - self.bounds[:, 0]))
                if not self.basin_archive or min(np.linalg.norm(cand - b["center"]) for b in self.basin_archive) > self.full_radius * 0.5: break
            self.center, self.radius, self.stagnant_rounds = cand, self.full_radius, 0
        else: action, self.center, self.radius = "refine", self.ctx["current_x"], self.radius * self.radius_decay
        self._record(round_, "Iterating", {"action": action}, t0); self.current_phase = None
        return hard_conv
    def run(self):
        if self.is_deterministic is None: self._probe_determinism()
        best = np.inf
        for round_ in range(1, self.max_rounds + 1):
            if self.eval_count >= self.budget: break
            if round_ > 2: self._adapt_budget()
            for phase in ["exploring", "sleuthing", "sifting", "figuring", "reckoning", "analyzing", "synthesizing", "crystallizing", "evaluating"]:
                getattr(self, phase)(round_)
            for phase in ["optimizing", "fine_tuning", "honing"]:
                if not self._should_skip(phase.capitalize().replace("_", "-")): getattr(self, phase)(round_)
            self.validating(round_); conv = self.iterating(round_, best); best = self.ctx["current_score"]
            self.ctx["best_score_so_far"] = best
            if conv: break
        return {"best_x": self.best_x_ever.tolist() if self.best_x_ever is not None else None, "best_score": self.best_score_ever, "rounds_run": round_, "converged": conv, "eval_count": self.eval_count, "is_deterministic": self.is_deterministic, "phase_stats": self.phase_stats, "seed": self.seed, "mode": self.mode, "basin_archive": self.basin_archive}

class TaskTyper:
    def __init__(self, objective_fn: Callable, bounds: np.ndarray, seed: int = 42):
        self.f, self.bounds, self.d, self.rng = objective_fn, bounds, bounds.shape[0], np.random.default_rng(seed)
    def characterize(self, n_probes: int = 20) -> Dict[str, Any]:
        pts = self.bounds[:, 0] + self.rng.uniform(size=(n_probes, self.d)) * (self.bounds[:, 1] - self.bounds[:, 0])
        v1, v2 = self.f(pts[0]), self.f(pts[0]); is_det = abs(v1 - v2) <= 1e-12 * max(1.0, abs(v1))
        noise_std = float(np.std([self.f(pts[0]) for _ in range(10)])) if not is_det else 0.0
        scores = np.array([self.f(p) for p in pts])
        ruggedness = float(np.std(np.diff(scores)) / (np.std(scores) + 1e-9)) if len(scores) > 1 else 0.0
        is_discrete = all(np.all(np.isclose(p, np.round(p))) for p in pts)
        return {"is_deterministic": is_det, "noise_std": noise_std, "ruggedness": ruggedness, "is_discrete": is_discrete, "d": self.d}

class Governor:
    def __init__(self, objective_fn: Callable, bounds: np.ndarray, budget: int = 5000, seed: int = 0):
        self.f, self.bounds, self.budget, self.rng = objective_fn, bounds, budget, np.random.default_rng(seed)
        self.typer, self.best_x, self.best_score = TaskTyper(objective_fn, bounds, seed=seed), None, np.inf
    def run(self):
        char = self.typer.characterize(10); used = 20; surrogate = DiscreteSurrogate() if char["is_discrete"] else None
        if not char["is_deterministic"] or char["noise_std"] > 1e-6: modes = ["NoisyMode", "GlobalExploring"]
        elif char["ruggedness"] > 1.2: modes = ["GlobalExploring", "SurrogateGuided", "LocalRefining"]
        elif char["d"] > 12: modes = ["GlobalExploring", "SurrogateGuided"]
        else: modes = ["SurrogateGuided", "LocalRefining"]
        b_per_mode = (self.budget - used) // len(modes); results = []
        for mode in modes:
            agent = PipelineAgent(self.f, self.bounds, mode=mode, surrogate_model=surrogate, budget=b_per_mode, seed=self.rng.integers(10000))
            res = agent.run(); used += res["eval_count"]; results.append(res)
            if res["best_score"] < self.best_score: self.best_score, self.best_x = res["best_score"], res["best_x"]
            if used >= self.budget: break
        return {"best_x": self.best_x, "best_score": self.best_score, "evals": used, "modes": modes, "char": char, "portfolio_results": results}
