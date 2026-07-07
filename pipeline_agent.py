"""
pipeline_agent.py

A 14-phase execution agent with an adaptive meta-controller, hierarchical governor,
and pluggable surrogate architecture.
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
        self.coef = None
        self.cov = None
        self.n_params = 0
        self.lam = 0.0

    def _get_feats(self, X):
        n, d = X.shape
        feats = [np.ones((n, 1)), X, X ** 2]
        cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if self.model_interactions and cross_idx:
            cross = np.column_stack([X[:, i] * X[:, j] for i, j in cross_idx])
            feats.append(cross)
        return np.hstack(feats)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        A = self._get_feats(X)
        self.n_params = A.shape[1]
        ratio = n / self.n_params
        self.lam = 0.05 / max(ratio - 1.0, 0.05)
        A_aug = np.vstack([A, np.sqrt(self.lam) * np.eye(self.n_params)])
        y_aug = np.concatenate([y, np.zeros(self.n_params)])
        self.coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)

        residuals = y - A @ self.coef
        sigma2 = np.sum(residuals**2) / max(n - self.n_params, 1)
        self.cov = sigma2 * np.linalg.inv(A.T @ A + (self.lam + 1e-6) * np.eye(self.n_params))

    def propose(self, base_x: np.ndarray, radius: float, bounds: np.ndarray) -> np.ndarray:
        d = len(base_x)
        b_lin = self.coef[1 : 1 + d]
        b_quad = self.coef[1 + d : 1 + 2 * d]
        grad = b_lin + 2 * b_quad * base_x
        cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if self.model_interactions and cross_idx:
            b_cross = self.coef[1 + 2 * d :]
            for (i, j), c in zip(cross_idx, b_cross):
                grad[i] += c * base_x[j]
                grad[j] += c * base_x[i]

        gnorm = np.linalg.norm(grad) + 1e-12
        steps = np.linspace(0.1, 0.5, 5) * radius
        proposals = [np.clip(base_x - s * grad / gnorm, bounds[:, 0], bounds[:, 1]) for s in steps]

        best_lcb = np.inf
        best_p = proposals[0]
        for p in proposals:
            phi = self._get_feats(p[None, :])[0]
            pred = phi @ self.coef
            sigma = np.sqrt(max(0, phi @ self.cov @ phi))
            lcb = pred - 2.0 * sigma
            if lcb < best_lcb:
                best_lcb, best_p = lcb, p
        return best_p

    def get_summary(self) -> dict:
        return {"n_params": self.n_params, "ridge_lambda": self.lam}

class DiscreteSurrogate:
    def __init__(self, step_size: float = 1.0):
        self.step_size = step_size
        self.best_x = None
        self.best_y = np.inf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        idx = np.argmin(y)
        self.best_x, self.best_y = X[idx], y[idx]

    def propose(self, base_x: np.ndarray, radius: float, bounds: np.ndarray) -> np.ndarray:
        d = len(base_x)
        proposal = base_x.copy()
        dim = np.random.randint(d)
        offset = np.random.choice([-1, 1]) * self.step_size
        proposal[dim] = np.clip(round(proposal[dim] + offset), bounds[dim, 0], bounds[dim, 1])
        return proposal

    def get_summary(self) -> dict:
        return {"model": "DiscreteNeighborhood", "best_val": float(self.best_y)}

class PipelineAgent:
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        mode: str = "SurrogateGuided",
        surrogate_model: SurrogateModel = None,
        seed: int = 0,
        max_rounds: int = 20,
        tol: float = 1e-6,
    ):
        self.f = objective_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.d = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.max_rounds = max_rounds
        self.tol = tol

        self.explore_n = 40
        self.sleuth_samples = 8
        self.noise_repeats = 5
        self.radius_decay = 0.6
        model_interactions = True

        if mode == "GlobalExploring":
            self.explore_n, self.sleuth_samples, model_interactions, self.radius_decay = 80, 4, False, 0.8
        elif mode == "LocalRefining":
            self.explore_n, self.sleuth_samples, self.radius_decay = 20, 8, 0.4
        elif mode == "NoisyMode":
            self.noise_repeats = 10

        self.surrogate = surrogate_model or QuadraticSurrogate(model_interactions=model_interactions)
        self.adaptive_explore_n, self.adaptive_sleuth_samples = self.explore_n, self.sleuth_samples
        self.opt_rounds, self.ft_rounds, self.honing_samples = 3, 20, 9
        self.log, self.ctx = [], {}
        self.full_radius = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0])) / 2.0
        self.radius, self.center = self.full_radius, self.bounds.mean(axis=1)
        self.best_x_ever, self.best_score_ever = None, np.inf
        self.stagnant_rounds, self.eval_count, self.current_phase = 0, 0, None
        self.phase_stats = {p: {"improvement": 0.0, "evals": 0} for p in PHASES}
        self.is_deterministic, self.basin_archive = None, []

    def _clip(self, x): return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    def _eval(self, x):
        score = float(self.f(x))
        self.eval_count += 1
        if self.current_phase:
            self.phase_stats[self.current_phase]["evals"] += 1
            if self.best_score_ever != np.inf and score < self.best_score_ever:
                self.phase_stats[self.current_phase]["improvement"] += (self.best_score_ever - score)
        if score < self.best_score_ever: self.best_score_ever, self.best_x_ever = score, x.copy()
        return score
    def _probe_determinism(self):
        self.current_phase = "Probing"
        x0 = self.center
        a, b = self._eval(x0), self._eval(x0)
        self.is_deterministic = abs(a - b) <= 1e-12 * max(1.0, abs(a))
        self.current_phase = None
    def _record(self, round_, phase, summary, t0): self.log.append(PhaseLog(round_, phase, summary, time.time() - t0))
    def _adapt_budget(self):
        def get_util(p):
            s = self.phase_stats[p]
            return s["improvement"] / max(s["evals"], 1)
        tunable = ["Exploring", "Sleuthing", "Optimizing", "Fine-tuning", "Honing"]
        utils = {p: get_util(p) for p in tunable}
        mean_u = sum(utils.values()) / len(tunable) + 1e-12
        def scale(val, u, min_val, max_val):
            if u > mean_u * 1.5: return min(max_val, int(val * 1.2) + 1)
            if u < mean_u * 0.5: return max(min_val, int(val * 0.8))
            return val
        self.adaptive_explore_n = scale(self.adaptive_explore_n, utils["Exploring"], 10, 200)
        self.adaptive_sleuth_samples = scale(self.adaptive_sleuth_samples, utils["Sleuthing"], 2, 32)
        self.opt_rounds = scale(self.opt_rounds, utils["Optimizing"], 1, 10)
        self.ft_rounds = scale(self.ft_rounds, utils["Fine-tuning"], 5, 100)
        self.honing_samples = scale(self.honing_samples, utils["Honing"], 3, 21)
    def _should_skip(self, phase):
        stats = self.phase_stats[phase]
        return stats["evals"] >= 50 and stats["improvement"] == 0
    def exploring(self, round_):
        t0 = time.time()
        self.current_phase = "Exploring"
        pts = self._clip(self.center + self.rng.uniform(-1, 1, size=(self.adaptive_explore_n, self.d)) * self.radius)
        scores = np.array([self._eval(p) for p in pts])
        self.ctx["explore_pts"], self.ctx["explore_scores"] = pts, scores
        self._record(round_, "Exploring", {"best": float(scores.min())}, t0)
        self.current_phase = None
    def sleuthing(self, round_):
        t0 = time.time()
        self.current_phase = "Sleuthing"
        pts, scores = self.ctx["explore_pts"], self.ctx["explore_scores"]
        idx = np.argsort(scores)[: 5]
        leads = pts[idx]
        new_pts, new_scores = [], []
        for lead in leads:
            cand = self._clip(lead + self.rng.uniform(-1, 1, size=(self.adaptive_sleuth_samples, self.d)) * self.radius * 0.2)
            for c in cand: new_pts.append(c); new_scores.append(self._eval(c))
        self.ctx["sleuth_pts"] = np.vstack([pts, np.array(new_pts)])
        self.ctx["sleuth_scores"] = np.concatenate([scores, np.array(new_scores)])
        self._record(round_, "Sleuthing", {"best": float(self.ctx["sleuth_scores"].min())}, t0)
        self.current_phase = None
    def sifting(self, round_):
        t0 = time.time()
        self.current_phase = "Sifting"
        pts, scores = self.ctx["sleuth_pts"], self.ctx["sleuth_scores"]
        keep_n = min(len(scores), max(30, int(0.3 * len(scores))))
        idx = np.argsort(scores)[:keep_n]
        self.ctx["sift_pts"], self.ctx["sift_scores"] = pts[idx], scores[idx]
        self._record(round_, "Sifting", {"kept": keep_n}, t0)
        self.current_phase = None
    def figuring(self, round_):
        t0 = time.time()
        self.current_phase = "Figuring"
        self.surrogate.fit(self.ctx["sift_pts"], self.ctx["sift_scores"])
        self._record(round_, "Figuring", self.surrogate.get_summary(), t0)
        self.current_phase = None
    def reckoning(self, round_):
        t0 = time.time()
        self.current_phase = "Reckoning"
        best_x = self.ctx["sift_pts"][np.argmin(self.ctx["sift_scores"])]
        proposal = self.surrogate.propose(best_x, self.radius, self.bounds)
        self.ctx["reckoning_proposal"] = proposal
        self.ctx["reckoning_proposal_true_score"] = self._eval(proposal)
        self._record(round_, "Reckoning", {"score": self.ctx["reckoning_proposal_true_score"]}, t0)
        self.current_phase = None
    def analyzing(self, round_):
        t0 = time.time()
        self.current_phase = "Analyzing"
        self._record(round_, "Analyzing", {"mean": float(self.ctx["sift_scores"].mean())}, t0)
        self.current_phase = None
    def synthesizing(self, round_):
        t0 = time.time()
        self.current_phase = "Synthesizing"
        self.ctx["synth_pts"] = np.vstack([self.ctx["sift_pts"], self.ctx["reckoning_proposal"][None, :]])
        self.ctx["synth_scores"] = np.concatenate([self.ctx["sift_scores"], [self.ctx["reckoning_proposal_true_score"]]])
        self._record(round_, "Synthesizing", {"pool_size": len(self.ctx["synth_scores"])}, t0)
        self.current_phase = None
    def crystallizing(self, round_):
        t0 = time.time()
        self.current_phase = "Crystallizing"
        i = np.argmin(self.ctx["synth_scores"])
        self.ctx["current_x"], self.ctx["current_score"] = self.ctx["synth_pts"][i], float(self.ctx["synth_scores"][i])
        self._record(round_, "Crystallizing", {"score": self.ctx["current_score"]}, t0)
        self.current_phase = None
    def evaluating(self, round_):
        t0 = time.time()
        self.current_phase = "Evaluating"
        prev = self.ctx.get("best_score_so_far", np.inf)
        self.ctx["evaluation_improvement"] = prev - self.ctx["current_score"]
        self._record(round_, "Evaluating", {"improvement": self.ctx["evaluation_improvement"]}, t0)
        self.current_phase = None
    def optimizing(self, round_):
        t0 = time.time()
        self.current_phase = "Optimizing"
        x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]
        step = self.radius * 0.15
        for _ in range(self.opt_rounds):
            for i in range(self.d):
                for sign in (+1, -1):
                    cand = x.copy(); cand[i] = np.clip(x[i] + sign * step, self.bounds[i, 0], self.bounds[i, 1])
                    s = self._eval(cand)
                    if s < best_score: best_score, x = s, cand
            step *= 0.6
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Optimizing", {"score": best_score}, t0)
        self.current_phase = None
    def fine_tuning(self, round_):
        t0 = time.time()
        self.current_phase = "Fine-tuning"
        x, best_score = self.ctx["current_x"], self.ctx["current_score"]
        step = self.radius * 0.03
        for _ in range(self.ft_rounds):
            cand = self._clip(x + self.rng.normal(0, step, size=self.d))
            s = self._eval(cand)
            if s < best_score: best_score, x = s, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Fine-tuning", {"score": best_score}, t0)
        self.current_phase = None
    def honing(self, round_):
        t0 = time.time()
        self.current_phase = "Honing"
        x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]
        eps = self.radius * 0.01 + 1e-6
        sens = [abs(self._eval(np.clip(x+eps*np.eye(self.d)[i], self.bounds[:,0], self.bounds[:,1])) - best_score) for i in range(self.d)]
        target_dim, step = np.argmax(sens), self.radius * 0.05
        for s_ in np.linspace(-step, step, self.honing_samples):
            cand = x.copy(); cand[target_dim] = np.clip(x[target_dim] + s_, self.bounds[target_dim, 0], self.bounds[target_dim, 1])
            sc = self._eval(cand)
            if sc < best_score: best_score, x = sc, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Honing", {"score": best_score}, t0)
        self.current_phase = None
    def validating(self, round_):
        t0 = time.time()
        self.current_phase = "Validating"
        repeats = [self._eval(self.ctx["current_x"])] if self.is_deterministic else [self._eval(self.ctx["current_x"]) for _ in range(self.noise_repeats)]
        self.ctx["current_score"] = float(np.mean(repeats))
        self._record(round_, "Validating", {"mean": self.ctx["current_score"]}, t0)
        self.current_phase = None
    def iterating(self, round_, prev_best):
        t0 = time.time()
        self.current_phase = "Iterating"
        cur = self.ctx["current_score"]
        if cur < self.best_score_ever: self.best_score_ever, self.best_x_ever = cur, self.ctx["current_x"].copy()
        stalled = prev_best - cur < self.tol and prev_best != np.inf
        self.stagnant_rounds = self.stagnant_rounds + 1 if stalled else 0
        hard_conv = stalled and self.radius < self.full_radius * 0.02
        hop = stalled and not hard_conv and self.stagnant_rounds >= 1
        if hard_conv: action = "stop"
        elif hop:
            action = "basin_hop"
            self.basin_archive.append(self.center.copy())
            for _ in range(10):
                cand = self._clip(self.bounds[:, 0] + self.rng.uniform(size=self.d) * (self.bounds[:, 1] - self.bounds[:, 0]))
                if not self.basin_archive or min(np.linalg.norm(cand - b) for b in self.basin_archive) > self.full_radius * 0.5: break
            self.center, self.radius, self.stagnant_rounds = cand, self.full_radius, 0
        else: action, self.center, self.radius = "refine", self.ctx["current_x"], self.radius * self.radius_decay
        self._record(round_, "Iterating", {"action": action}, t0)
        self.current_phase = None
        return hard_conv
    def run(self):
        if self.is_deterministic is None: self._probe_determinism()
        best = np.inf
        for round_ in range(1, self.max_rounds + 1):
            if round_ > 2: self._adapt_budget()
            self.exploring(round_); self.sleuthing(round_); self.sifting(round_); self.figuring(round_); self.reckoning(round_)
            self.analyzing(round_); self.synthesizing(round_); self.crystallizing(round_); self.evaluating(round_)
            if not self._should_skip("Optimizing"): self.optimizing(round_)
            if not self._should_skip("Fine-tuning"): self.fine_tuning(round_)
            if not self._should_skip("Honing"): self.honing(round_)
            self.validating(round_); conv = self.iterating(round_, best); best = self.ctx["current_score"]
            self.ctx["best_score_so_far"] = best
            if conv: break
        return {"best_x": self.best_x_ever.tolist() if self.best_x_ever is not None else None, "best_score": self.best_score_ever, "rounds_run": round_, "converged": conv, "eval_count": self.eval_count, "is_deterministic": self.is_deterministic, "phase_stats": self.phase_stats}
    def dump_log(self, path):
        with open(path, "w") as fh: json.dump([asdict(p) for p in self.log], fh, indent=2, default=str)

class TaskTyper:
    def __init__(self, objective_fn: Callable, bounds: np.ndarray, seed: int = 42):
        self.f, self.bounds, self.d, self.rng = objective_fn, bounds, bounds.shape[0], np.random.default_rng(seed)
    def characterize(self, n_probes: int = 20) -> Dict[str, Any]:
        pts = self.bounds[:, 0] + self.rng.uniform(size=(n_probes, self.d)) * (self.bounds[:, 1] - self.bounds[:, 0])
        v1, v2 = self.f(pts[0]), self.f(pts[0])
        is_det = abs(v1 - v2) <= 1e-12 * max(1.0, abs(v1))
        # Detect discrete dimensions (heuristic: check if all sampled values are integers)
        is_discrete = all(np.all(np.isclose(p, np.round(p))) for p in pts)
        scores = np.array([self.f(p) for p in pts])
        return {"is_deterministic": is_det, "is_discrete": is_discrete, "d": self.d, "val_range": (float(np.min(scores)), float(np.max(scores)))}

class Governor:
    def __init__(self, objective_fn: Callable, bounds: np.ndarray, budget: int = 5000, seed: int = 0):
        self.f, self.bounds, self.budget, self.rng = objective_fn, bounds, budget, np.random.default_rng(seed)
        self.typer = TaskTyper(objective_fn, bounds, seed=seed)
        self.best_x, self.best_score = None, np.inf
    def run(self):
        char = self.typer.characterize(10)
        used = 20
        surrogate = DiscreteSurrogate() if char["is_discrete"] else None
        modes = ["NoisyMode"] if not char["is_deterministic"] else (["GlobalExploring", "SurrogateGuided"] if char["d"] > 10 else ["SurrogateGuided", "LocalRefining"])
        b_per_mode = (self.budget - used) // len(modes)
        for mode in modes:
            agent = PipelineAgent(self.f, self.bounds, mode=mode, surrogate_model=surrogate, max_rounds=max(5, b_per_mode//200), seed=self.rng.integers(10000))
            res = agent.run()
            used += res["eval_count"]
            if res["best_score"] < self.best_score: self.best_score, self.best_x = res["best_score"], res["best_x"]
            if used >= self.budget: break
        return {"best_x": self.best_x, "best_score": self.best_score, "evals": used, "modes": modes, "char": char}
