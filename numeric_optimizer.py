"""
numeric_optimizer.py - Specialized Numeric Optimization Engine
Enhanced with Observability, Parallelism, Structured Goals, and Steerability.
"""

from __future__ import annotations
import time
import json
import numpy as np
import threading
from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Dict, List, Tuple, Protocol, Iterator, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

PHASES = [
    "Probing", "Exploring", "Sleuthing", "Sifting", "Figuring", "Reckoning",
    "Analyzing", "Synthesizing", "Crystallizing", "Evaluating",
    "Optimizing", "Fine-tuning", "Honing", "Validating", "Iterating",
]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

@dataclass
class PhaseLog:
    round: int
    phase: str
    summary: dict
    duration_s: float

@dataclass
class OptimizationGoal:
    objective: Callable
    bounds: np.ndarray
    constraints: List[Callable] = field(default_factory=list)
    penalty_factor: float = 1e6
    target_score: float = -np.inf
    success_criteria: Optional[Callable[[float, np.ndarray], bool]] = None

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
        d = len(base_x); b_lin, b_quad = self.coef[1 : 1 + d], self.coef[1 + d : 1 + 2 * d]; grad = b_lin + 2 * b_quad * base_x
        if self.actual_interactions:
            idx = [(i, j) for i in range(d) for j in range(i + 1, d)]; bc = self.coef[1 + 2 * d :]
            for (i, j), c in zip(idx, bc): grad[i] += c * base_x[j]; grad[j] += c * base_x[i]
        gnorm = np.linalg.norm(grad) + 1e-12; steps = np.linspace(0.1, 0.5, 5) * radius
        proposals = [np.clip(base_x - s * grad / gnorm, bounds[:, 0], bounds[:, 1]) for s in steps]
        best_lcb, best_p = np.inf, proposals[0]
        for p in proposals:
            phi = self._get_feats(p[None, :], self.actual_interactions)[0]
            lcb = phi @ self.coef - 2.0 * np.sqrt(max(0, phi @ self.get_cov() @ phi))
            if lcb < best_lcb: best_lcb, best_p = lcb, p
        return best_p
    def get_cov(self): return self.cov if self.cov is not None else np.eye(self.n_params)
    def get_summary(self) -> dict: return {"model": "Quadratic", "n_params": self.n_params, "ridge_lambda": self.lam, "interactions": self.actual_interactions}

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

class GlobalResultTracker:
    def __init__(self):
        self.best_score = np.inf
        self.best_x = None
        self.lock = threading.Lock()
    def update(self, score, x):
        with self.lock:
            if score < self.best_score:
                self.best_score, self.best_x = score, x.copy()
    def get_best(self):
        with self.lock:
            return self.best_score, self.best_x

class NumericOptimizer:
    def __init__(
        self, goal: OptimizationGoal, mode: str = "SurrogateGuided",
        surrogate_model: SurrogateModel = None, seed: int = 0, max_rounds: int = 20,
        tol: float = 1e-6, budget: int = 2000,
        callback: Optional[Callable[[dict], None]] = None,
        shared_tracker: Optional[GlobalResultTracker] = None
    ):
        self.goal = goal
        self.f, self.bounds = goal.objective, np.asarray(goal.bounds, dtype=float)
        self.d, self.rng = self.bounds.shape[0], np.random.default_rng(seed)
        self.mode, self.max_rounds, self.tol, self.budget, self.seed = mode, max_rounds, tol, budget, seed
        self.callback = callback
        self.shared_tracker = shared_tracker

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
        self.phase_stats = {p: {"improvement": 0.0, "evals": 0, "history": []} for p in PHASES}
        self._last_stats_evals = {p: 0 for p in PHASES}
        self._last_stats_imp = {p: 0.0 for p in PHASES}
        self.is_deterministic, self.basin_archive = None, []
        self.last_round, self.converged = 0, False

        # Steerability
        self.disabled_phases: Set[str] = set()
        self.manual_pause: bool = False

    def _clip(self, x): return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    def _eval(self, x):
        raw_score = float(self.f(x)); self.eval_count += 1
        penalty = 0.0
        for c in self.goal.constraints:
            val = c(x)
            if val > 0: penalty += val * self.goal.penalty_factor
        score = raw_score + penalty

        if self.shared_tracker:
            g_score, g_x = self.shared_tracker.get_best()
            if g_score < self.best_score_ever:
                self.best_score_ever, self.best_x_ever = g_score, g_x.copy()

        if self.current_phase:
            self.phase_stats[self.current_phase]["evals"] += 1
            if self.best_score_ever != np.inf and score < self.best_score_ever:
                imp = self.best_score_ever - score
                self.phase_stats[self.current_phase]["improvement"] += imp

        if score < self.best_score_ever:
            self.best_score_ever, self.best_x_ever = score, x.copy()
            if self.shared_tracker:
                self.shared_tracker.update(score, x)

        return score

    def _probe_determinism(self):
        self.current_phase = "Probing"; x0 = self.center; a, b = self._eval(x0), self._eval(x0)
        self.is_deterministic = abs(a - b) <= 1e-12 * max(1.0, abs(a)); self.current_phase = None
    def _record(self, r, p, s, t0):
        log_entry = PhaseLog(r, p, s, time.time() - t0)
        self.log.append(log_entry)
        # Track utility history
        stats = self.phase_stats[p]
        last_evals = stats["evals"] - self._last_stats_evals[p]
        last_imp = stats["improvement"] - self._last_stats_imp[p]
        stats["history"].append({"round": r, "improvement": last_imp, "evals": last_evals})
        self._last_stats_evals[p] = stats["evals"]
        self._last_stats_imp[p] = stats["improvement"]

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
    def _should_skip(self, p):
        if p in self.disabled_phases: return True
        return self.phase_stats[p]["evals"] >= 50 and self.phase_stats[p]["improvement"] == 0

    def set_surrogate(self, model: SurrogateModel): self.surrogate = model

    def exploring(self, r):
        t0 = time.time(); self.current_phase = "Exploring"
        pts = self._clip(self.center + self.rng.uniform(-1, 1, (self.adaptive_explore_n, self.d)) * self.radius)
        scores = np.array([self._eval(p) for p in pts]); self.ctx["explore_pts"], self.ctx["explore_scores"] = pts, scores
        self._record(r, "Exploring", {"best": float(scores.min())}, t0); self.current_phase = None
    def sleuthing(self, r):
        t0 = time.time(); self.current_phase = "Sleuthing"
        pts, scores = self.ctx.get("explore_pts", np.empty((0, self.d))), self.ctx.get("explore_scores", np.empty(0))
        if len(pts) == 0:
            self._record(r, "Sleuthing", {"skipped": "No explore_pts"}, t0); self.current_phase = None; return
        idx = np.argsort(scores)[: 5]; leads = pts[idx]; new_pts, new_scores = [], []
        for lead in leads:
            cand = self._clip(lead + self.rng.uniform(-1, 1, (self.adaptive_sleuth_samples, self.d)) * self.radius * 0.2)
            for c in cand: new_pts.append(c); new_scores.append(self._eval(c))
        self.ctx["sleuth_pts"] = np.vstack([pts, np.array(new_pts)]); self.ctx["sleuth_scores"] = np.concatenate([scores, np.array(new_scores)])
        self._record(r, "Sleuthing", {"best": float(self.ctx["sleuth_scores"].min())}, t0); self.current_phase = None
    def sifting(self, r):
        t0 = time.time(); self.current_phase = "Sifting"
        pts, scores = self.ctx.get("sleuth_pts", np.empty((0, self.d))), self.ctx.get("sleuth_scores", np.empty(0))
        if len(pts) == 0:
            # Fallback: maybe just use the center or nothing
            self._record(r, "Sifting", {"skipped": "No sleuth_pts"}, t0); self.current_phase = None; return
        keep_n = min(len(scores), max(30, int(0.3 * len(scores)))); idx = np.argsort(scores)[:keep_n]
        self.ctx["sift_pts"], self.ctx["sift_scores"] = pts[idx], scores[idx]; self._record(r, "Sifting", {"kept": keep_n}, t0); self.current_phase = None
    def figuring(self, r):
        t0 = time.time(); self.current_phase = "Figuring"
        pts, scores = self.ctx.get("sift_pts", np.empty((0, self.d))), self.ctx.get("sift_scores", np.empty(0))
        if len(pts) < 3:
            self._record(r, "Figuring", {"skipped": "Too few pts"}, t0); self.current_phase = None; return
        self.surrogate.fit(pts, scores)
        self._record(r, "Figuring", self.surrogate.get_summary(), t0); self.current_phase = None
    def reckoning(self, r):
        t0 = time.time(); self.current_phase = "Reckoning"
        pts, scores = self.ctx.get("sift_pts", np.empty((0, self.d))), self.ctx.get("sift_scores", np.empty(0))
        if len(pts) == 0 or self.surrogate.get_summary().get("model") != "Quadratic" and len(self.log) < 2:
            self._record(r, "Reckoning", {"skipped": "Requirements not met"}, t0); self.current_phase = None; return
        best_x = pts[np.argmin(scores)]
        proposal = self.surrogate.propose(best_x, self.radius, self.bounds); self.ctx["reckoning_proposal"] = proposal
        self.ctx["reckoning_proposal_true_score"] = self._eval(proposal); self._record(r, "Reckoning", {"score": self.ctx["reckoning_proposal_true_score"]}, t0); self.current_phase = None
    def analyzing(self, r):
        t0 = time.time(); self.current_phase = "Analyzing"
        scores = self.ctx.get("sift_scores", np.empty(0))
        mean_val = float(scores.mean()) if len(scores) > 0 else 0.0
        self._record(r, "Analyzing", {"mean": mean_val}, t0); self.current_phase = None
    def synthesizing(self, r):
        t0 = time.time(); self.current_phase = "Synthesizing"
        pts, scores = self.ctx.get("sift_pts", np.empty((0, self.d))), self.ctx.get("sift_scores", np.empty(0))
        if "reckoning_proposal" in self.ctx:
            pts = np.vstack([pts, self.ctx["reckoning_proposal"][None, :]])
            scores = np.concatenate([scores, [self.ctx["reckoning_proposal_true_score"]]])
        self.ctx["synth_pts"], self.ctx["synth_scores"] = pts, scores
        self._record(r, "Synthesizing", {"pool_size": len(scores)}, t0); self.current_phase = None
    def crystallizing(self, r):
        t0 = time.time(); self.current_phase = "Crystallizing"
        pts, scores = self.ctx.get("synth_pts", np.empty((0, self.d))), self.ctx.get("synth_scores", np.empty(0))
        if len(pts) == 0:
            if self.best_x_ever is not None:
                self.ctx["current_x"], self.ctx["current_score"] = self.best_x_ever, self.best_score_ever
            else:
                self.ctx["current_x"], self.ctx["current_score"] = self.center, self._eval(self.center)
        else:
            i = np.argmin(scores)
            self.ctx["current_x"], self.ctx["current_score"] = pts[i], float(scores[i])
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
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score; self._record(r, "Optimizing", {"score": best_score}, t0); self.current_phase = None
    def fine_tuning(self, r):
        t0 = time.time(); self.current_phase = "Fine-tuning"; x, best_score = self.ctx["current_x"], self.ctx["current_score"]; step = self.radius * 0.03
        for _ in range(self.ft_rounds):
            cand = self._clip(x + self.rng.normal(0, step, size=self.d)); s = self._eval(cand)
            if s < best_score: best_score, x = s, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score; self._record(r, "Fine-tuning", {"score": best_score}, t0); self.current_phase = None
    def honing(self, r):
        t0 = time.time(); self.current_phase = "Honing"; x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]; eps = self.radius * 0.01 + 1e-6
        sens = [abs(self._eval(np.clip(x+eps*np.eye(self.d)[i], self.bounds[:,0], self.bounds[:,1])) - best_score) for i in range(self.d)]
        target_dim, step = np.argmax(sens), self.radius * 0.05
        for s_ in np.linspace(-step, step, self.honing_samples):
            cand = x.copy(); cand[target_dim] = np.clip(x[target_dim] + s_, self.bounds[target_dim, 0], self.bounds[target_dim, 1])
            sc = self._eval(cand)
            if sc < best_score: best_score, x = sc, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score; self._record(r, "Honing", {"score": best_score}, t0); self.current_phase = None
    def validating(self, r):
        t0 = time.time(); self.current_phase = "Validating"; repeats = [self._eval(self.ctx["current_x"])] if self.is_deterministic else [self._eval(self.ctx["current_x"]) for _ in range(self.noise_repeats)]
        self.ctx["current_score"] = float(np.mean(repeats)); self._record(r, "Validating", {"mean": self.ctx["current_score"]}, t0); self.current_phase = None
    def iterating(self, round_, prev_best):
        t0 = time.time(); self.current_phase = "Iterating"; cur = self.ctx["current_score"]
        if cur < self.best_score_ever: self.best_score_ever, self.best_x_ever = cur, self.ctx["current_x"].copy()
        stalled = (prev_best - cur < self.tol) and prev_best != np.inf; self.stagnant_rounds = self.stagnant_rounds + 1 if stalled else 0
        hard_conv = stalled and (self.radius < self.full_radius * 0.01 or self.stagnant_rounds >= 3); hop = stalled and not hard_conv and self.stagnant_rounds >= 1
        reason = ""
        if hard_conv: action, reason = "stop", f"Converged: Stalled {self.stagnant_rounds}x at min radius."
        elif hop:
            action, reason = "basin_hop", f"Stalled {self.stagnant_rounds}x; hopping to new area."
            self.basin_archive.append({"center": self.center.copy(), "best_x": self.ctx["current_x"].copy(), "best_score": cur})
            for _ in range(10):
                cand = self._clip(self.bounds[:, 0] + self.rng.uniform(size=self.d) * (self.bounds[:, 1] - self.bounds[:, 0]))
                if not self.basin_archive or min(np.linalg.norm(cand - b["center"]) for b in self.basin_archive) > self.full_radius * 0.5: break
            self.center, self.radius, self.stagnant_rounds = cand, self.full_radius, 0
        else: action, reason, self.center, self.radius = "refine", "Improvement found; refining locally.", self.ctx["current_x"], self.radius * self.radius_decay
        self._record(round_, "Iterating", {"action": action, "reason": reason}, t0); self.current_phase = None; return hard_conv

    def _met_success(self) -> bool:
        if self.best_score_ever <= self.goal.target_score: return True
        if self.goal.success_criteria and self.best_x_ever is not None:
            return self.goal.success_criteria(self.best_score_ever, self.best_x_ever)
        return False

    def run_iterator(self) -> Iterator[dict]:
        if self.is_deterministic is None:
            self._probe_determinism()
            yield {"type": "phase", "phase": "Probing", "round": 0, "state": self.get_state()}

        best = np.inf
        for round_ in range(1, self.max_rounds + 1):
            while self.manual_pause: time.sleep(0.1)
            self.last_round = round_
            if self.eval_count >= self.budget: break
            if self._met_success(): break
            if round_ > 2: self._adapt_budget()

            for p in ["exploring", "sleuthing", "sifting", "figuring", "reckoning", "analyzing", "synthesizing", "crystallizing", "evaluating"]:
                p_name = p.capitalize()
                if p_name not in self.disabled_phases:
                    getattr(self, p)(round_)
                    yield {"type": "phase", "phase": p_name, "round": round_, "state": self.get_state()}
                else: yield {"type": "skip", "phase": p_name, "round": round_, "state": self.get_state()}

            for p in ["optimizing", "fine_tuning", "honing"]:
                p_name = p.capitalize().replace("_", "-")
                if not self._should_skip(p_name):
                    getattr(self, p.replace("-", "_"))(round_)
                    yield {"type": "phase", "phase": p_name, "round": round_, "state": self.get_state()}
                else:
                    yield {"type": "skip", "phase": p_name, "round": round_, "state": self.get_state()}

            self.validating(round_)
            yield {"type": "phase", "phase": "Validating", "round": round_, "state": self.get_state()}

            conv = self.iterating(round_, best)
            self.converged = conv
            yield {"type": "phase", "phase": "Iterating", "round": round_, "state": self.get_state()}

            best = self.ctx["current_score"]; self.ctx["best_score_so_far"] = best
            if self.callback: self.callback(self.get_state())
            if conv: break

    def run(self):
        for _ in self.run_iterator(): pass
        return self._finalize()

    def _finalize(self) -> dict:
        return {
            "best_x": self.best_x_ever.tolist() if self.best_x_ever is not None else None,
            "best_score": self.best_score_ever,
            "rounds_run": self.last_round,
            "converged": self.converged,
            "eval_count": self.eval_count,
            "is_deterministic": self.is_deterministic,
            "phase_stats": self.phase_stats,
            "seed": self.seed,
            "mode": self.mode,
            "basin_archive": self.basin_archive,
            "log": [asdict(l) for l in self.log]
        }

    def get_state(self) -> dict:
        return {
            "best_score": self.best_score_ever,
            "best_x": self.best_x_ever.tolist() if self.best_x_ever is not None else None,
            "evals": self.eval_count,
            "radius": self.radius,
            "current_phase": self.current_phase,
            "basin_count": len(self.basin_archive),
            "log": [asdict(l) for l in self.log[-5:]]
        }

    def export_data(self) -> dict:
        return {
            "config": {"mode": self.mode, "seed": self.seed, "budget": self.budget},
            "results": {"best_score": self.best_score_ever, "evals": self.eval_count},
            "history": [asdict(l) for l in self.log],
            "phase_stats": self.phase_stats,
            "basin_archive": self.basin_archive
        }

class GlobalResultTracker:
    def __init__(self):
        self.best_score = np.inf
        self.best_x = None
        self.lock = threading.Lock()
    def update(self, score, x):
        with self.lock:
            if score < self.best_score:
                self.best_score, self.best_x = score, x.copy()
    def get_best(self):
        with self.lock:
            return self.best_score, self.best_x

class TaskTyper:
    def __init__(self, objective_fn: Callable, bounds: np.ndarray, seed: int = 42):
        self.f, self.bounds, self.d, self.rng = objective_fn, bounds, bounds.shape[0], np.random.default_rng(seed)
    def characterize(self, n_probes: int = 20) -> Dict[str, Any]:
        pts = self.bounds[:, 0] + self.rng.uniform(size=(n_probes, self.d)) * (self.bounds[:, 1] - self.bounds[:, 0])
        v1, v2 = self.f(pts[0]), self.f(pts[0]); is_det = abs(v1 - v2) <= 1e-12 * max(1.0, abs(v1))
        noise_std = float(np.std([self.f(pts[0]) for _ in range(10)])) if not is_det else 0.0
        scores = np.array([self.f(p) for p in pts]); ruggedness = float(np.std(np.diff(scores)) / (np.std(scores) + 1e-9)) if len(scores) > 1 else 0.0
        is_discrete = all(np.all(np.isclose(p, np.round(p))) for p in pts)
        return {"is_deterministic": is_det, "noise_std": noise_std, "ruggedness": ruggedness, "is_discrete": is_discrete, "d": self.d}

class SolverOrchestrator:
    def __init__(
        self, goal: OptimizationGoal, budget: int = 5000, seed: int = 0,
        callback: Optional[Callable] = None, parallel: bool = False
    ):
        self.goal = goal
        self.budget, self.rng = budget, np.random.default_rng(seed)
        self.callback = callback
        self.parallel = parallel
        self.typer = TaskTyper(goal.objective, goal.bounds, seed=seed)
        self.best_x, self.best_score = None, np.inf
        self.portfolio_results = []
        self.optimization_report = []
        self.tracker = GlobalResultTracker()

    def run_iterator(self) -> Iterator[dict]:
        char = self.typer.characterize(10); used = 20; surrogate = DiscreteSurrogate() if char["is_discrete"] else None
        yield {"type": "characterization", "data": char}

        if not char["is_deterministic"] or char["noise_std"] > 1e-6: modes = ["NoisyMode", "GlobalExploring"]
        elif char["ruggedness"] > 1.2: modes = ["GlobalExploring", "SurrogateGuided", "LocalRefining"]
        elif char["d"] > 12: modes = ["GlobalExploring", "SurrogateGuided"]
        else: modes = ["SurrogateGuided", "LocalRefining"]

        b_per_mode = (self.budget - used) // len(modes); report = []
        report.append(f"Orchestrator: Portfolio selection: {modes}. Parallel={self.parallel}")
        self.optimization_report = report

        if self.parallel:
            with ThreadPoolExecutor(max_workers=len(modes)) as executor:
                futures = {}
                for mode in modes:
                    agent = NumericOptimizer(
                        self.goal, mode=mode, surrogate_model=surrogate,
                        budget=b_per_mode, seed=self.rng.integers(10000),
                        callback=self.callback, shared_tracker=self.tracker
                    )
                    futures[executor.submit(agent.run)] = mode
                for future in as_completed(futures):
                    res = future.result()
                    self.portfolio_results.append(res)
                    if res["best_score"] < self.best_score:
                        self.best_score, self.best_x = res["best_score"], res["best_x"]
                    yield {"type": "mode_end", "mode": futures[future], "best_score": self.best_score}
        else:
            for mode in modes:
                if used >= self.budget or self.best_score <= self.goal.target_score: break
                yield {"type": "mode_start", "mode": mode}
                agent = NumericOptimizer(
                    self.goal, mode=mode, surrogate_model=surrogate,
                    budget=b_per_mode, seed=self.rng.integers(10000),
                    callback=self.callback, shared_tracker=self.tracker
                )
                for msg in agent.run_iterator():
                    yield msg

                final_res = agent._finalize()
                used += final_res["eval_count"]
                self.portfolio_results.append(final_res)
                if final_res["best_score"] < self.best_score:
                    self.best_score, self.best_x = final_res["best_score"], final_res["best_x"]
                yield {"type": "mode_end", "mode": mode, "best_score": self.best_score}

    def run(self):
        for _ in self.run_iterator(): pass
        return {
            "best_x": self.best_x, "best_score": self.best_score, "evals": sum(r["eval_count"] for r in self.portfolio_results),
            "modes": [r["mode"] for r in self.portfolio_results], "portfolio_results": self.portfolio_results,
            "optimization_report": self.optimization_report
        }

    def save_report(self, path: str):
        data = {
            "best_score": self.best_score,
            "best_x": self.best_x,
            "report": self.optimization_report,
            "portfolio": [r for r in self.portfolio_results]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=NpEncoder)
