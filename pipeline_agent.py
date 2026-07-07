"""
pipeline_agent.py

A 14-phase execution agent with an adaptive meta-controller and uncertainty-aware surrogate.
Exploring -> Sleuthing -> Sifting -> Figuring -> Reckoning -> Analyzing ->
Synthesizing -> Crystallizing -> Evaluating -> Optimizing -> Fine-tuning ->
Honing -> Validating -> Iterating (loop back or stop)
"""

from __future__ import annotations
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Callable, Any

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


class PipelineAgent:
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        seed: int = 0,
        max_rounds: int = 20,
        tol: float = 1e-6,
        explore_n: int = 40,
        sleuth_k: int = 5,
        sleuth_samples: int = 8,
        model_interactions: bool = True,
        noise_repeats: int = 5,
    ):
        self.f = objective_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.d = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.max_rounds = max_rounds
        self.tol = tol
        self.explore_n = explore_n
        self.sleuth_k = sleuth_k
        self.sleuth_samples = sleuth_samples
        self.model_interactions = model_interactions
        self.noise_repeats = noise_repeats

        self.adaptive_explore_n = explore_n
        self.adaptive_sleuth_samples = sleuth_samples
        self.opt_rounds = 3
        self.ft_rounds = 20
        self.honing_samples = 9

        self.n_surrogate_params = 1 + 2 * self.d + (
            self.d * (self.d - 1) // 2 if model_interactions else 0
        )

        self.log: list[PhaseLog] = []
        self.ctx: dict[str, Any] = {}
        self.full_radius = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0])) / 2.0
        self.radius = self.full_radius
        self.center = self.bounds.mean(axis=1)

        self.best_x_ever = None
        self.best_score_ever = np.inf
        self.stagnant_rounds = 0

        self.eval_count = 0
        self.current_phase = None
        self.phase_stats = {p: {"improvement": 0.0, "evals": 0} for p in PHASES}
        self.is_deterministic = None
        self.basin_archive = []

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _eval(self, x):
        score = float(self.f(x))
        self.eval_count += 1
        if self.current_phase:
            self.phase_stats[self.current_phase]["evals"] += 1
            if self.best_score_ever != np.inf and score < self.best_score_ever:
                self.phase_stats[self.current_phase]["improvement"] += (self.best_score_ever - score)

        if score < self.best_score_ever:
            self.best_score_ever = score
            self.best_x_ever = x.copy()
        return score

    def _probe_determinism(self):
        self.current_phase = "Probing"
        x0 = self.center
        a, b = self._eval(x0), self._eval(x0)
        self.is_deterministic = abs(a - b) <= 1e-12 * max(1.0, abs(a))
        self.current_phase = None

    def _record(self, round_, phase, summary, t0):
        self.log.append(PhaseLog(round_, phase, summary, time.time() - t0))

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
        if stats["evals"] < 50: return False
        if stats["improvement"] == 0: return True
        return False

    def exploring(self, round_):
        t0 = time.time()
        self.current_phase = "Exploring"
        n = self.adaptive_explore_n
        pts = self.center + self.rng.uniform(-1, 1, size=(n, self.d)) * self.radius
        pts = self._clip(pts)
        scores = np.array([self._eval(p) for p in pts])
        self.ctx["explore_pts"], self.ctx["explore_scores"] = pts, scores
        self._record(round_, "Exploring", {
            "n_samples": n, "radius": round(self.radius, 4),
            "best_score": float(scores.min()),
        }, t0)
        self.current_phase = None

    def sleuthing(self, round_):
        t0 = time.time()
        self.current_phase = "Sleuthing"
        pts, scores = self.ctx["explore_pts"], self.ctx["explore_scores"]
        idx = np.argsort(scores)[: self.sleuth_k]
        leads = pts[idx]
        local_r = self.radius * 0.2
        new_pts, new_scores = [], []
        n_samples = self.adaptive_sleuth_samples
        for lead in leads:
            cand = lead + self.rng.uniform(-1, 1, size=(n_samples, self.d)) * local_r
            cand = self._clip(cand)
            for c in cand:
                new_pts.append(c)
                new_scores.append(self._eval(c))
        new_pts, new_scores = np.array(new_pts), np.array(new_scores)
        self.ctx["sleuth_pts"] = np.vstack([pts, new_pts])
        self.ctx["sleuth_scores"] = np.concatenate([scores, new_scores])
        self._record(round_, "Sleuthing", {
            "leads_investigated": self.sleuth_k,
            "samples_per_lead": n_samples,
            "new_samples": len(new_scores),
            "best_score": float(self.ctx["sleuth_scores"].min()),
        }, t0)
        self.current_phase = None

    def sifting(self, round_):
        t0 = time.time()
        self.current_phase = "Sifting"
        pts, scores = self.ctx["sleuth_pts"], self.ctx["sleuth_scores"]
        keep_n = max(2 * self.n_surrogate_params, int(0.3 * len(scores)))
        keep_n = min(keep_n, len(scores))
        idx = np.argsort(scores)[:keep_n]
        self.ctx["sift_pts"], self.ctx["sift_scores"] = pts[idx], scores[idx]
        self._record(round_, "Sifting", {
            "kept": keep_n, "discarded": len(scores) - keep_n,
            "points_per_param": round(keep_n / self.n_surrogate_params, 2),
            "best_score": float(scores[idx].min()),
        }, t0)
        self.current_phase = None

    def _get_feats(self, X):
        n, d = X.shape
        feats = [np.ones((n, 1)), X, X ** 2]
        cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if self.model_interactions and cross_idx:
            cross = np.column_stack([X[:, i] * X[:, j] for i, j in cross_idx])
            feats.append(cross)
        return np.hstack(feats)

    def figuring(self, round_):
        t0 = time.time()
        self.current_phase = "Figuring"
        X, y = self.ctx["sift_pts"], self.ctx["sift_scores"]
        n, d = X.shape
        A = self._get_feats(X)
        n_params = A.shape[1]
        ratio = n / n_params
        lam = 0.05 / max(ratio - 1.0, 0.05)
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n_params)])
        y_aug = np.concatenate([y, np.zeros(n_params)])
        coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)

        residuals = y - A @ coef
        sigma2 = np.sum(residuals**2) / max(n - n_params, 1)
        cov = sigma2 * np.linalg.inv(A.T @ A + (lam + 1e-6) * np.eye(n_params))

        self.ctx["surrogate_coef"] = coef
        self.ctx["surrogate_cov"] = cov

        best_x = X[np.argmin(y)]
        b_lin = coef[1 : 1 + d]
        b_quad = coef[1 + d : 1 + 2 * d]
        grad = b_lin + 2 * b_quad * best_x
        cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if self.model_interactions and cross_idx:
            b_cross = coef[1 + 2 * d :]
            for (i, j), c in zip(cross_idx, b_cross):
                grad[i] += c * best_x[j]
                grad[j] += c * best_x[i]
        self.ctx["surrogate_grad_at_best"] = grad
        self.ctx["figuring_base_x"] = best_x
        self._record(round_, "Figuring", {
            "n_params": n_params, "n_points": n, "ridge_lambda": lam,
            "grad_norm": float(np.linalg.norm(grad)),
        }, t0)
        self.current_phase = None

    def reckoning(self, round_):
        t0 = time.time()
        self.current_phase = "Reckoning"
        base_x = self.ctx["figuring_base_x"]
        grad = self.ctx["surrogate_grad_at_best"]
        gnorm = np.linalg.norm(grad) + 1e-12
        coef = self.ctx["surrogate_coef"]
        cov = self.ctx["surrogate_cov"]

        # Uncertainty-aware selection (LCB)
        # We sample a few points along the gradient and pick the one with best score - 2*sigma
        steps = np.linspace(0.1, 0.5, 5) * self.radius
        proposals = [self._clip(base_x - s * grad / gnorm) for s in steps]

        best_lcb = np.inf
        best_p = proposals[0]

        for p in proposals:
            phi = self._get_feats(p[None, :])[0]
            pred = phi @ coef
            sigma = np.sqrt(max(0, phi @ cov @ phi))
            lcb = pred - 2.0 * sigma
            if lcb < best_lcb:
                best_lcb, best_p = lcb, p

        self.ctx["reckoning_proposal"] = best_p
        self.ctx["reckoning_proposal_true_score"] = self._eval(best_p)
        self._record(round_, "Reckoning", {
            "proposal_score": self.ctx["reckoning_proposal_true_score"],
            "lcb_estimated": float(best_lcb),
        }, t0)
        self.current_phase = None

    def analyzing(self, round_):
        t0 = time.time()
        self.current_phase = "Analyzing"
        scores = self.ctx["sift_scores"]
        stats = {
            "mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "max": float(scores.max()),
        }
        self.ctx["analysis_stats"] = stats
        self._record(round_, "Analyzing", stats, t0)
        self.current_phase = None

    def synthesizing(self, round_):
        t0 = time.time()
        self.current_phase = "Synthesizing"
        pts = np.vstack([self.ctx["sift_pts"], self.ctx["reckoning_proposal"][None, :]])
        scores = np.concatenate([self.ctx["sift_scores"], [self.ctx["reckoning_proposal_true_score"]]])
        self.ctx["synth_pts"], self.ctx["synth_scores"] = pts, scores
        self._record(round_, "Synthesizing", {"pool_size": len(scores)}, t0)
        self.current_phase = None

    def crystallizing(self, round_):
        t0 = time.time()
        self.current_phase = "Crystallizing"
        pts, scores = self.ctx["synth_pts"], self.ctx["synth_scores"]
        i = int(np.argmin(scores))
        self.ctx["current_x"] = pts[i]
        self.ctx["current_score"] = float(scores[i])
        self._record(round_, "Crystallizing", {"selected_score": self.ctx["current_score"]}, t0)
        self.current_phase = None

    def evaluating(self, round_):
        t0 = time.time()
        self.current_phase = "Evaluating"
        prev = self.ctx.get("best_score_so_far", np.inf)
        cur = self.ctx["current_score"]
        self.ctx["evaluation_improvement"] = prev - cur
        self._record(round_, "Evaluating", {"improvement": prev - cur}, t0)
        self.current_phase = None

    def optimizing(self, round_):
        t0 = time.time()
        self.current_phase = "Optimizing"
        x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]
        step = self.radius * 0.15
        for _ in range(self.opt_rounds):
            for i in range(self.d):
                for sign in (+1, -1):
                    cand = x.copy()
                    cand[i] = np.clip(cand[i] + sign * step, self.bounds[i, 0], self.bounds[i, 1])
                    s = self._eval(cand)
                    if s < best_score: best_score, x = s, cand
            step *= 0.6
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Optimizing", {"score_after": best_score}, t0)
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
        self._record(round_, "Fine-tuning", {"score_after": best_score}, t0)
        self.current_phase = None

    def honing(self, round_):
        t0 = time.time()
        self.current_phase = "Honing"
        x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]
        eps = self.radius * 0.01 + 1e-6
        sens = np.zeros(self.d)
        for i in range(self.d):
            xp = x.copy(); xp[i] = np.clip(xp[i] + eps, self.bounds[i, 0], self.bounds[i, 1])
            sens[i] = abs(self._eval(xp) - best_score)
        target_dim = int(np.argmax(sens))
        step = self.radius * 0.05
        for s_ in np.linspace(-step, step, self.honing_samples):
            cand = x.copy()
            cand[target_dim] = np.clip(cand[target_dim] + s_, self.bounds[target_dim, 0], self.bounds[target_dim, 1])
            sc = self._eval(cand)
            if sc < best_score: best_score, x = sc, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Honing", {"score_after": best_score}, t0)
        self.current_phase = None

    def validating(self, round_):
        t0 = time.time()
        self.current_phase = "Validating"
        x = self.ctx["current_x"]
        repeats = [self._eval(x)] if self.is_deterministic else [self._eval(x) for _ in range(self.noise_repeats)]
        mean_score = float(np.mean(repeats))
        self.ctx["current_score"] = mean_score
        self._record(round_, "Validating", {"mean": mean_score}, t0)
        self.current_phase = None

    def iterating(self, round_, prev_best):
        t0 = time.time()
        self.current_phase = "Iterating"
        cur = self.ctx["current_score"]
        if cur < self.best_score_ever:
            self.best_score_ever = cur
            self.best_x_ever = self.ctx["current_x"].copy()
        delta = prev_best - cur if prev_best != np.inf else float("inf")
        stalled = delta < self.tol and prev_best != np.inf
        self.stagnant_rounds = self.stagnant_rounds + 1 if stalled else 0
        hard_converged = stalled and self.radius < self.full_radius * 0.02
        restart = stalled and not hard_converged and self.stagnant_rounds >= 1
        if hard_converged:
            action = "stop"
        elif restart:
            action = "basin_hop"
            self.basin_archive.append(self.center.copy())
            for _ in range(10):
                cand = self._clip(self.bounds[:, 0] + self.rng.uniform(size=self.d) * (self.bounds[:, 1] - self.bounds[:, 0]))
                if not self.basin_archive or min(np.linalg.norm(cand - b) for b in self.basin_archive) > self.full_radius * 0.5:
                    break
            self.center = cand
            self.radius = self.full_radius
            self.stagnant_rounds = 0
        else:
            action = "refine"
            self.center = self.ctx["current_x"]
            self.radius *= 0.6
        self._record(round_, "Iterating", {"action": action}, t0)
        self.current_phase = None
        return hard_converged

    def run(self):
        if self.is_deterministic is None: self._probe_determinism()
        best = np.inf
        for round_ in range(1, self.max_rounds + 1):
            if round_ > 2: self._adapt_budget()
            self.exploring(round_)
            self.sleuthing(round_)
            self.sifting(round_)
            self.figuring(round_)
            self.reckoning(round_)
            self.analyzing(round_)
            self.synthesizing(round_)
            self.crystallizing(round_)
            self.evaluating(round_)
            if not self._should_skip("Optimizing"): self.optimizing(round_)
            if not self._should_skip("Fine-tuning"): self.fine_tuning(round_)
            if not self._should_skip("Honing"): self.honing(round_)
            self.validating(round_)
            converged = self.iterating(round_, best)
            best = self.ctx["current_score"]
            self.ctx["best_score_so_far"] = best
            if converged: break
        return {
            "best_x": self.best_x_ever.tolist() if self.best_x_ever is not None else None,
            "best_score": self.best_score_ever, "rounds_run": round_,
            "converged": converged, "eval_count": self.eval_count,
            "is_deterministic": self.is_deterministic, "phase_stats": self.phase_stats,
        }

    def dump_log(self, path):
        with open(path, "w") as fh: json.dump([asdict(p) for p in self.log], fh, indent=2, default=str)
