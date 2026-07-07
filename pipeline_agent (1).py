"""
pipeline_agent.py

A 14-phase execution agent:
Exploring -> Sleuthing -> Sifting -> Figuring -> Reckoning -> Analyzing ->
Synthesizing -> Crystallizing -> Evaluating -> Optimizing -> Fine-tuning ->
Honing -> Validating -> Iterating (loop back or stop)

Design choice: every phase does real, checkable work on a candidate pool and
a scalar objective. Nothing here is descriptive placeholder text -- each
phase transforms actual numbers, and the run log records exactly what each
phase produced, so the pipeline's behavior can be judged from its output
rather than from how it's described.

Pluggable target: any (dim,) -> float objective to MINIMIZE.
"""

from __future__ import annotations
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Callable, Any

PHASES = [
    "Exploring", "Sleuthing", "Sifting", "Figuring", "Reckoning",
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
        self.bounds = np.asarray(bounds, dtype=float)  # shape (d, 2)
        self.d = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.max_rounds = max_rounds
        self.tol = tol
        self.explore_n = explore_n
        self.sleuth_k = sleuth_k
        self.sleuth_samples = sleuth_samples
        self.model_interactions = model_interactions
        self.noise_repeats = noise_repeats

        # surrogate parameter count drives how many points Sifting must keep:
        # diagonal quadratic = 1 + 2d terms; full quadratic adds d(d-1)/2 cross terms.
        self.n_surrogate_params = 1 + 2 * self.d + (
            self.d * (self.d - 1) // 2 if model_interactions else 0
        )

        self.log: list[PhaseLog] = []
        self.ctx: dict[str, Any] = {}
        self.full_radius = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0])) / 2.0
        self.radius = self.full_radius
        self.center = self.bounds.mean(axis=1)

        # elitist archive: the pipeline is allowed to wander away from a good
        # basin (to escape it) without ever reporting a worse final answer.
        self.best_x_ever = None
        self.best_score_ever = np.inf
        self.stagnant_rounds = 0

        self.eval_count = 0
        # unknown until probed empirically in run() -- never assumed.
        self.is_deterministic = None

    # ---------- infra ----------

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _eval(self, x):
        self.eval_count += 1
        return float(self.f(x))

    def _probe_determinism(self):
        """Evaluate the same point twice and check for exact/near-exact agreement.
        This replaces assuming noise exists (or doesn't) with a direct measurement."""
        x0 = self.center
        a, b = self._eval(x0), self._eval(x0)
        self.is_deterministic = abs(a - b) <= 1e-12 * max(1.0, abs(a))

    def _record(self, round_, phase, summary, t0):
        self.log.append(PhaseLog(round_, phase, summary, time.time() - t0))

    # ---------- the 14 phases ----------

    def exploring(self, round_):
        """Broad divergent sampling around current center at current radius."""
        t0 = time.time()
        pts = self.center + self.rng.uniform(-1, 1, size=(self.explore_n, self.d)) * self.radius
        pts = self._clip(pts)
        scores = np.array([self._eval(p) for p in pts])
        self.ctx["explore_pts"], self.ctx["explore_scores"] = pts, scores
        self._record(round_, "Exploring", {
            "n_samples": self.explore_n, "radius": round(self.radius, 4),
            "best_score": float(scores.min()),
        }, t0)

    def sleuthing(self, round_):
        """Investigate the top-k leads with tighter local sampling."""
        t0 = time.time()
        pts, scores = self.ctx["explore_pts"], self.ctx["explore_scores"]
        idx = np.argsort(scores)[: self.sleuth_k]
        leads = pts[idx]
        local_r = self.radius * 0.2
        new_pts, new_scores = [], []
        for lead in leads:
            cand = lead + self.rng.uniform(-1, 1, size=(self.sleuth_samples, self.d)) * local_r
            cand = self._clip(cand)
            for c in cand:
                new_pts.append(c)
                new_scores.append(self._eval(c))
        new_pts, new_scores = np.array(new_pts), np.array(new_scores)
        self.ctx["sleuth_pts"] = np.vstack([pts, new_pts])
        self.ctx["sleuth_scores"] = np.concatenate([scores, new_scores])
        self._record(round_, "Sleuthing", {
            "leads_investigated": self.sleuth_k,
            "new_samples": len(new_scores),
            "best_score": float(self.ctx["sleuth_scores"].min()),
        }, t0)

    def sifting(self, round_):
        """Discard the weak majority; keep enough of the pool to fit the surrogate
        with a real safety margin (2x its parameter count), or the top 30%,
        whichever is larger. A barely-determined fit (~1 point/param) lets
        ridge regularization mask overfitting rather than prevent it."""
        t0 = time.time()
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

    def figuring(self, round_):
        """Fit a local surrogate to work out the shape of the landscape.

        Diagonal-only quadratics can't see interactions between dimensions --
        they're blind to curved ridges where two variables are coupled (e.g.
        Rosenbrock's valley). When model_interactions=True, cross terms
        x_i*x_j are added so the surrogate can represent that coupling.
        This roughly triples the parameter count for d=5, so the fit is
        ridge-regularized (not plain lstsq) to stay stable even when Sifting
        hasn't produced enough points to make the system well-determined.
        """
        t0 = time.time()
        X, y = self.ctx["sift_pts"], self.ctx["sift_scores"]
        n, d = X.shape

        feats = [np.ones((n, 1)), X, X ** 2]
        cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if self.model_interactions and cross_idx:
            cross = np.column_stack([X[:, i] * X[:, j] for i, j in cross_idx])
            feats.append(cross)
        else:
            cross_idx = []
        A = np.hstack(feats)
        n_params = A.shape[1]

        # ridge regularization scaled continuously by how many points are
        # available per parameter -- a system that's technically solvable
        # (n >= n_params) but only barely (e.g. ratio 1.1x) still overfits
        # badly on cross terms; a binary "underdetermined or not" check
        # missed exactly that case.
        ratio = n / n_params
        lam = 0.05 / max(ratio - 1.0, 0.05)
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n_params)])
        y_aug = np.concatenate([y, np.zeros(n_params)])
        coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)

        self.ctx["surrogate_coef"] = coef
        self.ctx["surrogate_cross_idx"] = cross_idx

        best_x = X[np.argmin(y)]
        b_lin = coef[1 : 1 + d]
        b_quad = coef[1 + d : 1 + 2 * d]
        grad = b_lin + 2 * b_quad * best_x
        if cross_idx:
            b_cross = coef[1 + 2 * d :]
            for (i, j), c in zip(cross_idx, b_cross):
                grad[i] += c * best_x[j]
                grad[j] += c * best_x[i]

        self.ctx["surrogate_grad_at_best"] = grad
        self.ctx["figuring_base_x"] = best_x
        self._record(round_, "Figuring", {
            "model": "full_quadratic" if cross_idx else "diagonal_quadratic",
            "n_params": n_params, "n_points": n, "ridge_lambda": lam,
            "grad_norm": float(np.linalg.norm(grad)),
        }, t0)

    def reckoning(self, round_):
        """Compute a predicted improved point from the surrogate gradient and project its score."""
        t0 = time.time()
        base_x = self.ctx["figuring_base_x"]
        grad = self.ctx["surrogate_grad_at_best"]
        gnorm = np.linalg.norm(grad) + 1e-12
        step = self.radius * 0.3
        proposal = self._clip(base_x - step * grad / gnorm)
        self.ctx["reckoning_proposal"] = proposal
        self.ctx["reckoning_proposal_true_score"] = self._eval(proposal)
        self._record(round_, "Reckoning", {
            "step_size": round(step, 4),
            "proposal_score": self.ctx["reckoning_proposal_true_score"],
        }, t0)

    def analyzing(self, round_):
        """Characterize the sifted population statistically."""
        t0 = time.time()
        scores = self.ctx["sift_scores"]
        stats = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "spread_ratio": float(scores.std() / (abs(scores.mean()) + 1e-9)),
        }
        self.ctx["analysis_stats"] = stats
        self._record(round_, "Analyzing", stats, t0)

    def synthesizing(self, round_):
        """Merge raw-best and surrogate-projected candidates into one pool."""
        t0 = time.time()
        pts = np.vstack([self.ctx["sift_pts"], self.ctx["reckoning_proposal"][None, :]])
        scores = np.concatenate([self.ctx["sift_scores"], [self.ctx["reckoning_proposal_true_score"]]])
        self.ctx["synth_pts"], self.ctx["synth_scores"] = pts, scores
        self._record(round_, "Synthesizing", {
            "pool_size": len(scores), "best_score": float(scores.min()),
        }, t0)

    def crystallizing(self, round_):
        """Collapse the synthesized pool into a single current-best solution."""
        t0 = time.time()
        pts, scores = self.ctx["synth_pts"], self.ctx["synth_scores"]
        i = int(np.argmin(scores))
        self.ctx["current_x"] = pts[i]
        self.ctx["current_score"] = float(scores[i])
        self._record(round_, "Crystallizing", {
            "selected_score": self.ctx["current_score"],
        }, t0)

    def evaluating(self, round_):
        """Score the crystallized solution against the previous round's best."""
        t0 = time.time()
        prev = self.ctx.get("best_score_so_far", np.inf)
        cur = self.ctx["current_score"]
        improvement = prev - cur
        self.ctx["evaluation_improvement"] = improvement
        self._record(round_, "Evaluating", {
            "previous_best": None if prev == np.inf else prev,
            "current": cur, "improvement": None if prev == np.inf else improvement,
        }, t0)

    def optimizing(self, round_):
        """Local coordinate descent from the crystallized point."""
        t0 = time.time()
        x = self.ctx["current_x"].copy()
        best_score = self.ctx["current_score"]
        step = self.radius * 0.15
        for _ in range(3):
            for i in range(self.d):
                for sign in (+1, -1):
                    cand = x.copy()
                    cand[i] = np.clip(cand[i] + sign * step, self.bounds[i, 0], self.bounds[i, 1])
                    s = self._eval(cand)
                    if s < best_score:
                        best_score, x = s, cand
            step *= 0.6
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Optimizing", {"score_after": best_score}, t0)

    def fine_tuning(self, round_):
        """Small-step random perturbation search, finer than exploring/optimizing."""
        t0 = time.time()
        x, best_score = self.ctx["current_x"], self.ctx["current_score"]
        step = self.radius * 0.03
        for _ in range(20):
            cand = self._clip(x + self.rng.normal(0, step, size=self.d))
            s = self._eval(cand)
            if s < best_score:
                best_score, x = s, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Fine-tuning", {"score_after": best_score}, t0)

    def honing(self, round_):
        """Refine only the single most sensitive dimension."""
        t0 = time.time()
        x, best_score = self.ctx["current_x"].copy(), self.ctx["current_score"]
        eps = self.radius * 0.01 + 1e-6
        sens = np.zeros(self.d)
        for i in range(self.d):
            xp = x.copy(); xp[i] = np.clip(xp[i] + eps, self.bounds[i, 0], self.bounds[i, 1])
            sens[i] = abs(self._eval(xp) - best_score)
        target_dim = int(np.argmax(sens))
        step = self.radius * 0.05
        for s_ in np.linspace(-step, step, 9):
            cand = x.copy()
            cand[target_dim] = np.clip(cand[target_dim] + s_, self.bounds[target_dim, 0], self.bounds[target_dim, 1])
            sc = self._eval(cand)
            if sc < best_score:
                best_score, x = sc, cand
        self.ctx["current_x"], self.ctx["current_score"] = x, best_score
        self._record(round_, "Honing", {"target_dim": target_dim, "score_after": best_score}, t0)

    def validating(self, round_):
        """Independent re-evaluation. Only spends repeated evals on averaging
        when the objective is measurably noisy -- for a deterministic
        objective, repeating the same point five times confirms nothing and
        just burns eval budget."""
        t0 = time.time()
        x = self.ctx["current_x"]
        if self.is_deterministic:
            repeats = [self._eval(x)]
        else:
            repeats = [self._eval(x) for _ in range(self.noise_repeats)]
        mean_score = float(np.mean(repeats))
        self.ctx["current_score"] = mean_score
        self.ctx["validation_std"] = float(np.std(repeats))
        self._record(round_, "Validating", {
            "deterministic": self.is_deterministic,
            "repeats": repeats, "mean": mean_score, "std": self.ctx["validation_std"],
        }, t0)

    def iterating(self, round_, prev_best):
        """
        Decide: converged (stop), keep refining locally, or basin-hop.

        Uses an elitist archive so wandering away from the current basin to
        search elsewhere never loses the best solution found so far -- a
        stagnant round (no improvement) triggers a wide restart instead of
        just shrinking the radius forever, which is what let earlier runs
        get permanently trapped in the first basin they found.
        """
        t0 = time.time()
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
            self.center = self._clip(
                self.bounds[:, 0] + self.rng.uniform(size=self.d) * (self.bounds[:, 1] - self.bounds[:, 0])
            )
            self.radius = self.full_radius
            self.stagnant_rounds = 0
        else:
            action = "refine"
            self.center = self.ctx["current_x"]
            self.radius *= 0.6

        self._record(round_, "Iterating", {
            "delta": None if delta == float("inf") else delta,
            "action": action, "next_radius": self.radius,
            "best_score_ever": self.best_score_ever,
        }, t0)
        return hard_converged

    # ---------- driver ----------

    def run(self):
        if self.is_deterministic is None:
            self._probe_determinism()
        best = np.inf
        for round_ in range(1, self.max_rounds + 1):
            self.exploring(round_)
            self.sleuthing(round_)
            self.sifting(round_)
            self.figuring(round_)
            self.reckoning(round_)
            self.analyzing(round_)
            self.synthesizing(round_)
            self.crystallizing(round_)
            self.evaluating(round_)
            self.optimizing(round_)
            self.fine_tuning(round_)
            self.honing(round_)
            self.validating(round_)
            converged = self.iterating(round_, best)
            best = self.ctx["current_score"]
            self.ctx["best_score_so_far"] = best
            if converged:
                break
        return {
            "best_x": self.best_x_ever.tolist(),
            "best_score": self.best_score_ever,
            "rounds_run": round_,
            "converged": converged,
            "eval_count": self.eval_count,
            "is_deterministic": self.is_deterministic,
            "model_interactions": self.model_interactions,
        }

    def dump_log(self, path):
        with open(path, "w") as fh:
            json.dump([asdict(p) for p in self.log], fh, indent=2, default=str)
