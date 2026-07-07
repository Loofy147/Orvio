# Pipeline Agent

14-phase execution loop, fixed order, with a loop-back from the last phase:

Exploring → Sleuthing → Sifting → Figuring → Reckoning → Analyzing →
Synthesizing → Crystallizing → Evaluating → Optimizing → Fine-tuning →
Honing → Validating → Iterating

Reference implementation: `pipeline_agent.py`. Each phase is a real
transformation on a candidate pool and a scalar objective — not a
description of intent. If a phase doesn't change the state in a checkable
way, it isn't implemented, it's decoration.

## What each phase does

- **Exploring** — broad divergent sampling at the current radius. No filtering.
- **Sleuthing** — investigative deep-dive: tighter local sampling around the top-k leads from Exploring.
- **Sifting** — discard the weak majority; keep the top fraction of the combined pool.
- **Figuring** — fit a local model (surrogate) to work out the shape of the landscape from the sifted points. Ridge-regularized full quadratic (with cross terms) by default, so it can represent coupling between dimensions, not just per-dimension curvature; toggle `model_interactions=False` to fall back to a diagonal-only surrogate.
- **Reckoning** — use the model to compute a projected improved point and its predicted/actual score.
- **Analyzing** — statistical characterization of the sifted population (mean, std, spread).
- **Synthesizing** — merge the raw-best and model-projected candidates into one pool.
- **Crystallizing** — collapse the pool to a single current solution.
- **Evaluating** — score the current solution against the prior round's best.
- **Optimizing** — local coordinate descent from the crystallized point.
- **Fine-tuning** — smaller-step random perturbation refinement.
- **Honing** — identify the single most sensitive dimension and refine only that.
- **Validating** — independent re-evaluation, but only spends repeated evals on noise-averaging when a determinism probe (two evals at the same point, once per run) actually finds noise. A deterministic objective gets one confirmation eval, not five.
- **Iterating** — decide: stop (converged), refine locally (shrink radius), or basin-hop (stagnation → wide restart). An elitist archive means basin-hopping can never lose the best solution found so far.

## Operating principles

- Every phase must produce a number or a decision that the next phase consumes. No phase is allowed to be purely descriptive.
- Convergence claims are checked against multiple random seeds, not one lucky run — a single run can land in a good basin by chance.
- When a run's result is bad, the fix is structural (e.g. the pipeline could only shrink its search radius and got permanently trapped) — not just parameter retuning. Retuning a broken structure is polishing the wrong thing.
- A richer model is not assumed to help just because it's richer: the full-quadratic surrogate was A/B-tested against the diagonal one on a benchmark chosen specifically to need it (Rosenbrock's coupled-dimension valley). The first version of that upgrade was actually *worse* on average — the surrogate was underdetermined (~1.1 points per parameter) and ridge regularization wasn't scaled to catch that, so cross-term coefficients were mostly fit noise. Fixed by requiring a real points-per-parameter margin in Sifting and making regularization strength continuous in that ratio, not a binary underdetermined/not switch. See `RESULTS.md`.
- Determinism is measured (two evals at the same point), not assumed — Validating spends its repeat-eval budget only where noise is actually present.
- Default hyperparameters (`max_rounds=20`, radius decay 0.6, sift keep-rate max(2x surrogate params, 30%)) are set from sweeps in `results/`, not guessed.

## Extending to a new problem

Supply an `objective_fn: (x: np.ndarray) -> float` to minimize and a
`bounds: (d, 2)` array. Everything else (exploration radius, restart
triggers, convergence) is derived from `bounds` and adapts automatically.
For a non-numeric problem (e.g. discrete/combinatorial), the phase *roles*
still apply, but `figuring`/`reckoning`'s surrogate-gradient step needs
replacing with a domain-appropriate local model.
