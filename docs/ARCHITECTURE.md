# Architecture: Numeric Optimization Engine

## 1. Solver Orchestrator (Governor Layer)
The top-level orchestrator is responsible for:
- **Task Typing**: Probing the objective function to estimate noise (std), ruggedness, and detecting discrete dimensions.
- **Portfolio Selection**: Choosing which operational modes to run (SurrogateGuided, GlobalExploring, LocalRefining, NoisyMode).
- **Budget Orchestration**: Dividing the total evaluation budget across the chosen portfolio.

## 2. Adaptive Meta-Controller
Inside every run, the meta-controller:
- **Utility Tracking**: Measures the improvement in objective score per evaluation for every phase.
- **Dynamic Scaling**: Increases the budget (e.g., number of samples) for high-utility phases and shrinks or skips low-utility ones.

## 3. 14-Phase Base Engine
The core execution loop:
1. **Probing**: Initial measurement.
2. **Exploring**: Broad divergent sampling.
3. **Sleuthing**: Deep-dive around top leads.
4. **Sifting**: Filtering the candidate pool.
5. **Figuring**: Fitting the pluggable surrogate model.
6. **Reckoning**: Proposing a new point using uncertainty-aware logic (LCB).
7. **Analyzing**: Statistical characterization.
8. **Synthesizing**: Merging candidates.
9. **Crystallizing**: Selecting the current best.
10. **Evaluating**: Scoring against the prior best.
11. **Optimizing**: Local coordinate descent.
12. **Fine-tuning**: Small-step random search.
13. **Honing**: Dimensional sensitivity refinement.
14. **Validating**: Noise-aware re-evaluation.
15. **Iterating**: Loop-back, refine, or basin-hop decision.

## Pluggable Surrogates
- **QuadraticSurrogate**: Ridge-regularized quadratic fit with automatic interaction fallback.
- **DiscreteSurrogate**: Neighborhood search for non-numeric domains.
