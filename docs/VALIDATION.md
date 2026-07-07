# Validation and Verification

The Numeric Optimization Engine is validated through multiple layers of verification to ensure its reliability and effectiveness.

## 1. Unit & Functional Tests
Located in the `tests/` directory, these scripts verify individual components and overall engine logic:
- `test_agent.py`: Core 14-phase logic and state transitions.
- `test_adaptive.py`: Verifies that the Meta-Controller correctly scales budgets based on utility.
- `test_diversity.py`: Ensures basin-hopping explores new regions using the archive.
- `test_governor.py`: Validates the Orchestrator's strategy selection on smooth vs. noisy problems.
- `test_hybrid.py`: Confirms pluggable surrogate logic for discrete domains.

## 2. Standard Benchmarks
The `examples/benchmark_harness.py` script runs the engine on a suite of standard test functions (Sphere, Rosenbrock, Ackley) across multiple dimensions and random seeds.
**Success Criteria**:
- Consistently reaching global minima on convex/smooth functions.
- Escaping local minima on multimodal landscapes (Ackley).
- Low variance in results across multiple seeds.

## 3. Practical Application
The `examples/tune_controller.py` file demonstrates the engine's utility on a real-world engineering problem: tuning a PID controller to minimize Integral Absolute Error (IAE) in a dynamic simulation.

## 4. Auditability (Decision Records)
Every run of the `SolverOrchestrator` produces a structured `optimization_report`. This report serves as a human-readable verification log, explaining:
- Why certain search modes were chosen (based on Task Typing).
- Detected objective properties (noise level, ruggedness).
- Final portfolio results and evaluation efficiency.
