# Numeric Optimization Engine (Orvio)

A high-performance, hierarchical framework for solving complex numeric optimization problems. Designed for scenarios where search, modeling, and validation must be combined to find reliable solutions efficiently.

## Core Architecture

- **Solver Orchestrator**: High-level layer that analyzes problem characteristics (Task Typing) and selects the optimal search strategy portfolio.
- **Adaptive Meta-Controller**: Mid-level layer that monitors phase-level utility and dynamically reallocates evaluation budgets.
- **14-Phase Engine**: The base execution loop implementing a rigorous search-and-refine pipeline with pluggable surrogates.

## Key Features

- **Automated Task Typing**: Detects noise levels, ruggedness, and dimensionality to inform strategy.
- **Pluggable Surrogates**: Support for Ridge-regularized Quadratic models and Discrete neighborhood search.
- **Uncertainty-Aware Search**: Uses Lower Confidence Bound (LCB) logic for robust point selection.
- **Auditability**: Generates structured Optimization Reports explaining every strategic decision.
- **Elitist Basin Archive**: Ensures exploration diversity without losing the global best solution.

## Getting Started

See `examples/tune_controller.py` for a real-world PID tuning application, or run `examples/benchmark_harness.py` to see the engine in action on standard problems.
