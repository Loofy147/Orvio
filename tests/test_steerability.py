import numpy as np
import pytest
from numeric_optimizer import NumericOptimizer, OptimizationGoal

def sphere(x):
    return np.sum(x**2)

def test_disable_phases():
    bounds = np.array([[-5.0, 5.0]] * 2)
    goal = OptimizationGoal(objective=sphere, bounds=bounds)
    agent = NumericOptimizer(goal, max_rounds=2)

    # Disable a phase
    agent.disabled_phases.add("Exploring")

    messages = list(agent.run_iterator())

    # Check that 'Exploring' was skipped
    skipped = [m['phase'] for m in messages if m['type'] == 'skip']
    assert "Exploring" in skipped

    # Check that 'Exploring' was never run (no phase message)
    run_phases = [m['phase'] for m in messages if m['type'] == 'phase']
    assert "Exploring" not in run_phases

def test_mid_run_budget_adjustment():
    bounds = np.array([[-5.0, 5.0]] * 2)
    goal = OptimizationGoal(objective=sphere, bounds=bounds)
    agent = NumericOptimizer(goal, budget=100, max_rounds=10)

    # Run partially
    it = agent.run_iterator()
    for _ in range(5):
        next(it)

    # Adjust budget
    agent.budget = 1000

    # Continue
    for _ in it: pass

    assert agent.eval_count > 100

def test_phase_utility_history():
    bounds = np.array([[-5.0, 5.0]] * 2)
    goal = OptimizationGoal(objective=sphere, bounds=bounds)
    agent = NumericOptimizer(goal, max_rounds=3)
    agent.run()

    # Check history existence
    for p in ["Exploring", "Sleuthing", "Optimizing"]:
        history = agent.phase_stats[p]["history"]
        assert len(history) > 0
        assert "improvement" in history[0]
