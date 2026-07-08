import numpy as np
import pytest
from numeric_optimizer import NumericOptimizer, SolverOrchestrator, OptimizationGoal

def sphere(x):
    return np.sum(x**2)

def test_constrained_optimizer():
    d = 2
    bounds = np.array([[-5.0, 5.0]] * d)
    constraints = [lambda x: 2.0 - np.sum(x)]
    goal = OptimizationGoal(objective=sphere, bounds=bounds, constraints=constraints)

    agent = NumericOptimizer(goal, max_rounds=20)
    result = agent.run()

    assert result['best_score'] >= 1.9
    best_x = np.array(result['best_x'])
    assert np.sum(best_x) > 1.8

def test_orchestrator_with_constraints():
    bounds = np.array([[-5.0, 5.0]] * 2)
    constraints = [lambda x: 2.0 - np.sum(x)]
    goal = OptimizationGoal(objective=sphere, bounds=bounds, constraints=constraints)

    orchestrator = SolverOrchestrator(goal, budget=3000)
    res = orchestrator.run()

    assert res['best_score'] >= 1.9
    best_x = np.array(res['best_x'])
    assert np.sum(best_x) > 1.8
