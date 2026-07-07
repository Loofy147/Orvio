import numpy as np
import pytest
from numeric_optimizer import NumericOptimizer, SolverOrchestrator

def sphere(x):
    return np.sum(x**2)

def test_constrained_optimizer():
    # Minimize x^2 + y^2 subject to x + y >= 2
    # Optimum at x=1, y=1, score=2
    d = 2
    bounds = np.array([[-5.0, 5.0]] * d)

    # Constraint: x + y >= 2  =>  2 - (x + y) <= 0
    constraints = [lambda x: 2.0 - np.sum(x)]

    agent = NumericOptimizer(sphere, bounds, constraints=constraints, max_rounds=20)
    result = agent.run()

    # Allow some slack for penalty method
    assert result['best_score'] >= 1.9
    best_x = np.array(result['best_x'])
    assert np.sum(best_x) > 1.8

def test_orchestrator_with_constraints():
    bounds = np.array([[-5.0, 5.0]] * 2)
    constraints = [lambda x: 2.0 - np.sum(x)]

    orchestrator = SolverOrchestrator(sphere, bounds, budget=3000, constraints=constraints)
    res = orchestrator.run()

    assert res['best_score'] >= 1.9
    best_x = np.array(res['best_x'])
    assert np.sum(best_x) > 1.8
