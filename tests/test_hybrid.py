import numpy as np
from numeric_optimizer import SolverOrchestrator, OptimizationGoal

def discrete_sphere(x):
    return np.sum(np.round(x)**2)

def test_discrete_optimization():
    bounds = np.array([[0.0, 10.0]] * 2)
    goal = OptimizationGoal(objective=discrete_sphere, bounds=bounds)
    gov = SolverOrchestrator(goal, budget=1000)
    res = gov.run()
    assert res['best_score'] >= 0
    assert res['best_x'] is not None
