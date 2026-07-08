import numpy as np
from numeric_optimizer import NumericOptimizer, OptimizationGoal

def multi_modal(x):
    return np.sum(np.sin(5*x) + x**2)

def test_basin_hopping():
    d = 2
    bounds = np.array([[-2.0, 2.0]] * d)
    goal = OptimizationGoal(objective=multi_modal, bounds=bounds)
    agent = NumericOptimizer(goal, max_rounds=20, tol=1e-1)
    result = agent.run()
    assert 'basin_archive' in result
