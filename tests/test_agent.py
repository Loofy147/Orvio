import numpy as np
from numeric_optimizer import NumericOptimizer, OptimizationGoal

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def test_rosenbrock_optimization():
    d = 2
    bounds = np.array([[-2.0, 2.0]] * d)
    goal = OptimizationGoal(objective=rosenbrock, bounds=bounds)
    agent = NumericOptimizer(goal, max_rounds=10)
    result = agent.run()
    assert result['best_score'] < 1.0
    assert result['eval_count'] > 0
