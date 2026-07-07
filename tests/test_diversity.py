import numpy as np
from numeric_optimizer import NumericOptimizer

def multi_modal(x):
    return np.sum(np.sin(5*x) + x**2)

def test_basin_hopping():
    d = 2
    bounds = np.array([[-2.0, 2.0]] * d)
    agent = NumericOptimizer(multi_modal, bounds, max_rounds=20, tol=1e-1)
    result = agent.run()
    # Ensure basin archive exists
    assert 'basin_archive' in result
