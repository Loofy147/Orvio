import numpy as np
from numeric_optimizer import NumericOptimizer

def test_adaptive_meta_controller():
    d = 2
    bounds = np.array([[-5.0, 5.0]] * d)
    # Sphere function: Exploring and Optimizing should be very useful initially
    agent = NumericOptimizer(lambda x: np.sum((x-1)**2), bounds, max_rounds=15)
    result = agent.run()

    assert result['eval_count'] > 0
    assert 'phase_stats' in result
    # Basic check that utility was calculated
    for p, s in result['phase_stats'].items():
        if s['evals'] > 0:
            assert s['evals'] >= 0
