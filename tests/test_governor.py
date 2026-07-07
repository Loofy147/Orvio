import numpy as np
from numeric_optimizer import SolverOrchestrator

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def noisy_sphere(x):
    return np.sum(x**2) + np.random.normal(0, 0.01)

def test_orchestrator_deterministic():
    bounds = np.array([[-2.0, 2.0]] * 2)
    gov = SolverOrchestrator(rosenbrock, bounds, budget=2000)
    res = gov.run()
    assert res['best_score'] < 10.0
    assert len(res['modes']) > 0

def test_orchestrator_noisy():
    bounds = np.array([[-2.0, 2.0]] * 2)
    gov = SolverOrchestrator(noisy_sphere, bounds, budget=2000)
    res = gov.run()
    assert 'NoisyMode' in res['modes'] or 'GlobalExploring' in res['modes']
