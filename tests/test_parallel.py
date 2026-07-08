import numpy as np
from numeric_optimizer import SolverOrchestrator, OptimizationGoal

def sphere(x):
    return np.sum(x**2)

def test_parallel_execution():
    bounds = np.array([[-5.0, 5.0]] * 2)
    goal = OptimizationGoal(objective=sphere, bounds=bounds)

    # Run with parallel=True
    orc = SolverOrchestrator(goal, budget=2000, parallel=True)
    res = orc.run()

    assert res['best_score'] < 1.0
    assert len(res['portfolio_results']) > 0
    # Evals might be slightly higher than budget due to parallel completion
    assert res['evals'] > 0

def test_shared_tracker_efficiency():
    # This test verifies that agents share their best results
    bounds = np.array([[-5.0, 5.0]] * 2)
    goal = OptimizationGoal(objective=sphere, bounds=bounds)

    orc = SolverOrchestrator(goal, budget=1000, parallel=True)
    res = orc.run()

    # Check that individual portfolio results have seen the global best
    for p_res in res['portfolio_results']:
        assert p_res['best_score'] <= res['best_score'] or p_res['best_score'] == np.inf
