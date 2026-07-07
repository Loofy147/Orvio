import numpy as np
import pytest
from numeric_optimizer import NumericOptimizer, SolverOrchestrator

def sphere(x):
    return np.sum(x**2)

def test_target_score_early_exit():
    bounds = np.array([[-5.0, 5.0]] * 2)
    # Stop as soon as we find anything < 10
    agent = NumericOptimizer(sphere, bounds, target_score=10.0, max_rounds=5)
    res = agent.run()
    assert res['best_score'] <= 10.0
    # Should exit much earlier than full run
    assert res['rounds_run'] < 5

def test_callback_functionality():
    bounds = np.array([[-5.0, 5.0]] * 2)
    called_count = 0
    def my_callback(state):
        nonlocal called_count
        called_count += 1
        assert 'best_score' in state

    agent = NumericOptimizer(sphere, bounds, callback=my_callback, max_rounds=2)
    agent.run()
    assert called_count > 0

def test_run_iterator_api():
    bounds = np.array([[-5.0, 5.0]] * 2)
    agent = NumericOptimizer(sphere, bounds, max_rounds=1)
    messages = list(agent.run_iterator())

    # Check for basic message structure
    assert any(m['type'] == 'phase' for m in messages)
    assert any(m['phase'] == 'Exploring' for m in messages)

def test_export_and_json_serialization():
    bounds = np.array([[-5.0, 5.0]] * 2)
    orc = SolverOrchestrator(sphere, bounds, budget=500)
    orc.run()
    orc.save_report("test_report.json")

    import json
    with open("test_report.json") as f:
        data = json.load(f)
    assert 'best_score' in data
    assert 'portfolio' in data
