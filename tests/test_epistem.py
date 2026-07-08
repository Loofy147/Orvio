import pytest
import numpy as np
from epistem import ProfileBuilder, ConsensusSolver, StressTester, EpistemEngine

def test_profile_builder():
    builder = ProfileBuilder(n_dimensions=2)
    descriptions = [
        "fast and cheap",
        "slow and expensive",
        "medium speed and medium cost"
    ]
    profiles = builder.fit_transform(descriptions)
    assert profiles.shape == (3, 2)
    assert np.all(profiles >= 0) and np.all(profiles <= 1)
    assert len(builder.dimension_labels) == 2

def test_consensus_solver():
    solver = ConsensusSolver()
    # 2 options, 2 dimensions
    profiles = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    # 2 parties
    # Party 1 likes Dim 1
    # Party 2 likes Dim 2
    weights = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    q, mixture, satisfactions = solver.solve(profiles, weights)

    # Best compromise for two people wanting opposite things should be 50/50
    assert pytest.approx(q) == 0.5
    assert pytest.approx(mixture[0]) == 0.5
    assert pytest.approx(mixture[1]) == 0.5
    assert pytest.approx(satisfactions[0]) == 0.5
    assert pytest.approx(satisfactions[1]) == 0.5

def test_stress_tester():
    tester = StressTester()
    profiles = np.array([
        [1.0, 0.2, 0.5],
        [0.1, 0.9, 0.3]
    ])
    fragility = tester.calculate_fragility(profiles)
    assert np.array_equal(fragility, np.array([0.2, 0.1]))

    tension = tester.calculate_tension(np.array([0.5, 0.5]))
    assert tension == 0.0

    tension_spread = tester.calculate_tension(np.array([0.4, 0.6]))
    assert tension_spread > 0

def test_epistem_engine():
    options = ["Opt A", "Opt B"]
    descriptions = ["High quality low cost", "Low quality high cost"]
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])

    engine = EpistemEngine(options, descriptions, weights)
    result = engine.run()

    assert result.consensus_Q >= 0
    assert len(result.mixture) == 2
    assert len(result.option_names) == 2
    assert len(result.dimension_labels) > 0
    assert len(engine.log) >= 14 # Checking 14 phases
