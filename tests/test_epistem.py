import pytest
import numpy as np
from epistem import embed, lp_consensus, stress, q_worst, q_best, fragility, EpistemEngine

def test_embed():
    corpus = [
        ("A", "fast and cheap"),
        ("B", "slow and expensive"),
        ("C", "medium speed and medium cost")
    ]
    profiles = embed(corpus, n_dims=4)
    assert len(profiles) == 4 # Includes __padded_dims__
    assert profiles["A"].shape == (4,)
    assert np.all(profiles["A"] >= 0.15) and np.all(profiles["A"] <= 0.85)

def test_lp_consensus():
    profiles = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0])
    }
    # 2 parties
    weights = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    res = lp_consensus(profiles, weights)

    assert pytest.approx(res.consensus_Q) == 0.5
    assert pytest.approx(res.mixture["A"]) == 0.5
    assert pytest.approx(res.mixture["B"]) == 0.5

def test_adversarial():
    v = np.array([0.2, 0.8, 0.5])
    assert q_worst(v) == 0.2
    assert q_best(v) == 0.8
    assert pytest.approx(fragility(v)) == 0.6

def test_stress():
    profiles = {
        "A": np.array([0.2, 0.8]),
        "B": np.array([0.5, 0.5])
    }
    report = stress(profiles)
    assert "A" in report.results
    assert "B" in report.results
    assert report.results["A"]["fragility"] > report.results["B"]["fragility"]

def test_epistem_engine():
    options = ["Opt A", "Opt B"]
    descriptions = [
        "High quality low cost, very efficient and reliable solution for long term.",
        "Low quality high cost, expensive and prone to failure, not recommended."
    ]
    weights = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])

    engine = EpistemEngine(options, descriptions, weights)
    consensus, report = engine.run()

    assert consensus.consensus_Q >= 0
    assert len(consensus.mixture) >= 1
    assert len(engine.log) >= 14
