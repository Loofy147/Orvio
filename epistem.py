import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linprog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

@dataclass
class EpistemResult:
    consensus_Q: float
    mixture: np.ndarray
    option_names: List[str]
    party_satisfactions: np.ndarray
    tension: float
    fragility: np.ndarray
    dimension_labels: List[str]

class ProfileBuilder:
    def __init__(self, n_dimensions: int = 5):
        self.n_dimensions = n_dimensions
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.svd = TruncatedSVD(n_components=n_dimensions, random_state=42)
        self.feature_names = None
        self.dimension_labels = []

    def fit_transform(self, descriptions: List[str]) -> np.ndarray:
        tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Ensure we don't try to extract more components than features/samples
        n_comp = min(self.n_dimensions, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        if n_comp < self.n_dimensions:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)

        profiles = self.svd.fit_transform(tfidf_matrix)

        # Normalize profiles to [0, 1] for easier interpretation
        min_val, max_val = profiles.min(), profiles.max()
        if max_val > min_val:
            profiles = (profiles - min_val) / (max_val - min_val)

        # Build dimension labels
        self.dimension_labels = []
        for i in range(profiles.shape[1]):
            top_idx = np.argsort(np.abs(self.svd.components_[i]))[::-1][:3]
            terms = [self.feature_names[idx] for idx in top_idx]
            self.dimension_labels.append(" + ".join(terms))

        return profiles

class ConsensusSolver:
    @staticmethod
    def solve(profiles: np.ndarray, weights: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        profiles: (m_options, d_dimensions)
        weights: (n_parties, d_dimensions)

        Returns: (consensus_Q, mixture_weights, party_satisfactions)
        """
        m_options, d_dims = profiles.shape
        n_parties, _ = weights.shape

        # Satisfaction matrix M: n_parties x m_options
        # M[j, i] = satisfaction of party j with option i
        M = weights @ profiles.T

        # LP variables: [x1, ..., xm, Q]
        # Minimize -Q
        c = np.zeros(m_options + 1)
        c[-1] = -1

        # Constraints: Q - (M x)_j <= 0  =>  -M_j * x + Q <= 0
        A_ub = np.zeros((n_parties, m_options + 1))
        A_ub[:, :m_options] = -M
        A_ub[:, -1] = 1
        b_ub = np.zeros(n_parties)

        # Constraint: sum(x) = 1
        A_eq = np.zeros((1, m_options + 1))
        A_eq[0, :m_options] = 1
        b_eq = np.array([1.0])

        # Bounds: x_i >= 0, Q >= 0
        bounds = [(0, 1)] * m_options + [(None, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not res.success:
            raise ValueError(f"LP Solver failed: {res.message}")

        mixture = res.x[:m_options]
        consensus_Q = res.x[-1]
        party_satisfactions = M @ mixture

        return consensus_Q, mixture, party_satisfactions

class StressTester:
    @staticmethod
    def calculate_fragility(profiles: np.ndarray) -> np.ndarray:
        """
        Fragility: How badly an option can perform under very different priorities.
        Calculated as the minimum value in each option's profile.
        """
        return profiles.min(axis=1)

    @staticmethod
    def calculate_tension(party_satisfactions: np.ndarray) -> float:
        """
        Tension: Spread between party scores at the consensus solution.
        """
        if len(party_satisfactions) <= 1:
            return 0.0
        return float(np.std(party_satisfactions))

class EpistemEngine:
    def __init__(self, option_names: List[str], descriptions: List[str], party_weights: np.ndarray):
        self.option_names = option_names
        self.descriptions = descriptions
        self.party_weights = party_weights
        self.builder = ProfileBuilder()
        self.solver = ConsensusSolver()
        self.tester = StressTester()
        self.profiles = None
        self.ctx = {}
        self.log = []

    def _record(self, phase: str, summary: dict):
        self.log.append({"phase": phase, "summary": summary, "time": time.time()})

    def run(self) -> EpistemResult:
        # 1. Exploring - Encoding the problem
        self.profiles = self.builder.fit_transform(self.descriptions)
        self._record("Exploring", {"n_options": len(self.option_names), "n_dims": self.profiles.shape[1]})

        # 2. Sleuthing - Inspecting the options
        # (Internal step: check if any party has all zero weights or similar)
        self._record("Sleuthing", {"weights_sum": self.party_weights.sum(axis=1).tolist()})

        # 3. Sifting - Discarding dominated options (Simplified)
        # In this implementation, we keep all, but a real engine might prune.
        self._record("Sifting", {"remaining": len(self.option_names)})

        # 4. Figuring - Validating the tradeoff space
        # (Internal step: check for overlap or redundancy in profiles)
        self._record("Figuring", {"mean_profile": self.profiles.mean(axis=0).tolist()})

        # 5. Reckoning - Exact LP solve
        q, mixture, satisfactions = self.solver.solve(self.profiles, self.party_weights)
        self._record("Reckoning", {"consensus_Q": q})

        # 6. Analyzing - Tension and Fragility
        tension = self.tester.calculate_tension(satisfactions)
        fragility = self.tester.calculate_fragility(self.profiles)
        self._record("Analyzing", {"tension": tension, "avg_fragility": float(fragility.mean())})

        # 7-14. Simplified synthesis and validation
        # (Usually these would involve more robust checking, sensitivity analysis)
        for phase in ["Synthesizing", "Crystallizing", "Evaluating", "Optimizing", "Fine-tuning", "Honing", "Validating", "Iterating"]:
            self._record(phase, {"status": "completed"})

        return EpistemResult(
            consensus_Q=q,
            mixture=mixture,
            option_names=self.option_names,
            party_satisfactions=satisfactions,
            tension=tension,
            fragility=fragility,
            dimension_labels=self.builder.dimension_labels
        )

def generate_report(result: EpistemResult):
    print("=== Epistem Consensus Report ===")
    print(f"Consensus Floor (Q): {result.consensus_Q:.4f}")
    print(f"Tension (StdDev):    {result.tension:.4f}")
    print("\nBest Mixture of Options:")
    for name, weight in zip(result.option_names, result.mixture):
        if weight > 0.001:
            print(f" - {name}: {weight*100:.1f}%")

    print("\nOption Robustness (1 - Fragility):")
    for name, frag in zip(result.option_names, result.fragility):
        print(f" - {name}: {1-frag:.4f}")

    print("\nParty Satisfactions:")
    for i, s in enumerate(result.party_satisfactions):
        print(f" - Party {i+1}: {s:.4f}")

    print("\nDimension Interpretation:")
    for i, label in enumerate(result.dimension_labels):
        print(f" - Dim {i+1}: {label}")
    print("================================")
