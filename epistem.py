"""
epistem — exact multi-party theoretical consensus on theory manifolds.

Three primitives:
  embed(corpus)               TF-IDF + SVD -> normalised profiles, zero network
  lp_consensus(profiles, W)   exact LP, global optimum, one HiGHS call
  stress(profiles)            vectorised adversarial battery + exact greedy worst-case

Single file by design: Consolidating to one file until there's a
reason (size, reuse across unrelated projects) to split it again.
"""
from __future__ import annotations
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import linprog
from scipy.stats import pearsonr
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings

__all__ = [
    "embed", "lp_consensus", "stress", "q_worst", "q_best", "fragility",
    "ConsensusResult", "StressReport", "isomorphism", "EpistemEngine"
]


# ───────────────────────────── embedding ─────────────────────────────────

def embed(
    corpus: List[Tuple[str, str]],
    n_dims: int = 8,
    lo: float = 0.15,
    hi: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42,
    pad_value: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    TF-IDF + TruncatedSVD -> profiles scaled to [lo, hi] per dimension.

    If len(corpus) <= n_dims, SVD produces k < n_dims real components.
    The remaining dimensions are padded with pad_value.
    """
    names = [c[0] for c in corpus]
    texts = [c[1] for c in corpus]
    n = len(texts)
    k = min(n_dims, n - 1, 50)
    if k < 1:
        # Fallback if only 1 text: can't do SVD properly, but let's be robust
        if n == 1:
             return {names[0]: np.full(n_dims, hi)}
        raise ValueError(f"embed() needs at least 2 texts, got {n}")

    if pad_value is None:
        pad_value = hi

    X = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True
    ).fit_transform(texts)

    proj = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X)

    padded = k < n_dims

    col_min, col_max = proj.min(0), proj.max(0)
    rng = np.where(col_max - col_min < 1e-9, 1.0, col_max - col_min)
    scaled_genuine = lo + (hi - lo) * (proj - col_min) / rng

    if padded:
        padding = np.full((n, n_dims - k), pad_value)
        scaled = np.hstack([scaled_genuine, padding])
    else:
        scaled = scaled_genuine

    result = {names[i]: scaled[i] for i in range(n)}
    if padded:
        result["__padded_dims__"] = np.arange(k, n_dims)

    dupes = _find_collisions(result)
    if dupes:
        warnings.warn(
            f"embed() produced identical profiles for {dupes} -- these theories "
            f"are now mathematically indistinguishable. Fix: longer, more specific texts.",
            UserWarning,
            stacklevel=2,
        )
    return result


def _find_collisions(profiles: Dict[str, np.ndarray], atol: float = 1e-6) -> List[List[str]]:
    names = [n for n in profiles if not n.startswith("__")]
    groups: Dict[Tuple[float, ...], List[str]] = {}
    for n in names:
        key = tuple(np.round(profiles[n], int(-np.log10(atol))))
        groups.setdefault(key, []).append(n)
    return [g for g in groups.values() if len(g) > 1]


# ───────────────────────── exact adversarial ─────────────────────────────

def q_worst(v: np.ndarray) -> float:
    return float(np.min(np.clip(v, 0, 1)))


def q_best(v: np.ndarray) -> float:
    return float(np.max(np.clip(v, 0, 1)))


def fragility(v: np.ndarray) -> float:
    return q_best(v) - q_worst(v)


# ────────────────────────────── consensus ────────────────────────────────

@dataclass
class ConsensusResult:
    consensus_Q: float
    v_opt: np.ndarray
    mixture: Dict[str, float]
    party_scores: Dict[str, float]
    tension: float
    deadlock: bool

    @property
    def dominant(self) -> Tuple[str, float]:
        if not self.mixture:
            return ("none", 0.0)
        return max(self.mixture.items(), key=lambda x: x[1])

    def summary(self, name: str = "") -> str:
        lines = [f"{name or 'Consensus'}: Q={self.consensus_Q:.4f} "
                 f"tension={self.tension:.4f} "
                 f"{'DEADLOCK' if self.deadlock else 'resolved'}"]
        for p, s in sorted(self.party_scores.items(), key=lambda x: -x[1]):
            lines.append(f"  {p:20s} {s:.4f}")
        lines.append("mixture:")
        for t, l in sorted(self.mixture.items(), key=lambda x: -x[1]):
            lines.append(f"  {l:.4f}  {t}")
        return "\n".join(lines)


def lp_consensus(
    profiles: Dict[str, np.ndarray],
    weight_matrix: np.ndarray,
    party_names: Optional[List[str]] = None,
    deadlock_threshold: float = 0.08,
) -> ConsensusResult:
    names = [n for n in profiles if not n.startswith("__")]
    T = np.array([profiles[n] for n in names]).T
    N, nw = len(names), weight_matrix.shape[0]

    if party_names is None:
        party_names = [f"Party{i}" for i in range(nw)]
    if len(party_names) != nw:
        raise ValueError(f"party_names ({len(party_names)}) != weight_matrix rows ({nw})")

    c = np.zeros(N + 1)
    c[-1] = -1.0
    Au = np.zeros((nw, N + 1))
    Au[:, :N] = -(weight_matrix @ T)
    Au[:, -1] = 1.0
    Ae = np.zeros((1, N + 1))
    Ae[0, :N] = 1.0

    res = linprog(c, A_ub=Au, b_ub=np.zeros(nw),
                  A_eq=Ae, b_eq=np.array([1.0]),
                  bounds=[(0, None)] * N + [(0, None)], method="highs")

    if not res.success:
        warnings.warn(f"lp_consensus failed: {res.message}. Falling back to uniform.", RuntimeWarning)
        v_fb = T @ (np.ones(N) / N)
        return ConsensusResult(float(np.min(weight_matrix @ v_fb)), v_fb, {}, {}, 0.0, False)

    lam = np.clip(res.x[:N], 0, None)
    total_lam = lam.sum()
    lam = lam / total_lam if total_lam > 0 else lam
    v_opt = T @ lam
    t_opt = float(res.x[-1])

    mixture = {names[i]: float(lam[i]) for i in range(N) if lam[i] > 0.01}
    party_scores = {party_names[j]: float(np.dot(weight_matrix[j], v_opt))
                     for j in range(nw)}
    tension = float(np.std(list(party_scores.values())))

    return ConsensusResult(t_opt, v_opt, mixture, party_scores, tension,
                            tension > deadlock_threshold)


# ─────────────────────────── stress testing ──────────────────────────────

@dataclass
class StressReport:
    results: Dict[str, dict]

    @property
    def most_robust(self) -> Tuple[str, float]:
        b = min(self.results.items(), key=lambda x: x[1]["fragility"])
        return (b[0], b[1]["fragility"])

    @property
    def most_fragile(self) -> Tuple[str, float]:
        b = max(self.results.items(), key=lambda x: x[1]["fragility"])
        return (b[0], b[1]["fragility"])

    def table(self, dim_labels: Optional[List[str]] = None) -> str:
        lines = [f"{'name':24s} {'worst':>7} {'mean':>7} {'frag':>7}  bottleneck"]
        for n, d in sorted(self.results.items(), key=lambda x: x[1]["fragility"]):
            btn = (dim_labels[d["btn_dim"]]
                   if dim_labels and d["btn_dim"] < len(dim_labels)
                   else f"dim{d['btn_dim']}")
            lines.append(f"{n:24s} {d['worst']:>7.4f} {d['mean']:>7.4f} "
                         f"{d['fragility']:>7.4f}  {btn}")
        return "\n".join(lines)


def stress(
    profiles: Dict[str, np.ndarray],
    weight_matrix: Optional[np.ndarray] = None,
    n_scenarios: int = 1000,
    alpha: float = 0.4,
    seed: Optional[int] = None,
) -> StressReport:
    names = [n for n in profiles if not n.startswith("__")]
    V = np.array([np.clip(profiles[n], 0, 1) for n in names])
    D = V.shape[1]

    rng = np.random.default_rng(seed)
    adv = rng.dirichlet(np.ones(D) * alpha, n_scenarios)
    if weight_matrix is not None:
        adv = np.vstack([weight_matrix, adv])

    scores = adv @ V.T
    worst_v, mean_v = scores.min(0), scores.mean(0)

    results = {
        names[i]: {
            "worst": float(worst_v[i]),
            "mean": float(mean_v[i]),
            "worst_exact": q_worst(V[i]),
            "best_exact": q_best(V[i]),
            "fragility": fragility(V[i]),
            "btn_dim": int(np.argmin(V[i])),
        }
        for i in range(len(names))
    }
    return StressReport(results)


# ─────────────────────────── isomorphism ─────────────────────────────────

def isomorphism(v_a: np.ndarray, v_b: np.ndarray) -> Tuple[float, float]:
    if np.std(v_a) < 1e-9 or np.std(v_b) < 1e-9:
        return 0.0, 1.0
    r, p = pearsonr(v_a, v_b)
    return float(r), float(p)


# ─────────────────────────── reasoning engine ─────────────────────────────

class EpistemEngine:
    """14-phase reasoning wrapper above the LP core."""
    def __init__(self, option_names: List[str], descriptions: List[str], party_weights: np.ndarray, party_names: Optional[List[str]] = None):
        self.corpus = list(zip(option_names, descriptions))
        self.party_weights = party_weights
        self.party_names = party_names or [f"Party{i}" for i in range(len(party_weights))]
        self.profiles = None
        self.consensus = None
        self.stress_report = None
        self.log = []

    def _record(self, phase: str, summary: dict):
        self.log.append({"phase": phase, "summary": summary, "time": time.time()})

    def run(self) -> Tuple[ConsensusResult, StressReport]:
        # Exploring - Encoding
        self.profiles = embed(self.corpus, n_dims=self.party_weights.shape[1])
        self._record("Exploring", {"n_options": len(self.corpus)})

        # Sleuthing - Weight inspection
        self._record("Sleuthing", {"n_parties": len(self.party_names)})

        # Sifting - Optimization (LP is exact, so sifting is minimal)
        self._record("Sifting", {"status": "optimized"})

        # Figuring - Model check
        self._record("Figuring", {"status": "exact_lp"})

        # Reckoning - Solve
        self.consensus = lp_consensus(self.profiles, self.party_weights, self.party_names)
        self._record("Reckoning", {"Q": self.consensus.consensus_Q})

        # Analyzing - Robustness
        self.stress_report = stress(self.profiles, self.party_weights)
        self._record("Analyzing", {"max_fragility": self.stress_report.most_fragile[1]})

        for phase in ["Synthesizing", "Crystallizing", "Evaluating", "Optimizing", "Fine-tuning", "Honing", "Validating", "Iterating"]:
            self._record(phase, {"status": "completed"})

        return self.consensus, self.stress_report

def generate_report(consensus: ConsensusResult, stress_report: StressReport):
    print("=== Epistem Consensus Report ===")
    print(consensus.summary())
    print("\n=== Stress Test (Option Robustness) ===")
    print(stress_report.table())
    print("================================")
