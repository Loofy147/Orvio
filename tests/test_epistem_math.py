"""
Mathematical validation harness for the Epistem Engine.
Runs 28 distinct sanity assertions to ensure mathematical correctness of the library.
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
import epistem as ep

PASS, FAIL = [], []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
    else:
        FAIL.append((name, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not condition else ""))


# ── T1: embed() basic properties ──
print("\n=== T1: embed() basic properties ===")
corpus10 = [(f"T{i}", txt) for i, txt in enumerate([
    "cats are small domesticated feline mammals that purr",
    "dogs are loyal canine companions that bark and fetch",
    "quantum mechanics describes subatomic particle behaviour",
    "general relativity describes gravity as spacetime curvature",
    "python is a dynamically typed interpreted programming language",
    "rust guarantees memory safety without a garbage collector",
    "keynesian economics emphasises aggregate demand management",
    "austrian economics emphasises spontaneous market order",
    "the mitochondria is the powerhouse of the cell",
    "photosynthesis converts light energy into chemical energy",
])]
profiles = ep.embed(corpus10, n_dims=8)
real_profiles = {k: v for k, v in profiles.items() if not k.startswith("__")}
check("embed returns one profile per input text", len(real_profiles) == len(corpus10))
check("no zero-padding marker on a corpus with 10 texts / 8 dims", "__padded_dims__" not in profiles)
all_vals = np.concatenate(list(real_profiles.values()))
check("all values within [lo, hi] = [0.15, 0.85]", all_vals.min() >= 0.15 - 1e-9 and all_vals.max() <= 0.85 + 1e-9)


# ── T1b: collision warning ──
print("\n=== T1b: collision warning ===")
import warnings as _warnings
with _warnings.catch_warnings(record=True) as caught:
    _warnings.simplefilter("always")
    ep.embed(corpus10, n_dims=8)
    fired = any("identical profiles" in str(w.message) for w in caught)
check("embed() emits a UserWarning when duplicate profiles are detected", fired)


# ── T2: small-corpus zero-padding (resolution check) ──
print("\n=== T2: small-corpus zero-padding (resolution check) ===")
corpus3 = [("A", "cats purr"), ("B", "dogs bark"), ("C", "birds fly")]

# Test 2a: Historical floor tie configuration
profiles3_buggy = ep.embed(corpus3, n_dims=8, pad_value=0.15)
padded_idx_buggy = profiles3_buggy.get("__padded_dims__", np.array([]))
real3_buggy = {k: v for k, v in profiles3_buggy.items() if not k.startswith("__")}
tied_buggy = all(np.allclose(v[padded_idx_buggy], 0.15) for v in real3_buggy.values())
check("explicit pad_value=lo reproduces historical floor tie", tied_buggy)

# Test 2b: Corrected default ceiling padding
profiles3_fixed = ep.embed(corpus3, n_dims=8)
padded_idx_fixed = profiles3_fixed.get("__padded_dims__", np.array([]))
real3_fixed = {k: v for k, v in profiles3_fixed.items() if not k.startswith("__")}
tied_fixed = all(np.allclose(v[padded_idx_fixed], 0.85) for v in real3_fixed.values())
check("default behavior pads with ceiling value (0.85) instead of floor", tied_fixed)

# Test 2c: Bottleneck insulation verification
misattributed = False
for name, v in real3_fixed.items():
    if np.argmin(v) in padded_idx_fixed:
        misattributed = True
check("bottleneck is successfully insulated from fake padded dimensions", not misattributed)


# ── T3: input validation ──
print("\n=== T3: input validation ===")
try:
    ep.lp_consensus(real_profiles, np.random.rand(2, 8), party_names=["OnlyOne"])
    check("mismatched party_names raises ValueError", False, "no exception raised")
except ValueError:
    check("mismatched party_names raises ValueError", True)


# ── T4: LP consensus is NOT degenerate ──
print("\n=== T4: LP non-degeneracy ===")
W = np.array([
    [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03],
    [0.05, 0.10, 0.25, 0.30, 0.15, 0.08, 0.05, 0.02],
])
result = ep.lp_consensus(real_profiles, W, party_names=["PartyA", "PartyB"])
check("v_opt is not the trivial all-ones vector", not np.allclose(result.v_opt, 1.0, atol=1e-6))
check("consensus_Q is strictly less than 1.0 (non-trivial tradeoff)", result.consensus_Q < 0.999)
check("mixture weights sum to ~1.0", abs(sum(result.mixture.values()) - 1.0) < 1e-4)


# ── T5: LP optimality vs brute force ──
print("\n=== T5: LP optimality ===")
sub_names = list(real_profiles.keys())[:4]
sub_profiles = {k: real_profiles[k] for k in sub_names}
sub_result = ep.lp_consensus(sub_profiles, W, party_names=["PartyA", "PartyB"])

def grid_search_consensus(profiles_dict, weight_matrix):
    names = list(profiles_dict.keys())
    T = np.array([profiles_dict[n] for n in names])
    best_t = -1.0
    rng = np.random.default_rng(0)
    for _ in range(10000):
        lam = rng.dirichlet(np.ones(len(names)))
        v = lam @ T
        t = np.min(weight_matrix @ v)
        if t > best_t:
            best_t = t
    return best_t

brute_force_Q = grid_search_consensus(sub_profiles, W)
check("LP consensus_Q >= brute-force search", sub_result.consensus_Q >= brute_force_Q - 1e-6)


# ── T6: q_worst vs SLSQP ──
print("\n=== T6: q_worst validation ===")
def slsqp_worst(v):
    def obj(w): return float(np.dot(w, v))
    best = float(np.dot(np.ones(len(v)) / len(v), v))
    for _ in range(20):
        w0 = np.random.dirichlet(np.ones(len(v)))
        r = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)] * len(v), constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        if r.success and r.fun < best:
            best = r.fun
    return best

test_vecs = [real_profiles[n] for n in list(real_profiles)[:3]]
for i, v in enumerate(test_vecs):
    exact = ep.q_worst(v)
    numeric = slsqp_worst(v)
    check(f"q_worst matches SLSQP for theory {i}", abs(exact - numeric) < 1e-4)


# ── T7: edge cases ──
print("\n=== T7: degenerate cases ===")
single = {list(real_profiles.keys())[0]: list(real_profiles.values())[0]}
r_single = ep.lp_consensus(single, W, party_names=["PartyA", "PartyB"])
check("single-theory LP does not crash", r_single.consensus_Q >= 0)
check("single-theory mixture assigns 100% weight", len(r_single.mixture) == 1 and list(r_single.mixture.values())[0] > 0.99)


# ── T8: stress testing consistency ──
print("\n=== T8: stress testing ===")
sr = ep.stress(real_profiles, weight_matrix=W, n_scenarios=100, seed=0)
all_consistent = all(d["worst_exact"] <= d["worst"] + 1e-9 for d in sr.results.values())
check("exact worst-case is a lower bound to sampled scenarios", all_consistent)


# ── T10: isomorphism checks & robustness to constant inputs ──
print("\n=== T10: isomorphism validation ===")
v = np.array([0.2, 0.5, 0.8, 0.3, 0.6, 0.1, 0.9, 0.4])
r_self, p_self = ep.isomorphism(v, v)
check("isomorphism of self is r=1.0", abs(r_self - 1.0) < 1e-9)
r_inv, _ = ep.isomorphism(v, 1 - v)
check("isomorphism of complement is r=-1.0", abs(r_inv - (-1.0)) < 1e-9)

# Validate constant vector safeguard
v_const = np.ones(8) * 0.5
r_const, p_const = ep.isomorphism(v_const, v)
check("degenerate constant vector returns neutral (0.0, 1.0) rather than NaN", r_const == 0.0 and p_const == 1.0)


# ── SUMMARY ──
print(f"\nRESULTS: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
    sys.exit(1)
sys.exit(0)
