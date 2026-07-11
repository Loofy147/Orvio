import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from mlp_ae import MLPAutoencoder

train_facts_p2 = [
    "earth orbits the sun", "moon orbits the earth", "mars orbits the sun",
    "jupiter is a gas giant", "saturn has prominent rings", "stars produce light",
    "galaxies contain many stars", "the milky way is a galaxy",
    "paris is the capital of france", "berlin is the capital of germany",
    "rome is the capital of italy", "tokyo is the capital of japan",
    "madrid is the capital of spain", "london is the capital of the uk",
    "cairo is the capital of egypt", "ottawa is the capital of canada",
    "hydrogen is the lightest element", "helium is a noble gas",
    "diamonds are made of carbon", "water freezes at zero celsius",
    "water boils at one hundred celsius", "gravity pulls objects down",
    "sound travels slower than light", "electrons have negative charge",
    "cats are domestic felines", "dogs are domestic canines",
    "photosynthesis produces oxygen", "plants need water to grow",
    "humans breathe oxygen to live", "whales are marine mammals",
    "birds have feathers and wings", "insects have six legs"
]
val_facts_p2 = [
    "venus orbits the sun", "vienna is the capital of austria",
    "lions are wild felines", "iron is a metallic element",
    "fish breathe oxygen in water"
]
vectorizer = TfidfVectorizer(stop_words='english')
X_train_raw = vectorizer.fit_transform(train_facts_p2)
X_val_raw = vectorizer.transform(val_facts_p2)
lsa = TruncatedSVD(n_components=16, random_state=42)
E_train = lsa.fit_transform(X_train_raw)
E_val = lsa.transform(X_val_raw)
E_train = E_train / np.linalg.norm(E_train, axis=1, keepdims=True)
E_val = E_val / np.linalg.norm(E_val, axis=1, keepdims=True)

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def avg_cosine(orig, recon):
    dot = np.sum(orig*recon, axis=1)
    return np.mean(dot / (np.linalg.norm(orig,axis=1)*np.linalg.norm(recon,axis=1)+1e-9))

def smoothness_score(model, E, n_pairs=100, n_steps=6, seed=7, tol=0.02):
    """For random pairs (A,B), walk the latent interpolation and check whether
    similarity-to-A is non-increasing and similarity-to-B is non-decreasing
    (within tol, to allow for noise). Returns fraction of monotonic-both-ways
    paths, and the average number of monotonicity violations per path."""
    rng = np.random.RandomState(seed)
    N = E.shape[0]
    alphas = np.linspace(0, 1, n_steps)
    monotonic_count = 0
    total_violations = 0
    for _ in range(n_pairs):
        i, j = rng.choice(N, size=2, replace=False)
        A, B = E[i:i+1], E[j:j+1]
        zA = model.encode(A); zB = model.encode(B)
        sims_A, sims_B = [], []
        for a in alphas:
            z_interp = (1 - a) * zA + a * zB
            recon = model.decode(z_interp)[0]
            sims_A.append(cos(recon, A[0]))
            sims_B.append(cos(recon, B[0]))
        sims_A = np.array(sims_A); sims_B = np.array(sims_B)
        violations = np.sum(np.diff(sims_A) > tol) + np.sum(np.diff(sims_B) < -tol)
        total_violations += violations
        if violations == 0:
            monotonic_count += 1
    return monotonic_count / n_pairs, total_violations / n_pairs

print("Smoothness vs reconstruction quality tradeoff, z=8, 100 random interpolation pairs from E_train:")
print(f"{'noise_std':>9} | {'TrCos':>7} {'ValCos':>7} | {'MonotonicPathFrac':>18} {'AvgViolations':>14}")
for noise_std in [0.0, 0.05, 0.1, 0.2, 0.3]:
    model = MLPAutoencoder(encoder_sizes=[16, 8], decoder_sizes=[8, 16, 16], seed=42)
    model.fit(E_train, epochs=10000, lr=0.02, l2=1e-4, noise_std=noise_std)
    tr_cos = avg_cosine(E_train, model.reconstruct(E_train))
    v_cos = avg_cosine(E_val, model.reconstruct(E_val))
    frac, avg_viol = smoothness_score(model, E_train, n_pairs=100)
    print(f"{noise_std:9.2f} | {tr_cos:7.1%} {v_cos:7.1%} | {frac:18.1%} {avg_viol:14.2f}")
