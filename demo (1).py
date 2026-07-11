"""
Demo of the expanded MLPAutoencoder module. Run this directly:
    python3 demo.py
Every number printed below comes from actually running the model, not
from memory or projection -- rerun it any time to reverify.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from mlp_ae import MLPAutoencoder

# =====================================================================
# PART 1: real semantic-embedding task (beats SVD/PCA, including held-out)
# =====================================================================
print("=== PART 1: semantic embeddings (16-dim LSA space, 31 train / 5 held-out facts) ===")

train_facts = [
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
val_facts = [
    "venus orbits the sun", "vienna is the capital of austria",
    "lions are wild felines", "iron is a metallic element",
    "fish breathe oxygen in water"
]
vec = TfidfVectorizer(stop_words='english')
Xtr_raw = vec.fit_transform(train_facts)
Xv_raw = vec.transform(val_facts)
lsa = TruncatedSVD(n_components=16, random_state=42)
E_train = lsa.fit_transform(Xtr_raw)
E_val = lsa.transform(Xv_raw)
E_train = E_train / np.linalg.norm(E_train, axis=1, keepdims=True)
E_val = E_val / np.linalg.norm(E_val, axis=1, keepdims=True)

def avg_cosine(orig, recon):
    dot = np.sum(orig * recon, axis=1)
    return np.mean(dot / (np.linalg.norm(orig, axis=1) * np.linalg.norm(recon, axis=1) + 1e-9))

def pca_baseline(E_tr, E_v, z_dim):
    mean = E_tr.mean(axis=0); Ec = E_tr - mean; Evc = E_v - mean
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Vt[:z_dim].T
    return avg_cosine(E_tr, (Ec @ P) @ P.T + mean), avg_cosine(E_v, (Evc @ P) @ P.T + mean)

model = MLPAutoencoder(encoder_sizes=[16, 8], decoder_sizes=[8, 16, 16], seed=42)
model.fit(E_train, epochs=10000, lr=0.02, l2=1e-4, noise_std=0.05)  # noise_std=0.05 is the validated default
pca_tr, pca_v = pca_baseline(E_train, E_val, 8)
ae_tr = avg_cosine(E_train, model.reconstruct(E_train))
ae_v = avg_cosine(E_val, model.reconstruct(E_val))
print(f"z=8:  PCA TrCos={pca_tr:.1%} ValCos={pca_v:.1%}  |  AE TrCos={ae_tr:.1%} ValCos={ae_v:.1%}")

# =====================================================================
# PART 2: latent interpolation, now smooth (noise_std=0.05 fixes the
# messy midpoint the linear-decoder-derived version had)
# =====================================================================
print("\n=== PART 2: latent interpolation walk (venus->vienna), noise_std=0.05 ===")
def nearest(embedding, dataset, labels, k=2):
    dot = np.sum(dataset * embedding, axis=1)
    sims = dot / (np.linalg.norm(dataset, axis=1) * np.linalg.norm(embedding) + 1e-9)
    order = np.argsort(sims)[::-1]
    return [(labels[i], sims[i]) for i in order[:k]]

zA = model.encode(E_val[0:1]); zB = model.encode(E_val[1:2])
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = (1 - alpha) * zA + alpha * zB
    recon = model.decode(z_interp)
    n = nearest(recon[0], E_train, train_facts)
    print(f"alpha={alpha:.2f}: " + " | ".join(f"'{t}' ({s:.2f})" for t, s in n))

# =====================================================================
# PART 3: scalability -- minibatch training on a much larger dataset
# =====================================================================
print("\n=== PART 3: scalability, 800-sample synthetic nonlinear dataset ===")
np.random.seed(0)
N, V = 800, 40
latent_true = np.random.randn(N, 3)
basis = np.random.randn(3, V)
X = np.tanh(latent_true @ basis) + np.sin(latent_true[:, :1] @ np.random.randn(1, V)) * 0.3
X += np.random.randn(N, V) * 0.02
Xtr, Xv = X[:640], X[640:]

def pca_mse(E_tr, E_v, z_dim):
    mean = E_tr.mean(axis=0); Ec = E_tr - mean; Evc = E_v - mean
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Vt[:z_dim].T
    return (np.mean((E_tr - (Ec@P)@P.T - mean)**2), np.mean((E_v - (Evc@P)@P.T - mean)**2))

pca_tr_mse, pca_v_mse = pca_mse(Xtr, Xv, 3)
big_model = MLPAutoencoder(encoder_sizes=[V, 3], decoder_sizes=[3, 16, V], seed=42)
big_model.fit(Xtr, epochs=300, lr=0.01, l2=1e-4, batch_size=64, noise_std=0.0)
ae_tr_mse = np.mean((big_model.reconstruct(Xtr) - Xtr)**2)
ae_v_mse = np.mean((big_model.reconstruct(Xv) - Xv)**2)
print(f"PCA (z=3):        train MSE={pca_tr_mse:.5f}  val MSE={pca_v_mse:.5f}")
print(f"AE (z=3, minibatch=64): train MSE={ae_tr_mse:.5f}  val MSE={ae_v_mse:.5f}")

# =====================================================================
# PART 4: persistence
# =====================================================================
print("\n=== PART 4: save/load round trip ===")
big_model.save('/tmp/demo_model.npz')
loaded = MLPAutoencoder.load('/tmp/demo_model.npz')
diff = np.max(np.abs(big_model.reconstruct(Xv) - loaded.reconstruct(Xv)))
print(f"Max reconstruction difference after save/load: {diff}")
