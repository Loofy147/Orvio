import numpy as np
import time
from mlp_ae import MLPAutoencoder

# Scalability: synthetic dataset much bigger than the 31-example toy corpus,
# with genuine nonlinear structure (so a real autoencoder should beat PCA).
np.random.seed(0)
N, V = 800, 40
latent_true = np.random.randn(N, 3)
basis = np.random.randn(3, V)
X = np.tanh(latent_true @ basis) + np.sin(latent_true[:, :1] @ np.random.randn(1, V)) * 0.3
X += np.random.randn(N, V) * 0.02
X_train, X_val = X[:640], X[640:]

def linear_pca_mse(E_tr, E_v, z_dim):
    mean = E_tr.mean(axis=0); Ec = E_tr - mean; Evc = E_v - mean
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Vt[:z_dim].T
    tr_recon = (Ec @ P) @ P.T + mean
    v_recon = (Evc @ P) @ P.T + mean
    return np.mean((E_tr - tr_recon)**2), np.mean((E_v - v_recon)**2)

pca_tr, pca_v = linear_pca_mse(X_train, X_val, 3)
print(f"800-sample, 40-dim synthetic nonlinear dataset (train=640, val=160):")
print(f"PCA baseline (z=3): train MSE={pca_tr:.5f}  val MSE={pca_v:.5f}")

print("\nFull-batch vs minibatch training, z=3:")
for batch_size, label in [(None, "full-batch (640)"), (64, "minibatch=64"), (16, "minibatch=16")]:
    model = MLPAutoencoder(encoder_sizes=[V, 3], decoder_sizes=[3, 16, V], seed=42)
    t0 = time.time()
    model.fit(X_train, epochs=300, lr=0.01, l2=1e-4, batch_size=batch_size)
    elapsed = time.time() - t0
    tr_mse = np.mean((model.reconstruct(X_train) - X_train)**2)
    v_mse = np.mean((model.reconstruct(X_val) - X_val)**2)
    print(f"{label:<20}: train MSE={tr_mse:.5f}  val MSE={v_mse:.5f}  ({elapsed:.1f}s for 300 epochs)")

print("\nSave/load round-trip check:")
model = MLPAutoencoder(encoder_sizes=[V, 3], decoder_sizes=[3, 16, V], seed=42)
model.fit(X_train, epochs=300, lr=0.01, l2=1e-4, batch_size=64)
recon_before = model.reconstruct(X_val)
model.save('/tmp/test_model.npz')
loaded = MLPAutoencoder.load('/tmp/test_model.npz')
recon_after = loaded.reconstruct(X_val)
print("Max abs difference before vs after save/load:", np.max(np.abs(recon_before - recon_after)))
