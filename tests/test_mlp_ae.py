import os
import tempfile
import numpy as np
import pytest
from mlp_ae import MLPAutoencoder

def numgrad(f, p, eps=1e-4):
    g = np.zeros_like(p)
    it = np.nditer(p, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = p[idx]
        p[idx] = orig + eps; fp = f()
        p[idx] = orig - eps; fm = f()
        p[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g

def relerr(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))

def test_mlp_ae_initialization():
    # Test valid initializations
    model = MLPAutoencoder(encoder_sizes=[6, 4, 2], decoder_sizes=[2, 4, 6], seed=42)
    assert len(model.enc_W) == 2
    assert len(model.enc_b) == 2
    assert len(model.dec_W) == 2
    assert len(model.dec_b) == 2

    assert model.enc_W[0].shape == (6, 4)
    assert model.enc_b[0].shape == (1, 4)
    assert model.enc_W[1].shape == (4, 2)
    assert model.enc_b[1].shape == (1, 2)

    assert model.dec_W[0].shape == (2, 4)
    assert model.dec_b[0].shape == (1, 4)
    assert model.dec_W[1].shape == (4, 6)
    assert model.dec_b[1].shape == (1, 6)

    # Test mismatch assert
    with pytest.raises(AssertionError):
        MLPAutoencoder(encoder_sizes=[6, 4, 3], decoder_sizes=[2, 4, 6])

def test_mlp_ae_gradient_checks():
    np.random.seed(0)
    X = np.random.randn(4, 6)

    configs = [
        ("1 layer", [6, 4, 2], [2, 4, 6], 'tanh'),
        ("2 layers", [6, 5, 4, 2], [2, 4, 5, 6], 'tanh'),
        ("linear bottleneck", [6, 4, 2], [2, 4, 6], 'linear'),
    ]

    for name, enc_sizes, dec_sizes, bottleneck_act in configs:
        model = MLPAutoencoder(enc_sizes, dec_sizes, seed=1, bottleneck_activation=bottleneck_act)
        loss, genc_W, genc_b, gdec_W, gdec_b = model.loss_and_grad(X, l2=0.1)

        def f():
            l, *_ = model.loss_and_grad(X, l2=0.1)
            return l

        for i, W in enumerate(model.enc_W):
            err = relerr(genc_W[i], numgrad(f, W))
            assert err < 1e-3, f"Encoder W grad check failed for config {name}, layer {i}: err = {err}"
        for i, b in enumerate(model.enc_b):
            err = relerr(genc_b[i], numgrad(f, b))
            assert err < 1e-3, f"Encoder b grad check failed for config {name}, layer {i}: err = {err}"
        for i, W in enumerate(model.dec_W):
            err = relerr(gdec_W[i], numgrad(f, W))
            assert err < 1e-3, f"Decoder W grad check failed for config {name}, layer {i}: err = {err}"
        for i, b in enumerate(model.dec_b):
            err = relerr(gdec_b[i], numgrad(f, b))
            assert err < 1e-3, f"Decoder b grad check failed for config {name}, layer {i}: err = {err}"

def test_mlp_ae_fit_full_batch_and_minibatch():
    np.random.seed(42)
    # Generate simple synthetic 2D data embedded in 4D space
    N = 100
    latent = np.random.randn(N, 2)
    basis = np.random.randn(2, 4)
    X = np.tanh(latent @ basis)

    # Full batch check
    model_fb = MLPAutoencoder(encoder_sizes=[4, 2], decoder_sizes=[2, 4], seed=42)
    loss_before = np.mean((model_fb.reconstruct(X) - X) ** 2)
    model_fb.fit(X, epochs=50, lr=0.1, l2=1e-5, batch_size=None, noise_std=0.0)
    loss_after = np.mean((model_fb.reconstruct(X) - X) ** 2)
    assert loss_after < loss_before, "Full-batch fitting failed to reduce reconstruction error"

    # Minibatch check
    model_mb = MLPAutoencoder(encoder_sizes=[4, 2], decoder_sizes=[2, 4], seed=42)
    loss_before_mb = np.mean((model_mb.reconstruct(X) - X) ** 2)
    model_mb.fit(X, epochs=50, lr=0.1, l2=1e-5, batch_size=16, noise_std=0.0)
    loss_after_mb = np.mean((model_mb.reconstruct(X) - X) ** 2)
    assert loss_after_mb < loss_before_mb, "Minibatch fitting failed to reduce reconstruction error"

def test_mlp_ae_save_load_persistence():
    np.random.seed(42)
    X = np.random.randn(10, 8)
    model = MLPAutoencoder(encoder_sizes=[8, 4, 3], decoder_sizes=[3, 4, 8], seed=42)
    model.fit(X, epochs=10, lr=0.01, noise_std=0.02)

    recon_before = model.reconstruct(X)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        loaded = MLPAutoencoder.load(temp_path)

        assert list(loaded.encoder_sizes) == list(model.encoder_sizes)
        assert list(loaded.decoder_sizes) == list(model.decoder_sizes)

        recon_after = loaded.reconstruct(X)
        max_diff = np.max(np.abs(recon_before - recon_after))
        assert max_diff == 0.0, f"Loaded model is not bit-exact, max difference: {max_diff}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_mlp_ae_forward_and_latent_processing():
    model = MLPAutoencoder(encoder_sizes=[6, 3], decoder_sizes=[3, 6], seed=42)
    X = np.ones((5, 6))

    # Test forward returns correct lengths and shapes
    enc_pre, enc_act, dec_pre, dec_act = model._forward(X)
    assert len(enc_pre) == 1
    assert len(enc_act) == 2  # [X, z]
    assert len(dec_pre) == 1
    assert len(dec_act) == 2  # [z, E_recon]

    assert enc_act[0].shape == (5, 6)
    assert enc_act[1].shape == (5, 3)
    assert dec_act[0].shape == (5, 3)
    assert dec_act[1].shape == (5, 6)

    # Test encode, decode, reconstruct methods
    z = model.encode(X)
    assert z.shape == (5, 3)
    assert np.allclose(z, enc_act[-1])

    X_recon = model.decode(z)
    assert X_recon.shape == (5, 6)
    assert np.allclose(X_recon, dec_act[-1])

    recon_direct = model.reconstruct(X)
    assert np.allclose(recon_direct, X_recon)
