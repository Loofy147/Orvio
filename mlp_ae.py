"""
General-purpose nonlinear autoencoder: arbitrary-depth encoder -> bottleneck
-> arbitrary-depth decoder -> linear readout, hand-rolled backprop + Adam.

Generalized from a validated 3-layer prototype that was built to test one
specific finding: an autoencoder with a NONLINEAR ENCODER but LINEAR DECODER
can never beat PCA/SVD at the same bottleneck size (the decoder's linearity
caps the achievable reconstruction to the same affine subspace PCA already
optimizes exactly). A genuinely nonlinear DECODER breaks that ceiling.

VALIDATED FINDINGS (see gradcheck_generic.py, smoothness_test.py,
scale_and_persist_test.py in this directory for the actual runs):

1. Backprop is correct for arbitrary encoder/decoder depth and for both
   tanh and linear bottleneck activations. Gradient-checked across 4
   architectures, max relative error ~1e-7 to ~6e-5 (the latter is a
   float64 central-difference precision artifact confirmed via an eps
   sweep -- error is U-shaped in eps, which a real bug would not produce).

2. On the real 31-fact / 5-held-out-fact semantic embedding task (16-dim
   LSA space), this beats SVD/PCA at every bottleneck size z, including on
   held-out data (z=8: 100%/99.5% cosine vs SVD's 82%/87%).

3. LATENT SMOOTHNESS: the raw (noise_std=0) version reconstructs very
   well but its interpolation paths aren't fully monotonic -- 86% of
   random interpolation paths had zero similarity-monotonicity violations,
   14% had at least one (occasionally a path jumps through an unrelated
   concept at the midpoint instead of smoothly blending). Injecting small
   Gaussian noise into the bottleneck code during training (noise_std=0.05)
   fixes most of this almost for free: monotonic-path fraction rises to
   ~95%, average violations drop ~3x (0.18->0.06), while reconstruction
   cosine similarity drops less than 0.5 percentage points (sometimes
   improves on held-out data). Robust across 3 seeds tested. This is why
   noise_std defaults to 0.05 below rather than 0.

4. WIDTH VS DEPTH on small data: for the 31-example task, going from a
   1-hidden-layer decoder with 8 units to 16 units took TrCos from 82.8%
   to 99.6%. Adding a SECOND hidden layer at similar total capacity
   (8,8 or 12,8) did NOT recover that gain (stayed ~82.7-82.8%) under the
   same epoch budget. On small datasets, prefer a single wide hidden
   layer over multiple narrow ones unless you also raise epochs/tune lr
   for the deeper net.

5. SCALABILITY: on an 800-sample synthetic nonlinear dataset (real minibatch
   test, not toy-sized), minibatch training (batch_size=64) reached ~3x
   lower MSE than full-batch training in the same 300 epochs (0.008 vs
   0.026), and both comfortably beat the PCA baseline (0.033). Use
   batch_size on anything larger than a few hundred rows.

6. save()/load() round-trip verified bit-exact (0.0 max difference in
   reconstructions before vs after).

RECOMMENDED DEFAULTS: l2=1e-4 (l2=1e-2 was tested and collapses training --
too strong under Adam's adaptive scaling), noise_std=0.05, prefer width
over depth on small data, use batch_size on data much larger than ~200 rows.
"""
import numpy as np

class MLPAutoencoder:
    def __init__(self, encoder_sizes, decoder_sizes, seed=42, bottleneck_activation='tanh'):
        assert encoder_sizes[-1] == decoder_sizes[0], "bottleneck size mismatch"
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.bottleneck_activation = bottleneck_activation
        rng = np.random.RandomState(seed)
        self.enc_W, self.enc_b = [], []
        for i in range(len(encoder_sizes) - 1):
            fan_in = encoder_sizes[i]
            self.enc_W.append(rng.randn(encoder_sizes[i], encoder_sizes[i+1]) * np.sqrt(2.0 / fan_in))
            self.enc_b.append(np.zeros((1, encoder_sizes[i+1])))
        self.dec_W, self.dec_b = [], []
        for i in range(len(decoder_sizes) - 1):
            fan_in = decoder_sizes[i]
            self.dec_W.append(rng.randn(decoder_sizes[i], decoder_sizes[i+1]) * np.sqrt(2.0 / fan_in))
            self.dec_b.append(np.zeros((1, decoder_sizes[i+1])))
        self._init_adam()
        self.rng = rng

    def _all_params(self):
        return self.enc_W + self.enc_b + self.dec_W + self.dec_b

    def _init_adam(self):
        params = self._all_params()
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def _forward(self, X, noise_std=0.0, rng=None):
        """Returns (pre_activations, activations) lists for encoder and decoder,
        plus the final reconstruction. activations[0] = X."""
        enc_pre, enc_act = [], [X]
        a = X
        n_enc_layers = len(self.enc_W)
        for i, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
            h = a @ W + b
            enc_pre.append(h)
            is_bottleneck = (i == n_enc_layers - 1)
            if is_bottleneck:
                a = np.tanh(h) if self.bottleneck_activation == 'tanh' else h
            else:
                a = np.tanh(h)
            enc_act.append(a)
        z = enc_act[-1]
        if noise_std > 0:
            assert rng is not None
            z = z + rng.randn(*z.shape) * noise_std
        dec_pre, dec_act = [], [z]
        a = z
        n_dec_layers = len(self.dec_W)
        for i, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            h = a @ W + b
            dec_pre.append(h)
            is_last = (i == n_dec_layers - 1)
            a = h if is_last else np.tanh(h)
            dec_act.append(a)
        return enc_pre, enc_act, dec_pre, dec_act

    def reconstruct(self, X):
        *_, dec_act = self._forward(X)
        return dec_act[-1]

    def encode(self, X):
        _, enc_act, _, _ = self._forward(X)
        return enc_act[-1]

    def decode(self, Z):
        a = Z
        n_dec_layers = len(self.dec_W)
        for i, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            h = a @ W + b
            is_last = (i == n_dec_layers - 1)
            a = h if is_last else np.tanh(h)
        return a

    def loss_and_grad(self, X, l2=0.0, noise_std=0.0, rng=None):
        N, V = X.shape
        enc_pre, enc_act, dec_pre, dec_act = self._forward(X, noise_std=noise_std, rng=rng)
        E_recon = dec_act[-1]
        l2_term = 0.0
        if l2 > 0:
            for W in self.enc_W + self.dec_W:
                l2_term += 0.5 * l2 * np.sum(W ** 2)
        loss = np.mean((E_recon - X) ** 2) + l2_term

        grads_dec_W = [None] * len(self.dec_W)
        grads_dec_b = [None] * len(self.dec_b)
        grads_enc_W = [None] * len(self.enc_W)
        grads_enc_b = [None] * len(self.enc_b)

        d_out = 2.0 * (E_recon - X) / (N * V)
        n_dec = len(self.dec_W)
        delta = d_out
        for i in reversed(range(n_dec)):
            a_prev = dec_act[i]
            grads_dec_W[i] = a_prev.T @ delta + l2 * self.dec_W[i]
            grads_dec_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                d_a_prev = delta @ self.dec_W[i].T
                delta = d_a_prev * (1 - dec_act[i] ** 2)   # tanh derivative at layer i (dec_act[i]=tanh(pre))
        d_z = delta @ self.dec_W[0].T   # gradient flowing back into bottleneck code z

        n_enc = len(self.enc_W)
        if self.bottleneck_activation == 'tanh':
            delta = d_z * (1 - enc_act[-1] ** 2)
        else:
            delta = d_z
        for i in reversed(range(n_enc)):
            a_prev = enc_act[i]
            grads_enc_W[i] = a_prev.T @ delta + l2 * self.enc_W[i]
            grads_enc_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                d_a_prev = delta @ self.enc_W[i].T
                delta = d_a_prev * (1 - enc_act[i] ** 2)
        return loss, grads_enc_W, grads_enc_b, grads_dec_W, grads_dec_b

    def step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        genc_W, genc_b, gdec_W, gdec_b = grads
        params = self._all_params()
        all_grads = genc_W + genc_b + gdec_W + gdec_b
        self.t += 1
        new_params = []
        for i, (p, g) in enumerate(zip(params, all_grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g ** 2
            mhat = self.m[i] / (1 - beta1 ** self.t)
            vhat = self.v[i] / (1 - beta2 ** self.t)
            new_params.append(p - lr * mhat / (np.sqrt(vhat) + eps))
        n1, n2, n3, n4 = len(self.enc_W), len(self.enc_b), len(self.dec_W), len(self.dec_b)
        self.enc_W = new_params[0:n1]
        self.enc_b = new_params[n1:n1+n2]
        self.dec_W = new_params[n1+n2:n1+n2+n3]
        self.dec_b = new_params[n1+n2+n3:n1+n2+n3+n4]

    def fit(self, X, epochs=8000, lr=0.02, l2=1e-4, batch_size=None, noise_std=0.05, X_val=None, verbose_every=0, seed_shuffle=123):
        rng_shuffle = np.random.RandomState(seed_shuffle)
        rng_noise = np.random.RandomState(seed_shuffle + 1)
        N = X.shape[0]
        history = []
        for epoch in range(1, epochs + 1):
            if batch_size is None or batch_size >= N:
                batches = [X]
            else:
                idx = rng_shuffle.permutation(N)
                batches = [X[idx[i:i+batch_size]] for i in range(0, N, batch_size)]
            for Xb in batches:
                loss, genc_W, genc_b, gdec_W, gdec_b = self.loss_and_grad(Xb, l2=l2, noise_std=noise_std, rng=rng_noise)
                self.step((genc_W, genc_b, gdec_W, gdec_b), lr=lr)
            if verbose_every and epoch % verbose_every == 0:
                tr_loss = np.mean((self.reconstruct(X) - X) ** 2)
                v_loss = np.mean((self.reconstruct(X_val) - X_val) ** 2) if X_val is not None else None
                history.append((epoch, tr_loss, v_loss))
        return history

    def save(self, path):
        arrs = {}
        for i, W in enumerate(self.enc_W): arrs[f'encW{i}'] = W
        for i, b in enumerate(self.enc_b): arrs[f'encb{i}'] = b
        for i, W in enumerate(self.dec_W): arrs[f'decW{i}'] = W
        for i, b in enumerate(self.dec_b): arrs[f'decb{i}'] = b
        arrs['encoder_sizes'] = np.array(self.encoder_sizes)
        arrs['decoder_sizes'] = np.array(self.decoder_sizes)
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        encoder_sizes = list(d['encoder_sizes'])
        decoder_sizes = list(d['decoder_sizes'])
        model = cls(encoder_sizes, decoder_sizes)
        model.enc_W = [d[f'encW{i}'] for i in range(len(encoder_sizes) - 1)]
        model.enc_b = [d[f'encb{i}'] for i in range(len(encoder_sizes) - 1)]
        model.dec_W = [d[f'decW{i}'] for i in range(len(decoder_sizes) - 1)]
        model.dec_b = [d[f'decb{i}'] for i in range(len(decoder_sizes) - 1)]
        model._init_adam()
        return model
