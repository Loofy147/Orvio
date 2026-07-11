import numpy as np
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
        g[idx] = (fp - fm) / (2*eps)
        it.iternext()
    return g

def relerr(a, b): return np.max(np.abs(a-b) / (np.abs(a)+np.abs(b)+1e-8))

np.random.seed(0)
X = np.random.randn(8, 6)

configs = [
    ("1 enc layer / 1 dec layer (matches validated prototype)", [6,4,2], [2,4,6]),
    ("2 enc layers / 2 dec layers (deeper, new)",                [6,5,4,2], [2,4,5,6]),
    ("3 enc layers / 1 dec layer (asymmetric, new)",             [6,5,4,3,2], [2,6]),
    ("linear bottleneck variant",                                 [6,4,2], [2,4,6]),
]

for name, enc_sizes, dec_sizes in configs:
    bottleneck_act = 'linear' if 'linear bottleneck' in name else 'tanh'
    model = MLPAutoencoder(enc_sizes, dec_sizes, seed=1, bottleneck_activation=bottleneck_act)
    loss, genc_W, genc_b, gdec_W, gdec_b = model.loss_and_grad(X, l2=0.1)  # l2>0 so that term is checked too

    def f():
        l, *_ = model.loss_and_grad(X, l2=0.1)
        return l

    max_err = 0.0
    for i, W in enumerate(model.enc_W):
        err = relerr(genc_W[i], numgrad(f, W)); max_err = max(max_err, err)
    for i, b in enumerate(model.enc_b):
        err = relerr(genc_b[i], numgrad(f, b)); max_err = max(max_err, err)
    for i, W in enumerate(model.dec_W):
        err = relerr(gdec_W[i], numgrad(f, W)); max_err = max(max_err, err)
    for i, b in enumerate(model.dec_b):
        err = relerr(gdec_b[i], numgrad(f, b)); max_err = max(max_err, err)
    status = "PASS" if max_err < 1e-3 else "FAIL"
    print(f"[{status}] {name}: enc={enc_sizes} dec={dec_sizes}  max relative error = {max_err:.2e}")
