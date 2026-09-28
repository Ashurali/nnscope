"""Spectral bias: fit an image as a function f(x, y) -> (r, g, b), with raw coordinates vs Fourier features.

    python 05_inr_spectral_bias.py              # raw (x, y): the loss plateaus, fine detail never appears
    python 05_inr_spectral_bias.py --fourier    # gamma(v) = [sin 2 pi vB, cos 2 pi vB]: the loss keeps falling

The Fourier mapping is a fixed input transform, so the network nnscope shows starts at the 2m encoded features.
Compare the two runs' loss curves and gradient flow.
"""
import time

import numpy as np

import nnscope
from _common import finish, parse, scope_kwargs
from _mlp import Linear, ReLU, Sequential, mse

args = parse(__doc__, "runs/inr.html", 300, fourier=(False, "use Fourier features"), sigma=(1.0, "Fourier bandwidth sigma (coords in [-1, 1])"))
rng = np.random.default_rng(4)
S = 64                                                    # procedural 64x64 image with sharp edges: rings + a star
ys, xs = np.mgrid[-1:1:S * 1j, -1:1:S * 1j]
r, th = np.hypot(xs, ys), np.arctan2(ys, xs)
img = np.stack([(np.sin(12 * r) > 0) * 0.9, (r < 0.45 + 0.2 * np.cos(5 * th)) * 0.9, 0.2 + 0.6 * (xs > 0)], -1)
coords, colors = np.column_stack([xs.ravel(), ys.ravel()]), img.reshape(-1, 3)
perm = rng.permutation(len(coords))
tr, te = perm[: int(0.9 * len(perm))], perm[int(0.9 * len(perm)):]

if args.fourier:
    B = rng.normal(0, args.sigma, (2, 64))
    F = np.concatenate([np.sin(2 * np.pi * coords @ B), np.cos(2 * np.pi * coords @ B)], axis=1)
else:
    F = coords
model = Sequential(Linear(F.shape[1], 128, rng), ReLU(), Linear(128, 128, rng), ReLU(), Linear(128, 128, rng), ReLU(),
                   Linear(128, 3, rng, init="xavier"))
velocity = {id(l): (np.zeros_like(l.W), np.zeros_like(l.b)) for l in model.layers if hasattr(l, "W")}

scope = nnscope.Scope("inr-fourier" if args.fourier else "inr-raw", **scope_kwargs(args))
for epoch in range(args.epochs):
    order = rng.permutation(tr)
    for i in range(0, len(order), 128):
        b = order[i:i + 128]
        _, g = mse(model.forward(F[b]), colors[b])
        model.backward(g)
        for l in model.layers:                                # SGD with momentum 0.9
            if hasattr(l, "W"):
                vW, vb = velocity[id(l)]
                vW *= 0.9
                vW -= 0.05 * l.dW
                vb *= 0.9
                vb -= 0.05 * l.db
                l.W += vW
                l.b += vb
    test_loss, _ = mse(model.forward(F[te]), colors[te])
    loss, g = mse(model.forward(F[tr]), colors[tr])           # full train pass: fresh grads + cached activations
    model.backward(g)
    scope.log(epoch, model=model, loss=loss, metrics={"test_loss": test_loss})
    if not args.headless:
        time.sleep(args.delay)
finish(scope, args)
