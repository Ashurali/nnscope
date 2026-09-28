"""Watch a representation form: a 2-class classifier with a k-unit tanh bottleneck before the output layer.

Run with different --k (2, 3, 8, 32) and compare the latent-space panel along the timeline: the classes start mixed
and are pulled apart along the one direction the output layer uses. A small k squeezes each class into a tight cluster.
"""
import time

import numpy as np

import nnscope
from _common import finish, parse, scope_kwargs
from _mlp import Linear, ReLU, Sequential, Tanh, softmax_ce

args = parse(__doc__, "runs/latent.html", 400, k=(2, "units in the bottleneck (latent) layer"))
rng = np.random.default_rng(2)
centers = rng.normal(0, 1.2, (4, 20))


def make(n):
    """Each class is a mixture of two Gaussian blobs in 20-D."""
    y = rng.integers(0, 2, n)
    X = centers[2 * y + rng.integers(0, 2, n)] + rng.normal(0, 1.0, (n, 20))
    return X, y


X, y = make(700)
Xtr, ytr, Xte, yte = X[:560], y[:560], X[560:], y[560:]
model = Sequential(Linear(20, 32, rng), ReLU(), Linear(32, 16, rng), ReLU(),
                   Linear(16, args.k, rng, init="xavier"), Tanh(),          # <- the latent layer
                   Linear(args.k, 2, rng, init="xavier"))

scope = nnscope.Scope(f"latent-k{args.k}", **scope_kwargs(args))
for epoch in range(args.epochs):
    order = rng.permutation(len(Xtr))
    for i in range(0, len(Xtr), 16):
        b = order[i:i + 16]
        _, g, _ = softmax_ce(model.forward(Xtr[b]), ytr[b])
        model.backward(g)
        model.step(0.02)
    test_loss, _, p = softmax_ce(model.forward(Xte), yte)
    loss, g, _ = softmax_ce(model.forward(Xtr), ytr)          # full train pass: fresh grads + cached activations
    model.backward(g)
    latent = model.layers[-1].x                                 # input of the output layer = the latent features
    scope.log(epoch, model=model, loss=loss, latent=latent, labels=ytr,
              metrics={"test_loss": test_loss, "test_error": float(np.mean(p.argmax(1) != yte))})
    if not args.headless:
        time.sleep(args.delay)
finish(scope, args)
