"""Vanishing gradients, made visible: a deep sigmoid network vs the same network with ReLU.

    python 04_vanishing_gradients.py --act sigmoid   # gradient-flow bars shrink towards the input; an insight explains why
    python 04_vanishing_gradients.py --act relu      # bars of similar height; the network actually learns
"""
import time

import numpy as np

import nnscope
from _common import finish, parse, scope_kwargs
from _mlp import Linear, ReLU, Sequential, Sigmoid, softmax_ce

args = parse(__doc__, "runs/vanishing.html", 200, act=("sigmoid", "sigmoid or relu"), depth=(6, "number of hidden layers"))
rng = np.random.default_rng(3)
n = 600                                                   # two interleaved spirals
t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
cls = rng.integers(0, 2, n)
X = np.column_stack([t * np.cos(t + np.pi * cls), t * np.sin(t + np.pi * cls)]) / (3 * np.pi)
X += rng.normal(0, 0.02, X.shape)

Act, init = (Sigmoid, "xavier") if args.act == "sigmoid" else (ReLU, "he")
layers, d = [], 2
for _ in range(args.depth):
    layers += [Linear(d, 32, rng, init=init), Act()]
    d = 32
model = Sequential(*layers, Linear(d, 2, rng, init="xavier"))

scope = nnscope.Scope(f"deep-{args.act}", **scope_kwargs(args))
for epoch in range(args.epochs):
    order = rng.permutation(n)
    for i in range(0, n, 32):
        b = order[i:i + 32]
        _, g, _ = softmax_ce(model.forward(X[b]), cls[b])
        model.backward(g)
        model.step(0.1)
    loss, g, p = softmax_ce(model.forward(X), cls)
    model.backward(g)
    scope.log(epoch, model=model, loss=loss, metrics={"error": float(np.mean(p.argmax(1) != cls))})
    if not args.headless:
        time.sleep(args.delay)
finish(scope, args)
