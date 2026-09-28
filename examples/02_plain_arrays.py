"""No model object at all: a 2-layer network written with raw arrays, logged explicitly.

This is the "CS231n style" (W1, b1, W2, b2 and hand-written backprop). nnscope only needs the arrays:

    scope.log(step, weights=[W1, W2], biases=[b1, b2], grads=[dW1, dW2],
              activations=[h], deltas=[dz1, dz2], loss=loss)

The task is regression: fit the surface t = sin(3x) * cos(3y) with squared error.
"""
import time

import numpy as np

import nnscope
from _common import finish, parse, scope_kwargs

args = parse(__doc__, "runs/plain_arrays.html", 400)
rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, (800, 2))
T = (np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]))[:, None]
Xtr, Ttr, Xte, Tte = X[:640], T[:640], X[640:], T[640:]

H = 32
W1, b1 = rng.normal(0, np.sqrt(1 / 2), (2, H)), np.zeros(H)
W2, b2 = rng.normal(0, np.sqrt(1 / H), (H, 1)), np.zeros(1)
lr = 0.1

scope = nnscope.Scope("plain-arrays", **scope_kwargs(args))
for epoch in range(args.epochs):
    for i in range(0, len(Xtr), 32):
        x, t = Xtr[i:i + 32], Ttr[i:i + 32]
        z1 = x @ W1 + b1
        h = np.tanh(z1)
        y = h @ W2 + b2                                   # forward
        dz2 = 2 * (y - t) / len(x)                        # dL/dy for the mean squared error
        dW2, db2 = h.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ W2.T) * (1 - h ** 2)                 # backprop through tanh
        dW1, db1 = x.T @ dz1, dz1.sum(0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2                                    # SGD
    h = np.tanh(Xtr @ W1 + b1)
    loss = float(np.mean((h @ W2 + b2 - Ttr) ** 2))
    test = float(np.mean((np.tanh(Xte @ W1 + b1) @ W2 + b2 - Tte) ** 2))
    scope.log(epoch, weights=[W1, W2], biases=[b1, b2], grads=[dW1, dW2], bias_grads=[db1, db2],
              activations=[h], deltas=[dz1, dz2], loss=loss, metrics={"test_loss": test})
    if not args.headless:
        time.sleep(args.delay)
finish(scope, args)
