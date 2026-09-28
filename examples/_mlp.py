"""A tiny NumPy MLP in the style of a typical deep-learning homework (used by the examples).

Each Linear layer stores W (in, out), b, their gradients dW/db, and caches its input x, which is
everything nnscope needs, so `scope.log(step, model=model, loss=loss)` just works.
"""
import numpy as np


class Linear:
    def __init__(self, n_in, n_out, rng, init="he"):
        scale = np.sqrt((2.0 if init == "he" else 1.0) / n_in)
        self.W = rng.normal(0.0, scale, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW[...] = self.x.T @ g
        self.db[...] = g.sum(axis=0)
        return g @ self.W.T


class ReLU:
    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, g):
        return g * self.mask


class Tanh:
    def forward(self, z):
        self.y = np.tanh(z)
        return self.y

    def backward(self, g):
        return g * (1 - self.y ** 2)


class Sigmoid:
    def forward(self, z):
        self.y = 0.5 * (1 + np.tanh(0.5 * z))
        return self.y

    def backward(self, g):
        return g * self.y * (1 - self.y)


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db


def softmax_ce(logits, y):
    """Mean cross-entropy and its gradient w.r.t. the logits (y = integer labels)."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    g = p.copy()
    g[np.arange(n), y] -= 1
    return loss, g / n, p


def mse(pred, t):
    d = pred - t
    return float(np.mean(d ** 2)), 2 * d / d.size
