"""Quickstart: watch a small NumPy MLP learn XOR-like data, live.

    python 01_quickstart_xor.py              # live dashboard in your browser; keeps serving after training (Ctrl+C to stop)
    python 01_quickstart_xor.py --no-open    # live server, but don't open a browser tab
    python 01_quickstart_xor.py --headless   # no server: just train and write runs/xor.html + runs/xor.svg
"""
import argparse
import time

import numpy as np

import nnscope
from _mlp import Linear, ReLU, Sequential, Tanh, softmax_ce

ap = argparse.ArgumentParser()
ap.add_argument("--no-open", action="store_true", help="don't open a browser tab")
ap.add_argument("--headless", action="store_true", help="no live server; only write the replay HTML/SVG")
ap.add_argument("--delay", type=float, default=0.01, help="seconds to sleep per epoch so you can watch")
ap.add_argument("--out", default="runs/xor.html")
ap.add_argument("--epochs", type=int, default=300)
args = ap.parse_args()

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (600, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]

model = Sequential(Linear(2, 16, rng), ReLU(), Linear(16, 8, rng), Tanh(), Linear(8, 2, rng, init="xavier"))

scope = nnscope.Scope("xor-mlp", live=not args.headless, open_browser=not (args.no_open or args.headless))
for epoch in range(args.epochs):
    for i in range(0, len(Xtr), 32):                             # plain mini-batch SGD
        loss, g, _ = softmax_ce(model.forward(Xtr[i:i + 32]), ytr[i:i + 32])
        model.backward(g)
        model.step(0.1)
    train_loss, g, _ = softmax_ce(model.forward(Xtr), ytr)       # full-batch pass: fresh grads + cached activations
    model.backward(g)
    hidden = model.layers[3].forward(model.layers[2].forward(model.layers[1].forward(model.layers[0].forward(Xtr))))
    test_loss, _, p = softmax_ce(model.forward(Xte), yte)
    scope.log(epoch, model=model, loss=train_loss,
              metrics={"test_loss": test_loss, "test_error": float(np.mean(p.argmax(1) != yte))},
              latent=hidden, labels=ytr)
    if not args.headless:
        time.sleep(args.delay)                                    # slow down a little so you can watch

print("saved", scope.save_html(args.out))
scope.snapshot(args.out.replace(".html", ".svg"))
if not args.headless:
    scope.wait()
