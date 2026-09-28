"""Plain-language diagnostics: turn the logged numbers into hints a student can act on.

Each rule fires once when its condition appears and re-arms after the condition clears, so the feed reads like a
training diary ("step 120: layer 2 has 12 of 16 dead ReLUs") rather than a stream of repeats.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _fmt_ratio(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


class InsightEngine:
    def __init__(self, patience: int = 3, warmup: int = 5):
        self.patience = patience            # a condition must hold for this many logs in a row before it is reported
        self.warmup = warmup                # rules about learning speed wait this many logs (early steps are special)
        self.active: Dict[str, int] = {}    # key → consecutive logs the condition held
        self.clear: Dict[str, int] = {}     # key → consecutive logs it has been clear (re-arm needs several)
        self.reported: set = set()
        self.best_test: Optional[tuple] = None   # (loss, step, train_loss at that step)
        self.n = 0

    def _check(self, key: str, cond: bool, step, level: str, panel: str, message: str, out: List[dict], immediate=False,
               clear: Optional[bool] = None):
        """Report `key` once when `cond` holds; re-arm only after `clear` (default: not cond) held for 3×patience logs.

        The separate clear condition and the re-arm delay (hysteresis) stop a value hovering at a threshold from
        re-firing the same hint over and over.
        """
        if not cond:
            self.active.pop(key, None)
            is_clear = (not cond) if clear is None else clear
            self.clear[key] = self.clear.get(key, 0) + 1 if is_clear else 0
            if self.clear[key] >= 3 * self.patience:
                self.reported.discard(key)
            return
        self.clear[key] = 0
        self.active[key] = self.active.get(key, 0) + 1
        if key not in self.reported and (immediate or self.active[key] >= self.patience):
            self.reported.add(key)
            out.append({"step": step, "level": level, "kind": key.split(":")[0], "panel": panel, "message": message})

    def update(self, record: dict, structure: dict) -> List[dict]:
        out: List[dict] = []
        self.n += 1
        step, layers, scalars = record["step"], record["layers"], record["scalars"]
        names = [f"layer {k + 1}" for k in range(len(layers))]

        loss = scalars.get("loss")
        self._check("nan", loss is not None and not np.isfinite(loss), step, "error", "loss",
                    "The loss is NaN/inf: training diverged. Lower the learning rate or check for log(0) / division by zero.",
                    out, immediate=True)

        g = [l.get("g_mean_abs") for l in layers]
        if len(g) >= 2 and all(v is not None for v in g) and g[-1] and g[-1] > 0:
            ratio = g[-1] / max(g[0], 1e-30)
            self._check("vanishing", ratio > 1e3, step, "warn", "health", clear=ratio < 3e2, message=
                        f"Vanishing gradients: layer 1 receives {_fmt_ratio(ratio)}× smaller gradients than layer {len(g)}. "
                        "Early layers barely learn. Saturating activations (sigmoid/tanh) or poor initialisation are the "
                        "usual causes; try ReLU/tanh or He/Xavier init.", out=out)
            self._check("exploding", max(v for v in g) > 1e2, step, "warn", "health",
                        "Exploding gradients: mean |∂L/∂W| is above 100. Expect unstable jumps in the loss; lower the learning "
                        "rate or clip gradients.", out)

        ratios = [l.get("update_ratio") for l in layers if l.get("update_ratio") is not None]
        if ratios and self.n > self.warmup:
            med = float(np.median(ratios))
            self._check("lr_high", med > 0.1, step, "warn", "health", clear=med < 0.03, message=
                        f"Weights keep changing by ~{med:.0%} between logs (update ratio ‖ΔW‖/‖W‖). Per single SGD step a "
                        "healthy ratio is about 0.1%, so even when logging once per epoch this is high: the learning rate is "
                        "probably too large (watch for a noisy or rising loss).", out=out)
            self._check("lr_low", med < 1e-6, step, "info", "health",
                        f"Weights barely move (update ratio {med:.1e}). The learning rate may be too small, or training has converged.",
                        out)

        for k, l in enumerate(layers):
            act = (structure["layers"][k].get("act") if k < len(structure["layers"]) else None) or "unit"
            n, dead = structure["layers"][k]["out"], l.get("dead")
            kind = l.get("dead_kind") or "dead"
            if dead is None or n <= 1:
                continue
            word = "dead" if kind == "dead" else "saturated"
            self._check(f"{word}:{k}", dead / n >= 0.5, step, "warn", "activations",
                        f"{dead} of {n} {act} units in {names[k]} are {word} on the logged batch "
                        + ("(never positive → zero gradient forever). Lower the learning rate, use He init or LeakyReLU."
                           if word == "dead" else "(stuck near ±1 → gradient ≈ 0). Inputs to this layer are too large; "
                           "normalise inputs or use smaller initial weights."), out)

        test_key = next((k for k in scalars if ("test" in k or "val" in k) and "loss" in k), None)
        if test_key and loss is not None and scalars[test_key] is not None and np.isfinite(scalars[test_key]):
            t = scalars[test_key]
            if self.best_test is None or t < self.best_test[0]:
                self.best_test = (t, step, loss)
            b_loss, b_step, b_train = self.best_test
            self._check("overfit", t > 1.15 * b_loss and loss < 0.9 * b_train, step, "warn", "loss",
                        f"Overfitting: {test_key} has risen {t / b_loss - 1:.0%} above its minimum at step {b_step} while the "
                        "training loss keeps falling. Early stopping at that step, more data, or weight decay would help.", out)
        return out
