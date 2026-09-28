"""Numbers the dashboard shows: norms, update ratios, histograms, activation statistics and 2-D projections."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

SATURATING = {"tanh": (-0.97, 0.97), "sigmoid": (0.03, 0.97)}   # outside this range the unit's gradient ≈ 0


def sig(x: np.ndarray, digits: int = 4) -> list:
    """Round to `digits` significant figures and return a JSON-friendly nested list."""
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a == 0, 0, np.floor(np.log10(np.abs(a))))
    scale = 10.0 ** (digits - 1 - mag)
    r = np.round(a * scale) / scale
    r[~np.isfinite(r)] = 0.0
    return r.tolist()


def fnum(x: Optional[float], digits: int = 5) -> Optional[float]:
    if x is None or not np.isfinite(x):
        return None
    return float(f"{float(x):.{digits}g}")


def select_neurons(n: int, max_n: int) -> List[int]:
    """Indices of at most `max_n` neurons, evenly spread over `n` (all of them if n ≤ max_n)."""
    if n <= max_n:
        return list(range(n))
    return sorted(set(np.round(np.linspace(0, n - 1, max_n)).astype(int).tolist()))


def histogram(x: Optional[np.ndarray], bins: int = 30) -> Optional[dict]:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    counts, _ = np.histogram(a, bins=bins, range=(lo, hi))
    return {"lo": fnum(lo), "hi": fnum(hi), "counts": counts.tolist()}


def layer_numbers(W: np.ndarray, dW: Optional[np.ndarray], prev_W: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Per-layer scalars logged every step."""
    w_norm = float(np.linalg.norm(W))
    out = {
        "w_norm": fnum(w_norm),
        "w_mean_abs": fnum(np.mean(np.abs(W))),
        "g_norm": fnum(np.linalg.norm(dW)) if dW is not None else None,
        "g_mean_abs": fnum(np.mean(np.abs(dW))) if dW is not None else None,
        "update_ratio": None,
    }
    if prev_W is not None and prev_W.shape == W.shape and w_norm > 0:
        out["update_ratio"] = fnum(np.linalg.norm(W - prev_W) / max(np.linalg.norm(prev_W), 1e-12))
    return out


def activation_numbers(A: Optional[np.ndarray], act: Optional[str]) -> Optional[dict]:
    """Per-neuron statistics of a batch of activations A (batch, units).

    * relu-like: a unit is *active* when its output > 0; a unit never active on the batch is *dead*.
    * tanh / sigmoid: a unit is *active* when it is not saturated; one saturated on every sample is flagged *saturated*.
    """
    if A is None:
        return None
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    A = A.reshape(A.shape[0], -1)
    if act in SATURATING:
        lo, hi = SATURATING[act]
        active = (A > lo) & (A < hi)
        kind = "saturated"
    else:
        active = A > 0 if act in (None, "relu", "leaky_relu", "elu", "gelu", "swish", "softplus") else np.abs(A) > 1e-8
        kind = "dead"
    frac = active.mean(axis=0)
    dead = int(np.sum(frac == 0))
    return {
        "mean": A.mean(axis=0), "std": A.std(axis=0), "frac_active": frac,
        "layer_mean": fnum(A.mean()), "layer_frac_active": fnum(frac.mean()), "dead": dead, "dead_kind": kind,
        "batch": int(A.shape[0]),
    }


def project_2d(Z: np.ndarray, max_points: int = 1000, seed: int = 0):
    """2-D view of features Z (n, k): raw coordinates if k ≤ 2, else PCA via SVD.

    Returns (coords (m, 2), explained_variance_ratio or None, indices of the m points used).
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z.reshape(Z.shape[0], -1)
    idx = np.arange(Z.shape[0])
    if Z.shape[0] > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(Z.shape[0], max_points, replace=False))
    Z = Z[idx]
    if Z.shape[1] == 1:
        return np.column_stack([Z[:, 0], np.zeros(len(Z))]), None, idx
    if Z.shape[1] == 2:
        return Z, None, idx
    Zc = Z - Z.mean(axis=0)
    _, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    Vt = Vt[:2] * np.sign(Vt[np.arange(2), np.argmax(np.abs(Vt[:2]), axis=1)])[:, None]   # fixed sign → no flipping
    var = s ** 2
    explained = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    return Zc @ Vt[:2].T, explained, idx
