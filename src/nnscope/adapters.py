"""Turn "whatever the user has" into a uniform list of dense layers.

nnscope never runs your network. It only *reads* arrays you already have: weights, biases, their gradients and
(optionally) activations. This module finds them, either from explicit lists or by duck-typing common NumPy model
layouts, and normalises every weight matrix to the ``(in_features, out_features)`` orientation used for ``x @ W``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

WEIGHT_ATTRS = ("W", "weight", "weights", "kernel", "w")
BIAS_ATTRS = ("b", "bias", "biases")
WEIGHT_GRAD_ATTRS = ("dW", "grad_W", "W_grad", "dweight", "grad_weight", "weight_grad", "dw")
BIAS_GRAD_ATTRS = ("db", "grad_b", "b_grad", "dbias", "grad_bias", "bias_grad")
INPUT_CACHE_ATTRS = ("x", "input", "inputs", "a_prev", "x_cache", "cache_x")
ACTIVATION_NAMES = {
    "relu": "relu", "leakyrelu": "leaky_relu", "tanh": "tanh", "sigmoid": "sigmoid", "logistic": "sigmoid",
    "softmax": "softmax", "elu": "elu", "gelu": "gelu", "softplus": "softplus", "sine": "sine", "sin": "sine",
    "linear": "linear", "identity": "linear", "swish": "swish", "silu": "swish",
}

HELP = """nnscope could not find dense-layer weights. Pass them explicitly, for example

    scope.log(step, weights=[W1, W2], biases=[b1, b2], grads=[dW1, dW2], loss=loss)

or give a model object with a `.layers` list whose layers expose `W`/`b` (and `dW`/`db`),
a `.params` dict like {"W1": ..., "b1": ...}, or a list of (W, b) pairs."""


@dataclass
class LayerView:
    """One dense layer, normalised so that ``W.shape == (in_features, out_features)``."""

    W: np.ndarray
    b: Optional[np.ndarray] = None
    dW: Optional[np.ndarray] = None
    db: Optional[np.ndarray] = None
    act: Optional[str] = None          # activation applied to this layer's output ("relu", "tanh", ...)
    name: str = ""
    source: Any = None                 # the original layer object, if any (used to find cached inputs)

    @property
    def in_features(self) -> int:
        return int(self.W.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.W.shape[1])


def _arr(x: Any) -> Optional[np.ndarray]:
    return np.asarray(x) if isinstance(x, np.ndarray) or (x is not None and hasattr(x, "__array__")) else None


def _first_attr(obj: Any, names: Sequence[str], ndim: Optional[int] = None) -> Optional[np.ndarray]:
    for n in names:
        v = getattr(obj, n, None)
        a = _arr(v) if v is not None else None
        if a is not None and a.dtype.kind in "fiu" and (ndim is None or a.ndim == ndim):
            return a
    return None


def activation_name(obj: Any) -> Optional[str]:
    """Recognise an activation layer by its class name (ReLU, Tanh, Sigmoid, ...)."""
    key = re.sub(r"[^a-z]", "", type(obj).__name__.lower())
    return ACTIVATION_NAMES.get(key)


# ------------------------------------------------------------------ collection from different model layouts
def _from_layer_objects(layers: Sequence[Any]) -> List[LayerView]:
    views: List[LayerView] = []
    for obj in layers:
        if hasattr(obj, "layers") and not isinstance(obj, (list, tuple)):       # nested Sequential
            views.extend(_from_layer_objects(list(obj.layers)))
            continue
        W = _first_attr(obj, WEIGHT_ATTRS, ndim=2)
        if W is not None:
            views.append(LayerView(
                W=W, b=_first_attr(obj, BIAS_ATTRS), dW=_first_attr(obj, WEIGHT_GRAD_ATTRS, ndim=2),
                db=_first_attr(obj, BIAS_GRAD_ATTRS), name=type(obj).__name__, source=obj))
            continue
        act = activation_name(obj)
        if act and views and views[-1].act is None:
            views[-1].act = act
    return views


_KEY = re.compile(r"^(?P<kind>[A-Za-z]+)[_.]?(?P<idx>\d+)$")


def _from_dict(params: dict, grads: Optional[dict] = None) -> List[LayerView]:
    """CS231n-style dicts: {"W1": ..., "b1": ..., "W2": ...} (+ an optional grads dict with the same keys)."""
    layers: dict = {}
    for key, val in params.items():
        m = _KEY.match(str(key))
        if not m:
            continue
        kind, idx = m.group("kind"), int(m.group("idx"))
        slot = layers.setdefault(idx, {})
        if kind in WEIGHT_ATTRS:
            slot["W"], slot["kW"] = np.asarray(val), key
        elif kind in BIAS_ATTRS:
            slot["b"], slot["kb"] = np.asarray(val), key
    views = []
    for idx in sorted(layers):
        slot = layers[idx]
        if "W" not in slot or slot["W"].ndim != 2:
            continue
        dW = _arr(grads.get(slot["kW"])) if grads else None
        db = _arr(grads.get(slot.get("kb"))) if grads and slot.get("kb") else None
        views.append(LayerView(W=slot["W"], b=slot.get("b"), dW=dW, db=db, name=f"layer {idx}"))
    return views


def _from_sequence(items: Sequence[Any]) -> List[LayerView]:
    """A list of arrays ([W1, b1, W2, b2] or [W1, W2]), of (W, b) pairs, or of layer objects."""
    if items and not any(isinstance(it, (np.ndarray, tuple, list)) for it in items):
        return _from_layer_objects(items)
    views: List[LayerView] = []
    for it in items:
        if isinstance(it, (tuple, list)):
            W = np.asarray(it[0])
            b = np.asarray(it[1]) if len(it) > 1 and it[1] is not None else None
            views.append(LayerView(W=W, b=b))
        else:
            a = np.asarray(it)
            if a.ndim == 2:
                views.append(LayerView(W=a))
            elif a.ndim == 1 and views and views[-1].b is None:
                views[-1].b = a
    return views


# ------------------------------------------------------------------ orientation
def _chains(mats: List[np.ndarray], in_out: bool) -> bool:
    if in_out:
        return all(mats[k].shape[1] == mats[k + 1].shape[0] for k in range(len(mats) - 1))
    return all(mats[k].shape[0] == mats[k + 1].shape[1] for k in range(len(mats) - 1))


def detect_layout(views: List[LayerView]) -> str:
    """Return "in_out" (x @ W, W is (in, out)) or "out_in" (W @ x, W is (out, in))."""
    mats = [v.W for v in views]
    if len(mats) >= 2:
        io, oi = _chains(mats, True), _chains(mats, False)
        if io and not oi:
            return "in_out"
        if oi and not io:
            return "out_in"
    votes = 0                                            # ambiguous → let the biases decide
    for v in views:
        if v.b is not None and v.W.shape[0] != v.W.shape[1]:
            n = v.b.size
            votes += (n == v.W.shape[1]) - (n == v.W.shape[0])
    return "out_in" if votes < 0 else "in_out"


def normalise(views: List[LayerView], layout: str = "auto") -> List[LayerView]:
    if layout not in ("auto", "in_out", "out_in"):
        raise ValueError('layout must be "auto", "in_out" or "out_in"')
    if layout == "auto":
        layout = detect_layout(views)
    if layout == "out_in":
        for v in views:
            v.W = v.W.T
            if v.dW is not None:
                v.dW = v.dW.T
    for k, v in enumerate(views):
        if not v.name:
            v.name = f"layer {k + 1}"
        for attr in ("b", "db"):
            a = getattr(v, attr)
            if a is not None:
                setattr(v, attr, np.asarray(a).reshape(-1))
        if v.dW is not None and v.dW.shape != v.W.shape:
            v.dW = v.dW.T if v.dW.T.shape == v.W.shape else None
    return views


# ------------------------------------------------------------------ public entry point
def extract_layers(model: Any = None, *, weights: Optional[Sequence] = None, biases: Optional[Sequence] = None,
                   grads: Any = None, bias_grads: Optional[Sequence] = None, layout: str = "auto") -> List[LayerView]:
    """Collect dense layers from ``model`` or from explicit arrays. Raises ``TypeError`` with usage help if none are found."""
    if weights is not None:
        views = [LayerView(W=np.asarray(W)) for W in weights]
        for k, v in enumerate(views):
            if biases is not None and k < len(biases) and biases[k] is not None:
                v.b = np.asarray(biases[k])
            if isinstance(grads, (list, tuple)) and k < len(grads) and grads[k] is not None:
                v.dW = np.asarray(grads[k])
            if bias_grads is not None and k < len(bias_grads) and bias_grads[k] is not None:
                v.db = np.asarray(bias_grads[k])
    elif model is None:
        raise TypeError(HELP)
    elif hasattr(model, "layers") and not isinstance(model, dict):
        views = _from_layer_objects(list(model.layers))
    elif isinstance(getattr(model, "params", None), dict):
        g = grads if isinstance(grads, dict) else getattr(model, "grads", None)
        views = _from_dict(model.params, g if isinstance(g, dict) else None)
    elif isinstance(model, dict):
        views = _from_dict(model, grads if isinstance(grads, dict) else None)
    elif isinstance(model, (list, tuple)):
        views = _from_sequence(model)
        if isinstance(grads, (list, tuple)):
            for v, g in zip(views, grads):
                v.dW = np.asarray(g) if g is not None else None
    else:
        raise TypeError(HELP)
    views = [v for v in views if v.W.ndim == 2]
    if not views:
        raise TypeError(HELP)
    return normalise(views, layout)


def cached_activations(views: List[LayerView]) -> List[Optional[np.ndarray]]:
    """Best-effort: the output of hidden layer k is the cached *input* of layer k+1 (e.g. HW-style ``Linear.x``).

    Returns one entry per layer (the last one, the network output, is usually unknown → None).
    """
    out: List[Optional[np.ndarray]] = [None] * len(views)
    for k in range(len(views) - 1):
        nxt = views[k + 1].source
        if nxt is None:
            continue
        x = _first_attr(nxt, INPUT_CACHE_ATTRS)
        if x is not None and x.ndim == 2 and x.shape[1] == views[k].out_features:
            out[k] = x
    return out
