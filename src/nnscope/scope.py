"""The `Scope`: record what a NumPy network does during training and show it live or as a replay."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .adapters import LayerView, cached_activations, extract_layers
from .insights import InsightEngine
from .stats import activation_numbers, fnum, histogram, layer_numbers, project_2d, select_neurons, sig


class Scope:
    """Watch a NumPy neural network learn.

    >>> scope = nnscope.Scope("my-mlp")                # live dashboard at http://127.0.0.1:8765
    >>> for epoch in range(epochs):
    ...     ...                                          # your own forward / backward / SGD
    ...     scope.log(epoch, model=model, loss=loss, metrics={"test_loss": test_loss})
    >>> scope.save_html("run.html")                      # standalone replay, no server needed

    Parameters
    ----------
    name: shown in the dashboard header.
    live: start the local live dashboard (a tiny stdlib HTTP server in a background thread).
    port: first port to try (the next free one is used if taken). ``host`` defaults to localhost only.
    open_browser: open the dashboard in the default browser when the server starts.
    max_neurons: at most this many neurons per layer are drawn (evenly spaced; statistics always use all of them).
    snapshot_every: record a full snapshot (edges, histograms, latent) every N logs. Default: adaptive, keeping at
        most ``max_snapshots`` snapshots spread over the whole run.
    layout: weight orientation, "auto" (default), "in_out" (x @ W) or "out_in" (W @ x).
    """

    def __init__(self, name: str = "run", *, live: bool = True, port: int = 8765, host: str = "127.0.0.1",
                 open_browser: bool = True, max_neurons: int = 32, snapshot_every: Optional[int] = None,
                 max_snapshots: int = 300, max_records: int = 5000, layout: str = "auto", latent_points: int = 1000,
                 quiet: bool = False):
        self.name = name
        self.max_neurons, self.layout, self.latent_points, self.quiet = max_neurons, layout, latent_points, quiet
        self.max_snapshots, self.max_records = max_snapshots, max_records
        self._fixed_every = snapshot_every
        self._every = snapshot_every or 1
        self._record_every = 1
        self.structure: Optional[dict] = None
        self.records: List[dict] = []
        self.snapshots: List[dict] = []
        self.insights: List[dict] = []
        self._engine = InsightEngine()
        self._prev_W: Optional[List[np.ndarray]] = None
        self._n = 0
        self._lock = threading.Lock()
        self._listeners: List[Any] = []
        self._server = None
        self.url: Optional[str] = None
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        if live:
            from .server import start_server
            self._server, self.url = start_server(self, host, port)
            if not quiet:
                print(f"nnscope: live dashboard at {self.url}")
            if open_browser:
                import webbrowser
                threading.Thread(target=webbrowser.open, args=(self.url,), daemon=True).start()

    # ------------------------------------------------------------------ logging
    def log(self, step: Optional[int] = None, *, model: Any = None, loss: Optional[float] = None,
            metrics: Optional[Dict[str, float]] = None, weights: Optional[Sequence] = None,
            biases: Optional[Sequence] = None, grads: Any = None, bias_grads: Optional[Sequence] = None,
            activations: Optional[Sequence] = None, deltas: Optional[Sequence] = None,
            latent: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
            snapshot: Optional[bool] = None) -> None:
        """Record one training step.

        Pass either ``model`` (auto-detected) or explicit ``weights`` (+ ``biases``, ``grads``). Optional extras:
        ``activations`` = per-layer outputs on a batch, ``deltas`` = per-layer ∂L/∂z on a batch, ``latent`` (+``labels``)
        = features of any layer for the latent scatter. ``snapshot`` forces (True) or skips (False) a full snapshot.
        """
        step = self._n if step is None else step
        views = extract_layers(model, weights=weights, biases=biases, grads=grads, bias_grads=bias_grads,
                               layout=self.layout)
        acts = list(activations) if activations is not None else cached_activations(views)
        acts += [None] * (len(views) - len(acts))
        act_stats = [activation_numbers(a, v.act) for a, v in zip(acts, views)]
        new_structure = self._structure_for(views)
        messages: List[dict] = []
        if self.structure is None or new_structure["signature"] != self.structure["signature"]:
            self.structure, self._prev_W = new_structure, None
            messages.append({"type": "structure", "structure": new_structure})

        scalars = {"loss": fnum(loss)} if loss is not None else {}
        for k, v in (metrics or {}).items():
            scalars[str(k)] = fnum(v)
        layers = []
        for k, v in enumerate(views):
            nums = layer_numbers(v.W, v.dW, self._prev_W[k] if self._prev_W else None)
            s = act_stats[k]
            if s is not None:
                nums.update(act_mean=s["layer_mean"], frac_active=s["layer_frac_active"], dead=s["dead"],
                            dead_kind=s["dead_kind"])
            if deltas is not None and k < len(deltas) and deltas[k] is not None:
                nums["delta_mean_abs"] = fnum(np.mean(np.abs(deltas[k])))
            layers.append(nums)
        record = {"type": "record", "step": step, "t": round(time.time(), 3), "scalars": scalars, "layers": layers}

        take = snapshot if snapshot is not None else (self._n % self._every == 0)
        snap = self._snapshot(step, views, act_stats, latent, labels) if take else None
        new_insights = self._engine.update(record, self.structure)
        self._prev_W = [v.W.copy() for v in views]
        self._n += 1

        with self._lock:
            if self._n % self._record_every == 0 or take:
                self.records.append(record)
                messages.append(record)
            if snap is not None:
                self.snapshots.append(snap)
                messages.append(snap)
            for ins in new_insights:
                self.insights.append(ins)
                messages.append({"type": "insight", **ins})
            self._thin()
        self._emit(messages)

    def _structure_for(self, views: List[LayerView]) -> dict:
        layers = []
        for k, v in enumerate(views):
            layers.append({
                "name": v.name, "in": v.in_features, "out": v.out_features, "act": v.act,
                "params": int(v.W.size + (v.b.size if v.b is not None else 0)),
                "shown_in": select_neurons(v.in_features, self.max_neurons),
                "shown_out": select_neurons(v.out_features, self.max_neurons),
                "has_grad": v.dW is not None, "has_bias": v.b is not None,
            })
        sig_ = tuple((l["in"], l["out"], l["act"]) for l in layers)
        return {"layers": layers, "signature": str(sig_)}

    def _snapshot(self, step, views, act_stats, latent, labels) -> dict:
        st = self.structure["layers"]
        snap: Dict[str, Any] = {"type": "snapshot", "step": step, "W": [], "G": [], "B": [], "GB": [], "nodes": [],
                                "hist": []}
        for k, v in enumerate(views):
            si, so = st[k]["shown_in"], st[k]["shown_out"]
            snap["W"].append(sig(v.W[np.ix_(si, so)]))
            snap["G"].append(sig(v.dW[np.ix_(si, so)]) if v.dW is not None else None)
            snap["B"].append(sig(v.b[so]) if v.b is not None else None)
            snap["GB"].append(sig(v.db[so]) if v.db is not None else None)
            s = act_stats[k]
            snap["nodes"].append({"mean": sig(s["mean"][so]), "std": sig(s["std"][so]),
                                  "frac": sig(s["frac_active"][so])} if s is not None else None)
            snap["hist"].append({"w": histogram(v.W), "g": histogram(v.dW)})
        if latent is not None:
            xy, explained, idx = project_2d(latent, self.latent_points)
            lab = np.asarray(labels).reshape(-1)[idx] if labels is not None else None
            if lab is not None and lab.dtype.kind not in "iub":
                lab = np.unique(lab, return_inverse=True)[1] if lab.dtype.kind in "OUS" else lab
            snap["latent"] = {"xy": sig(xy), "labels": lab.tolist() if lab is not None else None,
                              "explained": sig(explained) if explained is not None else None,
                              "dim": int(np.asarray(latent).reshape(len(np.asarray(latent)), -1).shape[1])}
        return snap

    def _thin(self) -> None:
        """Keep memory bounded on long runs: drop every other snapshot/record and halve the rate going forward."""
        if len(self.snapshots) > self.max_snapshots and self._fixed_every is None:
            self.snapshots = self.snapshots[::2]
            self._every *= 2
        if len(self.records) > self.max_records:
            self.records = self.records[::2]
            self._record_every *= 2

    # ------------------------------------------------------------------ outputs
    def data(self) -> dict:
        """Everything recorded so far, in the format the dashboard reads (JSON-serialisable)."""
        with self._lock:
            return {"name": self.name, "version": __version__, "created": self.created, "structure": self.structure,
                    "records": list(self.records), "snapshots": list(self.snapshots), "insights": list(self.insights)}

    def save_html(self, path: str) -> str:
        """Write a standalone, offline dashboard of the whole run (timeline replay). Returns the path."""
        from .export import save_html
        return save_html(self.data(), path)

    def snapshot(self, path: str, step: Optional[int] = None) -> str:
        """Save an image of the network structure + loss curve at ``step`` (default: latest). ``.svg`` or ``.png``."""
        from .export import snapshot_image
        return snapshot_image(self.data(), path, step)

    def show(self, height: int = 820):
        """Inline the live dashboard in a Jupyter notebook."""
        if self.url is None:
            raise RuntimeError("show() needs the live server: create the Scope with live=True")
        from IPython.display import IFrame
        return IFrame(self.url, width="100%", height=height)

    def wait(self) -> None:
        """Keep serving the live dashboard until Ctrl+C (useful at the end of a script)."""
        if self._server is None:
            return
        if not self.quiet:
            print(f"nnscope: serving {self.url} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        self.close()

    def close(self) -> None:
        if self._server is not None:
            self._emit([{"type": "end"}])
            time.sleep(0.2)
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ live streaming
    def _emit(self, messages: List[dict]) -> None:
        if not messages or not self._listeners:
            return
        with self._lock:
            for q in list(self._listeners):
                for m in messages:
                    q.put(m)
