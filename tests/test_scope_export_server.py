import json
import threading
import urllib.request

import numpy as np

import nnscope
from nnscope.insights import InsightEngine

rng = np.random.default_rng(0)


class Linear:
    def __init__(self, i, o):
        self.W, self.b = rng.normal(size=(i, o)) * 0.5, np.zeros(o)
        self.dW, self.db = np.zeros((i, o)), np.zeros(o)


class ReLU:
    pass


class Net:
    def __init__(self):
        self.layers = [Linear(2, 40), ReLU(), Linear(40, 1)]


def _train(scope, steps=30):
    net = Net()
    for s in range(steps):
        x = rng.normal(size=(16, 2))
        h = np.maximum(x @ net.layers[0].W, 0)
        net.layers[2].x = h                                  # HW-style cached input → activation stats
        net.layers[0].dW[...] = rng.normal(size=(2, 40)) * 0.01
        net.layers[2].dW[...] = rng.normal(size=(40, 1)) * 0.01
        net.layers[0].W -= 0.1 * net.layers[0].dW
        scope.log(s, model=net, loss=1.0 / (s + 1), metrics={"test_loss": 1.2 / (s + 1)}, latent=h, labels=x[:, 0] > 0)
    return net


def test_record_structure_snapshots_and_html(tmp_path):
    scope = nnscope.Scope("t", live=False, max_neurons=8)
    _train(scope)
    d = scope.data()
    assert d["structure"]["layers"][0]["out"] == 40 and len(d["structure"]["layers"][0]["shown_out"]) == 8
    assert d["structure"]["layers"][0]["act"] == "relu"
    assert len(d["records"]) == 30 and d["records"][5]["layers"][0]["update_ratio"] is not None
    assert d["records"][3]["layers"][0]["frac_active"] is not None
    snap = d["snapshots"][-1]
    assert np.array(snap["W"][0]).shape == (2, 8) and snap["latent"]["dim"] == 40
    html = open(scope.save_html(str(tmp_path / "run.html")), encoding="utf-8").read()
    boot = html.split("window.NNSCOPE = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(boot)["data"]["name"] == "t"
    svg = open(scope.snapshot(str(tmp_path / "net.svg")), encoding="utf-8").read()
    assert svg.startswith("<svg") and "polyline" in svg


def test_adaptive_snapshot_thinning():
    scope = nnscope.Scope("t", live=False, max_snapshots=10)
    _train(scope, steps=100)
    assert len(scope.snapshots) <= 10 and scope.snapshots[-1]["step"] > 80


def test_live_server_history_and_stream():
    scope = nnscope.Scope("live", port=0, open_browser=False, quiet=True)
    try:
        assert "<html" in urllib.request.urlopen(scope.url).read().decode()
        got = []

        def listen():
            with urllib.request.urlopen(scope.url + "api/stream", timeout=10) as r:
                for line in r:
                    if line.startswith(b"data: "):
                        got.extend(json.loads(line[6:]))
                        return

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        import time
        for _ in range(50):
            if scope._listeners:
                break
            time.sleep(0.05)
        _train(scope, steps=3)
        t.join(timeout=10)
        assert any(m["type"] == "record" for m in got)
        hist = json.loads(urllib.request.urlopen(scope.url + "api/history").read())
        assert len(hist["records"]) == 3
    finally:
        scope.close()


def _rec(step, g, ur, loss=1.0, test=None, dead=None):
    layers = [{"g_mean_abs": gi, "update_ratio": ur, "dead": dead, "dead_kind": "dead"} for gi in g]
    scalars = {"loss": loss, **({"test_loss": test} if test is not None else {})}
    return {"step": step, "layers": layers, "scalars": scalars}


STRUCT = {"layers": [{"out": 16, "act": "sigmoid"}, {"out": 16, "act": "relu"}, {"out": 1, "act": None}]}


def test_insights_fire_once_and_not_on_healthy_runs():
    e = InsightEngine()
    healthy = [e.update(_rec(s, [0.01, 0.02, 0.05], 1e-3, loss=1 / (s + 1), test=1.1 / (s + 1)), STRUCT) for s in range(20)]
    assert not any(healthy)
    e = InsightEngine()
    kinds = [i["kind"] for s in range(10) for i in e.update(_rec(s, [1e-7, 1e-3, 1e-2], 0.2, dead=12), STRUCT)]
    assert kinds.count("vanishing") == 1 and kinds.count("lr_high") == 1 and kinds.count("dead") == 2   # once per affected layer
    e = InsightEngine()
    out = [i for s in range(40) for i in e.update(_rec(s, [0.01] * 3, 1e-3, loss=1 / (s + 1),
                                                   test=0.5 + abs(s - 10) * 0.05), STRUCT)]
    assert [i["kind"] for i in out] == ["overfit"]


def test_insight_hysteresis_no_flapping_near_threshold():
    e = InsightEngine()
    ratios = [1.2e3 if s % 7 else 0.8e3 for s in range(60)]           # hovers around the 1e3 threshold
    kinds = [i["kind"] for s, r in enumerate(ratios) for i in e.update(_rec(s, [0.01 / r, 1e-3, 0.01], 1e-3), STRUCT)]
    assert kinds.count("vanishing") == 1
