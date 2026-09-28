import numpy as np
import pytest

from nnscope.adapters import cached_activations, extract_layers

rng = np.random.default_rng(0)


# ---- a tiny HW-style NumPy model: Linear stores W (in, out), dW, and caches its input x
class Linear:
    def __init__(self, i, o):
        self.W, self.b = rng.normal(size=(i, o)), np.zeros(o)
        self.dW, self.db = np.zeros((i, o)), np.zeros(o)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b


class ReLU:
    def forward(self, z):
        return np.maximum(z, 0)


class Tanh:
    def forward(self, z):
        return np.tanh(z)


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


def test_sequential_duck_typing_and_activation_names():
    m = Sequential(Linear(2, 8), ReLU(), Linear(8, 4), Tanh(), Linear(4, 3))
    views = extract_layers(m)
    assert [v.W.shape for v in views] == [(2, 8), (8, 4), (4, 3)]
    assert [v.act for v in views] == ["relu", "tanh", None]
    assert views[0].dW is not None and views[0].b.shape == (8,)


def test_cached_activations_from_next_layer_input():
    m = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    m.forward(rng.normal(size=(5, 2)))
    acts = cached_activations(extract_layers(m))
    assert acts[0].shape == (5, 8) and np.all(acts[0] >= 0) and acts[1] is None


def test_params_dict_cs231n_style_with_grads():
    params = {"W1": rng.normal(size=(4, 10)), "b1": np.zeros(10), "W2": rng.normal(size=(10, 3)), "b2": np.zeros(3)}
    grads = {k: np.ones_like(v) for k, v in params.items()}
    views = extract_layers(params, grads=grads)
    assert [v.W.shape for v in views] == [(4, 10), (10, 3)]
    assert np.all(views[1].dW == 1) and views[1].db.shape == (3,)


def test_explicit_arrays_and_out_in_orientation():
    W1, W2 = rng.normal(size=(16, 2)), rng.normal(size=(1, 16))   # torch-like (out, in)
    views = extract_layers(weights=[W1, W2], biases=[np.zeros(16), np.zeros(1)], grads=[W1 * 0, W2 * 0])
    assert [v.W.shape for v in views] == [(2, 16), (16, 1)]
    assert views[0].dW.shape == (2, 16)


def test_single_layer_orientation_from_bias():
    views = extract_layers(weights=[rng.normal(size=(3, 7))], biases=[np.zeros(3)])
    assert views[0].W.shape == (7, 3)
    views = extract_layers(weights=[rng.normal(size=(3, 7))], biases=[np.zeros(7)])
    assert views[0].W.shape == (3, 7)


def test_list_of_pairs_and_interleaved():
    W1, W2 = rng.normal(size=(2, 5)), rng.normal(size=(5, 1))
    assert [v.W.shape for v in extract_layers([(W1, np.zeros(5)), (W2, np.zeros(1))])] == [(2, 5), (5, 1)]
    assert [v.b.shape for v in extract_layers([W1, np.zeros(5), W2, np.zeros(1)])] == [(5,), (1,)]


def test_forced_layout():
    W = rng.normal(size=(4, 4))
    assert np.allclose(extract_layers(weights=[W], layout="out_in")[0].W, W.T)


def test_helpful_error():
    with pytest.raises(TypeError, match="scope.log"):
        extract_layers(object())
    with pytest.raises(TypeError):
        extract_layers()
