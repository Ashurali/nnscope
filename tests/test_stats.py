import numpy as np

from nnscope.stats import activation_numbers, histogram, layer_numbers, project_2d, select_neurons, sig


def test_sig_rounding_and_nonfinite():
    assert sig(np.array([123456.0, 0.000123456, 0.0, np.nan])) == [123500.0, 0.0001235, 0.0, 0.0]


def test_select_neurons():
    assert select_neurons(5, 32) == [0, 1, 2, 3, 4]
    idx = select_neurons(256, 32)
    assert len(idx) == 32 and idx[0] == 0 and idx[-1] == 255


def test_histogram_counts_and_constant_input():
    h = histogram(np.arange(100.0), bins=10)
    assert sum(h["counts"]) == 100 and h["lo"] == 0 and h["hi"] == 99
    assert sum(histogram(np.ones(7))["counts"]) == 7
    assert histogram(None) is None


def test_layer_numbers_update_ratio():
    W = np.ones((3, 3))
    out = layer_numbers(W * 1.1, np.full((3, 3), 0.5), W)
    assert abs(out["update_ratio"] - 0.1) < 1e-9 and abs(out["g_norm"] - 1.5) < 1e-9
    assert layer_numbers(W, None, None)["g_norm"] is None


def test_activation_numbers_dead_relu_and_saturated_tanh():
    A = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0]])
    s = activation_numbers(A, "relu")
    assert s["dead"] == 1 and np.allclose(s["frac_active"], [0, 2 / 3])
    T = np.array([[0.999, 0.1], [-0.999, -0.2]])
    s = activation_numbers(T, "tanh")
    assert s["dead"] == 1 and s["dead_kind"] == "saturated"


def test_project_2d_pca_recovers_dominant_axis():
    rng = np.random.default_rng(0)
    t = rng.normal(size=500)
    Z = np.column_stack([t, 2 * t, rng.normal(scale=0.01, size=500), np.zeros(500)])
    xy, explained, idx = project_2d(Z)
    assert xy.shape == (500, 2) and explained[0] > 0.99 and len(idx) == 500
    xy2, explained2, _ = project_2d(Z * 1.0001)
    assert np.sign(xy[0, 0]) == np.sign(xy2[0, 0])            # stable sign between calls
    assert project_2d(Z[:, :2])[1] is None                    # 2-D input is shown as-is
    assert project_2d(np.ones((3000, 5)), max_points=1000)[0].shape == (1000, 2)
