# Changelog

## 0.1.1 (2026-09-29)

Fixes found while using nnscope on a real homework (the HW1 companion notebooks).

- `snapshot("x.png")` no longer calls `matplotlib.use("Agg")`. It used to switch the global backend, which silently stopped
  inline plots from showing in Jupyter afterwards; it now draws on a standalone `Figure` with an Agg canvas.
- Loss chart: the log scale stays on when a series reaches exactly 0 (e.g. a training error of 0); non-positive points are
  skipped instead of forcing a linear axis.
- Latent space: the class legend moved above the plot so it no longer hides points.

## 0.1.0 (2026-09-29)

First release.

- `Scope`: record dense NumPy networks during training (`log`), with auto-detection of `.layers` models, CS231n-style
  parameter dicts, lists of `(W, b)` pairs, or plain arrays; automatic `(in, out)` / `(out, in)` orientation.
- Live dashboard (stdlib HTTP server + Server-Sent Events) and standalone HTML replay (`save_html`).
- Views: network graph (weights / gradients / updates), neuron inspector with live formula, loss & metrics, gradient flow,
  layer health (norms, update ratio, histograms), activation statistics (dead / saturated units), latent-space scatter (PCA).
- Insights feed: vanishing / exploding gradients, dead or saturated units, learning rate too high / low, overfitting, NaN loss.
- Exports: network SVG/PNG, loss PNG, whole-dashboard PNG, run JSON; `snapshot()` to SVG (no deps) or PNG (matplotlib).
- Responsive layout (vertical network on phones), light/dark theme, keyboard timeline controls.
