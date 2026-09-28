# Changelog

## 0.1.0 (unreleased)

First release.

- `Scope`: record dense NumPy networks during training (`log`), with auto-detection of `.layers` models, CS231n-style
  parameter dicts, lists of `(W, b)` pairs, or plain arrays; automatic `(in, out)` / `(out, in)` orientation.
- Live dashboard (stdlib HTTP server + Server-Sent Events) and standalone HTML replay (`save_html`).
- Views: network graph (weights / gradients / updates), neuron inspector with live formula, loss & metrics, gradient flow,
  layer health (norms, update ratio, histograms), activation statistics (dead / saturated units), latent-space scatter (PCA).
- Insights feed: vanishing / exploding gradients, dead or saturated units, learning rate too high / low, overfitting, NaN loss.
- Exports: network SVG/PNG, loss PNG, whole-dashboard PNG, run JSON; `snapshot()` to SVG (no deps) or PNG (matplotlib).
- Responsive layout (vertical network on phones), light/dark theme, keyboard timeline controls.
