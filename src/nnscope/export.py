"""Standalone HTML replay and static images (SVG without dependencies, PNG via matplotlib if installed)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

BLUE, ORANGE, MID = (8, 119, 189), (245, 147, 34), (232, 234, 235)
BOOT_MARKER = "/*__NNSCOPE_BOOT__*/"


def _template() -> str:
    try:
        from importlib.resources import files
        return files("nnscope").joinpath("static/index.html").read_text(encoding="utf-8")
    except (ImportError, AttributeError):                              # Python < 3.9 fallback
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def render_page(data: Optional[dict], live: bool) -> str:
    boot = {"live": live, "data": data}
    payload = json.dumps(boot, separators=(",", ":")).replace("</", "<\\/")
    return _template().replace(BOOT_MARKER, f"window.NNSCOPE = {payload};", 1)


def save_html(data: dict, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_page(data, live=False), encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ static snapshot image
def _color(v: float) -> str:
    t = max(-1.0, min(1.0, v))
    c = BLUE if t >= 0 else ORANGE
    a = abs(t)
    return "#%02x%02x%02x" % tuple(int(MID[i] + (c[i] - MID[i]) * a) for i in range(3))


def _pick_snapshot(data: dict, step: Optional[int]) -> Optional[dict]:
    snaps = data.get("snapshots") or []
    if not snaps:
        return None
    if step is None:
        return snaps[-1]
    return min(snaps, key=lambda s: abs(s["step"] - step))


def _layout(structure: dict, x0: float, x1: float, y0: float, y1: float):
    layers = structure["layers"]
    cols = [layers[0]["shown_in"]] + [l["shown_out"] for l in layers]
    xs = [x0 + (x1 - x0) * c / max(1, len(cols) - 1) for c in range(len(cols))]
    pos = []
    for c, shown in enumerate(cols):
        n = len(shown)
        gap = min(24.0, (y1 - y0) / max(n, 1))
        top = (y0 + y1) / 2 - gap * (n - 1) / 2
        pos.append([(xs[c], top + gap * j) for j in range(n)])
    return pos, xs


def _loss_series(data: dict):
    series = {}
    for r in data.get("records", []):
        for k, v in r["scalars"].items():
            if v is not None and math.isfinite(v):
                series.setdefault(k, []).append((r["step"], v))
    return series


def snapshot_svg(data: dict, step: Optional[int] = None) -> str:
    structure, snap = data.get("structure"), _pick_snapshot(data, step)
    if structure is None or snap is None:
        raise ValueError("nothing to draw yet: call scope.log(...) at least once")
    W, H = 1100, 480
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="24" y="28" font-size="16" fill="#222">{data.get("name", "run")} — step {snap["step"]}</text>']
    pos, xs = _layout(structure, 70, 650, 70, H - 50)
    for l, Wl in enumerate(snap["W"]):
        m = max((abs(v) for row in Wl for v in row), default=1.0) or 1.0
        for i, row in enumerate(Wl):
            for j, w in enumerate(row):
                (xa, ya), (xb, yb) = pos[l][i], pos[l + 1][j]
                xm = (xa + xb) / 2
                out.append(f'<path d="M{xa:.1f},{ya:.1f} C{xm:.1f},{ya:.1f} {xm:.1f},{yb:.1f} {xb:.1f},{yb:.1f}" fill="none" '
                           f'stroke="{_color(1 if w >= 0 else -1)}" stroke-opacity="{0.15 + 0.85 * abs(w) / m:.2f}" '
                           f'stroke-width="{0.3 + 3.5 * abs(w) / m:.2f}"/>')
    for c, col in enumerate(pos):
        node = snap["nodes"][c - 1] if c > 0 and c - 1 < len(snap["nodes"]) else None
        for j, (x, y) in enumerate(col):
            v = node["mean"][j] if node else 0.0
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{_color(v)}" stroke="#555" stroke-width="0.8"/>')
        layer = structure["layers"][max(c - 1, 0)]
        n = layer["in"] if c == 0 else layer["out"]
        shown = len(col)
        label = ("input" if c == 0 else (layer.get("act") or "linear")) + f" · {n}"
        out.append(f'<text x="{xs[c]:.1f}" y="56" text-anchor="middle" fill="#555">{label}</text>')
        if shown < n:
            out.append(f'<text x="{xs[c]:.1f}" y="{H - 28}" text-anchor="middle" fill="#999">+{n - shown} not drawn</text>')
    # loss chart
    cx0, cy0, cw, ch = 720, 70, 350, 330
    out.append(f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" fill="#fafafa" stroke="#ddd"/>')
    series = _loss_series(data)
    vals = [v for s in series.values() for _, v in s]
    if vals:
        use_log = min(vals) > 0 and max(vals) / min(vals) > 100
        f = (lambda v: math.log10(v)) if use_log else (lambda v: v)
        lo, hi = min(f(v) for v in vals), max(f(v) for v in vals)
        hi = hi if hi > lo else lo + 1
        steps = [s for ser in series.values() for s, _ in ser]
        s0, s1 = min(steps), max(steps) if max(steps) > min(steps) else min(steps) + 1
        palette = ["#222222", "#f59322", "#0877bd", "#2ca02c", "#9467bd", "#8c564b"]
        for k, (name, ser) in enumerate(series.items()):
            pts = " ".join(f"{cx0 + cw * (s - s0) / (s1 - s0):.1f},{cy0 + ch - ch * (f(v) - lo) / (hi - lo):.1f}" for s, v in ser)
            col = palette[k % len(palette)]
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
            out.append(f'<text x="{cx0 + 8}" y="{cy0 + 18 + 16 * k}" fill="{col}">{name}</text>')
        out.append(f'<text x="{cx0}" y="{cy0 + ch + 18}" fill="#777">step {s0} … {s1}{" (log scale)" if use_log else ""}</text>')
    out.append(f'<text x="{cx0}" y="{cy0 - 10}" fill="#555">loss &amp; metrics</text>')
    out.append("</svg>")
    return "\n".join(out)


def snapshot_image(data: dict, path: str, step: Optional[int] = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".svg":
        p.write_text(snapshot_svg(data, step), encoding="utf-8")
        return str(p)
    if p.suffix.lower() != ".png":
        raise ValueError("snapshot path must end in .svg or .png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
    except ImportError:
        raise ImportError("PNG snapshots need matplotlib: pip install 'nnscope[png]' (or save as .svg)") from None
    structure, snap = data.get("structure"), _pick_snapshot(data, step)
    if structure is None or snap is None:
        raise ValueError("nothing to draw yet: call scope.log(...) at least once")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})
    pos, xs = _layout(structure, 0.0, 1.0, 0.0, 1.0)
    for l, Wl in enumerate(snap["W"]):
        m = max((abs(v) for row in Wl for v in row), default=1.0) or 1.0
        for i, row in enumerate(Wl):
            for j, w in enumerate(row):
                (xa, ya), (xb, yb) = pos[l][i], pos[l + 1][j]
                xm = (xa + xb) / 2
                path = MPath([(xa, ya), (xm, ya), (xm, yb), (xb, yb)], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
                ax.add_patch(PathPatch(path, fill=False, lw=0.3 + 2.5 * abs(w) / m, alpha=0.15 + 0.85 * abs(w) / m,
                                       color=_color(1 if w >= 0 else -1)))
    for c, col in enumerate(pos):
        node = snap["nodes"][c - 1] if c > 0 and c - 1 < len(snap["nodes"]) else None
        ax.scatter([x for x, _ in col], [y for _, y in col], s=45, zorder=3, edgecolors="#555",
                   c=[_color(node["mean"][j] if node else 0.0) for j in range(len(col))])
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05); ax.axis("off")
    ax.set_title(f"{data.get('name', 'run')} — step {snap['step']}")
    for name, ser in _loss_series(data).items():
        bx.plot([s for s, _ in ser], [v for _, v in ser], label=name)
    bx.set_xlabel("step"); bx.set_title("loss & metrics"); bx.grid(alpha=0.3); bx.legend()
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)
