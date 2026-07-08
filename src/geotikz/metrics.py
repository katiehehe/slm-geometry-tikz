"""Objective metrics for the Behavior Spec.

- extract_tikz / is_figure_only  -> spec-adherence (figure only, no prose)
- mse / ssim                     -> render-and-diff
- parse_coords / coord_match     -> coordinate assertion against ground truth
"""

from __future__ import annotations

import re

import numpy as np

_BEGIN = r"\begin{tikzpicture}"
_END = r"\end{tikzpicture}"


def extract_tikz(text: str) -> str | None:
    """Pull the first \\begin{tikzpicture}...\\end{tikzpicture} block."""
    i = text.find(_BEGIN)
    j = text.find(_END)
    if i == -1 or j == -1 or j < i:
        return None
    return text[i : j + len(_END)]


def is_figure_only(text: str) -> bool:
    """True iff the output is essentially just the figure (allow code fences/whitespace)."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", stripped).strip()
    return stripped.startswith(_BEGIN) and stripped.endswith(_END)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on two [0,1] grayscale images of equal shape."""
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return float(num / den) if den else 0.0


# match \filldraw (x,y) circle ... node ... {$NAME$}  and generic coordinate literals
_NODE_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"  # coordinate
    r"[^;]*?node[^;]*?\{\$?([A-Za-z]\w*)\$?\}",  # ... node {$NAME$}
    re.DOTALL,
)


def parse_named_coords(tikz: str) -> dict[str, tuple[float, float]]:
    """Recover {name: (x,y)} for labeled points in a TikZ figure."""
    out: dict[str, tuple[float, float]] = {}
    for m in _NODE_RE.finditer(tikz):
        x, y, name = float(m.group(1)), float(m.group(2)), m.group(3)
        out[name] = (x, y)
    return out


def coord_match(
    pred_tikz: str, gt_points: dict[str, list[float]], atol: float = 0.05
) -> dict:
    """Compare predicted named coords to ground truth within tolerance."""
    pred = parse_named_coords(pred_tikz)
    total = len(gt_points)
    hits = 0
    per_point = {}
    for name, (gx, gy) in gt_points.items():
        if name in pred:
            px, py = pred[name]
            err = max(abs(px - gx), abs(py - gy))
            ok = err <= atol
            per_point[name] = {"ok": ok, "err": round(err, 4)}
            hits += int(ok)
        else:
            per_point[name] = {"ok": False, "err": None}
    return {
        "matched": hits,
        "total": total,
        "accuracy": hits / total if total else 0.0,
        "all_correct": hits == total and total > 0,
        "per_point": per_point,
    }
