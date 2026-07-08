"""Scene representation for spec-first geometry generation.

A Scene is built forward from *known exact coordinates*. From one scene we emit:
  - ground-truth TikZ  (computed straight from the coordinates -> what we grade against)
  - a constraint-level description (coordinates stripped -> the model input)

The gap between the two is the difficulty of the task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def fmt(x: float) -> str:
    """Format a coordinate to 2 decimals, avoiding '-0.00'."""
    v = round(float(x), 2)
    if v == 0:
        v = 0.0
    return f"{v:.2f}"


@dataclass
class Scene:
    """Accumulates named points, a natural-language constraint spec, and TikZ."""

    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)  # coord-free, model input
    draw_cmds: list[str] = field(default_factory=list)  # ground-truth TikZ body
    steps: int = 0  # derivation-chain length (difficulty)
    tags: list[str] = field(default_factory=list)  # construction types used

    def add_point(self, name: str, x: float, y: float, label_pos: str = "above right") -> None:
        self.points[name] = (round(float(x), 2), round(float(y), 2))
        self.draw_cmds.append(
            f"  \\filldraw ({fmt(x)},{fmt(y)}) circle (1.5pt) "
            f"node[{label_pos}] {{${name}$}};"
        )

    def circle(self, cx: float, cy: float, r: float) -> None:
        self.draw_cmds.append(f"  \\draw ({fmt(cx)},{fmt(cy)}) circle ({fmt(r)});")

    def segment(self, a: str, b: str) -> None:
        (ax, ay), (bx, by) = self.points[a], self.points[b]
        self.draw_cmds.append(f"  \\draw ({fmt(ax)},{fmt(ay)}) -- ({fmt(bx)},{fmt(by)});")

    def constrain(self, line: str) -> None:
        self.constraints.append(line)

    def bump(self, tag: str) -> None:
        self.steps += 1
        self.tags.append(tag)

    def to_tikz(self, scale: float = 0.8) -> str:
        body = "\n".join(self.draw_cmds)
        return f"\\begin{{tikzpicture}}[scale={scale}]\n{body}\n\\end{{tikzpicture}}"

    def ground_truth_points(self) -> dict[str, list[float]]:
        return {k: [v[0], v[1]] for k, v in self.points.items()}


def polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def midpoint(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
    return (p[0] + q[0]) / 2, (p[1] + q[1]) / 2


def line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Intersection of line (p1,p2) with line (p3,p4). None if parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py
