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
    """Accumulates named points, a natural-language constraint spec, and TikZ.

    Emits ground-truth TikZ in two interchangeable styles (grading is identical
    because exact coordinates are always stored):
      - numeric  (default): points placed at literal computed coordinates.
      - symbolic (symbolic=True): points placed via coordinate-free PGF
        constructions (calc projection, name intersections, ...), letting PGF
        compute the numbers. This is the v2 "construction" behavior.
    """

    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)  # coord-free, model input
    draw_cmds: list[str] = field(default_factory=list)  # numeric ground-truth TikZ body
    sym_cmds: list[str] = field(default_factory=list)  # symbolic (PGF-construction) body
    steps: int = 0  # derivation-chain length (difficulty)
    tags: list[str] = field(default_factory=list)  # construction types used
    symbolic: bool = False  # also record the symbolic construction body

    def add_point(self, name: str, x: float, y: float, label_pos: str = "above right",
                  at: str | None = None) -> None:
        self.points[name] = (round(float(x), 2), round(float(y), 2))
        self.draw_cmds.append(
            f"  \\filldraw ({fmt(x)},{fmt(y)}) circle (1.5pt) "
            f"node[{label_pos}] {{${name}$}};"
        )
        if self.symbolic and at is not None:
            self.sym_cmds.append(f"  \\coordinate ({name}) at ({at});")
            self.label(name, label_pos)

    def label(self, name: str, label_pos: str = "above right") -> None:
        """Dot + label for an already-defined symbolic coordinate (by name)."""
        self.sym_cmds.append(
            f"  \\filldraw ({name}) circle (1.5pt) node[{label_pos}] {{${name}$}};"
        )

    def sym(self, cmd: str) -> None:
        """Append a raw line to the symbolic body (e.g. name-path setup)."""
        self.sym_cmds.append(cmd)

    def circle(self, cx: float, cy: float, r: float) -> None:
        cmd = f"  \\draw ({fmt(cx)},{fmt(cy)}) circle ({fmt(r)});"
        self.draw_cmds.append(cmd)
        if self.symbolic:
            self.sym_cmds.append(cmd)

    def segment(self, a: str, b: str) -> None:
        (ax, ay), (bx, by) = self.points[a], self.points[b]
        self.draw_cmds.append(f"  \\draw ({fmt(ax)},{fmt(ay)}) -- ({fmt(bx)},{fmt(by)});")
        if self.symbolic:
            self.sym_cmds.append(f"  \\draw ({a}) -- ({b});")

    def constrain(self, line: str) -> None:
        self.constraints.append(line)

    def bump(self, tag: str) -> None:
        self.steps += 1
        self.tags.append(tag)

    def to_tikz(self, scale: float = 0.8, symbolic: bool | None = None) -> str:
        use_sym = self.symbolic if symbolic is None else symbolic
        body = "\n".join(self.sym_cmds if use_sym else self.draw_cmds)
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
