"""Generator expansion (v4) — broaden the GT-verified construction vocabulary.

olympiad.py isolates the single-derived-point triangle centers. This module adds
the higher-frequency competition constructions the vocabulary mine
(``outputs/construction_freq_llm.json``) says dominate real AIME/MATH geometry
but were missing: line/circle INTERSECTION, PARALLEL/parallelogram,
MIDPOINT/midsegment, REFLECTION, ROTATION, general foot-of-perpendicular,
SQUARES, regular POLYGONS, and TWO-CIRCLE configurations.

Every builder mirrors olympiad.py's contract exactly, so it drops straight into
``build_olympiad``/``extract.grade``:

  * exact Python ground-truth coordinates for every named point,
  * a natural-language, coordinate-free description (with light phrasing
    variety so the student doesn't overfit one template),
  * a ground-truth figure whose DERIVED points are built with coordinate-free
    tkz-euclide constructions (``\\tkzInterLL``, ``\\tkzDefPointBy[rotation=...]``,
    ``\\tkzDefMidPoint``, ``\\tkzInterCC``, ...), so the student learns
    constructions, not transcribed coordinates.

Each returned figure round-trips through the compile-extract grader
(emit -> compile -> read back == GT); ``build_illustrator_data.py`` drops any
that don't, and reports the round-trip yield per construction.
"""

from __future__ import annotations

import math
import random

from .olympiad import (
    Pt,
    _area2,
    _dist,
    _g,
    _pt,
    _rand_triangle,
    _round_pts,
    _sane,
    foot,
    midpoint,
)
from .scene import line_intersection

TYPES = [
    "line_intersection",
    "cevian_intersection",
    "midsegment",
    "midpoint_segment",
    "parallelogram",
    "foot_perp",
    "reflection_line",
    "rotation",
    "square",
    "regular_polygon",
    "two_circles",
    "antipode",
]


# --------------------------------------------------------------------------- #
# exact geometry helpers (ground truth; independent of TeX)
# --------------------------------------------------------------------------- #
def _rotate(p: Pt, o: Pt, deg: float) -> Pt:
    a = math.radians(deg)
    dx, dy = p[0] - o[0], p[1] - o[1]
    return (o[0] + dx * math.cos(a) - dy * math.sin(a),
            o[1] + dx * math.sin(a) + dy * math.cos(a))


def _reflect_over_line(p: Pt, a: Pt, b: Pt) -> Pt:
    f = foot(p, a, b)
    return (2 * f[0] - p[0], 2 * f[1] - p[1])


def _circle_circle(o1: Pt, r1: float, o2: Pt, r2: float) -> tuple[Pt, Pt] | None:
    d = _dist(o1, o2)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-9:
        return None
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return None
    h = math.sqrt(h2)
    mx = o1[0] + a * (o2[0] - o1[0]) / d
    my = o1[1] + a * (o2[1] - o1[1]) / d
    ox, oy = h * (o2[1] - o1[1]) / d, h * (o2[0] - o1[0]) / d
    return ((mx + ox, my - oy), (mx - ox, my + oy))


def _rand_convex_quad(rng: random.Random, bound: int = 7) -> tuple[Pt, Pt, Pt, Pt] | None:
    """Four integer points in strictly increasing angle order (convex, ccw)."""
    for _ in range(400):
        cx, cy = rng.randint(2, bound - 1), rng.randint(2, bound - 1)
        angs = sorted(rng.sample(range(0, 360, 15), 4))
        # need reasonable angular spread so the quad isn't a sliver
        if any((angs[(i + 1) % 4] - angs[i]) % 360 < 40 for i in range(4)):
            continue
        pts = []
        for ang in angs:
            r = rng.choice([2.5, 3, 3.5, 4])
            pts.append((round(cx + r * math.cos(math.radians(ang))),
                        round(cy + r * math.sin(math.radians(ang)))))
        if len({p for p in pts}) < 4:
            continue
        if any(_dist(pts[i], pts[(i + 1) % 4]) < 2 for i in range(4)):
            continue
        # verify convex (all cross products same sign)
        crosses = [_area2(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)]
        if all(c > 0 for c in crosses):
            return tuple(pts)  # type: ignore[return-value]
    return None


# --------------------------------------------------------------------------- #
# phrasing variety
# --------------------------------------------------------------------------- #
def _tri_intro(rng: random.Random, pts: dict[str, Pt], names=("A", "B", "C")) -> str:
    tri = "".join(names)
    coords = ", ".join(f"{n}=({_g(pts[n][0])},{_g(pts[n][1])})" for n in names)
    return rng.choice([
        f"Triangle {tri} has vertices {coords}.",
        f"Let {tri} be the triangle with {coords}.",
        f"Consider triangle {tri} whose vertices are {coords}.",
        f"In the plane, {tri} is a triangle with {coords}.",
    ])


def _draw_suffix(rng: random.Random, shape: str, names: list[str]) -> str:
    nm = ", ".join(names)
    return rng.choice([
        f"Output a single TikZ figure that draws {shape} and defines the named "
        f"points {nm} at their correct positions.",
        f"Produce one TikZ figure showing {shape}, with the points {nm} placed "
        f"at their exact locations.",
        f"Draw {shape} as a single TikZ figure, defining every named point "
        f"({nm}) at its correct position.",
    ])


def _fig(body: list[str]) -> str:
    return "\\begin{tikzpicture}\n  " + "\n  ".join(body) + "\n\\end{tikzpicture}"


# --------------------------------------------------------------------------- #
# builders  ->  Problem dict {tag, points, derived, description, tikz, unordered}
# --------------------------------------------------------------------------- #
def _b_line_intersection(rng):
    for _ in range(3000):
        quad = _rand_convex_quad(rng)
        if quad is None:
            continue
        a, b, c, d = quad
        x = line_intersection(a, c, b, d)  # diagonals AC, BD
        base = {"A": a, "B": b, "C": c, "D": d}
        if x is None or not _sane(x, base):
            continue
        pts = {**base, "X": x}
        body = [_pt("A", a), _pt("B", b), _pt("C", c), _pt("D", d),
                "\\tkzInterLL(A,C)(B,D)\\tkzGetPoint{X}",
                "\\tkzDrawPolygon(A,B,C,D)", "\\tkzDrawSegments(A,C B,D)",
                "\\tkzDrawPoints(A,B,C,D,X)", "\\tkzLabelPoints(A,B,C,D,X)"]
        coords = ", ".join(f"{n}=({_g(pts[n][0])},{_g(pts[n][1])})" for n in "ABCD")
        setup = rng.choice([
            f"Convex quadrilateral ABCD has vertices {coords}.",
            f"Let ABCD be the convex quadrilateral with {coords}.",
        ])
        desc = (f"{setup} Its diagonals AC and BD intersect at point X. "
                + _draw_suffix(rng, "quadrilateral ABCD with its two diagonals",
                               ["A", "B", "C", "D", "X"]))
        return {"tag": "line_intersection", "points": _round_pts(pts),
                "derived": ["X"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("line_intersection")


def _b_cevian_intersection(rng):
    """Foot of altitude from A and median from B meet at X (two cevians)."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        f = foot(a, b, c)             # foot of altitude from A onto BC
        m = midpoint(c, a)            # midpoint of CA (median from B)
        x = line_intersection(a, f, b, m)
        base = {"A": a, "B": b, "C": c}
        if x is None or not _sane(x, base) or not _sane(f, base) or not _sane(m, base):
            continue
        if _dist(f, m) < 0.6:
            continue
        pts = {"A": a, "B": b, "C": c, "X": x}
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefPointBy[projection=onto B--C](A)\\tkzGetPoint{F}",
                "\\tkzDefMidPoint(C,A)\\tkzGetPoint{M}",
                "\\tkzInterLL(A,F)(B,M)\\tkzGetPoint{X}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawSegments(A,F B,M)",
                "\\tkzDrawPoints(A,B,C,X)", "\\tkzLabelPoints(A,B,C,X)"]
        desc = (f"{_tri_intro(rng, pts)} Let AF be the altitude from A (F on BC) "
                "and let BM be the median from B (M the midpoint of CA). The "
                "altitude AF and the median BM meet at X. "
                + _draw_suffix(rng, "triangle ABC with the altitude AF and median BM",
                               ["A", "B", "C", "X"]))
        return {"tag": "cevian_intersection", "points": _round_pts(pts),
                "derived": ["X"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("cevian_intersection")


def _b_midsegment(rng):
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        m, n = midpoint(a, b), midpoint(a, c)
        base = {"A": a, "B": b, "C": c}
        if not (_sane(m, base) and _sane(n, base)) or _dist(m, n) < 1.2:
            continue
        pts = {"A": a, "B": b, "C": c, "M": m, "N": n}
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefMidPoint(A,B)\\tkzGetPoint{M}",
                "\\tkzDefMidPoint(A,C)\\tkzGetPoint{N}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawSegment(M,N)",
                "\\tkzDrawPoints(A,B,C,M,N)", "\\tkzLabelPoints(A,B,C,M,N)"]
        desc = (f"{_tri_intro(rng, pts)} Let M be the midpoint of AB and N the "
                "midpoint of AC, so MN is the midsegment parallel to BC. "
                + _draw_suffix(rng, "triangle ABC with midsegment MN",
                               ["A", "B", "C", "M", "N"]))
        return {"tag": "midsegment", "points": _round_pts(pts),
                "derived": ["M", "N"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("midsegment")


def _b_midpoint_segment(rng):
    for _ in range(3000):
        a = (rng.randint(-6, 6), rng.randint(-6, 6))
        b = (rng.randint(-6, 6), rng.randint(-6, 6))
        if _dist(a, b) < 4:
            continue
        m = midpoint(a, b)
        pts = {"A": a, "B": b, "M": m}
        body = [_pt("A", a), _pt("B", b),
                "\\tkzDefMidPoint(A,B)\\tkzGetPoint{M}",
                "\\tkzDrawSegment(A,B)", "\\tkzDrawPoints(A,B,M)",
                "\\tkzLabelPoints(A,B,M)"]
        setup = rng.choice([
            f"Segment AB has endpoints A=({_g(a[0])},{_g(a[1])}) and "
            f"B=({_g(b[0])},{_g(b[1])}).",
            f"Let A=({_g(a[0])},{_g(a[1])}) and B=({_g(b[0])},{_g(b[1])}) be two "
            "points.",
        ])
        desc = (f"{setup} Let M be the midpoint of AB. "
                + _draw_suffix(rng, "segment AB with its midpoint",
                               ["A", "B", "M"]))
        return {"tag": "midpoint_segment", "points": _round_pts(pts),
                "derived": ["M"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("midpoint_segment")


def _b_parallelogram(rng):
    for _ in range(3000):
        a, b, c = _rand_triangle(rng, bound=6)
        d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])  # ABCD parallelogram
        base = {"A": a, "B": b, "C": c}
        if not _sane(d, base):
            continue
        pts = {"A": a, "B": b, "C": c, "D": d}
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefPointBy[translation=from B to C](A)\\tkzGetPoint{D}",
                "\\tkzDrawPolygon(A,B,C,D)",
                "\\tkzDrawPoints(A,B,C,D)", "\\tkzLabelPoints(A,B,C,D)"]
        coords = ", ".join(f"{n}=({_g(pts[n][0])},{_g(pts[n][1])})" for n in "ABC")
        setup = rng.choice([
            f"Three vertices of parallelogram ABCD are {coords}.",
            f"ABCD is a parallelogram with {coords} (in order).",
        ])
        desc = (f"{setup} Let D be the fourth vertex, so that ABCD is a "
                "parallelogram (AD parallel to BC and equal in length). "
                + _draw_suffix(rng, "parallelogram ABCD", ["A", "B", "C", "D"]))
        return {"tag": "parallelogram", "points": _round_pts(pts),
                "derived": ["D"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("parallelogram")


def _b_foot_perp(rng):
    """Foot of perpendicular from a free point P onto a segment AB (general)."""
    for _ in range(3000):
        a = (rng.randint(-6, 0), rng.randint(-2, 2))
        b = (rng.randint(1, 7), rng.randint(-2, 2))
        p = (rng.randint(-4, 5), rng.randint(2, 7))
        if _dist(a, b) < 4:
            continue
        f = foot(p, a, b)
        base = {"A": a, "B": b, "P": p}
        if not _sane(f, base) or _dist(p, f) < 1.0:
            continue
        pts = {"A": a, "B": b, "P": p, "F": f}
        body = [_pt("A", a), _pt("B", b), _pt("P", p),
                "\\tkzDefPointBy[projection=onto A--B](P)\\tkzGetPoint{F}",
                "\\tkzDrawSegment(A,B)", "\\tkzDrawSegment(P,F)",
                "\\tkzDrawPoints(A,B,P,F)", "\\tkzLabelPoints(A,B,P,F)"]
        desc = (f"Points A=({_g(a[0])},{_g(a[1])}), B=({_g(b[0])},{_g(b[1])}) "
                f"define a line, and P=({_g(p[0])},{_g(p[1])}) is a point off it. "
                "Let F be the foot of the perpendicular from P to line AB. "
                + _draw_suffix(rng, "line AB and the perpendicular from P",
                               ["A", "B", "P", "F"]))
        return {"tag": "foot_perp", "points": _round_pts(pts),
                "derived": ["F"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("foot_perp")


def _b_reflection_line(rng):
    for _ in range(3000):
        a = (rng.randint(-6, 0), rng.randint(-3, 3))
        b = (rng.randint(1, 7), rng.randint(-3, 3))
        p = (rng.randint(-4, 5), rng.randint(1, 7))
        if _dist(a, b) < 4:
            continue
        q = _reflect_over_line(p, a, b)
        base = {"A": a, "B": b, "P": p}
        if not _sane(q, base) or _dist(p, q) < 1.0:
            continue
        pts = {"A": a, "B": b, "P": p, "Q": q}
        body = [_pt("A", a), _pt("B", b), _pt("P", p),
                "\\tkzDefPointBy[reflection=over A--B](P)\\tkzGetPoint{Q}",
                "\\tkzDrawLine(A,B)", "\\tkzDrawSegment(P,Q)",
                "\\tkzDrawPoints(A,B,P,Q)", "\\tkzLabelPoints(A,B,P,Q)"]
        desc = (f"Line AB passes through A=({_g(a[0])},{_g(a[1])}) and "
                f"B=({_g(b[0])},{_g(b[1])}); P=({_g(p[0])},{_g(p[1])}) is a point. "
                "Let Q be the reflection of P across line AB. "
                + _draw_suffix(rng, "line AB and the reflected point Q",
                               ["A", "B", "P", "Q"]))
        return {"tag": "reflection_line", "points": _round_pts(pts),
                "derived": ["Q"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("reflection_line")


def _b_rotation(rng):
    for _ in range(3000):
        o = (rng.randint(-3, 3), rng.randint(-3, 3))
        p = (o[0] + rng.randint(-5, 5), o[1] + rng.randint(-5, 5))
        deg = rng.choice([30, 45, 60, 90, 120, -30, -45, -60, -90])
        if _dist(o, p) < 2.5:
            continue
        q = _rotate(p, o, deg)
        base = {"O": o, "P": p}
        if not _sane(q, base) or _dist(p, q) < 1.0:
            continue
        pts = {"O": o, "P": p, "Q": q}
        body = [_pt("O", o), _pt("P", p),
                f"\\tkzDefPointBy[rotation=center O angle {deg}](P)\\tkzGetPoint{{Q}}",
                "\\tkzDrawSegments(O,P O,Q)", "\\tkzDrawPoints(O,P,Q)",
                "\\tkzLabelPoints(O,P,Q)"]
        desc = (f"Point P=({_g(p[0])},{_g(p[1])}) is rotated about the center "
                f"O=({_g(o[0])},{_g(o[1])}) by {deg} degrees to give point Q. "
                + _draw_suffix(rng, "the rotation taking P to Q about O",
                               ["O", "P", "Q"]))
        return {"tag": "rotation", "points": _round_pts(pts),
                "derived": ["Q"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("rotation")


def _b_square(rng):
    for _ in range(3000):
        a = (rng.randint(-5, 2), rng.randint(-5, 2))
        b = (a[0] + rng.randint(2, 5), a[1] + rng.randint(-2, 2))
        if _dist(a, b) < 2.5:
            continue
        w = (-(b[1] - a[1]), b[0] - a[0])  # ccw perpendicular of AB
        d = (a[0] + w[0], a[1] + w[1])
        c = (b[0] + w[0], b[1] + w[1])
        base = {"A": a, "B": b}
        if not (_sane(c, base) and _sane(d, base)):
            continue
        pts = {"A": a, "B": b, "C": c, "D": d}
        body = [_pt("A", a), _pt("B", b),
                "\\tkzDefPointBy[rotation=center A angle 90](B)\\tkzGetPoint{D}",
                "\\tkzDefPointBy[rotation=center B angle -90](A)\\tkzGetPoint{C}",
                "\\tkzDrawPolygon(A,B,C,D)",
                "\\tkzDrawPoints(A,B,C,D)", "\\tkzLabelPoints(A,B,C,D)"]
        desc = (f"ABCD is a square (vertices in order) with side AB from "
                f"A=({_g(a[0])},{_g(a[1])}) to B=({_g(b[0])},{_g(b[1])}). "
                + _draw_suffix(rng, "square ABCD", ["A", "B", "C", "D"]))
        return {"tag": "square", "points": _round_pts(pts),
                "derived": ["C", "D"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("square")


def _b_regular_polygon(rng):
    for _ in range(3000):
        n = rng.choice([5, 6, 8])
        o = (rng.randint(-2, 2), rng.randint(-2, 2))
        r = rng.choice([3, 4])
        start = rng.choice([0, 15, 30, 45, 90])
        step = 360 // n
        v0 = (o[0] + r * math.cos(math.radians(start)),
              o[1] + r * math.sin(math.radians(start)))
        verts = {f"P{k}": _rotate(v0, o, step * k) for k in range(n)}
        base = {"O": o}
        if any(abs(v[0]) > 12 or abs(v[1]) > 12 for v in verts.values()):
            continue
        names = list(verts)
        pts = {"O": o, "P0": verts["P0"], **verts}
        body = [_pt("O", o), _pt("P0", verts["P0"])]
        for k in range(1, n):
            body.append(f"\\tkzDefPointBy[rotation=center O angle {step * k}]"
                        f"(P0)\\tkzGetPoint{{P{k}}}")
        poly = ",".join(names)
        body += [f"\\tkzDrawPolygon({poly})",
                 f"\\tkzDrawPoints(O,{poly})", f"\\tkzLabelPoints({poly})"]
        shape = {5: "regular pentagon", 6: "regular hexagon", 8: "regular octagon"}[n]
        # grade a representative subset (center + three vertices)
        graded = ["O", "P0", f"P{n // 3}", f"P{2 * n // 3}"]
        desc = (f"A {shape} P0P1...P{n - 1} is inscribed in a circle of radius "
                f"{r} centered at O=({_g(o[0])},{_g(o[1])}), with P0 at angle "
                f"{start} degrees. The vertices are equally spaced (each is the "
                f"previous one rotated {step} degrees about O). "
                + _draw_suffix(rng, f"the {shape} and its center O", graded))
        return {"tag": "regular_polygon", "points": _round_pts(pts),
                "derived": [n for n in names if n != "P0"],
                "description": desc, "tikz": _fig(body),
                "unordered": None, "grade_only": graded}
    raise RuntimeError("regular_polygon")


def _b_two_circles(rng):
    for _ in range(4000):
        o1 = (rng.randint(-3, 0), rng.randint(-2, 2))
        o2 = (rng.randint(1, 4), rng.randint(-2, 2))
        r1 = rng.choice([3, 4])
        r2 = rng.choice([3, 4])
        inter = _circle_circle(o1, float(r1), o2, float(r2))
        if inter is None:
            continue
        x, y = inter
        base = {"A": o1, "B": o2}
        if not (_sane(x, base, sep=0.5) and _sane(y, base, sep=0.5)) or _dist(x, y) < 1.2:
            continue
        w1 = (o1[0] + r1, o1[1])
        w2 = (o2[0] + r2, o2[1])
        pts = {"A": o1, "B": o2, "X": x, "Y": y}
        body = [_pt("A", o1), _pt("B", o2), _pt("Wa", w1), _pt("Wb", w2),
                "\\tkzInterCC(A,Wa)(B,Wb)\\tkzGetPoints{X}{Y}",
                "\\tkzDrawCircle(A,Wa)", "\\tkzDrawCircle(B,Wb)",
                "\\tkzDrawPoints(A,B,X,Y)", "\\tkzLabelPoints(A,B,X,Y)"]
        desc = (f"Two circles are given: one centered at A=({_g(o1[0])},{_g(o1[1])}) "
                f"with radius {r1}, the other centered at B=({_g(o2[0])},{_g(o2[1])}) "
                f"with radius {r2}. They intersect at two points X and Y. "
                + _draw_suffix(rng, "both circles and their intersection points",
                               ["A", "B", "X", "Y"]))
        return {"tag": "two_circles", "points": _round_pts(pts),
                "derived": ["X", "Y"], "description": desc, "tikz": _fig(body),
                "unordered": [["X", "Y"]]}
    raise RuntimeError("two_circles")


def _b_antipode(rng):
    for _ in range(3000):
        o = (rng.randint(-2, 2), rng.randint(-2, 2))
        r = rng.choice([3, 4, 5])
        ang = rng.choice([20, 40, 55, 70, 110, 130, 160, 200, 250, 290, 320])
        a = (o[0] + r * math.cos(math.radians(ang)), o[1] + r * math.sin(math.radians(ang)))
        b = (2 * o[0] - a[0], 2 * o[1] - a[1])
        base = {"O": o}
        if not (_sane(a, base, sep=1.0) and _sane(b, base, sep=1.0)):
            continue
        pts = {"O": o, "A": a, "B": b}
        body = [_pt("O", o), _pt("A", a),
                "\\tkzDefPointBy[symmetry=center O](A)\\tkzGetPoint{B}",
                "\\tkzDrawCircle(O,A)", "\\tkzDrawSegment(A,B)",
                "\\tkzDrawPoints(O,A,B)", "\\tkzLabelPoints(O,A,B)"]
        desc = (f"A circle has center O=({_g(o[0])},{_g(o[1])}) and radius {r}; "
                f"A=({_g(a[0])},{_g(a[1])}) lies on it. Let B be the point "
                "diametrically opposite A (so AB is a diameter). "
                + _draw_suffix(rng, "the circle and the diameter AB",
                               ["O", "A", "B"]))
        return {"tag": "antipode", "points": _round_pts(pts),
                "derived": ["B"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("antipode")


_BUILDERS = {
    "line_intersection": _b_line_intersection,
    "cevian_intersection": _b_cevian_intersection,
    "midsegment": _b_midsegment,
    "midpoint_segment": _b_midpoint_segment,
    "parallelogram": _b_parallelogram,
    "foot_perp": _b_foot_perp,
    "reflection_line": _b_reflection_line,
    "rotation": _b_rotation,
    "square": _b_square,
    "regular_polygon": _b_regular_polygon,
    "two_circles": _b_two_circles,
    "antipode": _b_antipode,
}


def make_problem(rng: random.Random, tag: str) -> dict:
    if tag not in _BUILDERS:
        raise ValueError(f"unknown construction: {tag}")
    return _BUILDERS[tag](rng)


def generate_problems(n_per_type: int, seed: int = 0,
                      types: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    types = types or TYPES
    out: list[dict] = []
    pid = 0
    for tag in types:
        seen: set[str] = set()
        got = tries = 0
        while got < n_per_type and tries < n_per_type * 300 + 3000:
            tries += 1
            prob = make_problem(rng, tag)
            if prob["description"] in seen:
                continue
            seen.add(prob["description"])
            prob["id"] = pid
            out.append(prob)
            pid += 1
            got += 1
        if got < n_per_type:
            print(f"  WARN {tag}: only {got}/{n_per_type} unique")
    return out
