"""Generator expansion (v5) — HARDER, BROADER, round-trip-verified constructions.

``olympiad.py`` isolates single-derived-point triangle centers and ``olympiad_ext.py``
adds the higher-frequency single/double-point families. This module goes one level
harder to stress two axes the v2 illustrator should improve on:

  * MORE DERIVED POINTS per figure (2-4), so the model must place a whole
    configuration correctly rather than one point.
  * LIGHT COMPOSITIONS / CHAINS — a derived point built ON TOP of another derived
    point (nine-point centre = midpoint of O and H; contact points = projections
    of the incentre; reflect the orthocentre over a side; ...), plus more
    olympiad VOCABULARY (Euler line, medial/orthic triangle, nine-point centre,
    contact/intouch triangle, antipode, medians, ...).

Every builder mirrors the ``olympiad.py`` contract EXACTLY
(``{tag, points, derived, description, tikz, unordered, grade_only}``) and only
uses tkz-euclide macros that the existing generators already prove round-trip
(``\\tkzDefTriangleCenter``, ``\\tkzDefMidPoint``, ``\\tkzDefPointBy[projection|
symmetry|rotation|translation]``, ``\\tkzInterLL``, ``\\tkzDefLine[bisector]``).
Each returned figure still round-trips through the compile-extract grader
(emit -> compile -> read back == GT); ``build_illustrator_v2_data.py`` drops any
that don't and reports the per-construction yield, so labels stay
correct-by-construction.
"""

from __future__ import annotations

import math
import random

from .olympiad import (
    Pt,
    _dist,
    _g,
    _pt,
    _rand_triangle,
    _round_pts,
    _sane,
    bisector_foot,
    centroid,
    circumcenter,
    foot,
    incenter,
    midpoint,
    orthocenter,
)

TYPES = [
    "euler_line",
    "nine_point_center",
    "medial_triangle",
    "orthic_triangle",
    "incircle_contact",
    "bisector_incenter",
    "three_medians",
    "parallelogram_center",
    "midpoint_reflect_chain",
    "circumcircle_antipode",
    "square_center",
    "reflect_ortho_over_side",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _reflect_over_line(p: Pt, a: Pt, b: Pt) -> Pt:
    f = foot(p, a, b)
    return (2 * f[0] - p[0], 2 * f[1] - p[1])


def _symmetry(center: Pt, p: Pt) -> Pt:
    return (2 * center[0] - p[0], 2 * center[1] - p[1])


def _rotate(p: Pt, o: Pt, deg: float) -> Pt:
    a = math.radians(deg)
    dx, dy = p[0] - o[0], p[1] - o[1]
    return (o[0] + dx * math.cos(a) - dy * math.sin(a),
            o[1] + dx * math.sin(a) + dy * math.cos(a))


def _is_acute(a: Pt, b: Pt, c: Pt) -> bool:
    """All three interior angles < 90 deg (dot product at each vertex > 0)."""
    for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
        d = (v[0] - u[0]) * (w[0] - u[0]) + (v[1] - u[1]) * (w[1] - u[1])
        if d <= 0.3:
            return False
    return True


def _fig(body: list[str]) -> str:
    return "\\begin{tikzpicture}\n  " + "\n  ".join(body) + "\n\\end{tikzpicture}"


def _coords(pts: dict[str, Pt], names) -> str:
    return ", ".join(f"{n}=({_g(pts[n][0])},{_g(pts[n][1])})" for n in names)


def _tri_intro(rng: random.Random, pts: dict[str, Pt], names=("A", "B", "C")) -> str:
    tri = "".join(names)
    coords = _coords(pts, names)
    return rng.choice([
        f"Triangle {tri} has vertices {coords}.",
        f"Let {tri} be the triangle with {coords}.",
        f"Consider triangle {tri} whose vertices are {coords}.",
        f"In the plane, {tri} is a triangle with {coords}.",
        f"A triangle {tri} is given by {coords}.",
    ])


def _draw_suffix(rng: random.Random, shape: str, names: list[str]) -> str:
    nm = ", ".join(names)
    return rng.choice([
        f"Output a single TikZ figure that draws {shape} and defines the named "
        f"points {nm} at their correct positions.",
        f"Produce one TikZ figure showing {shape}, with the points {nm} placed at "
        f"their exact locations.",
        f"Draw {shape} as a single TikZ figure, defining every named point "
        f"({nm}) at its correct position.",
        f"Return one TikZ figure of {shape} in which {nm} are all defined at their "
        f"true coordinates.",
    ])


def _pairwise_ok(pts: dict[str, Pt], names: list[str], sep: float = 0.6) -> bool:
    """No two of the given points are near-coincident (keeps the figure legible)."""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if _dist(pts[names[i]], pts[names[j]]) < sep:
                return False
    return True


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _b_euler_line(rng):
    """Circumcentre O, centroid G, orthocentre H — all on the Euler line."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        o, g, h = circumcenter(a, b, c), centroid(a, b, c), orthocenter(a, b, c)
        pts = {**base, "O": o, "G": g, "H": h}
        if not all(_sane(pts[n], base) for n in ("O", "G", "H")):
            continue
        if not _pairwise_ok(pts, ["O", "G", "H"], sep=0.8):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}",
                "\\tkzDefTriangleCenter[centroid](A,B,C)\\tkzGetPoint{G}",
                "\\tkzDefTriangleCenter[ortho](A,B,C)\\tkzGetPoint{H}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawLine(O,H)",
                "\\tkzDrawPoints(A,B,C,O,G,H)", "\\tkzLabelPoints(A,B,C,O,G,H)"]
        desc = (f"{_tri_intro(rng, pts)} Let O, G, and H be the circumcenter, the "
                "centroid, and the orthocenter of the triangle, respectively. These "
                "three points are collinear (they lie on the Euler line). "
                + _draw_suffix(rng, "triangle ABC together with its Euler line OH",
                               ["A", "B", "C", "O", "G", "H"]))
        return {"tag": "euler_line", "points": _round_pts(pts),
                "derived": ["O", "G", "H"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("euler_line")


def _b_nine_point_center(rng):
    """N = midpoint of O (circumcentre) and H (orthocentre): the nine-point centre."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        o, h = circumcenter(a, b, c), orthocenter(a, b, c)
        n = midpoint(o, h)
        pts = {**base, "O": o, "H": h, "N": n}
        if not all(_sane(pts[k], base) for k in ("O", "H", "N")):
            continue
        if not _pairwise_ok(pts, ["O", "H", "N"], sep=0.8):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}",
                "\\tkzDefTriangleCenter[ortho](A,B,C)\\tkzGetPoint{H}",
                "\\tkzDefMidPoint(O,H)\\tkzGetPoint{N}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawSegment(O,H)",
                "\\tkzDrawPoints(A,B,C,O,H,N)", "\\tkzLabelPoints(A,B,C,O,H,N)"]
        desc = (f"{_tri_intro(rng, pts)} Let O be the circumcenter and H the "
                "orthocenter. Let N be the nine-point center, i.e. the midpoint of "
                "segment OH. "
                + _draw_suffix(rng, "triangle ABC with O, H and the nine-point "
                               "center N", ["A", "B", "C", "O", "H", "N"]))
        return {"tag": "nine_point_center", "points": _round_pts(pts),
                "derived": ["O", "H", "N"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("nine_point_center")


def _b_medial_triangle(rng):
    """Midpoints D,E,F of BC,CA,AB — the medial triangle."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        d, e, f = midpoint(b, c), midpoint(c, a), midpoint(a, b)
        pts = {**base, "D": d, "E": e, "F": f}
        if not all(_sane(pts[k], base) for k in ("D", "E", "F")):
            continue
        if not _pairwise_ok(pts, ["D", "E", "F"], sep=1.0):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefMidPoint(B,C)\\tkzGetPoint{D}",
                "\\tkzDefMidPoint(C,A)\\tkzGetPoint{E}",
                "\\tkzDefMidPoint(A,B)\\tkzGetPoint{F}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawPolygon(D,E,F)",
                "\\tkzDrawPoints(A,B,C,D,E,F)", "\\tkzLabelPoints(A,B,C,D,E,F)"]
        desc = (f"{_tri_intro(rng, pts)} Let D, E, and F be the midpoints of sides "
                "BC, CA, and AB, respectively, so that DEF is the medial triangle. "
                + _draw_suffix(rng, "triangle ABC and its medial triangle DEF",
                               ["A", "B", "C", "D", "E", "F"]))
        return {"tag": "medial_triangle", "points": _round_pts(pts),
                "derived": ["D", "E", "F"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("medial_triangle")


def _b_orthic_triangle(rng):
    """Feet D,E,F of the altitudes from A,B,C — the orthic triangle (acute tri)."""
    for _ in range(4000):
        a, b, c = _rand_triangle(rng)
        if not _is_acute(a, b, c):
            continue
        base = {"A": a, "B": b, "C": c}
        d, e, f = foot(a, b, c), foot(b, c, a), foot(c, a, b)
        pts = {**base, "D": d, "E": e, "F": f}
        if not all(_sane(pts[k], base) for k in ("D", "E", "F")):
            continue
        if not _pairwise_ok(pts, ["D", "E", "F"], sep=0.8):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefPointBy[projection=onto B--C](A)\\tkzGetPoint{D}",
                "\\tkzDefPointBy[projection=onto C--A](B)\\tkzGetPoint{E}",
                "\\tkzDefPointBy[projection=onto A--B](C)\\tkzGetPoint{F}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawPolygon(D,E,F)",
                "\\tkzDrawSegments(A,D B,E C,F)",
                "\\tkzDrawPoints(A,B,C,D,E,F)", "\\tkzLabelPoints(A,B,C,D,E,F)"]
        desc = (f"{_tri_intro(rng, pts)} Let D, E, and F be the feet of the "
                "altitudes from A, B, and C (D on BC, E on CA, F on AB), so that "
                "DEF is the orthic triangle. "
                + _draw_suffix(rng, "triangle ABC, its three altitudes, and the "
                               "orthic triangle DEF", ["A", "B", "C", "D", "E", "F"]))
        return {"tag": "orthic_triangle", "points": _round_pts(pts),
                "derived": ["D", "E", "F"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("orthic_triangle")


def _b_incircle_contact(rng):
    """Incentre I and the three contact points D,E,F (projections of I onto sides)."""
    for _ in range(4000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        i = incenter(a, b, c)
        d, e, f = foot(i, b, c), foot(i, c, a), foot(i, a, b)
        pts = {**base, "I": i, "D": d, "E": e, "F": f}
        if not all(_sane(pts[k], base) for k in ("I", "D", "E", "F")):
            continue
        if not _pairwise_ok(pts, ["I", "D", "E", "F"], sep=0.6):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefTriangleCenter[in](A,B,C)\\tkzGetPoint{I}",
                "\\tkzDefPointBy[projection=onto B--C](I)\\tkzGetPoint{D}",
                "\\tkzDefPointBy[projection=onto C--A](I)\\tkzGetPoint{E}",
                "\\tkzDefPointBy[projection=onto A--B](I)\\tkzGetPoint{F}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawCircle[in](A,B,C)",
                "\\tkzDrawPoints(A,B,C,I,D,E,F)", "\\tkzLabelPoints(A,B,C,I,D,E,F)"]
        desc = (f"{_tri_intro(rng, pts)} Let I be the incenter and let the incircle "
                "touch sides BC, CA, and AB at D, E, and F respectively (the contact "
                "triangle DEF). "
                + _draw_suffix(rng, "triangle ABC, its incircle, and the contact "
                               "points D, E, F", ["A", "B", "C", "I", "D", "E", "F"]))
        return {"tag": "incircle_contact", "points": _round_pts(pts),
                "derived": ["I", "D", "E", "F"], "description": desc,
                "tikz": _fig(body), "unordered": None}
    raise RuntimeError("incircle_contact")


def _b_bisector_incenter(rng):
    """I as the intersection of two angle bisectors (feet DA on BC, DB on CA)."""
    for _ in range(4000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        da = bisector_foot(a, b, c)   # bisector from A meets BC
        db = bisector_foot(b, c, a)   # bisector from B meets CA
        i = incenter(a, b, c)
        pts = {**base, "D": da, "E": db, "I": i}
        if not all(_sane(pts[k], base) for k in ("D", "E", "I")):
            continue
        if not _pairwise_ok(pts, ["D", "E", "I"], sep=0.6):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefLine[bisector](B,A,C)\\tkzGetPoint{ba}",
                "\\tkzInterLL(A,ba)(B,C)\\tkzGetPoint{D}",
                "\\tkzDefLine[bisector](C,B,A)\\tkzGetPoint{bb}",
                "\\tkzInterLL(B,bb)(C,A)\\tkzGetPoint{E}",
                "\\tkzInterLL(A,D)(B,E)\\tkzGetPoint{I}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawSegments(A,D B,E)",
                "\\tkzDrawPoints(A,B,C,D,E,I)", "\\tkzLabelPoints(A,B,C,D,E,I)"]
        desc = (f"{_tri_intro(rng, pts)} Let D be where the internal bisector of "
                "angle A meets BC, and E where the internal bisector of angle B "
                "meets CA. The two bisectors AD and BE meet at the incenter I. "
                + _draw_suffix(rng, "triangle ABC with the two angle bisectors and "
                               "their intersection I", ["A", "B", "C", "D", "E", "I"]))
        return {"tag": "bisector_incenter", "points": _round_pts(pts),
                "derived": ["D", "E", "I"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("bisector_incenter")


def _b_three_medians(rng):
    """Midpoints D,E,F of the sides and the centroid G where the medians meet."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        d, e, f = midpoint(b, c), midpoint(c, a), midpoint(a, b)
        g = centroid(a, b, c)
        pts = {**base, "D": d, "E": e, "F": f, "G": g}
        if not all(_sane(pts[k], base) for k in ("D", "E", "F", "G")):
            continue
        if not _pairwise_ok(pts, ["D", "E", "F", "G"], sep=0.6):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefMidPoint(B,C)\\tkzGetPoint{D}",
                "\\tkzDefMidPoint(C,A)\\tkzGetPoint{E}",
                "\\tkzDefMidPoint(A,B)\\tkzGetPoint{F}",
                "\\tkzDefTriangleCenter[centroid](A,B,C)\\tkzGetPoint{G}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawSegments(A,D B,E C,F)",
                "\\tkzDrawPoints(A,B,C,D,E,F,G)", "\\tkzLabelPoints(A,B,C,D,E,F,G)"]
        desc = (f"{_tri_intro(rng, pts)} Let D, E, F be the midpoints of BC, CA, AB, "
                "so AD, BE, CF are the three medians; they all pass through the "
                "centroid G. "
                + _draw_suffix(rng, "triangle ABC, its three medians, and the "
                               "centroid G", ["A", "B", "C", "D", "E", "F", "G"]))
        return {"tag": "three_medians", "points": _round_pts(pts),
                "derived": ["D", "E", "F", "G"], "description": desc,
                "tikz": _fig(body), "unordered": None}
    raise RuntimeError("three_medians")


def _b_parallelogram_center(rng):
    """4th vertex D of parallelogram ABCD + centre X = diagonal intersection."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng, bound=6)
        d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
        x = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2)  # centre = midpoint of AC (=BD)
        base = {"A": a, "B": b, "C": c}
        if not (_sane(d, base) and _sane(x, base)):
            continue
        pts = {"A": a, "B": b, "C": c, "D": d, "X": x}
        if not _pairwise_ok(pts, ["D", "X"], sep=0.6):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefPointBy[translation=from B to C](A)\\tkzGetPoint{D}",
                "\\tkzInterLL(A,C)(B,D)\\tkzGetPoint{X}",
                "\\tkzDrawPolygon(A,B,C,D)", "\\tkzDrawSegments(A,C B,D)",
                "\\tkzDrawPoints(A,B,C,D,X)", "\\tkzLabelPoints(A,B,C,D,X)"]
        coords = _coords(pts, "ABC")
        setup = rng.choice([
            f"Three vertices of parallelogram ABCD are {coords}.",
            f"ABCD is a parallelogram with {coords} given (in order).",
        ])
        desc = (f"{setup} Let D be the fourth vertex (so AD is parallel to BC), and "
                "let X be the center of the parallelogram, where the diagonals AC "
                "and BD cross. "
                + _draw_suffix(rng, "parallelogram ABCD with both diagonals and "
                               "their intersection X", ["A", "B", "C", "D", "X"]))
        return {"tag": "parallelogram_center", "points": _round_pts(pts),
                "derived": ["D", "X"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("parallelogram_center")


def _b_midpoint_reflect_chain(rng):
    """Chain: M = midpoint(B,C); N = reflection of A through M (central symmetry)."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        m = midpoint(b, c)
        n = _symmetry(m, a)
        pts = {**base, "M": m, "N": n}
        if not (_sane(m, base) and _sane(n, base)) or not _pairwise_ok(pts, ["M", "N"]):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefMidPoint(B,C)\\tkzGetPoint{M}",
                "\\tkzDefPointBy[symmetry=center M](A)\\tkzGetPoint{N}",
                "\\tkzDrawPolygon(A,B,C)", "\\tkzDrawSegments(A,N)",
                "\\tkzDrawPoints(A,B,C,M,N)", "\\tkzLabelPoints(A,B,C,M,N)"]
        desc = (f"{_tri_intro(rng, pts)} Let M be the midpoint of BC, and let N be "
                "the reflection of A through M (equivalently, ABNC is a "
                "parallelogram). "
                + _draw_suffix(rng, "triangle ABC, the midpoint M of BC, and the "
                               "reflected point N", ["A", "B", "C", "M", "N"]))
        return {"tag": "midpoint_reflect_chain", "points": _round_pts(pts),
                "derived": ["M", "N"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("midpoint_reflect_chain")


def _b_circumcircle_antipode(rng):
    """Circumcentre O + circumcircle; S = antipode of A on that circle."""
    for _ in range(3000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        o = circumcenter(a, b, c)
        s = _symmetry(o, a)
        pts = {**base, "O": o, "S": s}
        if not (_sane(o, base) and _sane(s, base)) or not _pairwise_ok(pts, ["O", "S"]):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}",
                "\\tkzDefPointBy[symmetry=center O](A)\\tkzGetPoint{S}",
                "\\tkzDrawCircle(O,A)", "\\tkzDrawPolygon(A,B,C)",
                "\\tkzDrawSegment(A,S)",
                "\\tkzDrawPoints(A,B,C,O,S)", "\\tkzLabelPoints(A,B,C,O,S)"]
        desc = (f"{_tri_intro(rng, pts)} Let O be the circumcenter of the triangle "
                "and draw the circumcircle through A, B, C. Let S be the antipode of "
                "A on the circumcircle (the point diametrically opposite A). "
                + _draw_suffix(rng, "triangle ABC, its circumcircle, and the "
                               "antipode S of A", ["A", "B", "C", "O", "S"]))
        return {"tag": "circumcircle_antipode", "points": _round_pts(pts),
                "derived": ["O", "S"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("circumcircle_antipode")


def _b_square_center(rng):
    """Square ABCD erected on side AB (rotations) + its center X."""
    for _ in range(3000):
        a = (rng.randint(-5, 2), rng.randint(-5, 2))
        b = (a[0] + rng.randint(2, 5), a[1] + rng.randint(-2, 2))
        if _dist(a, b) < 2.5:
            continue
        w = (-(b[1] - a[1]), b[0] - a[0])  # ccw perpendicular of AB
        d = (a[0] + w[0], a[1] + w[1])
        c = (b[0] + w[0], b[1] + w[1])
        x = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2)
        base = {"A": a, "B": b}
        if not all(_sane(p, base) for p in (c, d, x)):
            continue
        pts = {"A": a, "B": b, "C": c, "D": d, "X": x}
        if not _pairwise_ok(pts, ["C", "D", "X"], sep=0.6):
            continue
        body = [_pt("A", a), _pt("B", b),
                "\\tkzDefPointBy[rotation=center A angle 90](B)\\tkzGetPoint{D}",
                "\\tkzDefPointBy[rotation=center B angle -90](A)\\tkzGetPoint{C}",
                "\\tkzInterLL(A,C)(B,D)\\tkzGetPoint{X}",
                "\\tkzDrawPolygon(A,B,C,D)", "\\tkzDrawSegments(A,C B,D)",
                "\\tkzDrawPoints(A,B,C,D,X)", "\\tkzLabelPoints(A,B,C,D,X)"]
        desc = (f"ABCD is a square (vertices in order) with side AB from "
                f"A=({_g(a[0])},{_g(a[1])}) to B=({_g(b[0])},{_g(b[1])}). Let X be "
                "the center of the square, where its diagonals AC and BD meet. "
                + _draw_suffix(rng, "square ABCD with its diagonals and center X",
                               ["A", "B", "C", "D", "X"]))
        return {"tag": "square_center", "points": _round_pts(pts),
                "derived": ["C", "D", "X"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("square_center")


def _b_reflect_ortho_over_side(rng):
    """Compose: O, H, then H' = reflection of H over BC (lands on circumcircle)."""
    for _ in range(4000):
        a, b, c = _rand_triangle(rng)
        if not _is_acute(a, b, c):
            continue
        base = {"A": a, "B": b, "C": c}
        o = circumcenter(a, b, c)
        h = orthocenter(a, b, c)
        r = _reflect_over_line(h, b, c)
        pts = {**base, "O": o, "H": h, "R": r}
        if not all(_sane(pts[k], base) for k in ("O", "H", "R")):
            continue
        if not _pairwise_ok(pts, ["O", "H", "R"], sep=0.6):
            continue
        body = [_pt("A", a), _pt("B", b), _pt("C", c),
                "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}",
                "\\tkzDefTriangleCenter[ortho](A,B,C)\\tkzGetPoint{H}",
                "\\tkzDefPointBy[reflection=over B--C](H)\\tkzGetPoint{R}",
                "\\tkzDrawCircle(O,A)", "\\tkzDrawPolygon(A,B,C)",
                "\\tkzDrawSegment(H,R)",
                "\\tkzDrawPoints(A,B,C,O,H,R)", "\\tkzLabelPoints(A,B,C,O,H,R)"]
        desc = (f"{_tri_intro(rng, pts)} Let O be the circumcenter and H the "
                "orthocenter. Let R be the reflection of H over line BC; it lies on "
                "the circumcircle. "
                + _draw_suffix(rng, "triangle ABC, its circumcircle, the "
                               "orthocenter H, and its reflection R over BC",
                               ["A", "B", "C", "O", "H", "R"]))
        return {"tag": "reflect_ortho_over_side", "points": _round_pts(pts),
                "derived": ["O", "H", "R"], "description": desc, "tikz": _fig(body),
                "unordered": None}
    raise RuntimeError("reflect_ortho_over_side")


_BUILDERS = {
    "euler_line": _b_euler_line,
    "nine_point_center": _b_nine_point_center,
    "medial_triangle": _b_medial_triangle,
    "orthic_triangle": _b_orthic_triangle,
    "incircle_contact": _b_incircle_contact,
    "bisector_incenter": _b_bisector_incenter,
    "three_medians": _b_three_medians,
    "parallelogram_center": _b_parallelogram_center,
    "midpoint_reflect_chain": _b_midpoint_reflect_chain,
    "circumcircle_antipode": _b_circumcircle_antipode,
    "square_center": _b_square_center,
    "reflect_ortho_over_side": _b_reflect_ortho_over_side,
}


def make_problem(rng: random.Random, tag: str) -> dict:
    if tag not in _BUILDERS:
        raise ValueError(f"unknown construction: {tag}")
    return _BUILDERS[tag](rng)


def generate_problems(n_per_type: int, seed: int = 0,
                      types: list[str] | None = None) -> list[dict]:
    """Sample ``n_per_type`` distinct problems for each harder construction type."""
    rng = random.Random(seed)
    types = types or TYPES
    out: list[dict] = []
    pid = 0
    for tag in types:
        seen: set[str] = set()
        got = tries = 0
        while got < n_per_type and tries < n_per_type * 400 + 4000:
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
