"""Olympiad construction generator (v3 track — additive, coordinate-free targets).

Where v1/v2 stress compose-a-chain-of-transforms, this track isolates the
*named olympiad constructions* competition geometry actually leans on:

    circumcenter/circumcircle, incenter/incircle, orthocenter, centroid,
    angle bisector, foot of an altitude, median, tangent-from-a-point.

Each problem pins a base configuration with explicit coordinates (so grading has
a fixed frame, exactly like v1's "circle at the origin" anchor) and then asks —
purely in words, coordinate-free — for a DERIVED construction. From one sampled
scene we emit:

  * exact Python ground-truth coordinates for every named point,
  * a natural-language description naming the points the model must define,
  * a ground-truth tkz-euclide figure that CONSTRUCTS those points.

The ground truth round-trips through the compile-extract grader (emit -> compile
-> extract == GT), which cross-checks the Python math against TeX's own geometry
engine. Grading is done by ``extract.grade`` (compile-extract), never static
parsing, because tkz-euclide coordinates are not statically recoverable.
"""

from __future__ import annotations

import math
import random

TYPES = [
    "circumcenter",
    "incenter",
    "orthocenter",
    "centroid",
    "angle_bisector",
    "foot_altitude",
    "median",
    "tangent",
]

Pt = tuple[float, float]


# --------------------------------------------------------------------------- #
# exact geometry (independent of TeX; this is the ground truth we grade against)
# --------------------------------------------------------------------------- #
def _sub(p: Pt, q: Pt) -> Pt:
    return (p[0] - q[0], p[1] - q[1])


def _dist(p: Pt, q: Pt) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _area2(a: Pt, b: Pt, c: Pt) -> float:
    """Twice the signed triangle area (>0 ccw)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def circumcenter(a: Pt, b: Pt, c: Pt) -> Pt:
    d = 2 * _area2(a, b, c)
    ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
          + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
    uy = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
          + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
    return (ux, uy)


def incenter(a: Pt, b: Pt, c: Pt) -> Pt:
    la, lb, lc = _dist(b, c), _dist(c, a), _dist(a, b)  # sides opposite A,B,C
    s = la + lb + lc
    return ((la * a[0] + lb * b[0] + lc * c[0]) / s,
            (la * a[1] + lb * b[1] + lc * c[1]) / s)


def orthocenter(a: Pt, b: Pt, c: Pt) -> Pt:
    o = circumcenter(a, b, c)
    return (a[0] + b[0] + c[0] - 2 * o[0], a[1] + b[1] + c[1] - 2 * o[1])


def centroid(a: Pt, b: Pt, c: Pt) -> Pt:
    return ((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3)


def midpoint(p: Pt, q: Pt) -> Pt:
    return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)


def foot(apex: Pt, b: Pt, c: Pt) -> Pt:
    """Foot of the perpendicular from ``apex`` onto line b--c."""
    dx, dy = c[0] - b[0], c[1] - b[1]
    t = ((apex[0] - b[0]) * dx + (apex[1] - b[1]) * dy) / (dx * dx + dy * dy)
    return (b[0] + t * dx, b[1] + t * dy)


def bisector_foot(a: Pt, b: Pt, c: Pt) -> Pt:
    """Where the internal bisector of angle A meets side b--c (ratio AB:AC)."""
    lb, lc = _dist(c, a), _dist(a, b)  # AC, AB
    return ((lb * b[0] + lc * c[0]) / (lb + lc), (lb * b[1] + lc * c[1]) / (lb + lc))


def tangent_points(o: Pt, r: float, p: Pt) -> tuple[Pt, Pt]:
    """The two points where tangents from external ``p`` touch circle (o,r)."""
    d = _dist(o, p)
    ell = math.sqrt(max(d * d - r * r, 0.0))
    ux, uy = (p[0] - o[0]) / d, (p[1] - o[1]) / d
    px, py = -uy, ux
    fx, fy = o[0] + (r * r / d) * ux, o[1] + (r * r / d) * uy
    off = r * ell / d
    return ((fx + off * px, fy + off * py), (fx - off * px, fy - off * py))


# --------------------------------------------------------------------------- #
# emission helpers
# --------------------------------------------------------------------------- #
def _g(x: float) -> str:
    """Compact number: integers stay integers, else 4 decimals (no -0)."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    v = round(x, 4)
    return f"{v:g}"


def _pt(name: str, p: Pt) -> str:
    return f"\\tkzDefPoint({_g(p[0])},{_g(p[1])}){{{name}}}"


def _coords_str(pts: dict[str, Pt], names: list[str]) -> str:
    return ", ".join(f"{n}=({_g(pts[n][0])},{_g(pts[n][1])})" for n in names)


def _round_pts(pts: dict[str, Pt]) -> dict[str, list[float]]:
    return {k: [round(v[0], 4), round(v[1], 4)] for k, v in pts.items()}


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _rand_triangle(rng: random.Random, bound: int = 8) -> tuple[Pt, Pt, Pt]:
    """A non-degenerate, not-too-thin integer triangle."""
    for _ in range(2000):
        a = (rng.randint(0, bound), rng.randint(0, bound))
        b = (rng.randint(0, bound), rng.randint(0, bound))
        c = (rng.randint(0, bound), rng.randint(0, bound))
        if abs(_area2(a, b, c)) < 12:  # area >= 6
            continue
        if min(_dist(a, b), _dist(b, c), _dist(c, a)) < 3:
            continue
        return a, b, c
    return (0, 0), (6, 0), (2, 5)


def _sane(derived: Pt, base: dict[str, Pt], bound: float = 13.0, sep: float = 0.4) -> bool:
    """Derived point is drawable (in bounds) and not coincident with a base point."""
    if abs(derived[0]) > bound or abs(derived[1]) > bound:
        return False
    return all(_dist(derived, q) >= sep for q in base.values())


# --------------------------------------------------------------------------- #
# per-construction builders -> Problem dict
# --------------------------------------------------------------------------- #
def _tri_problem(rng, tag, derive, dname, dphrase, draw_extra):
    """Shared skeleton for the single-derived-point triangle constructions."""
    for _ in range(2000):
        a, b, c = _rand_triangle(rng)
        base = {"A": a, "B": b, "C": c}
        d = derive(a, b, c)
        if not _sane(d, base):
            continue
        pts = {**base, dname: d}
        body = [_pt("A", a), _pt("B", b), _pt("C", c)]
        body += draw_extra(dname)
        body += [
            "\\tkzDrawPolygon(A,B,C)",
            f"\\tkzDrawPoints(A,B,C,{dname})",
            f"\\tkzLabelPoints(A,B,C,{dname})",
        ]
        tikz = "\\begin{tikzpicture}\n  " + "\n  ".join(body) + "\n\\end{tikzpicture}"
        desc = (
            f"Triangle ABC has vertices {_coords_str(pts, ['A', 'B', 'C'])}. "
            f"{dphrase} "
            f"Output a single TikZ figure that draws triangle ABC and defines the "
            f"named points A, B, C, {dname} at their correct positions."
        )
        return {"tag": tag, "points": _round_pts(pts), "derived": [dname],
                "description": desc, "tikz": tikz, "unordered": None}
    raise RuntimeError(f"could not sample {tag}")


def _b_circumcenter(rng):
    return _tri_problem(
        rng, "circumcenter", circumcenter, "O",
        "Let O be the circumcenter of triangle ABC — the point equidistant from "
        "A, B, and C, i.e. the center of the circle passing through all three "
        "vertices. Also draw that circumcircle.",
        lambda d: [f"\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{{{d}}}",
                   f"\\tkzDrawCircle({d},A)"],
    )


def _b_incenter(rng):
    return _tri_problem(
        rng, "incenter", incenter, "I",
        "Let I be the incenter of triangle ABC — the point equidistant from all "
        "three sides, i.e. the center of the inscribed circle (the circle tangent "
        "to all three sides). Also draw that incircle.",
        lambda d: [f"\\tkzDefTriangleCenter[in](A,B,C)\\tkzGetPoint{{{d}}}",
                   f"\\tkzDrawCircle[in](A,B,C)"],
    )


def _b_orthocenter(rng):
    return _tri_problem(
        rng, "orthocenter", orthocenter, "H",
        "Let H be the orthocenter of triangle ABC — the common intersection of "
        "the three altitudes (each altitude passes through a vertex and is "
        "perpendicular to the opposite side).",
        lambda d: [f"\\tkzDefTriangleCenter[ortho](A,B,C)\\tkzGetPoint{{{d}}}"],
    )


def _b_centroid(rng):
    return _tri_problem(
        rng, "centroid", centroid, "G",
        "Let G be the centroid of triangle ABC — the intersection of the three "
        "medians (each median joins a vertex to the midpoint of the opposite "
        "side).",
        lambda d: [f"\\tkzDefTriangleCenter[centroid](A,B,C)\\tkzGetPoint{{{d}}}"],
    )


def _b_angle_bisector(rng):
    return _tri_problem(
        rng, "angle_bisector", bisector_foot, "D",
        "Let D be the point where the internal bisector of angle A (angle BAC) "
        "meets side BC. Also draw segment AD.",
        lambda d: [
            f"\\tkzDefLine[bisector](B,A,C)\\tkzGetPoint{{{d}bl}}",
            f"\\tkzInterLL(A,{d}bl)(B,C)\\tkzGetPoint{{{d}}}",
            f"\\tkzDrawSegment(A,{d})",
        ],
    )


def _b_foot_altitude(rng):
    return _tri_problem(
        rng, "foot_altitude", lambda a, b, c: foot(a, b, c), "F",
        "Let F be the foot of the altitude from A — the foot of the perpendicular "
        "dropped from vertex A onto line BC. Also draw segment AF.",
        lambda d: [
            f"\\tkzDefPointBy[projection=onto B--C](A)\\tkzGetPoint{{{d}}}",
            f"\\tkzDrawSegment(A,{d})",
        ],
    )


def _b_median(rng):
    return _tri_problem(
        rng, "median", lambda a, b, c: midpoint(b, c), "M",
        "Let M be the midpoint of side BC, so that AM is the median from A. "
        "Also draw segment AM.",
        lambda d: [f"\\tkzDefMidPoint(B,C)\\tkzGetPoint{{{d}}}",
                   f"\\tkzDrawSegment(A,{d})"],
    )


def _b_tangent(rng):
    """Circle (center O through helper W) + external P; tangency points T1,T2."""
    for _ in range(2000):
        o = (rng.randint(-1, 3), rng.randint(-1, 3))
        r = rng.randint(2, 4)
        ang = math.radians(rng.randint(0, 359))
        dd = r + rng.choice([2, 3, 4, 5])
        p = (o[0] + dd * math.cos(ang), o[1] + dd * math.sin(ang))
        p = (round(p[0]), round(p[1]))
        if _dist(o, p) <= r + 1.2:  # keep P safely external after rounding
            continue
        t1, t2 = tangent_points(o, float(r), p)
        base = {"O": o, "P": p}
        if not (_sane(t1, base, sep=0.5) and _sane(t2, base, sep=0.5)):
            continue
        if _dist(t1, t2) < 1.0:  # near-degenerate (P almost on circle)
            continue
        w = (o[0] + r, o[1])  # helper point on the circle (defines the radius)
        pts = {"O": o, "P": p, "T1": t1, "T2": t2}
        body = [
            _pt("O", o), _pt("P", p), _pt("W", w),
            "\\tkzDefTangent[from = P](O,W)\\tkzGetPoints{T1}{T2}",
            "\\tkzDrawCircle(O,W)",
            "\\tkzDrawSegments(P,T1 P,T2)",
            "\\tkzDrawPoints(O,P,T1,T2)",
            "\\tkzLabelPoints(O,P,T1,T2)",
        ]
        tikz = "\\begin{tikzpicture}\n  " + "\n  ".join(body) + "\n\\end{tikzpicture}"
        desc = (
            f"A circle has center O=({_g(o[0])},{_g(o[1])}) and radius {r}. "
            f"Point P=({_g(p[0])},{_g(p[1])}) lies outside the circle. From P there "
            f"are two tangent lines to the circle; let T1 and T2 be the two points "
            f"of tangency (where the tangents touch the circle). "
            f"Output a single TikZ figure that draws the circle and the two "
            f"tangent segments and defines the named points O, P, T1, T2 at their "
            f"correct positions."
        )
        return {"tag": "tangent", "points": _round_pts(pts), "derived": ["T1", "T2"],
                "description": desc, "tikz": tikz, "unordered": [["T1", "T2"]]}
    raise RuntimeError("could not sample tangent")


_BUILDERS = {
    "circumcenter": _b_circumcenter,
    "incenter": _b_incenter,
    "orthocenter": _b_orthocenter,
    "centroid": _b_centroid,
    "angle_bisector": _b_angle_bisector,
    "foot_altitude": _b_foot_altitude,
    "median": _b_median,
    "tangent": _b_tangent,
}


def make_problem(rng: random.Random, tag: str) -> dict:
    if tag not in _BUILDERS:
        raise ValueError(f"unknown construction: {tag}")
    return _BUILDERS[tag](rng)


def generate_problems(n_per_type: int, seed: int = 0,
                      types: list[str] | None = None) -> list[dict]:
    """Sample ``n_per_type`` distinct problems for each construction type."""
    rng = random.Random(seed)
    types = types or TYPES
    out: list[dict] = []
    pid = 0
    for tag in types:
        seen: set[str] = set()
        got = 0
        tries = 0
        while got < n_per_type and tries < n_per_type * 200 + 2000:
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
