"""Spec-first synthetic generator.

Samples a symbolic geometry scene with exact coordinates already known, then
emits a coordinate-free constraint description (model input) alongside the
ground-truth TikZ (what we grade against).

Difficulty dial:
  - chain length  = number of derivation steps that compose
  - number irregularity = round vs non-round angles/radii
"""

from __future__ import annotations

import math
import random

from .scene import Scene, line_intersection, midpoint, polar

# angle pools: "round" is fakeable, "irregular" forces real computation
ROUND_ANGLES = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315]
IRREGULAR_ANGLES = [17, 23, 38, 52, 68, 74, 83, 97, 113, 128, 142, 161, 199, 233, 287]


def _angles(rng: random.Random, irregular: bool) -> list[int]:
    return IRREGULAR_ANGLES if irregular else ROUND_ANGLES


def _radius(rng: random.Random, irregular: bool) -> float:
    return rng.choice([2.5, 3.5, 4.5, 5.5]) if irregular else rng.choice([3, 4, 5, 6])


def _coincident(x: float, y: float, pts: dict[str, tuple[float, float]], tol: float = 0.15) -> bool:
    """True if (x,y) lands on an existing point -> ambiguous/degenerate, reject."""
    return any(math.hypot(x - px, y - py) < tol for px, py in pts.values())


# Operation taxonomy. "Easy" ops (reflection/midpoint/second point) are near-
# trivial for frontier models; "hard" ops (line intersection, foot of an
# altitude) require real projection/solving and are what actually breaks them.
EASY_OPS = ["reflect_x", "reflect_y", "midpoint_center", "second_point"]
HARD_OPS = ["intersection", "foot_altitude"]
ALL_OPS = EASY_OPS + HARD_OPS

# Minimum chain length for a forced op to have enough points to act on.
MIN_CHAIN_FOR_OP = {"intersection": 3, "foot_altitude": 4}


def build_scene(rng: random.Random, chain: int, irregular: bool,
                force_op: str | None = None, easy_only: bool = False,
                symbolic: bool = False) -> Scene:
    """Compose `chain` derivation steps into one scene.

    force_op   : if set, the final step is forced to be this op and every earlier
                 step is an easy setup step (so the hard op has points to act on).
                 Use this to build the *shortest* scene that contains a given op.
    easy_only  : restrict to easy ops (a matched control with no hard reasoning).
    symbolic   : also emit coordinate-free PGF constructions (v2 behavior).
    """
    s = Scene(symbolic=symbolic)
    r = _radius(rng, irregular)
    s.circle(0, 0, r)
    if symbolic:
        s.sym("  \\coordinate (O) at (0,0);")  # named origin for constructions
    s.constrain(f"There is a circle centered at the origin with radius {r:g}.")

    # Step 1: a point on the circle at some angle (always present).
    ang = rng.choice(_angles(rng, irregular))
    ax, ay = polar(0, 0, r, ang)
    s.add_point("A", ax, ay, at=f"{ang}:{r:g}")
    s.constrain(
        f"Point A lies on the circle at an angle of {ang} degrees measured "
        f"counterclockwise from the positive x-axis."
    )
    s.bump("point_on_circle")

    available = ["A"]
    n_steps = max(0, chain - 1)

    for step_i in range(n_steps):
        remaining = n_steps - step_i
        if force_op is not None and remaining == 1:
            op = force_op  # final step carries the target (hard) op
        elif force_op is not None or easy_only:
            op = rng.choice(EASY_OPS)  # setup / control steps stay easy
        else:
            op = rng.choice(ALL_OPS)
        nxt = chr(ord("A") + len(s.points))  # B, C, D, ...

        if op == "reflect_x":
            src = rng.choice(available)
            px, py = s.points[src]
            if _coincident(px, -py, s.points):  # src on the x-axis -> reflection is itself
                continue
            s.add_point(nxt, px, -py, "below right", at=f"$({src})!2!({src} |- O)$")
            s.constrain(f"Point {nxt} is the reflection of {src} across the x-axis.")
            s.segment(src, nxt)
            s.bump("reflect_x")

        elif op == "reflect_y":
            src = rng.choice(available)
            px, py = s.points[src]
            if _coincident(-px, py, s.points):  # src on the y-axis, or double-reflection back
                continue
            s.add_point(nxt, -px, py, "above left", at=f"$({src})!2!(O |- {src})$")
            s.constrain(f"Point {nxt} is the reflection of {src} across the y-axis.")
            s.segment(src, nxt)
            s.bump("reflect_y")

        elif op == "midpoint_center":
            src = rng.choice(available)
            px, py = s.points[src]
            mx, my = midpoint((px, py), (0, 0))
            if _coincident(mx, my, s.points):
                continue
            s.add_point(nxt, mx, my, "below left", at=f"$({src})!0.5!(O)$")
            s.constrain(f"Point {nxt} is the midpoint of the segment from {src} to the origin.")
            s.bump("midpoint")

        elif op == "second_point":
            ang2 = rng.choice(_angles(rng, irregular))
            bx, by = polar(0, 0, r, ang2)
            if _coincident(bx, by, s.points):  # same angle as an existing on-circle point
                continue
            s.add_point(nxt, bx, by, "above", at=f"{ang2}:{r:g}")
            s.constrain(
                f"Point {nxt} lies on the circle at an angle of {ang2} degrees "
                f"counterclockwise from the positive x-axis."
            )
            s.bump("point_on_circle")

        elif op == "intersection":
            if len(available) >= 2:
                p, q = rng.sample(available, 2)
                pp, qq = s.points[p], s.points[q]
                # line through p and the origin, met by the *vertical* line x = qq_x
                inter = line_intersection(pp, (0, 0), qq, (qq[0], qq[1] - 1))
                if inter is None:  # line p-origin is itself vertical (parallel) -> skip
                    continue
                if abs(inter[0]) > 7 or abs(inter[1]) > 7:
                    continue  # far/near-parallel intersection: not cleanly drawable
                if _coincident(inter[0], inter[1], s.points):
                    continue
                s.add_point(nxt, inter[0], inter[1], "right")
                if s.symbolic:  # line through O and p, met by the vertical through q
                    s.sym(f"  \\path[name path=lpo{nxt}] ($(O)!-8!({p})$) -- ($(O)!8!({p})$);")
                    s.sym(f"  \\path[name path=lv{nxt}] ($({q})+(0,-15)$) -- ($({q})+(0,15)$);")
                    s.sym(f"  \\path[name intersections={{of=lpo{nxt} and lv{nxt}, by={nxt}}}];")
                    s.label(nxt, "right")
                s.constrain(
                    f"Point {nxt} is where line {p}-origin meets the vertical line through {q}."
                )
                s.bump("intersection")
            else:
                continue

        elif op == "foot_altitude":
            # Need a base line (two points) and an apex; drop the perpendicular
            # foot from the apex onto the line. Fully coordinate-free.
            if len(available) >= 3:
                a, b, c = rng.sample(available, 3)
                pa, pb, pc = s.points[a], s.points[b], s.points[c]
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                denom = dx * dx + dy * dy
                if denom < 1e-6:  # degenerate line
                    continue
                t = ((pc[0] - pa[0]) * dx + (pc[1] - pa[1]) * dy) / denom
                fx, fy = pa[0] + t * dx, pa[1] + t * dy
                # reject degenerate feet: apex on the line (foot == apex), or the
                # foot landing on an existing point (ambiguous coincident labels).
                if math.hypot(pc[0] - fx, pc[1] - fy) < 0.5:
                    continue
                if any(math.hypot(fx - px, fy - py) < 0.1 for px, py in s.points.values()):
                    continue
                s.add_point(nxt, fx, fy, "below", at=f"$({a})!({c})!({b})$")
                s.segment(a, b)
                s.segment(c, nxt)
                s.constrain(
                    f"Point {nxt} is the foot of the perpendicular dropped from {c} "
                    f"onto the line through {a} and {b} (the altitude from {c})."
                )
                s.bump("foot_altitude")
            else:
                continue

        available.append(nxt)

    return s


def make_example(rng: random.Random, chain: int, irregular: bool,
                 force_op: str | None = None, easy_only: bool = False,
                 symbolic: bool = False) -> dict:
    s = build_scene(rng, chain, irregular, force_op=force_op, easy_only=easy_only,
                    symbolic=symbolic)
    return {
        "constraints": s.constraints,
        "description": " ".join(s.constraints),
        "tikz": s.to_tikz(),
        "points": s.ground_truth_points(),
        "chain": s.steps,
        "irregular": irregular,
        "tags": s.tags,
    }


def generate_dataset(n: int, seed: int = 0, max_chain: int = 3) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        chain = rng.randint(1, max_chain)
        irregular = rng.random() < 0.5
        ex = make_example(rng, chain, irregular)
        ex["id"] = i
        out.append(ex)
    return out
