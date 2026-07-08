"""Spec-first synthetic generator.

Samples a symbolic geometry scene with exact coordinates already known, then
emits a coordinate-free constraint description (model input) alongside the
ground-truth TikZ (what we grade against).

Difficulty dial:
  - chain length  = number of derivation steps that compose
  - number irregularity = round vs non-round angles/radii
"""

from __future__ import annotations

import random

from .scene import Scene, line_intersection, midpoint, polar

# angle pools: "round" is fakeable, "irregular" forces real computation
ROUND_ANGLES = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315]
IRREGULAR_ANGLES = [17, 23, 38, 52, 68, 74, 83, 97, 113, 128, 142, 161, 199, 233, 287]


def _angles(rng: random.Random, irregular: bool) -> list[int]:
    return IRREGULAR_ANGLES if irregular else ROUND_ANGLES


def _radius(rng: random.Random, irregular: bool) -> float:
    return rng.choice([2.5, 3.5, 4.5, 5.5]) if irregular else rng.choice([3, 4, 5, 6])


def build_scene(rng: random.Random, chain: int, irregular: bool) -> Scene:
    """Compose `chain` derivation steps into one scene."""
    s = Scene()
    r = _radius(rng, irregular)
    s.circle(0, 0, r)
    s.constrain(f"There is a circle centered at the origin with radius {r:g}.")

    # Step 1: a point on the circle at some angle (always present).
    ang = rng.choice(_angles(rng, irregular))
    ax, ay = polar(0, 0, r, ang)
    s.add_point("A", ax, ay)
    s.constrain(
        f"Point A lies on the circle at an angle of {ang} degrees measured "
        f"counterclockwise from the positive x-axis."
    )
    s.bump("point_on_circle")

    available = ["A"]
    ops = ["reflect_x", "reflect_y", "midpoint_center", "second_point", "intersection"]

    for _ in range(max(0, chain - 1)):
        op = rng.choice(ops)
        nxt = chr(ord("A") + len(s.points))  # B, C, D, ...

        if op == "reflect_x":
            src = rng.choice(available)
            px, py = s.points[src]
            s.add_point(nxt, px, -py, "below right")
            s.constrain(f"Point {nxt} is the reflection of {src} across the x-axis.")
            s.segment(src, nxt)
            s.bump("reflect_x")

        elif op == "reflect_y":
            src = rng.choice(available)
            px, py = s.points[src]
            s.add_point(nxt, -px, py, "above left")
            s.constrain(f"Point {nxt} is the reflection of {src} across the y-axis.")
            s.segment(src, nxt)
            s.bump("reflect_y")

        elif op == "midpoint_center":
            src = rng.choice(available)
            px, py = s.points[src]
            mx, my = midpoint((px, py), (0, 0))
            s.add_point(nxt, mx, my, "below left")
            s.constrain(f"Point {nxt} is the midpoint of the segment from {src} to the origin.")
            s.bump("midpoint")

        elif op == "second_point":
            ang2 = rng.choice(_angles(rng, irregular))
            bx, by = polar(0, 0, r, ang2)
            s.add_point(nxt, bx, by, "above")
            s.constrain(
                f"Point {nxt} lies on the circle at an angle of {ang2} degrees "
                f"counterclockwise from the positive x-axis."
            )
            s.bump("point_on_circle")

        elif op == "intersection":
            if len(available) >= 2:
                p, q = rng.sample(available, 2)
                pp, qq = s.points[p], s.points[q]
                inter = line_intersection(pp, (0, 0), qq, (pp[0], pp[1] - 1))
                if inter is None:
                    continue
                s.add_point(nxt, inter[0], inter[1], "right")
                s.constrain(
                    f"Point {nxt} is where line {p}-origin meets the vertical line through {q}."
                )
                s.bump("intersection")
            else:
                continue

        available.append(nxt)

    return s


def make_example(rng: random.Random, chain: int, irregular: bool) -> dict:
    s = build_scene(rng, chain, irregular)
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
