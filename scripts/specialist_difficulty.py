"""Build a difficulty grid for the geometry SPECIALIST, in its NATIVE style.

The specialist (`qwen3-illustrator-4b`) was trained on olympiad/olympiad_ext
constructions: base points given as EXPLICIT integer coordinates, one or more
further points described only by a coordinate-free CONSTRUCTION, graded by the
compile-extract gate. (It scores 97% on that held-out distribution but ~0% on the
frontier sweep's `generator.py` circle+angle-chain grid, whose polar base-point
spec — "A lies on the circle at angle 30 degrees" — is out-of-distribution.)

To measure the specialist's *complexity ceiling* fairly we ramp difficulty inside
its own distribution, along the two axes the task calls out:

  * CHAIN LENGTH  — number of composed derived-point steps (1..5). Each step
                    defines a new named point from earlier ones.
  * OP COMPLEXITY — the "regime" of the steps:
      - affine : isometry/affine ops only (midpoint, point-symmetry, reflection
                 over a line, rotation, translation) — the EASY ops.
      - mixed  : the final step is a METRIC op (foot-of-perpendicular /
                 line-intersection), the HARD reasoning ops that require real
                 projection/solving. (Mirrors generator.py's EASY vs HARD
                 taxonomy, but in the specialist's explicit-coordinate style.)

Every step uses the exact coordinate-free tkz-euclide macros the specialist was
trained on (`\\tkzDefMidPoint`, `\\tkzDefPointBy[projection=onto ..]`,
`\\tkzInterLL`, ...). Ground-truth coordinates are computed in Python and the
ground-truth FIGURE is ROUND-TRIP VALIDATED through `extract.grade` (emit ->
compile -> read back == GT); any example whose own GT does not round-trip is
dropped, so the specialist is never graded against a wrong label.

A second slice re-samples the 20 native construction FAMILIES (olympiad +
olympiad_ext) at higher n, to read which single constructions the specialist is
strong/weak on and how it handles many-derived-point figures (regular_polygon,
two-point families).

READ-ONLY / additive: this is analysis tooling. It imports the app's generators
and grader but does not modify the app, training, data, or any adapter.

Usage:
  uv run python scripts/specialist_difficulty.py --out outputs/specialist_ceiling/grid.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract, olympiad, olympiad_ext  # noqa: E402
from geotikz.olympiad import (  # noqa: E402
    _dist, _g, _pt, _rand_triangle, _round_pts, _sane, foot, midpoint,
)
from geotikz.olympiad_ext import _draw_suffix, _fig, _reflect_over_line, _rotate  # noqa: E402
from geotikz.scene import line_intersection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [30, 45, 60, 90, 120, -30, -45, -60, -90, -120]
BOUND = 11.0   # keep every point drawable / in a sane frame
SEP = 0.6      # min separation so labels never collide / points never coincide


# --------------------------------------------------------------------------- #
# step operations: (pt, macro, clause, is_metric) | None
# Each consumes earlier named points and defines `name` coordinate-free.
# --------------------------------------------------------------------------- #
def _op_midpoint(rng, pts, avail, name):
    p, q = rng.sample(avail, 2)
    return (midpoint(pts[p], pts[q]),
            f"\\tkzDefMidPoint({p},{q})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the midpoint of segment {p}{q}.", False)


def _op_symmetry(rng, pts, avail, name):
    p, o = rng.sample(avail, 2)
    pp, oo = pts[p], pts[o]
    return ((2 * oo[0] - pp[0], 2 * oo[1] - pp[1]),
            f"\\tkzDefPointBy[symmetry=center {o}]({p})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the reflection of {p} through point {o}.", False)


def _op_reflect_line(rng, pts, avail, name):
    if len(avail) < 3:
        return None
    p, a, b = rng.sample(avail, 3)
    if _dist(pts[a], pts[b]) < 2.0:
        return None
    return (_reflect_over_line(pts[p], pts[a], pts[b]),
            f"\\tkzDefPointBy[reflection=over {a}--{b}]({p})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the reflection of {p} over line {a}{b}.", False)


def _op_rotation(rng, pts, avail, name):
    p, o = rng.sample(avail, 2)
    if _dist(pts[p], pts[o]) < 2.0:
        return None
    deg = rng.choice(ANGLES)
    return (_rotate(pts[p], pts[o], deg),
            f"\\tkzDefPointBy[rotation=center {o} angle {deg}]({p})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the image of {p} rotated {deg} degrees about {o}.", False)


def _op_translation(rng, pts, avail, name):
    if len(avail) < 3:
        return None
    p, a, b = rng.sample(avail, 3)
    va, vb, pp = pts[a], pts[b], pts[p]
    if _dist(va, vb) < 2.0:
        return None
    return ((pp[0] + vb[0] - va[0], pp[1] + vb[1] - va[1]),
            f"\\tkzDefPointBy[translation=from {a} to {b}]({p})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the image of {p} translated by the vector from {a} to {b}.", False)


def _op_foot(rng, pts, avail, name):
    if len(avail) < 3:
        return None
    p, a, b = rng.sample(avail, 3)
    if _dist(pts[a], pts[b]) < 2.0:
        return None
    f = foot(pts[p], pts[a], pts[b])
    if _dist(f, pts[p]) < 1.0:   # apex on the line -> foot == apex (degenerate)
        return None
    return (f, f"\\tkzDefPointBy[projection=onto {a}--{b}]({p})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the foot of the perpendicular from {p} to line {a}{b}.", True)


def _op_intersection(rng, pts, avail, name):
    if len(avail) < 4:
        return None
    a, b, c, d = rng.sample(avail, 4)
    x = line_intersection(pts[a], pts[b], pts[c], pts[d])
    if x is None:
        return None
    return (x, f"\\tkzInterLL({a},{b})({c},{d})\\tkzGetPoint{{{name}}}",
            f"Let {name} be the intersection of lines {a}{b} and {c}{d}.", True)


AFFINE_OPS = [_op_midpoint, _op_symmetry, _op_reflect_line, _op_rotation, _op_translation]
METRIC_OPS = [_op_foot, _op_intersection]

# Named registry so callers can restrict the op set. The specialist is robust on
# midpoint / reflect_line / foot / intersection (chain-1 ~100%) but brittle to
# paraphrase on the general transforms symmetry / rotation / translation (it emits
# malformed macros). The "robust" preset composes only ops it handles singly, so
# chain degradation isolates COMPOSITIONAL DEPTH rather than paraphrase brittleness.
_OP_BY_NAME = {
    "midpoint": _op_midpoint, "symmetry": _op_symmetry, "reflect_line": _op_reflect_line,
    "rotation": _op_rotation, "translation": _op_translation,
    "foot": _op_foot, "intersection": _op_intersection,
}
ROBUST_AFFINE = ["midpoint", "reflect_line"]
ROBUST_METRIC = ["foot", "intersection"]


def _next_name(pts: dict) -> str:
    for i in range(26):
        n = chr(ord("A") + i)
        if n not in pts:
            return n
    raise RuntimeError("ran out of point names")


def _try_step(rng, pool, pts, avail, name):
    for _ in range(60):
        op = rng.choice(pool)
        res = op(rng, pts, avail, name)
        if res is None:
            continue
        if _sane(res[0], pts, bound=BOUND, sep=SEP):
            return res
    return None


# --------------------------------------------------------------------------- #
# compositional chain example (native explicit-coordinate style)
# --------------------------------------------------------------------------- #
def make_chain_example(rng: random.Random, chain: int, regime: str,
                       affine_ops: list = AFFINE_OPS, metric_ops: list = METRIC_OPS) -> dict | None:
    """One native-style scene composing `chain` derived-point steps.

    regime "affine": every step is an isometry/affine op.
    regime "mixed" : the final step is a metric op (foot / intersection); earlier
                     steps are affine setup (so the hard op has points to act on).
    """
    for _ in range(200):
        a, b, c = _rand_triangle(rng, bound=7)
        pts: dict[str, tuple[float, float]] = {"A": a, "B": b, "C": c}
        avail = ["A", "B", "C"]
        body = [_pt("A", a), _pt("B", b), _pt("C", c)]
        clauses: list[str] = []
        ok = True
        for step in range(chain):
            name = _next_name(pts)
            last = step == chain - 1
            pool = (metric_ops if last else affine_ops) if regime == "mixed" else affine_ops
            res = _try_step(rng, pool, pts, avail, name)
            if res is None:
                ok = False
                break
            pt, macro, clause, _is_metric = res
            pts[name] = pt
            avail.append(name)
            body.append(macro)
            clauses.append(clause)
        if not ok or len(pts) != 3 + chain:
            continue
        names = list(pts.keys())
        body += [f"\\tkzDrawPolygon(A,B,C)",
                 f"\\tkzDrawPoints({','.join(names)})",
                 f"\\tkzLabelPoints({','.join(names)})"]
        coords = ", ".join(f"{n}=({_g(pts[n][0])},{_g(pts[n][1])})" for n in "ABC")
        desc = (f"Triangle ABC has vertices {coords}. " + " ".join(clauses) + " "
                + _draw_suffix(rng, "triangle ABC and the constructed points", names))
        return {"tag": f"chain_{regime}", "kind": "chain", "regime": regime,
                "chain": chain, "irregular": regime == "mixed",
                "force_op": "metric" if regime == "mixed" else None,
                "easy_only": regime == "affine",
                "cell": f"c{chain}_{'mix' if regime == 'mixed' else 'aff'}",
                "n_derived": chain, "points": _round_pts(pts), "derived": names[3:],
                "description": desc, "tikz": _fig(body),
                "unordered": None, "grade_only": None}
    return None


# --------------------------------------------------------------------------- #
# round-trip validation (correctness guarantee)
# --------------------------------------------------------------------------- #
def _grade_gt(prob: dict) -> dict:
    gt = prob["points"]
    if prob.get("grade_only"):
        gt = {k: v for k, v in gt.items() if k in prob["grade_only"]}
    return gt


def roundtrip_ok(prob: dict) -> bool:
    g = extract.grade(prob["tikz"], _grade_gt(prob), atol=0.05,
                      unordered=prob.get("unordered"))
    return bool(g["figure_only"] and g["compiles"] and g["coords_all_correct"])


def build_chain_cells(chains: list[int], regimes: list[str], k: int, seed: int,
                      workers: int, affine_ops: list = AFFINE_OPS,
                      metric_ops: list = METRIC_OPS) -> list[dict]:
    rng = random.Random(seed)
    kept: list[dict] = []
    for regime in regimes:
        for chain in chains:
            cell = f"c{chain}_{'mix' if regime == 'mixed' else 'aff'}"
            seen: set[str] = set()
            got: list[dict] = []
            rounds = 0
            while len(got) < k and rounds < 12:
                rounds += 1
                need = (k - len(got))
                cand: list[dict] = []
                tries = 0
                target = need + max(8, need // 2)  # ~1.5x buffer (round-trip yield ~100%)
                while len(cand) < target and tries < need * 60 + 400:
                    tries += 1
                    ex = make_chain_example(rng, chain, regime, affine_ops, metric_ops)
                    if ex is None or ex["description"] in seen:
                        continue
                    seen.add(ex["description"])
                    cand.append(ex)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    flags = list(pool.map(roundtrip_ok, cand))
                got += [c for c, ok in zip(cand, flags) if ok]
            got = got[:k]
            kept += got
            print(f"  {cell}: kept {len(got)}/{k} (round-trip validated)", flush=True)
    return kept


def build_family_cells(families: list[str], k: int, seed: int, workers: int) -> list[dict]:
    tri = [t for t in families if t in olympiad.TYPES]
    ext = [t for t in families if t in olympiad_ext.TYPES]
    probs: list[dict] = []
    if tri:
        probs += olympiad.generate_problems(k * 2, seed=seed, types=tri)
    if ext:
        probs += olympiad_ext.generate_problems(k * 2, seed=seed + 1, types=ext)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(pool.map(roundtrip_ok, probs))
    by_tag: dict[str, list[dict]] = {}
    for p, ok in zip(probs, flags):
        if ok:
            by_tag.setdefault(p["tag"], []).append(p)
    kept: list[dict] = []
    for tag in families:
        rows = by_tag.get(tag, [])[:k]
        for p in rows:
            n_der = len(p.get("derived") or [])
            kept.append({**p, "kind": "family", "regime": None, "chain": n_der,
                         "irregular": False, "force_op": None, "easy_only": False,
                         "cell": f"fam_{tag}", "n_derived": n_der})
        print(f"  fam_{tag}: kept {len(rows)}/{k}", flush=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/specialist_ceiling/grid.jsonl")
    ap.add_argument("--chains", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--regimes", nargs="+", default=["affine", "mixed"])
    ap.add_argument("--k-chain", type=int, default=40)
    ap.add_argument("--k-family", type=int, default=20)
    ap.add_argument("--families", nargs="+",
                    default=list(olympiad.TYPES) + list(olympiad_ext.TYPES))
    ap.add_argument("--no-chain", action="store_true")
    ap.add_argument("--no-family", action="store_true")
    ap.add_argument("--affine-ops", nargs="+", default=None,
                    help="restrict affine step ops (names from _OP_BY_NAME)")
    ap.add_argument("--metric-ops", nargs="+", default=None,
                    help="restrict metric step ops (names from _OP_BY_NAME)")
    ap.add_argument("--robust", action="store_true",
                    help="compose only ops the specialist handles singly (midpoint/"
                         "reflect_line + foot/intersection), isolating chain depth "
                         "from paraphrase brittleness")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.robust:
        affine_names, metric_names = ROBUST_AFFINE, ROBUST_METRIC
    else:
        affine_names = args.affine_ops or ["midpoint", "symmetry", "reflect_line",
                                           "rotation", "translation"]
        metric_names = args.metric_ops or ["foot", "intersection"]
    affine_ops = [_OP_BY_NAME[n] for n in affine_names]
    metric_ops = [_OP_BY_NAME[n] for n in metric_names]

    grid: list[dict] = []
    if not args.no_chain:
        print(f"building chain cells: chains={args.chains} regimes={args.regimes} "
              f"k={args.k_chain} affine={affine_names} metric={metric_names} ...")
        grid += build_chain_cells(args.chains, args.regimes, args.k_chain,
                                  args.seed, args.workers, affine_ops, metric_ops)
    if not args.no_family:
        print(f"building family cells: {len(args.families)} families k={args.k_family} ...")
        grid += build_family_cells(args.families, args.k_family, args.seed + 500,
                                   args.workers)

    for i, ex in enumerate(grid):
        ex["id"] = i
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(ex) for ex in grid) + "\n")
    from collections import Counter
    print(f"\nwrote {len(grid)} examples -> {out}")
    print("cells:", dict(sorted(Counter(e["cell"] for e in grid).items())))


if __name__ == "__main__":
    main()
