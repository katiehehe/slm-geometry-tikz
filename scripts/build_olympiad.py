"""v3 olympiad eval set: coordinate-free CONSTRUCTION targets, GT-validated.

Samples problems over the olympiad construction vocabulary (circumcenter,
incenter, orthocenter, centroid, angle bisector, foot of altitude, median,
tangent-from-a-point), then VALIDATES each ground truth by round-tripping it
through the compile-extract grader (emit -> tectonic -> read back == GT). Only
problems whose ground truth round-trips are written, so the eval set is
guaranteed clean. The round-trip rate is reported per construction.

Writes data/olympiad_eval.jsonl (v1/v2 datasets are left untouched).

Usage:
  uv run python scripts/build_olympiad.py --n 15
  uv run python scripts/build_olympiad.py --n 15 --types circumcenter incenter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract, olympiad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dhash(desc: str) -> str:
    return hashlib.md5(desc.encode()).hexdigest()[:12]


def validate(prob: dict, atol: float) -> tuple[dict, dict]:
    """Round-trip the GT figure through the grader."""
    g = extract.grade(prob["tikz"], prob["points"], atol=atol, unordered=prob["unordered"])
    return prob, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="problems per construction type")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--types", nargs="+", default=None, choices=olympiad.TYPES)
    ap.add_argument("--atol", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="data/olympiad_eval.jsonl")
    args = ap.parse_args()

    types = args.types or olympiad.TYPES
    # oversample so that, after dropping any non-round-tripping GT, we still hit --n
    raw = olympiad.generate_problems(int(args.n * 1.3) + 3, seed=args.seed, types=types)
    print(f"generated {len(raw)} candidate problems; validating round-trip ...")

    kept: dict[str, list[dict]] = defaultdict(list)
    seen_rt: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # tag -> [pass, total]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for prob, g in ex.map(lambda p: validate(p, args.atol), raw):
            tag = prob["tag"]
            rt_ok = g["figure_only"] and g["compiles"] and g["coords_all_correct"]
            seen_rt[tag][1] += 1
            seen_rt[tag][0] += int(rt_ok)
            if rt_ok and len(kept[tag]) < args.n:
                kept[tag].append(prob)

    print(f"\n{'construction':<16}{'roundtrip':>12}{'kept':>8}")
    print("-" * 36)
    tot_ok = tot = 0
    for tag in types:
        ok, n = seen_rt[tag]
        tot_ok += ok
        tot += n
        print(f"{tag:<16}{ok}/{n:<10}{len(kept[tag]):>6}")
    print("-" * 36)
    rate = tot_ok / (tot or 1)
    print(f"{'TOTAL':<16}{tot_ok}/{tot:<10}  round-trip rate = {rate:.1%}")

    # reassign contiguous ids and write
    problems: list[dict] = []
    pid = 0
    for tag in types:
        for prob in kept[tag]:
            prob = dict(prob)
            prob["id"] = pid
            prob["h"] = dhash(prob["description"])
            problems.append(prob)
            pid += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(p) for p in problems) + "\n")
    print(f"\nwrote {len(problems)} problems -> {out_path}")
    if rate < 0.95:
        print(f"WARNING: round-trip rate {rate:.1%} < 95% — inspect emission bugs")


if __name__ == "__main__":
    main()
