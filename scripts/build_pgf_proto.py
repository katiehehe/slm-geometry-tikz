"""v2 PGF-prototype dataset: coordinate-free CONSTRUCTION targets.

Same failure-region focus as build_dataset.py, but the ground-truth TikZ uses PGF
constructions (calc projection `($(a)!(c)!(b)$)`, `name intersections`, ...), so
the model emits the *construction* and PGF computes the coordinates. This tests
whether offloading the arithmetic cracks the foot-of-altitude wall.

Writes to data/*_pgf.jsonl — the v1 numeric datasets are left untouched.

Usage:
  uv run python scripts/build_pgf_proto.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz.generator import make_example  # noqa: E402
from geotikz.prompts import to_chat_example  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (chain, irregular, force_op, easy_only, count)
TRAIN_MIX = [
    (4, True, "foot_altitude", False, 500),
    (5, True, "foot_altitude", False, 400),
    (4, False, "foot_altitude", False, 150),
    (3, True, "intersection", False, 250),
    (4, True, "intersection", False, 250),
    (4, True, None, False, 250),
    (3, True, None, False, 150),
    (2, True, None, False, 100),
]
EVAL_MIX = [
    (4, True, "foot_altitude", False, 60),
    (5, True, "foot_altitude", False, 60),
    (3, True, "intersection", False, 40),
    (4, True, "intersection", False, 40),
    (4, True, None, False, 50),
    (3, True, None, False, 30),
]


def sample(rng: random.Random, mix, seen: set[str], blocklist: set[str]) -> list[dict]:
    rows: list[dict] = []
    for chain, irr, op, easy, count in mix:
        got, tries, cap = 0, 0, count * 400 + 5000
        while got < count and tries < cap:
            tries += 1
            ex = make_example(rng, chain, irr, force_op=op, easy_only=easy, symbolic=True)
            if ex["chain"] != chain:
                continue
            if op and op not in ex["tags"]:
                continue
            d = ex["description"]
            if d in seen or d in blocklist:
                continue
            seen.add(d)
            rows.append(ex)
            got += 1
        if got < count:
            print(f"  WARN c{chain} {op or 'mixed'}: only {got}/{count}")
    return rows


def write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(rows):
            f.write(json.dumps({**ex, "id": i}) + "\n")
    chat = out.with_name(out.stem + "_chat.jsonl")
    with chat.open("w") as f:
        for ex in rows:
            f.write(json.dumps(to_chat_example(ex)) + "\n")
    print(f"wrote {len(rows):>5} -> {out}  (+ {chat.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen: set[str] = set()

    ev = sample(rng, EVAL_MIX, seen, set())  # eval first
    blk = {e["description"] for e in ev}
    tr = sample(rng, TRAIN_MIX, seen, blk)  # train disjoint from eval

    overlap = {e["description"] for e in tr} & blk
    assert not overlap, f"train/eval overlap: {len(overlap)}"
    write(tr, ROOT / "data/train_pgf.jsonl")
    write(ev, ROOT / "data/eval_pgf.jsonl")
    print(f"disjoint OK: {len(tr)} train / {len(ev)} eval, 0 shared scenes")


if __name__ == "__main__":
    main()
