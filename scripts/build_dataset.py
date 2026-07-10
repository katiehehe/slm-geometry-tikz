"""Build the training set at the difficulty the frontier sweep says is worth it.

The difficulty sweep (scripts/difficulty_sweep.py) showed that prompting SOTA
models is only unreliable at a specific place: **irregular numbers + a hard
construction** (foot-of-altitude, then line-intersection), from chain ~4. Round
numbers and long chains of easy ops are already solved by prompting, so training
there is wasted. This script therefore samples a training mixture centered on
that failure region, with a minority of easier/round examples for robustness.

Held-out eval: by default we reuse the sweep grid (outputs/sweep/grid.jsonl) as
the eval set, so the fine-tuned model is measured on the *exact same items* the 9
SOTA baselines were — a clean base-vs-tuned-vs-SOTA comparison. The training set
is built disjoint from it (no shared scene descriptions).

Usage:
  uv run python scripts/build_dataset.py                       # defaults
  uv run python scripts/build_dataset.py --scale 1.5           # ~1.5x more train
  uv run python scripts/build_dataset.py --eval-from data/my_eval.jsonl
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

# Training mixture: (chain, irregular, force_op, easy_only, count). Weighted
# toward the sweep's failure region (irregular + foot-of-altitude / intersection),
# with round + easy + short examples kept as a robustness tail.
TRAIN_MIX: list[tuple[int, bool, str | None, bool, int]] = [
    (4, True,  "foot_altitude", False, 300),   # core failure region
    (5, True,  "foot_altitude", False, 300),   # core failure region
    (4, False, "foot_altitude", False, 120),   # same op, round -> generalize numbers
    (3, True,  "intersection",  False, 150),   # secondary hard op
    (4, True,  "intersection",  False, 150),
    (4, True,  None,            False, 200),   # mixed irregular (natural distribution)
    (5, True,  None,            False, 200),
    (3, True,  None,            False, 120),   # easier irregular anchor
    (4, False, None,            False, 120),   # mixed round anchor
    (2, True,  None,            False,  60),   # robustness: short + irregular
    (3, False, None,            True,   60),   # robustness: easy-only, round
]


def load_blocklist(paths: list[str]) -> set[str]:
    block: set[str] = set()
    for p in paths:
        fp = Path(p) if Path(p).is_absolute() else ROOT / p
        if fp.exists():
            for l in fp.read_text().splitlines():
                if l.strip():
                    block.add(json.loads(l)["description"])
    return block


def sample_spec(rng: random.Random, chain: int, irregular: bool, force_op: str | None,
                easy_only: bool, n: int, blocklist: set[str], seen: set[str]) -> list[dict]:
    """Sample n unique examples matching a spec, disjoint from blocklist+seen."""
    out: list[dict] = []
    cap = n * 400 + 5000
    tries = 0
    while len(out) < n and tries < cap:
        tries += 1
        ex = make_example(rng, chain, irregular, force_op=force_op, easy_only=easy_only)
        if ex["chain"] != chain:
            continue
        if force_op and force_op not in ex["tags"]:
            continue
        d = ex["description"]
        if d in blocklist or d in seen:
            continue
        seen.add(d)
        out.append(ex)
    if len(out) < n:
        print(f"  WARN chain={chain} irr={irregular} op={force_op} easy={easy_only}: "
              f"only {len(out)}/{n} unique found")
    return out


def write_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(rows):
            f.write(json.dumps({**ex, "id": i}) + "\n")
    chat = out.with_name(out.stem + "_chat.jsonl")
    with chat.open("w") as f:
        for ex in rows:
            f.write(json.dumps(to_chat_example(ex)) + "\n")
    print(f"wrote {len(rows):>5} -> {out}  (+ {chat.name})")


def materialize_eval(eval_from: str, out: Path) -> list[dict]:
    src = Path(eval_from) if Path(eval_from).is_absolute() else ROOT / eval_from
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    write_jsonl(rows, out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all train counts")
    ap.add_argument("--eval-from", type=str, default="outputs/sweep/grid.jsonl",
                    help="reuse this jsonl as the held-out eval (SOTA-comparable); "
                         "set to '' to skip and only build train")
    ap.add_argument("--train-out", type=str, default="data/train.jsonl")
    ap.add_argument("--eval-out", type=str, default="data/eval.jsonl")
    args = ap.parse_args()

    # Eval first, so train can be built disjoint from it.
    eval_rows: list[dict] = []
    blocklist: set[str] = set()
    if args.eval_from:
        eval_rows = materialize_eval(args.eval_from, ROOT / args.eval_out)
        blocklist = {e["description"] for e in eval_rows}
        print(f"eval: reused {len(eval_rows)} scenes from {args.eval_from} "
              f"(train will be disjoint from these)")

    rng = random.Random(args.seed)
    seen: set[str] = set()
    train_rows: list[dict] = []
    print("\nbuilding train mixture (weighted to the failure region):")
    for chain, irregular, force_op, easy_only, count in TRAIN_MIX:
        n = round(count * args.scale)
        rows = sample_spec(rng, chain, irregular, force_op, easy_only, n, blocklist, seen)
        tag = force_op or ("easy" if easy_only else "mixed")
        print(f"  c{chain} {'irr' if irregular else 'rnd':3s} {tag:13s} -> {len(rows):>4}")
        train_rows += rows

    rng.shuffle(train_rows)
    if blocklist:
        overlap = {e["description"] for e in train_rows} & blocklist
        assert not overlap, f"train/eval overlap: {len(overlap)}"
    write_jsonl(train_rows, ROOT / args.train_out)
    print(f"\ndisjoint OK: {len(train_rows)} train / {len(eval_rows)} eval, 0 shared scenes")


if __name__ == "__main__":
    main()
