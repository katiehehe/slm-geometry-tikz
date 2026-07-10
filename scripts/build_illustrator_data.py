"""Assemble the illustrator training set: distilled real problems + GT-verified
synthetic constructions, all as coordinate-free CONSTRUCTION-prompt chat records.

Two ingredients, one deliverable:

  1. DISTILLED (primary, closes the NL/coverage gap): (real problem text -> teacher
     figure) pairs written by scripts/distill.py, already hard-filtered to
     compile + non-degenerate + judge-plausible. Read from
     data/distill_illustrator.jsonl.

  2. SYNTHETIC (secondary, GT-grounded correctness + construction breadth): sampled
     across the full construction vocabulary (olympiad.py's triangle centers +
     olympiad_ext.py's intersection / parallel / midpoint / reflection / rotation /
     square / polygon / two-circle families). Every synthetic figure is
     round-trip VALIDATED through the compile-extract grader (emit -> compile ->
     read back == GT) and only kept if it passes, so the labels are provably
     correct. A disjoint slice is held out with GT coordinates as a synthetic
     pass-rate eval.

Outputs (all NEW files; nothing existing is overwritten):
  data/illustrator_train_chat.jsonl   distilled + synthetic-train, chat format
  data/illustrator_syn_eval.jsonl     held-out synthetic eval (GT coords)
  outputs/distill/dataset_report.md   composition + round-trip yields

Usage:
  uv run python scripts/build_illustrator_data.py                 # defaults
  uv run python scripts/build_illustrator_data.py --syn-per-type 150 --eval-per-type 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract, olympiad, olympiad_ext, serve  # noqa: E402
from geotikz.prompts import build_construction_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "distill" / "cache" / "roundtrip.jsonl"


def _grade_gt(prob: dict) -> dict:
    """GT coords to grade against (a subset for many-point polygons)."""
    gt = prob["points"]
    if prob.get("grade_only"):
        gt = {k: v for k, v in gt.items() if k in prob["grade_only"]}
    return gt


def roundtrip_ok(prob: dict, cache: serve.Cache) -> bool:
    """True iff the GT figure compiles and reads back == GT (cached by figure)."""
    ck = "rt:" + serve.dhash(prob["tikz"])
    hit = cache.get(ck)
    if hit is not None:
        return bool(hit["ok"])
    g = extract.grade(prob["tikz"], _grade_gt(prob), atol=0.05,
                      unordered=prob.get("unordered"))
    ok = bool(g["figure_only"] and g["compiles"] and g["coords_all_correct"])
    cache.put(ck, {"ok": ok, "reason": g["compile_reason"]})
    return ok


def gen_synthetic(per_type: int, seed: int, cache: serve.Cache,
                  workers: int) -> tuple[list[dict], Counter, Counter]:
    """Sample + round-trip-validate synthetic problems across all families."""
    probs: list[dict] = []
    # olympiad.py triangle centers
    probs += olympiad.generate_problems(per_type, seed=seed, types=olympiad.TYPES)
    # olympiad_ext.py expanded families
    probs += olympiad_ext.generate_problems(per_type, seed=seed + 1, types=olympiad_ext.TYPES)

    ok = Counter()
    tot = Counter()
    kept: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        flags = list(ex.map(lambda p: roundtrip_ok(p, cache), probs))
    for p, good in zip(probs, flags):
        tot[p["tag"]] += 1
        if good:
            ok[p["tag"]] += 1
            kept.append(p)
    return kept, ok, tot


def to_chat(description: str, tikz: str) -> dict:
    return {"messages": build_construction_messages(description)
            + [{"role": "assistant", "content": tikz}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-per-type", type=int, default=140,
                    help="synthetic TRAIN problems sampled per construction type")
    ap.add_argument("--eval-per-type", type=int, default=12,
                    help="held-out synthetic eval problems per type")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--distill", default="data/distill_illustrator.jsonl")
    ap.add_argument("--distill-repeat", type=int, default=1,
                    help="upsample distilled real-problem pairs N times (the "
                         "distillation is the primary lever for the NL gap)")
    ap.add_argument("--train-out", default="data/illustrator_train_chat.jsonl")
    ap.add_argument("--eval-out", default="data/illustrator_syn_eval.jsonl")
    args = ap.parse_args()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = serve.Cache(CACHE)

    # --- held-out synthetic eval first (so train is disjoint from it) ---
    print("generating + validating held-out synthetic eval ...")
    eval_probs, eok, etot = gen_synthetic(args.eval_per_type, args.seed, cache, args.workers)
    eval_descs = {p["description"] for p in eval_probs}

    print("generating + validating synthetic TRAIN ...")
    train_syn, sok, stot = gen_synthetic(args.syn_per_type, args.seed + 1000, cache, args.workers)
    train_syn = [p for p in train_syn if p["description"] not in eval_descs]

    # --- distilled real-problem pairs ---
    distill_path = ROOT / args.distill
    distilled: list[dict] = []
    if distill_path.exists():
        distilled = [json.loads(l) for l in distill_path.read_text().splitlines() if l.strip()]
    print(f"distilled pairs available: {len(distilled)}")

    # --- write combined training chat set (distilled + synthetic) ---
    train_records: list[dict] = []
    for r in distilled:
        if r.get("tikz"):
            for _ in range(max(1, args.distill_repeat)):
                train_records.append(to_chat(r["description"], r["tikz"]))
    for p in train_syn:
        train_records.append(to_chat(p["description"], p["tikz"]))

    import random
    random.Random(args.seed).shuffle(train_records)
    train_out = ROOT / args.train_out
    with train_out.open("w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    # --- write held-out synthetic eval (GT coords, grader format) ---
    eval_out = ROOT / args.eval_out
    with eval_out.open("w") as f:
        for i, p in enumerate(eval_probs):
            f.write(json.dumps({
                "id": i, "tag": p["tag"], "description": p["description"],
                "tikz": p["tikz"], "points": p["points"],
                "derived": p.get("derived"), "unordered": p.get("unordered"),
                "grade_only": p.get("grade_only"),
            }) + "\n")

    # --- report ---
    n_distill = sum(1 for r in distilled if r.get("tikz"))
    md = [
        "# Illustrator dataset composition\n",
        f"- Combined training records: **{len(train_records)}** "
        f"(distilled real = {n_distill} unique x{args.distill_repeat} repeat, "
        f"synthetic = {len(train_syn)})",
        f"- Held-out synthetic eval: **{len(eval_probs)}** (GT-graded)\n",
        "## Synthetic TRAIN round-trip yield (kept / sampled)\n",
        "| construction | kept | sampled |", "|---|---|---|",
    ]
    for t in list(olympiad.TYPES) + list(olympiad_ext.TYPES):
        md.append(f"| {t} | {sok[t]} | {stot[t]} |")
    md += ["\n## Held-out synthetic eval round-trip yield\n",
           "| construction | kept | sampled |", "|---|---|---|"]
    for t in list(olympiad.TYPES) + list(olympiad_ext.TYPES):
        md.append(f"| {t} | {eok[t]} | {etot[t]} |")
    (ROOT / "outputs" / "distill" / "dataset_report.md").write_text("\n".join(md) + "\n")

    print("\n".join(md[:6]))
    print(f"\nwrote {len(train_records)} train -> {train_out}")
    print(f"wrote {len(eval_probs)} syn-eval -> {eval_out}")
    print(f"report -> outputs/distill/dataset_report.md")


if __name__ == "__main__":
    main()
