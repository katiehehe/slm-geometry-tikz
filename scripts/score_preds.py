"""Score pre-generated eval predictions locally (no model needed).

Model inference of base+LoRA thrashes on an 8GB Mac, so we generate on the GPU
(scripts/train_modal.py::eval_infer -> preds on the Modal Volume) and score here.
Scoring is just TikZ compile + coordinate assertion, which is lightweight.

Usage:
  uv run python scripts/score_preds.py --preds outputs/eval_preds_tuned.jsonl --tag tuned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import harness  # noqa: E402


def _rate(evals: list, attr: str) -> float:
    return sum(getattr(e, attr) for e in evals) / len(evals) if evals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]

    def score_one(ex: dict):
        return ex, harness.evaluate_one(
            ex_id=ex["id"], description=ex["description"], model_output=ex["output"],
            gt_tikz=ex["tikz"], gt_points=ex["points"], render=False,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    evals: list = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(score_one, ex): i for i, ex in enumerate(rows)}
        for fut in as_completed(futs):
            evals[futs[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  scored {done}/{len(rows)}")

    allr = [r for _, r in evals]
    summary = harness.aggregate(allr)
    print(f"\n=== {args.tag}: OVERALL (n={len(allr)}) ===")
    for k in ("figure_only_rate", "compile_rate", "coord_accuracy_mean",
              "coords_all_correct_rate", "pass_rate"):
        print(f"  {k:<26} {summary[k]:.3f}")

    # by chain
    print(f"\n=== {args.tag}: by chain ===")
    print(f"  {'chain':<8}{'n':>5}{'pass':>8}{'compile':>9}{'coordAcc':>10}")
    by_chain: dict = {}
    for ex, r in evals:
        by_chain.setdefault(ex.get("chain"), []).append(r)
    for ch in sorted(k for k in by_chain if k is not None):
        es = by_chain[ch]
        print(f"  {ch:<8}{len(es):>5}{_rate(es, 'passed'):>8.2f}"
              f"{_rate(es, 'compiles'):>9.2f}{_rate(es, 'coord_accuracy'):>10.2f}")

    # by hard op (foot_altitude / intersection present in tags)
    print(f"\n=== {args.tag}: by hard op ===")
    print(f"  {'op':<16}{'n':>5}{'pass':>8}{'compile':>9}{'coordAcc':>10}")
    for op in ("intersection", "foot_altitude"):
        es = [r for ex, r in evals if op in (ex.get("tags") or [])]
        if es:
            print(f"  {op:<16}{len(es):>5}{_rate(es, 'passed'):>8.2f}"
                  f"{_rate(es, 'compiles'):>9.2f}{_rate(es, 'coord_accuracy'):>10.2f}")

    if args.out:
        payload = {"tag": args.tag, "model": args.model, "adapter": args.adapter,
                   "n": len(allr), "summary": summary, "results": harness.to_dicts(allr)}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
