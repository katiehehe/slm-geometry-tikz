"""Run a model over a dataset and score outputs against the Behavior Spec.

Usage:
  uv run python scripts/evaluate.py --data data/smoke.jsonl --n 8 \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --tag base --out outputs/eval_base.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import harness, infer  # noqa: E402


def load_jsonl(path: str, n: int | None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return rows[:n] if n else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/smoke.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--tag", type=str, default="base")
    ap.add_argument("--out", type=str, default="outputs/eval.json")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    rows = load_jsonl(args.data, args.n)
    model, tok, device = infer.load_model(args.model, args.adapter)

    results = []
    for ex in rows:
        out = infer.generate(model, tok, device, ex["description"])
        r = harness.evaluate_one(
            ex_id=ex["id"],
            description=ex["description"],
            model_output=out,
            gt_tikz=ex["tikz"],
            gt_points=ex["points"],
            use_judge=args.judge,
            render=not args.no_render,
        )
        results.append(r)
        print(f"  [{ex['id']:>3}] compiles={r.compiles} coords={r.coord_accuracy:.2f} passed={r.passed}")

    summary = harness.aggregate(results)
    payload = {"tag": args.tag, "model": args.model, "adapter": args.adapter,
               "summary": summary, "results": harness.to_dicts(results)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"\n=== {args.tag} summary ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
