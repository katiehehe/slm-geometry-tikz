"""Day-2 smoke test: run the full loop generate -> train -> eval end to end.

This closes the loop on ~50 JUNK examples with a tiny model. It proves the
plumbing, not quality. Real data + Qwen3 QLoRA come on Day 3.

Usage:
  uv run python scripts/run_smoke.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable]


def run(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 70}\n>>> {title}\n{'=' * 70}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="dataset size (junk examples)")
    ap.add_argument("--eval-n", type=int, default=6, help="examples to eval per model")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = ap.parse_args()

    data = "data/smoke.jsonl"
    chat = "data/smoke_chat.jsonl"
    adapter = "outputs/smoke-adapter"

    run(PY + ["scripts/generate.py", "--n", str(args.n), "--out", data], "STEP 1/4 — GENERATE")
    run(PY + ["scripts/train.py", "--data", chat, "--out", adapter,
              "--model", args.model, "--max-steps", str(args.max_steps)], "STEP 2/4 — TRAIN (smoke)")
    run(PY + ["scripts/evaluate.py", "--data", data, "--n", str(args.eval_n),
              "--model", args.model, "--tag", "base",
              "--out", "outputs/eval_base.json"], "STEP 3/4 — EVAL BASE")
    run(PY + ["scripts/evaluate.py", "--data", data, "--n", str(args.eval_n),
              "--model", args.model, "--adapter", adapter, "--tag", "tuned",
              "--out", "outputs/eval_tuned.json"], "STEP 4/4 — EVAL TUNED")

    base = json.loads(Path(ROOT / "outputs/eval_base.json").read_text())["summary"]
    tuned = json.loads(Path(ROOT / "outputs/eval_tuned.json").read_text())["summary"]

    print(f"\n{'=' * 70}\nBASE vs TUNED (smoke — numbers are meaningless, loop is the point)\n{'=' * 70}")
    keys = ["figure_only_rate", "compile_rate", "coord_accuracy_mean",
            "coords_all_correct_rate", "pass_rate", "ssim_mean"]
    print(f"{'metric':<28}{'base':>12}{'tuned':>12}")
    for k in keys:
        b, t = base.get(k), tuned.get(k)
        bs = f"{b:.3f}" if isinstance(b, (int, float)) else str(b)
        ts = f"{t:.3f}" if isinstance(t, (int, float)) else str(t)
        print(f"{k:<28}{bs:>12}{ts:>12}")
    print("\nDAY-2 CHECKPOINT: full loop generate -> train -> eval ran end to end.")


if __name__ == "__main__":
    main()
