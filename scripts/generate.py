"""Generate a spec-first geometry->TikZ dataset as JSONL.

Usage:
  uv run python scripts/generate.py --n 50 --seed 0 --out data/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz.generator import generate_dataset  # noqa: E402
from geotikz.prompts import to_chat_example  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chain", type=int, default=3)
    ap.add_argument("--out", type=str, default="data/smoke.jsonl")
    ap.add_argument("--chat-out", type=str, default=None, help="optional chat-format JSONL for SFT")
    args = ap.parse_args()

    data = generate_dataset(args.n, seed=args.seed, max_chain=args.max_chain)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

    chat_out = Path(args.chat_out) if args.chat_out else out.with_name(out.stem + "_chat.jsonl")
    with chat_out.open("w") as f:
        for ex in data:
            f.write(json.dumps(to_chat_example(ex)) + "\n")

    print(f"wrote {len(data)} examples -> {out}")
    print(f"wrote chat-format SFT records -> {chat_out}")


if __name__ == "__main__":
    main()
