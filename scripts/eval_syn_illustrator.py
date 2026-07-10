"""Held-out SYNTHETIC pass-rate eval for the illustrator (GT-verified correctness).

Coverage on AIME (compiles + non-degenerate) is a necessary but not sufficient
proxy: real problems have no ground-truth coordinates, so "correct" there is only
plausibility-filtered. This eval closes that gap on the synthetic slice, where we
DO have exact ground truth: it runs the illustrator over the disjoint held-out
synthetic set (data/illustrator_syn_eval.jsonl) and grades each output with the
compile-extract grader (figure-only AND compiles AND every named point within
atol of GT). Reports pass rate overall + per construction, base vs tuned — the
clean "did the data instill the behavior" comparison on provably-correct labels.

Usage:
  uv run python scripts/eval_syn_illustrator.py                 # tuned only
  uv run python scripts/eval_syn_illustrator.py --also-base     # base vs tuned
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/infer_illustrator_modal.py"


def run_modal(descs: list[str], adapter: str, max_new_tokens: int, batch_size: int,
              script: str = SCRIPT) -> list[str]:
    tmp = Path(tempfile.mkdtemp(prefix="syn_eval_"))
    inp, outp = tmp / "in.jsonl", tmp / "out.jsonl"
    inp.write_text("\n".join(json.dumps({"id": i, "description": d})
                             for i, d in enumerate(descs)) + "\n")
    cmd = ["modal", "run", script, "--input", str(inp), "--output", str(outp),
           "--max-new-tokens", str(max_new_tokens), "--batch-size", str(batch_size),
           "--adapter", adapter]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    by_i = {r["id"]: r["output"] for r in rows}
    return [by_i[i] for i in range(len(descs))]


def grade_all(rows: list[dict], outputs: list[str], workers: int) -> dict:
    def one(item):
        row, out = item
        gt = row["points"]
        if row.get("grade_only"):
            gt = {k: v for k, v in gt.items() if k in row["grade_only"]}
        g = extract.grade(out or "", gt, atol=0.05, unordered=row.get("unordered"))
        return row["tag"], bool(g["passed"]), bool(g["compiles"])

    passed = Counter()
    compiled = Counter()
    total = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tag, p, c in ex.map(one, zip(rows, outputs)):
            total[tag] += 1
            passed[tag] += int(p)
            compiled[tag] += int(c)
    return {"passed": passed, "compiled": compiled, "total": total}


def summarize(tag_label: str, res: dict) -> list[str]:
    total, passed, compiled = res["total"], res["passed"], res["compiled"]
    n = sum(total.values())
    np_ = sum(passed.values())
    nc = sum(compiled.values())
    lines = [f"### {tag_label}\n",
             f"- Overall pass: **{np_}/{n} = {np_ / n * 100:.1f}%**  |  "
             f"compile: {nc}/{n} = {nc / n * 100:.1f}%\n",
             "| construction | pass | compile | n |", "|---|---|---|---|"]
    for t in sorted(total):
        lines.append(f"| {t} | {passed[t]}/{total[t]} | {compiled[t]}/{total[t]} | {total[t]} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/illustrator_syn_eval.jsonl")
    ap.add_argument("--also-base", action="store_true",
                    help="also run the untuned base model for a base-vs-tuned delta")
    ap.add_argument("--script", default=SCRIPT,
                    help="Modal inference script (use scripts/infer_illustrator_4b_modal.py "
                         "for the 4B capacity probe)")
    ap.add_argument("--tuned-adapter", default="qwen3-illustrator",
                    help="tuned adapter RUN_NAME on the Volume (e.g. qwen3-illustrator-4b)")
    ap.add_argument("--tuned-label", default=None,
                    help="report label for the tuned variant (defaults to the adapter name)")
    ap.add_argument("--base-label", default="base Qwen3-1.7B (no adapter)",
                    help="report label for the base variant")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/syn_eval_illustrator")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.eval).read_text().splitlines() if l.strip()]
    descs = [r["description"] for r in rows]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = serve.Cache(out_dir / "cache.jsonl")
    print(f"held-out synthetic eval: {len(rows)} problems")

    report = ["# Held-out synthetic pass rate (GT-verified)\n",
              f"- Eval set: {len(rows)} problems, disjoint from training.\n"]

    tuned_label = args.tuned_label or f"tuned ({args.tuned_adapter})"
    variants = [(args.tuned_adapter, tuned_label)]
    if args.also_base:
        variants.append(("none", args.base_label))

    for adapter, label in variants:
        keyed = {i: f"syn:{adapter}:" + serve.dhash(d) for i, d in enumerate(descs)}
        outs: list[str | None] = [None] * len(descs)
        todo = [i for i in range(len(descs)) if cache.get(keyed[i]) is None]
        for i in range(len(descs)):
            hit = cache.get(keyed[i])
            if hit:
                outs[i] = hit["output"]
        if todo:
            gen = run_modal([descs[i] for i in todo], adapter, args.max_new_tokens,
                            args.batch_size, script=args.script)
            for i, o in zip(todo, gen):
                cache.put(keyed[i], {"adapter": adapter, "output": o})
                outs[i] = o
        res = grade_all(rows, outs, args.workers)
        n = sum(res["total"].values())
        np_ = sum(res["passed"].values())
        print(f"[{label}] pass {np_}/{n} = {np_ / n * 100:.1f}%")
        report += summarize(label, res) + ["\n"]

    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"report -> {out_dir / 'report.md'}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
