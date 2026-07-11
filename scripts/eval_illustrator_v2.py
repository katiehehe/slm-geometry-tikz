"""Head-to-head illustrator eval: base vs v1 vs v2 on TWO GT-verified gates.

Reuses the exact compile-extract grader (extract.grade: figure-only AND compiles
AND every named point within atol of GT) and the 4B Modal inference path
(scripts/infer_illustrator_4b_modal.py), so the ONLY moving part between columns
is the adapter. Two eval sets, both correct-by-construction:

  (i)  SYNTHETIC GATE v2   data/illustrator_syn_eval_v2.jsonl
       the v1 gate (240) + the harder held-out constructions -> coverage.
  (ii) PARAPHRASE GATE     data/illustrator_paraphrase_eval.jsonl
       loose rewordings of the v1 gate problems (unseen wordings of unseen
       problems) -> phrasing-robustness.

For each adapter, ALL problems across both gates are generated in ONE Modal call
(amortises model load), cached by (gate, adapter, scene-hash) so re-runs never
re-spend GPU. Prints an overall base/v1/v2 table for each gate plus a
per-construction breakdown, and writes a markdown report.

Usage:
  uv run python scripts/eval_illustrator_v2.py
  uv run python scripts/eval_illustrator_v2.py --variants base qwen3-illustrator-4b qwen3-illustrator-4b-v2
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
SCRIPT = "scripts/infer_illustrator_4b_modal.py"


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_modal(descs: list[str], adapter: str, max_new_tokens: int, batch_size: int,
              script: str) -> list[str]:
    tmp = Path(tempfile.mkdtemp(prefix="v2eval_"))
    inp, outp = tmp / "in.jsonl", tmp / "out.jsonl"
    inp.write_text("\n".join(json.dumps({"id": i, "description": d})
                             for i, d in enumerate(descs)) + "\n")
    cmd = ["modal", "run", script, "--input", str(inp), "--output", str(outp),
           "--max-new-tokens", str(max_new_tokens), "--batch-size", str(batch_size),
           "--adapter", adapter]
    print("  running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    rows = {json.loads(l)["id"]: json.loads(l)["output"]
            for l in outp.read_text().splitlines() if l.strip()}
    return [rows[i] for i in range(len(descs))]


def grade_all(rows: list[dict], outputs: list[str], workers: int) -> dict:
    def one(item):
        row, out = item
        gt = row["points"]
        if row.get("grade_only"):
            gt = {k: v for k, v in gt.items() if k in row["grade_only"]}
        g = extract.grade(out or "", gt, atol=0.05, unordered=row.get("unordered"))
        return row["tag"], bool(g["passed"]), bool(g["compiles"])

    passed, compiled, total = Counter(), Counter(), Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tag, p, c in ex.map(one, zip(rows, outputs)):
            total[tag] += 1
            passed[tag] += int(p)
            compiled[tag] += int(c)
    return {"passed": passed, "compiled": compiled, "total": total}


def generate_for_variant(adapter: str, gates: dict[str, list[dict]], cache: serve.Cache,
                         max_new_tokens: int, batch_size: int, script: str) -> dict[str, list[str]]:
    """Generate (cached) outputs for every gate for one adapter in ONE Modal call."""
    # collect all uncached (gate, idx) descriptions
    todo_keys, todo_descs = [], []
    outs: dict[str, list[str | None]] = {}
    for gate, rows in gates.items():
        outs[gate] = [None] * len(rows)
        for i, r in enumerate(rows):
            ck = f"{gate}:{adapter}:" + serve.dhash(r["description"])
            hit = cache.get(ck)
            if hit is not None:
                outs[gate][i] = hit["output"]
            else:
                todo_keys.append((gate, i, ck))
                todo_descs.append(r["description"])
    if todo_descs:
        print(f"[{adapter}] generating {len(todo_descs)} scenes ...", flush=True)
        gen = run_modal(todo_descs, adapter, max_new_tokens, batch_size, script)
        for (gate, i, ck), o in zip(todo_keys, gen):
            cache.put(ck, {"adapter": adapter, "output": o})
            outs[gate][i] = o
    else:
        print(f"[{adapter}] all cached", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["base", "qwen3-illustrator-4b", "qwen3-illustrator-4b-v2"],
                    help="adapters to compare ('base'/'none' = untuned Qwen3-4B)")
    ap.add_argument("--labels", nargs="+", default=["base 4B", "v1", "v2"])
    ap.add_argument("--gate", default="data/illustrator_syn_eval_v2.jsonl")
    ap.add_argument("--paraphrase", default="data/illustrator_paraphrase_eval.jsonl")
    ap.add_argument("--script", default=SCRIPT)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/syn_eval_illustrator_4b_v2")
    args = ap.parse_args()

    gates = {"gate": _load(ROOT / args.gate),
             "paraphrase": _load(ROOT / args.paraphrase)}
    print(f"gate: {len(gates['gate'])} problems | paraphrase: {len(gates['paraphrase'])} problems")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = serve.Cache(out_dir / "cache.jsonl")

    labels = dict(zip(args.variants, args.labels)) if len(args.labels) == len(args.variants) else {}
    # variant -> gate -> grade result
    scored: dict[str, dict[str, dict]] = {}
    for adapter in args.variants:
        outs = generate_for_variant(adapter, gates, cache, args.max_new_tokens,
                                     args.batch_size, args.script)
        scored[adapter] = {g: grade_all(gates[g], outs[g], args.workers) for g in gates}
        for g in gates:
            res = scored[adapter][g]
            n, npass = sum(res["total"].values()), sum(res["passed"].values())
            print(f"  [{labels.get(adapter, adapter)}] {g}: {npass}/{n} = "
                  f"{npass / max(n,1) * 100:.1f}%", flush=True)

    # ---- report ---------------------------------------------------------- #
    def lab(a):
        return labels.get(a, a)

    def overall(a, g):
        res = scored[a][g]
        n, npass = sum(res["total"].values()), sum(res["passed"].values())
        nc = sum(res["compiled"].values())
        return npass, nc, n

    md = ["# Illustrator v2 eval — base vs v1 vs v2 (GT-verified)\n",
          "Grader: figure-only AND compiles AND every named point within 0.05 of GT.\n",
          "## Headline pass rate\n",
          "| gate | " + " | ".join(lab(a) for a in args.variants) + " |",
          "|---|" + "---|" * len(args.variants)]
    for g, gname in [("gate", "Synthetic gate v2 (coord-verified)"),
                     ("paraphrase", "Paraphrase gate (unseen wordings)")]:
        cells = []
        for a in args.variants:
            npass, nc, n = overall(a, g)
            cells.append(f"{npass}/{n} = {npass / max(n,1) * 100:.1f}%")
        md.append(f"| {gname} | " + " | ".join(cells) + " |")

    # per-construction, per gate
    for g, gname in [("gate", "Synthetic gate v2"), ("paraphrase", "Paraphrase gate")]:
        tags = sorted(set().union(*[scored[a][g]["total"].keys() for a in args.variants]))
        md += [f"\n## {gname}: pass by construction\n",
               "| construction | " + " | ".join(lab(a) for a in args.variants) + " |",
               "|---|" + "---|" * len(args.variants)]
        for t in tags:
            cells = []
            for a in args.variants:
                res = scored[a][g]
                cells.append(f"{res['passed'][t]}/{res['total'][t]}")
            md.append(f"| {t} | " + " | ".join(cells) + " |")

    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nreport -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
