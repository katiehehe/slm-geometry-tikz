"""Olympiad litmus sweep: can frontier models draw olympiad geometry?

Runs a set of hosted frontier models over the v3 olympiad construction eval set
(circumcenter, incenter, orthocenter, centroid, angle bisector, foot of altitude,
median, tangent) via the gateway, grades each output with the COMPILE-EXTRACT
grader (compile the figure with tkz-euclide, read back the true coordinates of
the named points), and prints a per-construction pass-rate table.

pass = figure-only AND compiles AND every named coordinate correct within tol —
the same falsifiable gate the project trains toward. The question this answers:
which olympiad constructions do well-prompted frontier models already ace (so a
specialist isn't defensible), and which do they fail (the niche)?

Raw model outputs are cached per (model, problem) so re-scoring / adding a model
never re-spends API budget.

Usage:
  # litmus: 3 strong models, 12 problems per construction
  uv run python scripts/olympiad_sweep.py --preset frontier_small --n 12

  # explicit models
  uv run python scripts/olympiad_sweep.py \
      --models openai-group/gpt-5.5 claude-group/claude-opus-4-8 --n 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract, gateway, olympiad  # noqa: E402
from geotikz.prompts import build_construction_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PRESETS: dict[str, list[str]] = {
    "frontier": [
        "openai-group/gpt-5.5",
        "claude-group/claude-opus-4-8",
        "gemini-group/gemini-3.1-pro",
        "xai-group/grok-4.5",
        "claude-group/claude-sonnet-5",
    ],
    "frontier_small": [
        "openai-group/gpt-5.5",
        "claude-group/claude-opus-4-8",
        "gemini-group/gemini-3.1-pro",
    ],
}


def dhash(desc: str) -> str:
    return hashlib.md5(desc.encode()).hexdigest()[:12]


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


# --------------------------------------------------------------------------- #
# eval set: load the fixed dataset, or build + GT-validate it on first run
# --------------------------------------------------------------------------- #
def load_or_build_eval(path: Path, n_per_type: int, seed: int, atol: float,
                       workers: int) -> list[dict]:
    if path.exists():
        probs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"reusing eval set: {len(probs)} problems from {path}")
        return probs

    print(f"building eval set: {n_per_type}/type, validating GT round-trip ...")
    raw = olympiad.generate_problems(int(n_per_type * 1.3) + 3, seed=seed)
    kept: dict[str, list[dict]] = defaultdict(list)
    rt: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def _val(p: dict) -> tuple[dict, bool]:
        g = extract.grade(p["tikz"], p["points"], atol=atol, unordered=p["unordered"])
        return p, bool(g["figure_only"] and g["compiles"] and g["coords_all_correct"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p, ok in ex.map(_val, raw):
            rt[p["tag"]][1] += 1
            rt[p["tag"]][0] += int(ok)
            if ok and len(kept[p["tag"]]) < n_per_type:
                kept[p["tag"]].append(p)

    tot_ok = sum(v[0] for v in rt.values())
    tot = sum(v[1] for v in rt.values())
    print(f"GT round-trip rate = {tot_ok}/{tot} = {tot_ok / (tot or 1):.1%}")
    problems: list[dict] = []
    pid = 0
    for tag in olympiad.TYPES:
        for p in kept[tag]:
            p = dict(p)
            p["id"] = pid
            p["h"] = dhash(p["description"])
            problems.append(p)
            pid += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p) for p in problems) + "\n")
    print(f"wrote {len(problems)} problems -> {path}")
    return problems


# --------------------------------------------------------------------------- #
# raw model outputs (cached per model)
# --------------------------------------------------------------------------- #
def ensure_raw(model: str, probs: list[dict], cache_dir: Path, workers: int,
               max_tokens: int, temperature: float) -> dict[int, dict]:
    cache_path = cache_dir / f"{safe_name(model)}.jsonl"
    by_id = {p["id"]: p for p in probs}
    cache: dict[int, dict] = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line from an interrupted run
            p = by_id.get(r["id"])
            if p is None or r.get("h") != p["h"]:
                continue  # stale entry from a different eval set
            cache[r["id"]] = r
    todo = [p for p in probs if p["id"] not in cache]
    if todo:
        print(f"    calling {model} on {len(todo)} new prompts (workers={workers}) ...",
              flush=True)

        def worker(p: dict) -> dict:
            res = gateway.chat(build_construction_messages(p["description"]), model,
                               max_tokens=max_tokens, temperature=temperature)
            return {"id": p["id"], "h": p["h"], "raw": res.text, "ok": res.ok,
                    "error": res.error, "finish": res.finish_reason,
                    "latency_s": res.latency_s, "attempts": res.attempts}

        lock = Lock()
        done = 0
        with cache_path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(worker, p) for p in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    cache[rec["id"]] = rec
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done += 1
                    if done % 20 == 0 or done == len(todo):
                        print(f"      {done}/{len(todo)} done", flush=True)
        fails = sum(1 for p in todo if not cache[p["id"]]["ok"])
        if fails:
            print(f"    {fails}/{len(todo)} API failures (excluded from denominators)")
    return cache


# --------------------------------------------------------------------------- #
# scoring (compile-extract grade)
# --------------------------------------------------------------------------- #
def score_model(model: str, probs: list[dict], cache: dict[int, dict],
                workers: int, atol: float) -> list[dict]:
    def scorer(p: dict) -> dict:
        rec = cache[p["id"]]
        base = {"id": p["id"], "tag": p["tag"], "latency_s": rec.get("latency_s")}
        if not rec.get("ok"):
            return {**base, "api_fail": True, "passed": False, "compiles": False,
                    "figure_only": False, "coords_all_correct": False,
                    "coord_accuracy": 0.0}
        g = extract.grade(rec["raw"], p["points"], atol=atol, unordered=p["unordered"])
        return {**base, "api_fail": False, "passed": g["passed"],
                "compiles": g["compiles"], "figure_only": g["figure_only"],
                "coords_all_correct": g["coords_all_correct"],
                "coord_accuracy": g["coord_accuracy"], "per_point": g["per_point"]}

    return gateway.map_concurrent(scorer, probs, workers=workers)


def aggregate(detail: list[dict]) -> dict:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for d in detail:
        by_tag[d["tag"]].append(d)
    out: dict[str, dict] = {}
    for tag, rows in by_tag.items():
        live = [r for r in rows if not r["api_fail"]]
        n = len(live) or 1
        out[tag] = {
            "n": len(live),
            "n_api_fail": sum(r["api_fail"] for r in rows),
            "pass_rate": sum(r["passed"] for r in live) / n,
            "compile_rate": sum(r["compiles"] for r in live) / n,
            "figure_only_rate": sum(r["figure_only"] for r in live) / n,
            "coords_all_correct_rate": sum(r["coords_all_correct"] for r in live) / n,
            "coord_accuracy_mean": sum(r["coord_accuracy"] for r in live) / n,
        }
    return out


def print_table(models: list[str], results: dict, cells: list[str], metric: str,
                title: str) -> None:
    print(f"\n{'=' * (36 + 11 * len(cells))}\n{title}\n{'=' * (36 + 11 * len(cells))}")
    head = f"{'model':<30}" + "".join(f"{c[:10]:>11}" for c in cells)
    print(head)
    print("-" * len(head))
    for m in models:
        row = f"{m:<30}"
        for c in cells:
            v = results[m]["cells"].get(c, {}).get(metric)
            row += f"{('%.2f' % v) if v is not None else '  -':>11}"
        print(row)
    print("-" * len(head))
    for label, fn in (("MEAN", lambda xs: sum(xs) / len(xs)), ("BEST", max)):
        row = f"{label + ' across models':<30}"
        for c in cells:
            xs = [results[m]["cells"][c][metric] for m in models
                  if c in results[m]["cells"]]
            row += f"{('%.2f' % fn(xs)) if xs else '  -':>11}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help="explicit gateway model ids")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--n", type=int, default=12, help="problems per construction type")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--atol", type=float, default=0.05)
    ap.add_argument("--eval", type=str, default="data/olympiad_eval.jsonl")
    ap.add_argument("--out", type=str, default="outputs/olympiad_sweep")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--score-workers", type=int, default=6, help="concurrent compiles")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rescore", action="store_true", help="re-grade cached outputs")
    ap.add_argument("--gather-only", action="store_true",
                    help="only fetch+cache raw outputs; skip grading")
    args = ap.parse_args()

    models = args.models or (PRESETS[args.preset] if args.preset else PRESETS["frontier_small"])
    out_dir = ROOT / args.out
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "detail").mkdir(parents=True, exist_ok=True)

    probs = load_or_build_eval(ROOT / args.eval, args.n, args.seed, args.atol, args.score_workers)
    cells = [t for t in olympiad.TYPES if any(p["tag"] == t for p in probs)]

    results: dict = {}
    prior_models: list[str] = []
    results_path = out_dir / "results.json"
    if results_path.exists():
        prior = json.loads(results_path.read_text())
        results = prior.get("results", {})
        prior_models = prior.get("meta", {}).get("models", [])
    all_models = list(dict.fromkeys(prior_models + models))

    for model in models:
        detail_path = out_dir / "detail" / f"{safe_name(model)}.jsonl"
        if model in results and detail_path.exists() and not args.rescore:
            print(f"\n>>> {model} (already scored — skipping; use --rescore)")
            continue
        print(f"\n>>> {model}")
        t0 = time.time()
        cache = ensure_raw(model, probs, out_dir / "raw", args.workers,
                           args.max_tokens, args.temperature)
        if args.gather_only:
            print(f"    gather-only: {len(cache)}/{len(probs)} cached ({time.time() - t0:.0f}s)")
            continue
        detail = score_model(model, probs, cache, args.score_workers, args.atol)
        detail_path.write_text("\n".join(json.dumps(d) for d in detail) + "\n")
        lats = [d["latency_s"] for d in detail if d.get("latency_s")]
        cellsagg = aggregate(detail)
        n_live = sum(c["n"] for c in cellsagg.values())
        results[model] = {
            "cells": cellsagg,
            "overall": {
                "n": n_live,
                "n_api_fail": sum(c["n_api_fail"] for c in cellsagg.values()),
                "pass_rate": sum(d["passed"] for d in detail if not d["api_fail"]) / (n_live or 1),
                "compile_rate": sum(d["compiles"] for d in detail if not d["api_fail"]) / (n_live or 1),
                "coords_all_correct_rate": sum(d["coords_all_correct"] for d in detail if not d["api_fail"]) / (n_live or 1),
            },
            "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else None,
        }
        ov = results[model]["overall"]
        print(f"    overall pass={ov['pass_rate']:.2f} compile={ov['compile_rate']:.2f} "
              f"coords_ok={ov['coords_all_correct_rate']:.2f} api_fail={ov['n_api_fail']} "
              f"avg_lat={results[model]['avg_latency_s']}s ({time.time() - t0:.0f}s)")

        meta = {"models": all_models, "cells": cells, "n_per_type": args.n,
                "seed": args.seed, "atol": args.atol, "max_tokens": args.max_tokens,
                "temperature": args.temperature, "eval": args.eval}
        results_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    if args.gather_only:
        print("\ngather-only complete. Re-run without --gather-only to grade.")
        return

    shown = [m for m in all_models if m in results]
    print_table(shown, results, cells, "pass_rate",
                "PASS RATE  (figure-only AND compiles AND all coords within tol)")
    print_table(shown, results, cells, "coords_all_correct_rate",
                "COORDS-ALL-CORRECT RATE  (ignores figure-only/compile format gate)")
    print_table(shown, results, cells, "compile_rate", "COMPILE RATE")

    print("\nper-model overall:")
    for m in shown:
        ov = results[m]["overall"]
        print(f"  {m:<30} pass={ov['pass_rate']:.2f}  coords_ok={ov['coords_all_correct_rate']:.2f}"
              f"  compile={ov['compile_rate']:.2f}  (n={ov['n']}, api_fail={ov['n_api_fail']})")
    print(f"\nwrote -> {results_path}")


if __name__ == "__main__":
    main()
