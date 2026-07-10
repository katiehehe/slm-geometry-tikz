"""Teacher distillation for the arbitrary-competition-geometry ILLUSTRATOR.

The specialist trained on the narrow synthetic vocabulary illustrates only ~14%
of real AIME geometry, because real competition problems are out-of-distribution
in BOTH construction diversity and natural-language style. The canonical fix is
to distill a strong frontier TEACHER (fluent in tkz-euclide) on the *actual
problem text* of a large corpus of real competition geometry, then HARD-FILTER
for quality so the student learns from clean (problem -> figure) pairs.

Pipeline (every stage is cache-backed and resumable; re-running never re-spends):

  1. gather   real competition problems:
                * gneubig/aime-1983-2024      geometry, EXCLUDING the fixed 150
                  held-out eval sample (seed 20260709) so AIME stays a true test,
                * EleutherAI/hendrycks_math   config `geometry` (train+test),
              with any [asy]...[/asy] diagram code stripped so the model learns
              text -> figure (not asy -> tikz transliteration).
  2. teacher  gpt-5.5 emits ONE coordinate-free construction figure per problem
              via CONSTRUCTION_SYSTEM_PROMPT (build_construction_messages) --
              the SAME prompt the AIME illustrator uses for its frontier
              fallback. Concurrent over the corpus; cached.
  3. compile  keep only figures that COMPILE and are NON-DEGENERATE, reusing the
              exact degeneracy guards + tkz-euclide preamble the eval grades with
              (serve.compile_and_render). Cached by figure hash.
  4. judge    a SECOND frontier model (VISION) looks at the RENDERED figure + the
              problem text and decides whether the drawing faithfully depicts the
              configuration. Real problems have no ground-truth coordinates, so
              this is our stand-in for "correct" -- and, unlike a coordinate or
              3D-rejecting gate, it KEEPS faithful 3D / combinatorial / region
              figures (softer signal, but it widens coverage). Cached; falls back
              to a text-over-source judge if the gateway rejects images.
  5. write    data/distill_illustrator{,_chat}.jsonl (kept pairs, construction
              prompt) + a yield report at outputs/distill/report.md.

Usage (resumable -- safe to Ctrl-C and re-run):
  uv run python scripts/distill.py --workers 12               # all stages
  uv run python scripts/distill.py --stage teacher --workers 12
  uv run python scripts/distill.py --stage filter             # compile+judge+write
  uv run python scripts/distill.py --limit-per-source 50      # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import gateway, metrics, serve, vision_judge  # noqa: E402
from geotikz.prompts import build_construction_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "distill"
CACHE_DIR = OUT_DIR / "cache"
FIGS_DIR = OUT_DIR / "figs"

# The held-out AIME eval is sampled with THIS seed/size by scripts/illustrate_aime.py.
# We exclude exactly those problems from distillation so before/after coverage on
# them is a clean, leakage-free measurement.
EVAL_SEED = 20260709
EVAL_N = 150

_ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL | re.IGNORECASE)


def strip_asy(text: str) -> str:
    """Remove embedded Asymptote diagram code so the input is text-only."""
    return _ASY_RE.sub(" ", text or "").strip()


# --------------------------------------------------------------------------- #
# stage 1: gather real competition geometry problems
# --------------------------------------------------------------------------- #
def _aime_eval_ids() -> set[str]:
    """The exact ids scripts/illustrate_aime.py holds out for evaluation."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from illustrate_aime import load_problems  # noqa: E402

    return {p["id"] for p in load_problems(EVAL_N, EVAL_SEED)}


def gather_problems(limit_per_source: int | None, seed: int) -> list[dict]:
    """Real competition geometry problems, asy-stripped, eval excluded."""
    import random

    from datasets import load_dataset

    sys.path.insert(0, str(ROOT / "scripts"))
    from illustrate_aime import is_geometry  # noqa: E402

    rng = random.Random(seed)
    probs: list[dict] = []

    # --- AIME geometry, minus the held-out eval sample ---
    eval_ids = _aime_eval_ids()
    aime = load_dataset("gneubig/aime-1983-2024")["train"]
    aime_geo = []
    for r in aime:
        q = r.get("Question", "")
        if not is_geometry(q):
            continue
        pid = str(r.get("ID") or f"{r.get('Year')}-{r.get('Problem Number')}")
        if pid in eval_ids:
            continue  # leakage guard: never train on an eval problem
        aime_geo.append({"id": f"aime:{pid}", "source": "aime",
                         "description": strip_asy(q)})
    aime_geo.sort(key=lambda p: p["id"])
    if limit_per_source:
        aime_geo = aime_geo[:limit_per_source]
    print(f"[gather] AIME geometry (ex-eval): {len(aime_geo)} "
          f"(held out {len(eval_ids)} eval ids)")
    probs += aime_geo

    # --- MATH geometry (train + test) ---
    math_geo = []
    ds = load_dataset("EleutherAI/hendrycks_math", "geometry")
    for split in ("train", "test"):
        for i, r in enumerate(ds[split]):
            text = strip_asy(r.get("problem", ""))
            if len(text) < 20:
                continue
            math_geo.append({"id": f"math:{split}:{i}", "source": "math",
                             "level": r.get("level"), "description": text})
    math_geo.sort(key=lambda p: p["id"])
    if limit_per_source:
        math_geo = rng.sample(math_geo, min(limit_per_source, len(math_geo)))
        math_geo.sort(key=lambda p: p["id"])
    print(f"[gather] MATH geometry (train+test): {len(math_geo)}")
    probs += math_geo

    # de-dup on description text (a few MATH/AIME repeats exist across years)
    seen: set[str] = set()
    uniq = []
    for p in probs:
        h = serve.dhash(p["description"])
        if h in seen:
            continue
        seen.add(h)
        uniq.append(p)
    print(f"[gather] total unique problems: {len(uniq)}")
    return uniq


# --------------------------------------------------------------------------- #
# stage 2: teacher generation (cached, concurrent)
# --------------------------------------------------------------------------- #
def run_teacher(probs: list[dict], teacher: str, cache: serve.Cache,
                workers: int, max_tokens: int) -> dict[str, dict]:
    keyed = {p["id"]: f"{teacher}:" + serve.dhash(p["description"]) for p in probs}
    todo = [p for p in probs if cache.get(keyed[p["id"]]) is None]
    print(f"[teacher {teacher}] {len(probs) - len(todo)} cached, {len(todo)} to generate")

    if todo:
        done = [0]
        t0 = time.time()

        def gen(p: dict) -> None:
            res = gateway.chat(build_construction_messages(p["description"]),
                               teacher, max_tokens=max_tokens)
            cache.put(keyed[p["id"]], {"id": p["id"], "source": p["source"],
                                       "description": p["description"],
                                       "output": res.text, "ok": res.ok,
                                       "error": res.error, "finish": res.finish_reason,
                                       "latency_s": res.latency_s})
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(todo):
                rate = done[0] / max(time.time() - t0, 1e-6)
                print(f"    teacher {done[0]}/{len(todo)}  ({rate:.1f}/s)", flush=True)

        gateway.map_concurrent(gen, todo, workers=workers)

    return {p["id"]: cache.get(keyed[p["id"]]) for p in probs}


# --------------------------------------------------------------------------- #
# stage 3: compile + non-degenerate filter (cached by figure hash)
# Renders each compiling figure to a PERSISTENT PNG (needed by the vision judge).
# --------------------------------------------------------------------------- #
def _fig_png(tikz: str) -> Path:
    return FIGS_DIR / f"{serve.dhash(tikz)}.png"


def run_compile(gen: dict[str, dict], compile_cache: serve.Cache,
                workers: int) -> dict[str, dict]:
    import concurrent.futures as cf

    jobs = []
    results: dict[str, dict] = {}
    for pid, g in gen.items():
        tikz = metrics.extract_tikz((g or {}).get("output", "") or "")
        if not tikz:
            results[pid] = {"ok": False, "reason": "no-figure", "png": None}
            continue
        png = _fig_png(tikz)
        ck = "compile:" + serve.dhash(tikz)
        hit = compile_cache.get(ck)
        if hit is not None and (not hit["ok"] or png.exists()):
            results[pid] = {"ok": hit["ok"], "reason": hit["reason"],
                            "png": str(png) if hit["ok"] else None}
        else:
            jobs.append((pid, tikz, png, ck))

    print(f"[compile] {len(results)} cached/no-figure, {len(jobs)} to compile")
    if jobs:
        def one(job):
            pid, tikz, png, ck = job
            r = serve.compile_and_render(tikz, png)  # writes PNG at `png` when ok
            return pid, ck, tikz, {"ok": r.ok, "reason": r.reason}

        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, j) for j in jobs]
            done = 0
            for fut in cf.as_completed(futs):
                pid, ck, tikz, rec = fut.result()
                compile_cache.put(ck, rec)
                png = _fig_png(tikz)
                results[pid] = {"ok": rec["ok"], "reason": rec["reason"],
                                "png": str(png) if rec["ok"] else None}
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"    compiled {done}/{len(jobs)}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# stage 4: VISION plausibility judge (cached, concurrent)
# Renders + shows a capable vision model the problem TEXT + the FIGURE and asks
# whether it faithfully depicts the configuration. KEEPS 3D / combinatorial /
# region figures the judge approves (they are not coordinate-verifiable but are
# genuine illustrations), so the distilled set is not limited to clean-GT scenes.
# --------------------------------------------------------------------------- #
def run_judge(gen: dict[str, dict], comp: dict[str, dict], judge_model: str,
              cache: serve.Cache, workers: int, prefer_vision: bool) -> dict[str, dict]:
    compiled_ok = [pid for pid, r in comp.items() if r["ok"]]
    todo = []
    for pid in compiled_ok:
        g = gen[pid]
        tikz = metrics.extract_tikz(g.get("output", "") or "")
        ck = f"vj:{judge_model}:" + serve.dhash((g["description"] or "") + "|" + (tikz or ""))
        if cache.get(ck) is None:
            todo.append((pid, g["description"], tikz, comp[pid].get("png"), ck))
    print(f"[vision-judge {judge_model}] {len(compiled_ok) - len(todo)} cached, "
          f"{len(todo)} to judge")

    if todo:
        done = [0]

        def one(item):
            pid, desc, tikz, png, ck = item
            v = vision_judge.judge(desc, png, tikz or "", judge_model,
                                   prefer_vision=prefer_vision)
            cache.put(ck, {"id": pid, "approved": v.approved, "reason": v.reason,
                           "mode": v.mode})
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(todo):
                print(f"    judged {done[0]}/{len(todo)}", flush=True)
            return None

        gateway.map_concurrent(one, todo, workers=workers)

    out: dict[str, dict] = {}
    for pid in compiled_ok:
        g = gen[pid]
        tikz = metrics.extract_tikz(g.get("output", "") or "")
        ck = f"vj:{judge_model}:" + serve.dhash((g["description"] or "") + "|" + (tikz or ""))
        out[pid] = cache.get(ck) or {"approved": False, "reason": "missing", "mode": "error"}
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="openai-group/gpt-5.5")
    ap.add_argument("--judge-model", default="gemini-group/gemini-3.1-pro",
                    help="SECOND (vision) model for the plausibility filter")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the plausibility judge (keep all compiling figures)")
    ap.add_argument("--text-judge", action="store_true",
                    help="force the text-over-source judge instead of vision")
    ap.add_argument("--stage", choices=["all", "teacher", "filter"], default="all")
    ap.add_argument("--limit-per-source", type=int, default=0,
                    help="0 = use all; else cap AIME/MATH each (smoke test)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--judge-workers", type=int, default=12)
    ap.add_argument("--compile-workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6144)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    teacher_cache = serve.Cache(CACHE_DIR / "teacher.jsonl")
    compile_cache = serve.Cache(CACHE_DIR / "compile.jsonl")
    judge_cache = serve.Cache(CACHE_DIR / "judge_vision.jsonl")

    limit = args.limit_per_source or None
    probs = gather_problems(limit, args.seed)
    (OUT_DIR / "problems.jsonl").write_text(
        "\n".join(json.dumps(p) for p in probs) + "\n")

    # stage 2: teacher
    gen = run_teacher(probs, args.teacher, teacher_cache, args.workers, args.max_tokens)
    teacher_ok = sum(1 for g in gen.values() if g and g.get("ok"))
    print(f"[teacher] usable generations: {teacher_ok}/{len(probs)}")
    if args.stage == "teacher":
        print("stage=teacher done (run --stage filter next).")
        return

    # stage 3: compile + non-degenerate
    comp = run_compile(gen, compile_cache, args.compile_workers)
    compiled_ok = {pid for pid, r in comp.items() if r["ok"]}
    print(f"[compile] compiles+non-degenerate: {len(compiled_ok)}/{len(probs)}")

    # stage 4: VISION plausibility judge (keeps 3D / combinatorial if faithful)
    if args.no_judge:
        judged = {pid: {"approved": True, "reason": "judge-skipped", "mode": "skipped"}
                  for pid in compiled_ok}
    else:
        judged = run_judge(gen, comp, args.judge_model, judge_cache,
                           args.judge_workers, prefer_vision=not args.text_judge)

    kept_ids = [pid for pid in compiled_ok if judged[pid]["approved"]]
    print(f"[judge] kept (judge-approved): {len(kept_ids)}/{len(compiled_ok)}")

    # stage 5: write kept pairs (raw + chat) and a yield report
    kept_rows = []
    for pid in sorted(kept_ids):
        g = gen[pid]
        tikz = metrics.extract_tikz(g.get("output", "") or "")
        kept_rows.append({"id": pid, "source": g["source"],
                          "description": g["description"], "tikz": tikz,
                          "judge_mode": judged[pid]["mode"]})

    raw_out = ROOT / "data" / "distill_illustrator.jsonl"
    chat_out = ROOT / "data" / "distill_illustrator_chat.jsonl"
    raw_out.write_text("\n".join(json.dumps(r) for r in kept_rows) + "\n")
    with chat_out.open("w") as f:
        for r in kept_rows:
            rec = {"messages": build_construction_messages(r["description"])
                   + [{"role": "assistant", "content": r["tikz"]}]}
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(kept_rows)} distilled pairs -> {raw_out.name} (+ {chat_out.name})")

    _write_report(probs, gen, comp, judged, kept_rows, args)


def _write_report(probs, gen, comp, judged, kept_rows, args) -> None:
    from collections import Counter

    n = len(probs)
    by_src = Counter(p["source"] for p in probs)
    teacher_ok = sum(1 for g in gen.values() if g and g.get("ok"))
    compiled_ok = {pid for pid, r in comp.items() if r["ok"]}
    kept_by_src = Counter(r["source"] for r in kept_rows)
    approved_hist = Counter(bool(v["approved"]) for v in judged.values())
    mode_hist = Counter(v.get("mode") for v in judged.values())
    reason_hist = Counter(r["reason"] for r in comp.values() if not r["ok"])

    def pct(a, b):
        return f"{(a / b * 100):.1f}%" if b else "n/a"

    md = [
        "# Distillation yield\n",
        f"- Teacher: `{args.teacher}`  |  Vision judge: "
        f"`{'(skipped)' if args.no_judge else args.judge_model}` "
        f"({'text-over-source' if args.text_judge else 'vision on rendered PNG'})",
        f"- Corpus: {n} real competition problems "
        f"({', '.join(f'{k}={v}' for k, v in by_src.items())}), "
        "AIME eval sample held out.",
        "- The judge KEEPS faithful 3D / combinatorial / region figures (they are "
        "not coordinate-verifiable, but are genuine illustrations).\n",
        "## Funnel\n",
        "| stage | count | yield |",
        "|---|---|---|",
        f"| problems gathered | {n} | 100% |",
        f"| teacher produced output | {teacher_ok} | {pct(teacher_ok, n)} |",
        f"| compiles + non-degenerate | {len(compiled_ok)} | {pct(len(compiled_ok), n)} |",
        f"| kept after vision judge | {len(kept_rows)} | {pct(len(kept_rows), n)} |\n",
        "## Judge verdict (of compiling figures)\n",
        "| verdict | count |", "|---|---|",
        f"| approved | {approved_hist.get(True, 0)} |",
        f"| rejected | {approved_hist.get(False, 0)} |\n",
        "## Judge mode (vision vs text fallback)\n",
        "| mode | count |", "|---|---|",
    ]
    for mode, c in mode_hist.most_common():
        md.append(f"| {mode} | {c} |")
    md += ["\n## Kept pairs by source\n", "| source | kept |", "|---|---|"]
    for k, v in kept_by_src.items():
        md.append(f"| {k} | {v} |")
    md += ["\n## Top compile-failure reasons\n", "| reason | count |", "|---|---|"]
    for reason, c in reason_hist.most_common(8):
        md.append(f"| {reason} | {c} |")
    (OUT_DIR / "report.md").write_text("\n".join(md) + "\n")
    print(f"report -> {OUT_DIR / 'report.md'}")
    print("\n".join(md[:18]))


if __name__ == "__main__":
    main()
