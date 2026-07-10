"""LLM-classify the geometric constructions in competition problems (+ solutions).

Refines the keyword mine (scripts/mine_constructions.py): a keyword pass misses
IMPLICIT constructions ("center of the circle through A,B,C" == circumcenter) and
constructions that only appear in the SOLUTION. This asks an LLM to tag each
problem (and its solution, when available) against a controlled vocabulary.

Results are cached per problem hash (outputs/construction_llm_cache.jsonl), so
re-runs are free and the run is resumable.

Usage:
  uv run python scripts/classify_constructions.py --limit 40      # sanity check
  uv run python scripts/classify_constructions.py                 # full run
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import gateway  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/construction_llm_cache.jsonl"

VOCAB = [
    "midpoint", "intersection", "parallel", "perpendicular/foot", "tangent",
    "angle bisector", "perpendicular bisector", "altitude", "median",
    "circumcircle/circumcenter", "incircle/incenter", "excircle/excenter",
    "centroid", "orthocenter", "reflection", "rotation", "trisection",
    "diameter", "chord", "arc", "inscribed polygon", "circumscribed",
    "similar triangles", "congruent triangles", "concurrent/collinear",
    "power of a point", "cyclic quadrilateral",
]
_CANON = {v.lower(): v for v in VOCAB}

SYSTEM = (
    "You are a geometry problem classifier. Given a competition geometry problem "
    "(and its solution, if provided), list which geometric CONSTRUCTIONS are present "
    "OR required to solve it — INCLUDING ones that are implied but not named "
    "(e.g. 'the center of the circle through three points' IS a circumcenter; "
    "'the point equidistant from the sides' IS an incenter). "
    "Choose ONLY from this exact vocabulary:\n" + ", ".join(VOCAB) + "\n"
    "Return ONLY a JSON array of the applicable tag strings (verbatim from the list). "
    "If none apply, return []. No prose."
)


def build_msgs(problem: str, solution: str) -> list[dict]:
    user = f"PROBLEM:\n{problem}\n"
    if solution:
        user += f"\nSOLUTION:\n{solution[:2500]}\n"
    user += "\nJSON array of applicable construction tags:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_tags(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for item in arr:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key in _CANON:
            out.append(_CANON[key])
        else:  # fuzzy: match any vocab term contained in / containing the item
            for v in VOCAB:
                if v.lower() in key or key in v.lower():
                    out.append(v)
                    break
    return sorted(set(out))


def _hash(t: str) -> str:
    return hashlib.md5(t.encode()).hexdigest()[:16]


def load_corpora() -> list[dict]:
    from datasets import load_dataset

    recs: list[dict] = []
    g = load_dataset("EleutherAI/hendrycks_math", "geometry")
    for split in g.values():
        for r in split:
            recs.append({"corpus": "MATH", "problem": r["problem"],
                         "solution": r.get("solution", "")})
    geom = re.compile(
        r"triangl|circle|square|rectangl|quadrilateral|polygon|angle|perpendicular|"
        r"parallel|vert(ex|ices)|tangent|radius|radii|diameter|chord|centroid|bisect",
        re.I)
    try:
        a = load_dataset("gneubig/aime-1983-2024")["train"]
        for r in a:
            q = r.get("Question") or r.get("problem") or ""
            if geom.search(q):
                recs.append({"corpus": "AIME", "problem": q, "solution": ""})
    except Exception as e:  # noqa: BLE001
        print(f"(AIME skipped: {str(e)[:80]})")
    for rec in recs:
        rec["h"] = _hash(rec["problem"])
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-group/gpt-5-mini")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="classify only first N (sanity check)")
    ap.add_argument("--out", default="outputs/construction_freq_llm.json")
    args = ap.parse_args()

    recs = load_corpora()
    if args.limit:
        recs = recs[: args.limit]
    print(f"corpora: {Counter(r['corpus'] for r in recs)}  total={len(recs)}")

    cache: dict[str, list[str]] = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                cache[r["h"]] = r["tags"]
    todo = [r for r in recs if r["h"] not in cache]
    print(f"cached {len(recs) - len(todo)} / {len(recs)}; calling {len(todo)} via {args.model}")

    if todo:
        lock = threading.Lock()
        done = 0

        def worker(rec: dict) -> None:
            nonlocal done
            res = gateway.chat(build_msgs(rec["problem"], rec["solution"]), args.model,
                               max_tokens=600, temperature=0.0)
            tags = parse_tags(res.text) if res.ok else []
            with lock:
                cache[rec["h"]] = tags
                with CACHE.open("a") as f:
                    f.write(json.dumps({"h": rec["h"], "tags": tags, "ok": res.ok}) + "\n")
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}")

        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, todo))

    # aggregate document frequency
    n = len(recs) or 1
    combined: Counter = Counter()
    per_corpus: dict[str, Counter] = {}
    per_corpus_n: Counter = Counter()
    for r in recs:
        per_corpus_n[r["corpus"]] += 1
        tags = cache.get(r["h"], [])
        combined.update(tags)
        per_corpus.setdefault(r["corpus"], Counter()).update(tags)

    corpora = sorted(per_corpus_n)
    print(f"\n{'construction':<28}{'ALL %':>8}" + "".join(f"{c:>9}" for c in corpora))
    print("-" * (36 + 9 * len(corpora)))
    for name, cnt in combined.most_common():
        row = f"{name:<28}{100*cnt/n:>7.1f}%"
        for c in corpora:
            row += f"{100*per_corpus[c][name]/(per_corpus_n[c] or 1):>8.1f}%"
        print(row)

    payload = {"model": args.model, "totals": dict(per_corpus_n),
               "combined_docfreq": dict(combined.most_common()),
               "per_corpus_docfreq": {c: dict(per_corpus[c]) for c in corpora}}
    (ROOT / args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote -> {ROOT / args.out}")


if __name__ == "__main__":
    main()
