"""Build the v2 illustrator dataset: phrasing-robust + harder constructions.

ADDITIVE. Nothing existing is overwritten; every output is a NEW file. The v2
training set = the existing v1 set (data/illustrator_train_chat.jsonl, 3,996
records) PLUS two new ingredients aimed at the two weaknesses of v1
(format-locking to the synthetic template wording, and no coverage of
multi-point / composed constructions):

  (a) PARAPHRASE AUGMENTATION.  For every SYNTHETIC construction prompt in the v1
      set (the coordinate-grounded, template-worded ones), a frontier model
      rewrites the scene several ways (varying order / formality / vocabulary)
      while the assistant TARGET TikZ is kept BYTE-FOR-BYTE unchanged. Each
      rewrite is validated to preserve every number (coordinates/radii/angles)
      and every requested point name, so the (prompt -> figure) pair stays
      correct-by-construction. This teaches the model that phrasing is
      irrelevant to the figure.

  (b) HARDER / BROADER CONSTRUCTIONS.  src/geotikz/olympiad_hard.py emits
      compositions with 2-4 derived points and more olympiad vocabulary (Euler
      line, medial / orthic / contact triangles, nine-point centre, medians,
      antipode, ...). Every figure is ROUND-TRIP VALIDATED through the
      compile-extract grader (built forward from exact coords, coords stripped
      from the model input), so labels are correct-by-construction. A disjoint
      slice is held out (with GT coords) to extend the synthetic gate.

Held-out evals produced (GT-graded, disjoint from training):
  * data/illustrator_syn_eval_v2.jsonl    v1 gate (240) + harder held-out
  * data/illustrator_paraphrase_eval.jsonl  LOOSE rephrasings of the v1 gate
      problems (unseen wordings of unseen problems) -> phrasing-robustness test

Usage (resumable; gateway calls are cached):
  uv run python scripts/build_illustrator_v2_data.py --smoke      # tiny dry run
  uv run python scripts/build_illustrator_v2_data.py              # full build
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract, gateway, olympiad_hard, serve  # noqa: E402
from geotikz.prompts import build_construction_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RT_CACHE = ROOT / "outputs" / "distill" / "cache" / "roundtrip.jsonl"
PARA_CACHE = ROOT / "outputs" / "distill" / "cache" / "paraphrase.jsonl"
REPORT = ROOT / "outputs" / "distill" / "v2_dataset_report.md"

_SYN_SUFFIXES = ("at their correct positions.", "at its correct position.",
                 "at their exact locations.", "at their true coordinates.")
_SCENE_PREFIX = "Scene:\n"
_SCENE_SUFFIX = "\n\nReturn the TikZ figure."
_LABEL_RE = re.compile(r"\\tkzLabelPoints(?:\[[^\]]*\])?\(([^()]*)\)")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- #
# validation of a paraphrase against the original scene
# --------------------------------------------------------------------------- #
def _nums(text: str) -> list[str]:
    return _NUM_RE.findall((text or "").replace("\u2212", "-"))


def _multiset_subset(need: list[str], have: list[str]) -> bool:
    cn, ch = Counter(need), Counter(have)
    return all(ch[k] >= v for k, v in cn.items())


def _name_present(name: str, text: str) -> bool:
    if len(name) == 1:  # single letter -> require a standalone token
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text) is not None
    return name in text  # multi-char (P0, T1, ...) -> substring is safe


def valid_paraphrase(orig_inner: str, para: str, names: list[str]) -> bool:
    """A paraphrase is usable iff it preserves every number + every point name
    and still asks for a drawing (so the fixed TikZ target stays correct)."""
    if not para or len(para.strip()) < 20:
        return False
    if not _multiset_subset(_nums(orig_inner), _nums(para)):
        return False
    if not all(_name_present(n, para) for n in names):
        return False
    low = para.lower()
    if not any(w in low for w in ("tikz", "figure", "draw", "construct", "sketch", "plot")):
        return False
    return True


# --------------------------------------------------------------------------- #
# paraphrase generation (frontier model via gateway, JSON list, cached)
# --------------------------------------------------------------------------- #
_PARA_SYS = (
    "You rewrite geometry problem statements to create natural-language variety "
    "for a training set. You never solve the problem and never change its meaning."
)


def _para_prompt(inner: str, k: int, loose: bool) -> list[dict]:
    style = (
        "Make them read like a student or teacher casually posing the problem: "
        "some terse, some chatty, some formal. Reorder clauses and vary vocabulary "
        "aggressively, but keep it unambiguous."
        if loose else
        "Vary sentence order, formality (mix terse and verbose), and vocabulary."
    )
    rules = (
        f"Rewrite the SCENE below in {k} DIFFERENT ways. Every rewrite MUST:\n"
        "- keep EVERY number exactly as written (all coordinates, radii, angles, "
        "lengths) as digits — never round, drop, spell out, or invent numbers;\n"
        "- keep EVERY point name exactly (case-sensitive) and the SAME points to "
        "be defined/constructed;\n"
        "- preserve the geometry precisely (same construction, same relationships); "
        "do NOT add coordinates or facts, and do NOT solve it;\n"
        "- keep an instruction to output a single TikZ figure defining those points.\n"
        f"{style}\n"
        f"Return ONLY a JSON array of exactly {k} strings and nothing else.\n\n"
        f"SCENE:\n{inner}"
    )
    return [{"role": "system", "content": _PARA_SYS},
            {"role": "user", "content": rules}]


def _parse_json_list(text: str) -> list[str]:
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    i, j = t.find("["), t.rfind("]")
    if i == -1 or j == -1 or j < i:
        return []
    try:
        arr = json.loads(t[i : j + 1])
    except json.JSONDecodeError:
        return []
    return [s for s in arr if isinstance(s, str) and s.strip()]


def paraphrase(inner: str, names: list[str], model: str, cache: serve.Cache,
               k: int, loose: bool) -> list[str]:
    """Return the VALID paraphrases for one scene (cached by scene+style+k)."""
    ck = f"para:{model}:{'loose' if loose else 'train'}:k{k}:" + serve.dhash(inner)
    hit = cache.get(ck)
    if hit is None:
        res = gateway.chat(_para_prompt(inner, k, loose), model, max_tokens=2048)
        cands = _parse_json_list(res.text) if res.ok else []
        cache.put(ck, {"cands": cands, "ok": res.ok, "error": res.error})
        hit = {"cands": cands}
    return [p.strip() for p in hit.get("cands", []) if valid_paraphrase(inner, p, names)]


# --------------------------------------------------------------------------- #
# harder constructions (round-trip validated, cached by figure hash)
# --------------------------------------------------------------------------- #
def _grade_gt(prob: dict) -> dict:
    gt = prob["points"]
    if prob.get("grade_only"):
        gt = {k: v for k, v in gt.items() if k in prob["grade_only"]}
    return gt


def roundtrip_ok(prob: dict, cache: serve.Cache) -> bool:
    ck = "rt:" + serve.dhash(prob["tikz"])
    hit = cache.get(ck)
    if hit is not None:
        return bool(hit["ok"])
    g = extract.grade(prob["tikz"], _grade_gt(prob), atol=0.05,
                      unordered=prob.get("unordered"))
    ok = bool(g["figure_only"] and g["compiles"] and g["coords_all_correct"])
    cache.put(ck, {"ok": ok, "reason": g["compile_reason"]})
    return ok


def gen_hard(per_type: int, seed: int, cache: serve.Cache, workers: int):
    probs = olympiad_hard.generate_problems(per_type, seed=seed)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        flags = list(ex.map(lambda p: roundtrip_ok(p, cache), probs))
    ok, tot = Counter(), Counter()
    kept = []
    for p, good in zip(probs, flags):
        tot[p["tag"]] += 1
        if good:
            ok[p["tag"]] += 1
            kept.append(p)
    return kept, ok, tot


def _syn_to_chat(inner_desc: str, tikz: str) -> dict:
    return {"messages": build_construction_messages(inner_desc)
            + [{"role": "assistant", "content": tikz}]}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/illustrator_train_chat.jsonl")
    ap.add_argument("--syn-eval", default="data/illustrator_syn_eval.jsonl")
    ap.add_argument("--para-model", default="gemini-group/gemini-3.1-pro")
    ap.add_argument("--paraphrase-k", type=int, default=2)
    ap.add_argument("--hard-per-type", type=int, default=120)
    ap.add_argument("--hard-eval-per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rt-workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dry run: few synthetic, 2 hard/type, no full write")
    ap.add_argument("--limit-syn", type=int, default=0,
                    help="cap number of synthetic prompts to paraphrase (0=all)")
    ap.add_argument("--train-out", default="data/illustrator_train_chat_v2.jsonl")
    ap.add_argument("--syn-eval-out", default="data/illustrator_syn_eval_v2.jsonl")
    ap.add_argument("--para-eval-out", default="data/illustrator_paraphrase_eval.jsonl")
    args = ap.parse_args()

    RT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    rt_cache = serve.Cache(RT_CACHE)
    para_cache = serve.Cache(PARA_CACHE)

    # ---- load base v1 set + isolate synthetic records --------------------- #
    base = [json.loads(l) for l in (ROOT / args.base).read_text().splitlines() if l.strip()]
    syn_records = []
    for r in base:
        u = next(m["content"] for m in r["messages"] if m["role"] == "user")
        a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        if any(s in u for s in _SYN_SUFFIXES) and u.startswith(_SCENE_PREFIX):
            inner = u[len(_SCENE_PREFIX):].rsplit(_SCENE_SUFFIX, 1)[0]
            names = []
            for grp in _LABEL_RE.findall(a):
                for n in grp.split(","):
                    n = n.strip()
                    if n and n not in names:
                        names.append(n)
            syn_records.append({"inner": inner, "tikz": a, "names": names})
    print(f"base records: {len(base)}  |  synthetic prompts: {len(syn_records)}")

    if args.smoke:
        syn_records = syn_records[:20]
        args.hard_per_type = 2
        args.hard_eval_per_type = 2
    elif args.limit_syn:
        syn_records = syn_records[: args.limit_syn]

    # ---- (a) paraphrase augmentation -------------------------------------- #
    print(f"paraphrasing {len(syn_records)} synthetic prompts x{args.paraphrase_k} "
          f"via {args.para_model} ...")

    def _do_para(rec):
        paras = paraphrase(rec["inner"], rec["names"], args.para_model, para_cache,
                           args.paraphrase_k, loose=False)
        return rec, paras

    para_records = []
    n_valid = n_cand = 0
    results = gateway.map_concurrent(_do_para, syn_records, workers=args.workers)
    for rec, paras in results:
        n_cand += args.paraphrase_k
        for p in paras:
            n_valid += 1
            para_records.append(_syn_to_chat(p, rec["tikz"]))
    print(f"paraphrase yield: {n_valid} valid / ~{n_cand} requested "
          f"({n_valid / max(n_cand,1) * 100:.0f}%)")

    # ---- (b) harder constructions: held-out eval first, then train -------- #
    print("generating + round-trip-validating HARD held-out eval ...")
    hard_eval, heok, hetot = gen_hard(args.hard_eval_per_type, args.seed, rt_cache, args.rt_workers)
    hard_eval_descs = {p["description"] for p in hard_eval}
    print("generating + round-trip-validating HARD train ...")
    hard_train, htok, httot = gen_hard(args.hard_per_type, args.seed + 1000, rt_cache, args.rt_workers)
    hard_train = [p for p in hard_train if p["description"] not in hard_eval_descs]
    hard_train_records = [_syn_to_chat(p["description"], p["tikz"]) for p in hard_train]

    # ---- (ii) paraphrase EVAL: loose rewordings of the v1 gate ------------ #
    syn_eval = [json.loads(l) for l in (ROOT / args.syn_eval).read_text().splitlines() if l.strip()]
    if args.smoke:
        syn_eval_para_src = syn_eval[:8]
    else:
        syn_eval_para_src = syn_eval
    print(f"generating LOOSE paraphrase eval from {len(syn_eval_para_src)} gate problems ...")

    def _do_eval_para(row):
        names = list(row["points"].keys())
        paras = paraphrase(row["description"], names, args.para_model, para_cache,
                           k=2, loose=True)
        return row, (paras[0] if paras else None)

    para_eval_rows = []
    for row, para in gateway.map_concurrent(_do_eval_para, syn_eval_para_src, workers=args.workers):
        if para is None:
            continue
        para_eval_rows.append({"id": len(para_eval_rows), "tag": row["tag"],
                               "description": para, "points": row["points"],
                               "derived": row.get("derived"),
                               "unordered": row.get("unordered"),
                               "grade_only": row.get("grade_only")})
    print(f"paraphrase eval: {len(para_eval_rows)}/{len(syn_eval_para_src)} kept")

    # ---- compose + write -------------------------------------------------- #
    train_records = list(base) + para_records + hard_train_records
    import random
    random.Random(args.seed).shuffle(train_records)

    if args.smoke:
        print("\n[SMOKE] would write:")
        print(f"  train v2: {len(train_records)} = base {len(base)} + para "
              f"{len(para_records)} + hard {len(hard_train_records)}")
        print(f"  syn_eval_v2: {len(syn_eval)} + hard {len(hard_eval)}")
        print(f"  paraphrase_eval: {len(para_eval_rows)}")
        _print_hard_yield(htok, httot, heok, hetot)
        # show 3 sample paraphrases for eyeballing
        for rec, paras in results[:3]:
            if paras:
                print("\n  ORIG:", rec["inner"][:130].replace("\n", " "))
                print("  PARA:", paras[0][:130].replace("\n", " "))
        return

    train_out = ROOT / args.train_out
    with train_out.open("w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    syn_eval_out = ROOT / args.syn_eval_out
    with syn_eval_out.open("w") as f:
        for row in syn_eval:  # v1 gate verbatim (ids preserved)
            f.write(json.dumps(row) + "\n")
        for i, p in enumerate(hard_eval):
            f.write(json.dumps({"id": 10000 + i, "tag": p["tag"],
                                "description": p["description"], "tikz": p["tikz"],
                                "points": p["points"], "derived": p.get("derived"),
                                "unordered": p.get("unordered"),
                                "grade_only": p.get("grade_only")}) + "\n")

    para_eval_out = ROOT / args.para_eval_out
    with para_eval_out.open("w") as f:
        for row in para_eval_rows:
            f.write(json.dumps(row) + "\n")

    _write_report(base, para_records, hard_train_records, hard_train, syn_eval,
                  hard_eval, para_eval_rows, n_valid, n_cand, htok, httot,
                  heok, hetot, args)
    print(f"\nwrote {len(train_records)} train  -> {train_out}")
    print(f"wrote {len(syn_eval) + len(hard_eval)} syn-eval-v2 -> {syn_eval_out}")
    print(f"wrote {len(para_eval_rows)} paraphrase-eval -> {para_eval_out}")
    print(f"report -> {REPORT}")


def _print_hard_yield(htok, httot, heok, hetot) -> None:
    print("  hard round-trip yield (train | eval):")
    for t in olympiad_hard.TYPES:
        print(f"    {t:26s} {htok[t]}/{httot[t]}  |  {heok[t]}/{hetot[t]}")


def _write_report(base, para_records, hard_train_records, hard_train, syn_eval,
                  hard_eval, para_eval_rows, n_valid, n_cand, htok, httot,
                  heok, hetot, args) -> None:
    total = len(base) + len(para_records) + len(hard_train_records)
    md = [
        "# Illustrator v2 dataset composition\n",
        f"- **Total training records: {total}**",
        f"  - v1 base (distilled AIME/MATH + template synthetic): {len(base)}",
        f"  - paraphrase augmentation (synthetic, target unchanged): {len(para_records)}",
        f"  - harder/broader round-trip-validated constructions: {len(hard_train_records)}",
        f"- Paraphrase model: `{args.para_model}` (k={args.paraphrase_k}); "
        f"valid {n_valid}/{n_cand} = {n_valid / max(n_cand,1) * 100:.0f}%",
        "\n## Held-out evals (GT-graded, disjoint from training)\n",
        f"- Synthetic gate v2: {len(syn_eval)} (v1 gate) + {len(hard_eval)} harder = "
        f"{len(syn_eval) + len(hard_eval)}",
        f"- Paraphrase gate (loose rewordings of v1 gate): {len(para_eval_rows)}\n",
        "## Harder-construction round-trip yield (kept / sampled)\n",
        "| construction | train kept/sampled | eval kept/sampled |",
        "|---|---|---|",
    ]
    for t in olympiad_hard.TYPES:
        md.append(f"| {t} | {htok[t]}/{httot[t]} | {heok[t]}/{hetot[t]} |")
    REPORT.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
