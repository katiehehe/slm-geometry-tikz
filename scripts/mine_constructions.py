"""Mine the frequency of geometric constructions in competition math corpora.

Grounds the olympiad construction vocabulary in what actually appears in AMC/AIME/
MATH geometry problems, instead of guessing. Uses public HF datasets (no scraping):
  - EleutherAI/hendrycks_math  (config 'geometry'; labeled geometry problems)
  - gneubig/aime-1983-2024     (AIME 1983-2024; keyword-filtered to geometry)

Reports, per construction, the DOCUMENT frequency (how many geometry problems
mention it) and % — a ranked table that sets the generator/litmus vocabulary.

Usage:
  uv run python scripts/mine_constructions.py
  uv run python scripts/mine_constructions.py --out outputs/construction_freq.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Is an (unlabeled) problem a geometry problem? (used to filter AIME)
GEOM_HINT = re.compile(
    r"triangl|\bcircle|\bcircular|square|rectangl|quadrilateral|polygon|pentagon|"
    r"hexagon|octagon|\bangle|perpendicular|parallel|\bvertex|vertices|\bpoint\b|"
    r"coordinate|diagonal|tangent|radius|radii|diameter|\bchord|\barc\b|sphere|"
    r"cylinder|\bcone\b|\bprism\b|centroid|bisect", re.I)

# Construction / relation vocabulary -> regex (case-insensitive). Ordered roughly
# base-objects first, then derived constructions.
VOCAB: dict[str, str] = {
    "triangle": r"triangl",
    "circle": r"\bcircle|\bcircular\b",
    "square": r"\bsquare",
    "rectangle": r"rectangl",
    "quadrilateral": r"quadrilateral",
    "polygon (n-gon)": r"polygon|pentagon|hexagon|heptagon|octagon|decagon",
    "angle": r"\bangle",
    "midpoint": r"midpoint",
    "perpendicular / foot": r"perpendicular|foot of (the )?(perpendicular|altitude)",
    "parallel": r"parallel",
    "diagonal": r"diagonal",
    "tangent": r"tangent",
    "chord": r"\bchord",
    "diameter": r"diameter",
    "arc": r"\barc\b|minor arc|major arc",
    "intersection": r"intersect",
    "reflection": r"reflect",
    "inscribed": r"inscrib",
    "circumscribed": r"circumscrib",
    "circumcircle / circumcenter": r"circumcircle|circumcenter|circumscrib|circumradius",
    "incircle / incenter": r"incircle|incenter|inradius|\binscribed circle",
    "excircle / excenter": r"excircle|excenter|escrib",
    "angle bisector": r"angle[\s-]?bisect|bisector of (the )?angle|bisect(s|or|ing)?\s+(the\s+)?angle",
    "perpendicular bisector": r"perpendicular\s+bisector",
    "altitude": r"\baltitude",
    "median": r"\bmedian",
    "centroid": r"centroid",
    "orthocenter": r"orthocenter",
    "trisect": r"trisect",
    "similar triangles": r"\bsimilar\b",
    "congruent": r"congruen",
    "concurrent / collinear": r"concurren|collinear",
    "power of a point / radical": r"power of (a|the) point|radical axis",
}
PATTERNS = {name: re.compile(rx, re.I) for name, rx in VOCAB.items()}


def load_corpora() -> dict[str, list[str]]:
    from datasets import load_dataset

    corpora: dict[str, list[str]] = {}
    g = load_dataset("EleutherAI/hendrycks_math", "geometry")
    corpora["MATH-geometry"] = [r["problem"] for split in g.values() for r in split]
    try:
        a = load_dataset("gneubig/aime-1983-2024")["train"]
        aime = [r.get("Question") or r.get("problem") or "" for r in a]
        corpora["AIME-geometry"] = [t for t in aime if GEOM_HINT.search(t)]
    except Exception as e:  # noqa: BLE001
        print(f"(AIME load failed, skipping: {str(e)[:80]})")
    return corpora


def freq(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for name, pat in PATTERNS.items():
            if pat.search(t):
                c[name] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/construction_freq.json")
    args = ap.parse_args()

    corpora = load_corpora()
    per_corpus = {name: freq(texts) for name, texts in corpora.items()}
    totals = {name: len(texts) for name, texts in corpora.items()}
    combined = Counter()
    for c in per_corpus.values():
        combined.update(c)
    n_all = sum(totals.values())

    print(f"\ncorpora: " + ", ".join(f"{k}={v}" for k, v in totals.items()) + f"  (total {n_all})")
    print(f"\n{'construction':<30}{'ALL %':>8}" + "".join(f"{k.split('-')[0]:>10}" for k in corpora))
    print("-" * (38 + 10 * len(corpora)))
    for name, cnt in combined.most_common():
        row = f"{name:<30}{100*cnt/n_all:>7.1f}%"
        for cname in corpora:
            n = totals[cname] or 1
            row += f"{100*per_corpus[cname][name]/n:>9.1f}%"
        print(row)

    payload = {
        "totals": totals,
        "combined_docfreq": dict(combined.most_common()),
        "per_corpus_docfreq": {k: dict(v) for k, v in per_corpus.items()},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
