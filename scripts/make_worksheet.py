"""Classroom geometry WORKSHEET GENERATOR (additive; nothing existing is touched).

Why this plays to the specialist's strength
-------------------------------------------
Auto-illustrating arbitrary competition problems is mostly out-of-distribution
for a 0.6B model. A *worksheet* generator flips that: the problems are drawn from
OUR OWN generators (``geotikz.generator`` and ``geotikz.olympiad``), so every
problem is guaranteed to be IN the specialist's trained vocabulary and comes with
exact ground-truth coordinates. A teacher gets a printable worksheet with correct
figures + an answer key, instantly, locally, free.

What it emits
-------------
1. N in-vocab geometry problems (exact GT coordinates + a natural-language
   statement), from ``generator`` (coordinate-geometry constructions: reflection,
   midpoint, perpendicular foot, line intersection) and/or ``olympiad`` (named
   constructions: circumcenter, incenter, orthocenter, centroid, angle bisector,
   altitude foot, median, tangents).
2. One CONSTRUCTION figure per problem. Per the spec every figure is emitted as a
   coordinate-free construction (``generator`` symbolic PGF ``calc`` /
   ``intersections`` and ``olympiad`` ``tkz-euclide`` macros) — parametric and
   editable, never a hardcoded numeric coordinate dump. Each is compiled with
   tectonic to a vector PDF and rasterised to PNG.
3. A printable worksheet PDF (problem statements + figures) and a SEPARATE answer
   key PDF (exact coordinates + the asked quantity + the same construction), both
   compiled via tectonic.
4. OPTIONALLY runs the specialist (Qwen3-0.6B + LoRA ``qwen3-pgf-geotikz``) on
   each problem via the *training* prompt (``geotikz.prompts.build_messages``, as
   used by ``infer.generate``) and reports the specialist-vs-GT match rate as a
   reliability statistic. The figures printed on the worksheet are always the GT
   constructions; the specialist is only measured, never trusted for the figure.

Usage
-----
  # Fast, self-contained (ground-truth figures only):
  uv run python scripts/make_worksheet.py --source generator --n 8 --seed 7

  # Named olympiad constructions:
  uv run python scripts/make_worksheet.py --source olympiad \
      --topics circumcenter incenter centroid median --n 8

  # Also run the trained specialist and report reliability vs ground truth:
  uv run python scripts/make_worksheet.py --source generator --n 8 --specialist

Options: --source {generator,olympiad,mixed}  --topics ...  --n  --seed
         --chain-min/--chain-max  --irregular {mix,on,off}  --title
         --specialist [--adapter PATH] [--specialist-n K]  --out-dir  --dpi
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fitz  # pymupdf  # noqa: E402

from geotikz import extract, generator, metrics, olympiad, tex  # noqa: E402

# --------------------------------------------------------------------------- #
# Construction-figure compilation (tkz-euclide capable — ADDITIVE).
#
# tex.compile_tikz only loads plain tikz, but olympiad figures need tkz-euclide.
# We keep tex.py untouched and compile here against a richer preamble that also
# happily renders the generator's symbolic PGF (calc / intersections) figures.
# --------------------------------------------------------------------------- #
FIG_TEMPLATE = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{tkz-euclide}
\usetikzlibrary{calc,angles,quotes,intersections,through,positioning,arrows.meta,decorations.markings}
\begin{document}
__BODY__
\end{document}
"""


def prep_figure(tikz: str) -> str:
    """Make a construction figure crop cleanly as a standalone.

    The generator's symbolic ``intersection`` emits invisible helper lines via
    ``\\path[name path=...]`` that extend far past the drawing (``($(O)!8!(A)$)``,
    ``+(0,15)``). Those still expand the standalone bounding box, shrinking the
    visible figure to a dot. Marking them ``overlay`` removes them from the box
    without changing the geometry (the named intersection is still computed).
    """
    return re.sub(r"\\path\[name path=", r"\\path[overlay,name path=", tikz)


def compile_construction(tikz: str, out_pdf: Path, timeout: int = 90) -> tuple[bool, str]:
    """Compile ONE construction figure to a standalone vector PDF at ``out_pdf``.

    Returns (ok, log). Failure is non-fatal to the caller so one bad figure never
    sinks the whole worksheet.
    """
    if not tex.has_tectonic():
        return False, "tectonic-not-installed"
    tmp = Path(tempfile.mkdtemp(prefix="wksheet_fig_"))
    try:
        tex_path = tmp / "fig.tex"
        tex_path.write_text(FIG_TEMPLATE.replace("__BODY__", tikz))
        try:
            proc = subprocess.run(
                ["tectonic", "-X", "compile", "--outfmt", "pdf", "-o", str(tmp), str(tex_path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except FileNotFoundError:
            return False, "tectonic-not-installed"
        pdf = tmp / "fig.pdf"
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0 and pdf.exists():
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, out_pdf)
            return True, log
        return False, log
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> bool:
    """Rasterise the first page of a PDF to PNG (for quick viewing / previews)."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        page.get_pixmap(dpi=dpi).save(str(png_path))
        doc.close()
        return True
    except Exception:  # noqa: BLE001 - preview render is best-effort
        return False


# --------------------------------------------------------------------------- #
# Problem model
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    num: int                       # 1-based problem number on the worksheet
    source: str                    # "generator" | "olympiad"
    topic: str                     # construction tag, e.g. "circumcenter"
    title: str                     # human-friendly heading
    statement: str                 # student-facing prose (given configuration)
    question: str                  # what to find (LaTeX-safe, authored here)
    tikz: str                      # GROUND-TRUTH construction figure (coordinate-free)
    points: dict[str, list[float]] # exact GT coordinates
    given: list[str]               # given/base point names
    derived: list[str]             # point names the student must find
    unordered: list[list[str]] | None  # interchangeable point groups (tangents)
    scalar: tuple[str, str] | None     # (label, value) extra asked quantity
    spec_prompt: str               # description fed to the specialist (build_messages)
    grade_atol: float = 0.05
    # filled in later:
    fig_pdf: Path | None = None
    fig_png: Path | None = None
    fig_ok: bool = False
    specialist: dict | None = None


def _num(x: float) -> str:
    """Compact number: integers stay integers, else up to 4 decimals (no -0)."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    v = round(x, 4)
    if v == 0:
        v = 0.0
    return f"{v:g}"


def _dist(p: list[float] | tuple[float, float], q: list[float] | tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _tri_area(a, b, c) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


# --------------------------------------------------------------------------- #
# Olympiad problems (named constructions -> tkz-euclide figures)
# --------------------------------------------------------------------------- #
def _oly_scalar(tag: str, p: dict[str, list[float]]) -> tuple[str, str] | None:
    """A natural extra 'asked quantity' per construction, computed from exact GT."""
    if tag == "circumcenter":
        return ("Circumradius $R$", _num(_dist(p["O"], p["A"])))
    if tag == "incenter":
        a, b, c = p["A"], p["B"], p["C"]
        area = _tri_area(a, b, c)
        s = (_dist(b, c) + _dist(c, a) + _dist(a, b)) / 2.0
        return ("Inradius $r$", _num(area / s) if s else "0")
    if tag == "angle_bisector":
        return ("Length $AD$", _num(_dist(p["A"], p["D"])))
    if tag == "foot_altitude":
        return ("Altitude length $AF$", _num(_dist(p["A"], p["F"])))
    if tag == "median":
        return ("Median length $AM$", _num(_dist(p["A"], p["M"])))
    if tag == "tangent":
        return ("Tangent length $PT$", _num(_dist(p["P"], p["T1"])))
    return None  # orthocenter / centroid: coordinates only


_OLY_TITLE = {
    "circumcenter": "Circumcenter \\& circumcircle",
    "incenter": "Incenter \\& incircle",
    "orthocenter": "Orthocenter",
    "centroid": "Centroid",
    "angle_bisector": "Angle bisector",
    "foot_altitude": "Foot of an altitude",
    "median": "Median",
    "tangent": "Tangents from an external point",
}

_OLY_FIND = {
    "circumcenter": "the coordinates of the circumcenter $O$",
    "incenter": "the coordinates of the incenter $I$",
    "orthocenter": "the coordinates of the orthocenter $H$",
    "centroid": "the coordinates of the centroid $G$",
    "angle_bisector": "the coordinates of the point $D$ where the bisector of $\\angle A$ meets $BC$",
    "foot_altitude": "the coordinates of the foot $F$ of the altitude from $A$",
    "median": "the coordinates of the midpoint $M$ of $BC$",
    "tangent": "the coordinates of both points of tangency $T_1$ and $T_2$",
}


def build_olympiad_problem(rng: random.Random, tag: str, num: int) -> Problem:
    prob = olympiad.make_problem(rng, tag)
    points = prob["points"]
    derived = prob["derived"]
    given = [n for n in points if n not in derived]
    # Student statement = the geometric setup, minus the model-facing "Output a
    # single TikZ figure ..." instruction sentence.
    statement = prob["description"].split("Output a single TikZ figure")[0].strip()
    scalar = _oly_scalar(tag, points)
    q = f"Determine {_OLY_FIND[tag]}"
    if scalar:
        q += f", and compute the {scalar[0].split('$')[0].strip().lower() or 'quantity'} "
        q += f"(${scalar[0].split('$')[1]}$)."
    else:
        q += "."
    return Problem(
        num=num, source="olympiad", topic=tag, title=_OLY_TITLE[tag],
        statement=statement, question=q,
        tikz=prob["tikz"], points=points, given=given, derived=derived,
        unordered=prob.get("unordered"), scalar=scalar,
        spec_prompt=prob["description"],
    )


# --------------------------------------------------------------------------- #
# Generator problems (coordinate-geometry constructions -> symbolic PGF figures)
# --------------------------------------------------------------------------- #
_OP_TITLE = {
    "point_on_circle": "Points on a circle",
    "reflect_x": "Reflection across the x-axis",
    "reflect_y": "Reflection across the y-axis",
    "midpoint": "Midpoint",
    "intersection": "Line intersection",
    "foot_altitude": "Foot of a perpendicular",
}


def _generator_title(tags: list[str]) -> str:
    """Name the problem after its most interesting (hardest) operation."""
    for hard in ("foot_altitude", "intersection", "reflect_x", "reflect_y", "midpoint"):
        if hard in tags:
            return _OP_TITLE[hard]
    return "Coordinate construction"


def build_generator_problem(rng: random.Random, chain: int, irregular: bool,
                            num: int, force_op: str | None = None) -> Problem:
    # symbolic=True => coordinate-free PGF construction figure (per the spec).
    ex = generator.make_example(rng, chain, irregular, force_op=force_op, symbolic=True)
    points = ex["points"]
    names = list(points.keys())
    statement = ex["description"]
    q = ("Determine the exact coordinates of every labelled point: "
         f"${', '.join(names)}$.")
    return Problem(
        num=num, source="generator", topic="+".join(ex["tags"]),
        title=_generator_title(ex["tags"]),
        statement=statement, question=q,
        tikz=ex["tikz"], points=points, given=["A"],
        derived=[n for n in names if n != "A"], unordered=None, scalar=None,
        spec_prompt=ex["description"],
    )


# --------------------------------------------------------------------------- #
# Problem sampling
# --------------------------------------------------------------------------- #
_GEN_RECIPE = [  # (force_op, min_chain) — a spread of classroom constructions
    (None, 3), ("foot_altitude", 4), ("intersection", 3), (None, 3),
    ("foot_altitude", 4), ("intersection", 3), (None, 4), ("foot_altitude", 5),
]


def _min_pair_sep(points: dict[str, list[float]]) -> float:
    """Smallest distance between any two named points (a legibility proxy)."""
    pts = list(points.values())
    if len(pts) < 2:
        return 0.0
    return min(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
               for i in range(len(pts)) for j in range(i + 1, len(pts)))


def _pick_legible(build_one, tries: int = 16) -> Problem:
    """Best-of-K sampling: prefer candidates with MORE labelled points (so the
    problem isn't degenerate), then the most spread-out ones (so labels don't
    collide and the printed figure stays readable)."""
    best: Problem | None = None
    best_score = (-1, -1.0)
    for _ in range(tries):
        cand = build_one()
        # cap separation so it can't dominate the point-count preference
        score = (len(cand.points), min(_min_pair_sep(cand.points), 50.0))
        if score > best_score:
            best, best_score = cand, score
    assert best is not None
    return best


def sample_problems(source: str, topics: list[str] | None, n: int, seed: int,
                    chain_min: int, chain_max: int, irregular_mode: str) -> list[Problem]:
    rng = random.Random(seed)

    def irr() -> bool:
        return {"on": True, "off": False}.get(irregular_mode, rng.random() < 0.5)

    def gen_one(num: int) -> Problem:
        force_op, min_c = _GEN_RECIPE[(num - 1) % len(_GEN_RECIPE)]
        chain = max(min_c, rng.randint(chain_min, chain_max))
        irregular = irr()
        return _pick_legible(
            lambda: build_generator_problem(rng, chain, irregular, num, force_op=force_op))

    def oly_one(num: int, tag: str) -> Problem:
        return _pick_legible(lambda: build_olympiad_problem(rng, tag, num))

    def oly_types() -> list[str]:
        return topics or olympiad.TYPES

    problems: list[Problem] = []
    if source == "generator":
        for i in range(n):
            problems.append(gen_one(i + 1))
    elif source == "olympiad":
        types = oly_types()
        for i in range(n):
            problems.append(oly_one(i + 1, types[i % len(types)]))
    elif source == "mixed":
        types = oly_types()
        for i in range(n):
            if i % 2 == 0:
                problems.append(oly_one(i + 1, types[(i // 2) % len(types)]))
            else:
                problems.append(gen_one(i + 1))
    else:
        raise ValueError(f"unknown source: {source}")
    return problems


# --------------------------------------------------------------------------- #
# Specialist (optional): base Qwen3-0.6B + LoRA qwen3-pgf-geotikz.
# Uses infer.generate -> geotikz.prompts.build_messages (the TRAINING prompt).
# --------------------------------------------------------------------------- #
def summarize_specialist(problems: list[Problem], base_model: str, adapter: str) -> dict:
    """Aggregate whatever specialist results are attached to ``problems``."""
    scored = [p for p in problems if p.specialist is not None]
    if not scored:
        return {"ran": False, "reason": "no specialist results"}
    per_source: dict[str, list[int]] = {}
    n_ok = acc = 0.0
    for p in scored:
        r = int(bool(p.specialist["reproduced"]))
        n_ok += r
        acc += float(p.specialist["coord_accuracy"])
        b = per_source.setdefault(p.source, [0, 0])
        b[0] += r
        b[1] += 1
    n = len(scored)
    return {
        "ran": True, "base_model": base_model, "adapter": adapter,
        "n": n, "reproduced": int(n_ok),
        "match_rate": round(n_ok / n, 3),
        "mean_coord_accuracy": round(acc / n, 3),
        "per_source": {s: {"reproduced": v[0], "n": v[1], "match_rate": round(v[0] / v[1], 3)}
                       for s, v in per_source.items()},
    }


def save_specialist_cache(path: Path, problems: list[Problem], seed: int, source: str) -> None:
    data = {
        "seed": seed, "source": source,
        "results": {str(p.num): p.specialist for p in problems if p.specialist is not None},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_specialist_cache(path: Path, problems: list[Problem]) -> bool:
    """Attach cached specialist results by problem number. Returns True if any hit."""
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    res = data.get("results", {})
    hit = False
    for p in problems:
        if str(p.num) in res:
            p.specialist = res[str(p.num)]
            hit = True
    return hit


def run_specialist(problems: list[Problem], adapter: str, base_model: str,
                   limit: int | None, cache_path: Path | None = None,
                   seed: int = 0, source: str = "") -> dict:
    """Generate a figure for each problem with the specialist and grade vs GT.

    Never raises: on any failure it returns a summary with ``ran=False`` so the
    worksheet still builds. Results are cached incrementally (per problem) so a
    slow/interrupted local GPU run keeps its partial progress. The figures on the
    worksheet are always the GT constructions; this only measures how faithfully
    the trained model reproduces them (a reliability statistic).
    """
    from geotikz import infer  # local import: only pay the torch cost on demand

    adapter_path = adapter if Path(adapter).exists() else str(ROOT / adapter)
    if not Path(adapter_path).exists():
        return {"ran": False, "reason": f"adapter not found: {adapter}"}

    try:
        print(f"[specialist] loading {base_model} + adapter {adapter_path} ...", flush=True)
        model, tok, device = infer.load_model(base_model, adapter_path)
        print(f"[specialist] device={device}", flush=True)
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "reason": f"load failed: {type(e).__name__}: {e}"}

    targets = problems if limit is None else problems[:limit]
    for i, p in enumerate(targets, 1):
        try:
            out = infer.generate(model, tok, device, p.spec_prompt)
            g = extract.grade(out, p.points, atol=p.grade_atol, unordered=p.unordered)
        except Exception as e:  # noqa: BLE001
            g = {"compiles": False, "coords_all_correct": False,
                 "coord_accuracy": 0.0, "figure_only": False, "reason": str(e)}
            out = ""
        p.specialist = {
            "reproduced": bool(g.get("coords_all_correct")),
            "coord_accuracy": round(float(g.get("coord_accuracy", 0.0)), 3),
            "compiles": bool(g.get("compiles")),
            "figure_only": bool(g.get("figure_only")),
            "output": out,
        }
        print(f"  [{i}/{len(targets)} P{p.num} {p.source}/{p.topic}] "
              f"compiles={p.specialist['compiles']} "
              f"coord_acc={p.specialist['coord_accuracy']:.2f} "
              f"reproduced={p.specialist['reproduced']}", flush=True)
        if cache_path is not None:  # persist after every problem (crash-resilient)
            save_specialist_cache(cache_path, problems, seed, source)

    return summarize_specialist(problems, base_model, adapter_path)


# --------------------------------------------------------------------------- #
# LaTeX assembly (worksheet + answer key), compiled with tectonic.
# Figures are pre-compiled vector PDFs, embedded via \includegraphics so one bad
# figure can never break the whole document.
# --------------------------------------------------------------------------- #
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    out = []
    for ch in s:
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


_DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{tcolorbox}
\definecolor{brand}{RGB}{31,72,133}
\definecolor{soft}{RGB}{244,247,251}
\setlength{\parindent}{0pt}
\graphicspath{{figures/}}
"""


def _title_block(title: str, subtitle: str, kind: str) -> str:
    """Product-y header: title, subtitle, and a Name/Date/Score line."""
    tag = "ANSWER KEY" if kind == "key" else "WORKSHEET"
    info = ("" if kind == "key" else
            r"""\vspace{2pt}
{\color{brand!80}\small Name: \underline{\hspace{5.5cm}}\qquad
Date: \underline{\hspace{3cm}}\qquad Score: \underline{\hspace{1.6cm}}}
""")
    return (
        r"\begin{center}" "\n"
        r"{\color{brand}\Large\textbf{" + tex_escape(title) + r"}}\\[2pt]" "\n"
        r"{\color{brand!70}\small\textsc{" + tex_escape(subtitle) + r"} \;\textbullet\; " + tag + r"}" "\n"
        r"\end{center}" "\n"
        r"{\color{brand!30}\hrule height 1.2pt}" "\n" + info + r"\vspace{6pt}" "\n"
    )


def _fig_include(p: Problem, width: str) -> str:
    if p.fig_ok and p.fig_pdf is not None:
        return (r"\begin{center}\includegraphics[width=" + width +
                r",height=6.5cm,keepaspectratio]{" + p.fig_pdf.name + r"}\end{center}")
    return r"\begin{center}\fbox{\parbox{0.6\linewidth}{\centering\itshape figure unavailable}}\end{center}"


def _coords_math(p: Problem, names: list[str]) -> str:
    parts = []
    for nseq in names:
        xy = p.points.get(nseq)
        if xy is None:
            continue
        parts.append(f"{nseq} = ({_num(xy[0])},\\ {_num(xy[1])})")
    return r"\quad ".join(parts)


def build_worksheet_tex(problems: list[Problem], title: str, subtitle: str,
                        note: str) -> str:
    body = [_DOC_PREAMBLE, r"\begin{document}", _title_block(title, subtitle, "worksheet")]
    for p in problems:
        head = (r"\vspace{4pt}{\color{brand}\large\textbf{Problem " + str(p.num) +
                r".}} \textbf{" + p.title + r"}\\[2pt]")
        body.append(head)
        body.append(tex_escape(p.statement))
        body.append(r"\\[3pt]\textit{" + p.question + r"}")
        body.append(_fig_include(p, "0.62\\linewidth"))
        body.append(r"\vspace{2pt}{\color{brand!30}\hrule}")
    if note:
        body.append(r"\vfill{\color{brand!60}\footnotesize " + tex_escape(note) + r"}")
    body.append(r"\end{document}")
    return "\n".join(body)


def build_answer_key_tex(problems: list[Problem], title: str, subtitle: str,
                         specialist: dict | None) -> str:
    body = [_DOC_PREAMBLE, r"\begin{document}", _title_block(title, subtitle, "key")]
    for p in problems:
        body.append(r"\vspace{3pt}{\color{brand}\large\textbf{" + str(p.num) +
                    r".}} \textbf{" + p.title + r"}")
        # Given + answer coordinates, then the asked scalar.
        if p.given:
            body.append(r"\\[1pt]{\small\color{black!60}Given: $" +
                        _coords_math(p, p.given) + r"$}")
        ans = _coords_math(p, p.derived) if p.derived else _coords_math(p, list(p.points))
        body.append(r"\\[1pt]\textbf{Answer:} $" + ans + r"$")
        if p.scalar:
            body.append(r"\\[1pt]" + p.scalar[0] + r"\ $= " + p.scalar[1] + r"$")
        if p.specialist is not None:
            mark = "reproduced GT" if p.specialist["reproduced"] else "did not match GT"
            body.append(r"\\[1pt]{\footnotesize\color{brand!70}Specialist: " + mark +
                        r" (coord acc " + f"{p.specialist['coord_accuracy']:.2f}" + r")}")
        body.append(_fig_include(p, "0.4\\linewidth"))
        body.append(r"\vspace{1pt}{\color{brand!25}\hrule}")
    if specialist and specialist.get("ran"):
        line = (f"Specialist reliability (Qwen3-0.6B + {Path(specialist['adapter']).name}): "
                f"reproduced {specialist['reproduced']}/{specialist['n']} figures exactly "
                f"({specialist['match_rate']:.0%}); mean coordinate accuracy "
                f"{specialist['mean_coord_accuracy']:.0%}.")
        body.append(r"\vfill{\color{brand!70}\footnotesize " + tex_escape(line) + r"}")
    body.append(r"\end{document}")
    return "\n".join(body)


def compile_document(tex_source: str, work_dir: Path, stem: str,
                     timeout: int = 180) -> tuple[bool, Path | None, str]:
    """Write ``stem.tex`` into work_dir (so figures/ resolves) and compile it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    tex_path = work_dir / f"{stem}.tex"
    tex_path.write_text(tex_source)
    if not tex.has_tectonic():
        return False, None, "tectonic-not-installed"
    try:
        proc = subprocess.run(
            ["tectonic", "-X", "compile", "--outfmt", "pdf", "-o", str(work_dir), str(tex_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    pdf = work_dir / f"{stem}.pdf"
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode == 0 and pdf.exists():
        return True, pdf, log
    return False, None, log


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Classroom geometry worksheet generator")
    ap.add_argument("--source", choices=["generator", "olympiad", "mixed"], default="generator")
    ap.add_argument("--topics", nargs="+", default=None,
                    help="olympiad construction types (default: all)")
    ap.add_argument("--n", type=int, default=8, help="number of problems")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chain-min", type=int, default=2, help="generator: min chain length")
    ap.add_argument("--chain-max", type=int, default=4, help="generator: max chain length")
    ap.add_argument("--irregular", choices=["mix", "on", "off"], default="mix",
                    help="generator: round vs non-round numbers")
    ap.add_argument("--title", type=str, default="Geometry Constructions")
    ap.add_argument("--out-dir", type=str, default="outputs/worksheets")
    ap.add_argument("--dpi", type=int, default=200, help="PNG preview resolution")
    ap.add_argument("--specialist", action="store_true",
                    help="also run the trained specialist and report reliability vs GT")
    ap.add_argument("--adapter", type=str, default="outputs/qwen3-pgf-geotikz")
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--specialist-n", type=int, default=None,
                    help="limit specialist to the first K problems (default: all)")
    ap.add_argument("--specialist-cache", type=str, default=None,
                    help="JSON cache of specialist results; loaded if present "
                         "(skips inference), else written after a run")
    args = ap.parse_args()

    if args.topics:
        bad = [t for t in args.topics if t not in olympiad.TYPES]
        if bad:
            ap.error(f"unknown topics {bad}; choose from {olympiad.TYPES}")

    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not tex.has_tectonic():
        print("ERROR: tectonic is not installed; cannot compile figures/worksheet.")
        sys.exit(1)

    # 1) sample in-vocab problems --------------------------------------------
    print(f"sampling {args.n} problems (source={args.source}) ...")
    problems = sample_problems(
        args.source, args.topics, args.n, args.seed,
        args.chain_min, args.chain_max, args.irregular,
    )

    # 2) compile each GROUND-TRUTH construction figure -> vector PDF + PNG ----
    print("compiling ground-truth construction figures ...")
    n_fig_ok = 0
    for p in problems:
        p.fig_pdf = fig_dir / f"fig_{p.num:02d}.pdf"
        p.fig_png = fig_dir / f"fig_{p.num:02d}.png"
        fig_src = prep_figure(p.tikz)  # coordinate-free construction, box-cleaned
        ok, log = compile_construction(fig_src, p.fig_pdf)
        p.fig_ok = ok
        if ok:
            pdf_to_png(p.fig_pdf, p.fig_png, dpi=args.dpi)
            n_fig_ok += 1
        else:
            p.fig_pdf = None
            print(f"  WARN P{p.num} ({p.source}/{p.topic}) figure failed: "
                  f"{log.strip().splitlines()[-1] if log.strip() else 'unknown'}")
        # persist the editable construction source alongside the render
        (fig_dir / f"fig_{p.num:02d}.tikz").write_text(fig_src)
    print(f"  figures compiled: {n_fig_ok}/{len(problems)}")

    # 3) optional specialist reliability -------------------------------------
    specialist = None
    cache_path: Path | None = None
    if args.specialist_cache:
        cache_path = (Path(args.specialist_cache) if Path(args.specialist_cache).is_absolute()
                      else ROOT / args.specialist_cache)
    adapter_resolved = (args.adapter if Path(args.adapter).exists()
                        else str(ROOT / args.adapter))
    if cache_path is not None and cache_path.exists() and load_specialist_cache(cache_path, problems):
        specialist = summarize_specialist(problems, args.base_model, adapter_resolved)
        print(f"[specialist] loaded cached results from {cache_path}")
    elif args.specialist:
        specialist = run_specialist(problems, args.adapter, args.base_model, args.specialist_n,
                                    cache_path=cache_path, seed=args.seed, source=args.source)
    if specialist is not None:
        if specialist.get("ran"):
            print(f"[specialist] match_rate={specialist['match_rate']:.0%} "
                  f"({specialist['reproduced']}/{specialist['n']}), "
                  f"mean_coord_acc={specialist['mean_coord_accuracy']:.0%}")
        else:
            print(f"[specialist] not run: {specialist.get('reason')}")

    # 4) assemble worksheet + answer key -------------------------------------
    subtitle = {
        "generator": "Coordinate-plane constructions",
        "olympiad": "Triangle & circle constructions",
        "mixed": "Mixed geometry practice",
    }[args.source]
    note = ("Figures are coordinate-free constructions (tkz-euclide / PGF), "
            "compiled locally with tectonic. Generated by the geotikz worksheet generator.")

    print("assembling worksheet.pdf ...")
    ws_ok, ws_pdf, ws_log = compile_document(
        build_worksheet_tex(problems, args.title, subtitle, note), out_dir, "worksheet")
    if not ws_ok:
        print("ERROR: worksheet failed to compile:\n" + ws_log[-1500:])

    print("assembling answer_key.pdf ...")
    ak_ok, ak_pdf, ak_log = compile_document(
        build_answer_key_tex(problems, args.title, subtitle, specialist), out_dir, "answer_key")
    if not ak_ok:
        print("ERROR: answer key failed to compile:\n" + ak_log[-1500:])

    # 5) persist metadata (problems + specialist results) --------------------
    meta = {
        "title": args.title, "source": args.source, "n": len(problems),
        "seed": args.seed, "figures_ok": n_fig_ok,
        "worksheet_pdf": str(ws_pdf) if ws_ok else None,
        "answer_key_pdf": str(ak_pdf) if ak_ok else None,
        "specialist": specialist,
        "problems": [
            {
                "num": p.num, "source": p.source, "topic": p.topic, "title": p.title,
                "statement": p.statement, "points": p.points, "derived": p.derived,
                "scalar": p.scalar, "figure_ok": p.fig_ok,
                "specialist": (None if p.specialist is None
                               else {k: v for k, v in p.specialist.items() if k != "output"}),
            }
            for p in problems
        ],
    }
    meta_path = out_dir / "worksheet_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # 6) report --------------------------------------------------------------
    def size(pth: Path | None) -> int:
        return pth.stat().st_size if pth and pth.exists() else 0

    print("\n" + "=" * 68)
    print(f"WORKSHEET: {args.title}  ({args.source}, {len(problems)} problems)")
    print(f"  figures compiled : {n_fig_ok}/{len(problems)}")
    print(f"  worksheet.pdf    : {ws_pdf}  ({size(ws_pdf):,} bytes)" if ws_ok else
          "  worksheet.pdf    : FAILED")
    print(f"  answer_key.pdf   : {ak_pdf}  ({size(ak_pdf):,} bytes)" if ak_ok else
          "  answer_key.pdf   : FAILED")
    print(f"  figures dir      : {fig_dir}")
    print(f"  metadata         : {meta_path}")
    if specialist and specialist.get("ran"):
        print(f"  specialist       : reproduced {specialist['reproduced']}/{specialist['n']} "
              f"({specialist['match_rate']:.0%}); mean coord acc "
              f"{specialist['mean_coord_accuracy']:.0%}")
        for s, v in specialist["per_source"].items():
            print(f"      - {s:10s}: {v['reproduced']}/{v['n']} ({v['match_rate']:.0%})")
    print("=" * 68)

    if not (ws_ok and ak_ok):
        sys.exit(2)


if __name__ == "__main__":
    main()
