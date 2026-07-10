"""Compile-extract grader for olympiad geometry (tkz-euclide capable).

The v1/v2 grader (``metrics.parse_named_coords``) statically parses calc /
``name intersections`` TikZ. Olympiad constructions (circumcenter, incenter,
orthocenter, tangents, ...) are far more naturally expressed with tkz-euclide
macros (``\\tkzCircumCenter``, ``\\tkzInCenter``, ...) whose coordinates we
CANNOT recover statically. Instead we let TeX place the points and read the
truth back out:

  * wrap the (model or ground-truth) figure with tkz-euclide + a write stream,
  * inside the picture, measure three reference points (0,0),(1,0),(0,1) and each
    requested named point via ``\\path let ... in \\pgfextra{\\immediate\\write}``,
  * normalize out the (possibly scaled / translated) drawing transform,
  * return ``{name: (x,y)}`` in scene units, or ``None`` per undefined point.

tectonic SUPPRESSES ``\\typeout`` but honors ``\\openout`` file writes, which is
why this works where a log-scrape would not. Undefined points are guarded with
``\\@ifundefined`` so one missing name never aborts extraction of the rest.

This is ADDITIVE: it does not touch ``tex.compile_tikz`` or the v1/v2 grader.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import metrics, tex

COORDS_FILE = "geocoords.txt"

# Standalone doc that (a) supports tkz-euclide + the usual tikz libraries and
# (b) exposes \gtextractpt{name}: a guarded "write this point's canvas coords"
# helper defined in the preamble where @ is a letter. __BODY__ (not %s: the
# template itself is full of literal TeX '%' comments) is the injected picture.
EXTRACT_TEMPLATE = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{tkz-euclide}
\usetikzlibrary{calc,angles,quotes,intersections,through,positioning,arrows.meta,decorations.markings}
\newwrite\cf
\makeatletter
\newcommand{\gtextractpt}[1]{%
  \@ifundefined{pgf@sh@ns@#1}%
    {\immediate\write\cf{#1 undef}}%
    {\path let \p1=(#1) in \pgfextra{\immediate\write\cf{#1 \x1 \y1}};}%
}
\makeatother
\immediate\openout\cf=geocoords.txt
\begin{document}
__BODY__
\immediate\closeout\cf
\end{document}
"""

_BEGIN = r"\begin{tikzpicture}"
_END = r"\end{tikzpicture}"
# Reference triangle written first: lets us cancel scale + translation (and even
# independent x/y unit scaling) at read time. Rotation/shear are not handled
# (pathological for these figures) and surface as a normalization failure.
_REF_LINE = (
    r"\path let \p1=(0,0),\p2=(1,0),\p3=(0,1) in "
    r"\pgfextra{\immediate\write\cf{REF \x1 \y1 \x2 \y2 \x3 \y3}};"
)
_NUM = re.compile(r"-?\d*\.?\d+")
# point names we are willing to inject verbatim into TeX (single tokens only)
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


@dataclass
class ExtractResult:
    """Outcome of one compile-extract pass over a figure."""

    compiled: bool  # figure body executed to the end (REF line was written)
    reason: str  # tectonic exit / failure reason (for debugging)
    coords: dict[str, tuple[float, float] | None] = field(default_factory=dict)


def _inject(tikz: str, names: list[str]) -> str:
    """Insert the REF measurement + one guarded write per name before \\end{...}."""
    idx = tikz.rfind(_END)
    head, tail = (tikz[:idx], tikz[idx:]) if idx != -1 else (tikz, "")
    calls = "".join(f"\\gtextractpt{{{n}}}" for n in names if _SAFE_NAME.match(n))
    return f"{head}\n{_REF_LINE}\n{calls}\n{tail}"


def _parse_pt(tok: str) -> float | None:
    m = _NUM.search(tok)
    return float(m.group()) if m else None


def parse_coords_file(text: str) -> tuple[list[float] | None, dict[str, tuple[float, float] | None]]:
    """Parse geocoords.txt -> (ref[x1,y1,x2,y2,x3,y3] | None, {name: (x,y)|None})."""
    ref: list[float] | None = None
    raw: dict[str, tuple[float, float] | None] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "REF" and len(parts) >= 7:
            vals = [_parse_pt(p) for p in parts[1:7]]
            if all(v is not None for v in vals):
                ref = vals  # type: ignore[assignment]
        elif len(parts) == 2 and parts[1] == "undef":
            raw[parts[0]] = None
        elif len(parts) >= 3:
            x, y = _parse_pt(parts[1]), _parse_pt(parts[2])
            raw[parts[0]] = (x, y) if x is not None and y is not None else None
    return ref, raw


def _normalize(
    ref: list[float], raw: dict[str, tuple[float, float] | None]
) -> dict[str, tuple[float, float] | None] | None:
    """Map canvas pt -> scene units: x=(px-x1)/(x2-x1), y=(py-y1)/(y3-y1)."""
    x1, y1, x2, y2, x3, y3 = ref
    ux, uy = x2 - x1, y3 - y1
    if abs(ux) < 1e-6 or abs(uy) < 1e-6:
        return None  # degenerate / rotated reference frame
    out: dict[str, tuple[float, float] | None] = {}
    for name, val in raw.items():
        if val is None:
            out[name] = None
        else:
            px, py = val
            out[name] = ((px - x1) / ux, (py - y1) / uy)
    return out


def _run_tectonic(full_tex: str, timeout: int) -> tuple[str | None, str]:
    """Compile a full standalone doc; return (geocoords.txt text | None, reason)."""
    if not tex.has_tectonic():
        return None, "tectonic-not-installed"
    tmp = Path(tempfile.mkdtemp(prefix="geoextract_"))
    try:
        tex_path = tmp / "fig.tex"
        tex_path.write_text(full_tex)
        try:
            proc = subprocess.run(
                ["tectonic", "-X", "compile", "--outfmt", "pdf", "-o", str(tmp), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except FileNotFoundError:
            return None, "tectonic-not-installed"
        cf_path = tmp / COORDS_FILE
        # Read the coords file whenever it exists: tectonic may exit nonzero on an
        # unrelated late warning while the (last) injected writes already happened.
        if cf_path.exists():
            return cf_path.read_text(), f"exit={proc.returncode}"
        return None, f"exit={proc.returncode}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract(tikz: str, point_names, timeout: int = 60) -> ExtractResult:
    """Compile a figure and recover the scene coords of each named point.

    ``compiled`` is True iff the figure body executed to completion (the REF line
    was written), which is our proxy for "this figure compiles". Each name maps
    to its (x,y) in scene units, or None if the point is undefined in the figure
    or the figure failed to compile.
    """
    names = list(point_names)
    none_map: dict[str, tuple[float, float] | None] = {n: None for n in names}
    if not tikz or _BEGIN not in tikz or _END not in tikz:
        return ExtractResult(False, "no-figure", dict(none_map))
    doc = EXTRACT_TEMPLATE.replace("__BODY__", _inject(tikz, names))
    text, reason = _run_tectonic(doc, timeout)
    if text is None:
        return ExtractResult(False, reason, dict(none_map))
    ref, raw = parse_coords_file(text)
    if ref is None:  # never reached the injection -> figure did not compile
        return ExtractResult(False, reason, dict(none_map))
    norm = _normalize(ref, raw) or {}
    return ExtractResult(True, reason, {n: norm.get(n) for n in names})


def extract_named_coords(
    tikz: str, point_names, timeout: int = 60
) -> dict[str, tuple[float, float] | None]:
    """Public shim: {name: (x,y) | None} for the given figure and names."""
    return extract(tikz, point_names, timeout).coords


def _err(pt: tuple[float, float], gt: list[float] | tuple[float, float]) -> float:
    return max(abs(pt[0] - gt[0]), abs(pt[1] - gt[1]))


def _match_unordered(
    group: list[str],
    coords: dict[str, tuple[float, float] | None],
    gt_points: dict[str, list[float]],
    atol: float,
) -> dict[str, dict]:
    """Best assignment of a group's predicted points to its GT points.

    Some constructions produce a SET of interchangeable points (e.g. the two
    tangency points from an external point): which one is "T1" vs "T2" is
    arbitrary. We score the permutation that maximizes hits so a model is not
    penalized for labeling order. Groups are tiny, so brute force is fine.
    """
    import itertools

    gts = [gt_points[n] for n in group]
    best: tuple[int, dict] | None = None
    for perm in itertools.permutations(range(len(group))):
        pp: dict[str, dict] = {}
        hits = 0
        for name, gi in zip(group, perm):
            pt = coords.get(name)
            gt = gts[gi]
            if pt is None:
                pp[name] = {"ok": False, "err": None}
            else:
                e = _err(pt, gt)
                ok = e <= atol
                pp[name] = {"ok": ok, "err": round(e, 4)}
                hits += int(ok)
        if best is None or hits > best[0]:
            best = (hits, pp)
    return best[1] if best else {n: {"ok": False, "err": None} for n in group}


def grade(model_output: str, gt_points: dict[str, list[float]], atol: float = 0.05,
          timeout: int = 60, unordered: list[list[str]] | None = None) -> dict:
    """Grade a raw model output against known ground-truth coordinates.

    ``model_output`` may include prose / markdown fences; the first tikzpicture
    is extracted. Returns the same falsifiable gate the project trains toward:
    ``passed = figure_only AND compiles AND every named coord within atol``.

    ``unordered`` optionally lists groups of interchangeable point names (e.g.
    ``[["T1","T2"]]`` for tangency points): within a group, coords are matched to
    GT as an unordered set so labeling order is not penalized. All other names
    are matched strictly by name (identical to the v1/v2 harness).
    """
    figure_only = metrics.is_figure_only(model_output)
    tikz = metrics.extract_tikz(model_output)
    names = list(gt_points.keys())
    res = extract(tikz or "", names, timeout)

    grouped = {n for g in (unordered or []) for n in g}
    per_point: dict[str, dict] = {}
    for name in names:
        if name in grouped:
            continue
        pt = res.coords.get(name)
        if pt is None:
            per_point[name] = {"ok": False, "err": None}
        else:
            e = _err(pt, gt_points[name])
            per_point[name] = {"ok": e <= atol, "err": round(e, 4)}
    for group in unordered or []:
        per_point.update(_match_unordered(group, res.coords, gt_points, atol))

    hits = sum(1 for name in names if per_point[name]["ok"])
    total = len(names)
    coords_all_correct = hits == total and total > 0
    passed = bool(figure_only and res.compiled and coords_all_correct)
    return {
        "figure_only": figure_only,
        "has_tikz": tikz is not None,
        "compiles": res.compiled,
        "compile_reason": res.reason,
        "coord_accuracy": hits / total if total else 0.0,
        "coords_all_correct": coords_all_correct,
        "matched": hits,
        "total": total,
        "per_point": per_point,
        "passed": passed,
    }
