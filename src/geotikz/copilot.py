"""Importable core for the Geometry Figure Copilot.

This is the shared engine behind BOTH product surfaces:

  * ``scripts/copilot.py``        -- the LOCAL laptop app (frontier-first; can
    optionally borrow the Modal specialist via a toggle / ``--modal-specialist``).
  * ``scripts/copilot_modal.py``  -- the HOSTED website (a GPU-served local
    specialist + the Gradio UI mounted as a Modal web endpoint).

Everything the two surfaces share lives here so there is exactly one copy of the
routing / rendering / attribution logic:

  * TEXT scene  -> the specialist first (an injected ``specialist_fn`` — local
    base+LoRA OR a Modal GPU call), else a frontier model prompted for
    coordinate-free constructions. Escalates to frontier if the specialist's
    figure doesn't compile / is degenerate.
  * IMAGE (screenshot) / PDF -> when the specialist is enabled, a frontier vision
    model *reads* the scene as text, then the text router can send in-vocab
    constructions to the trained illustrator; otherwise frontier vision draws
    TikZ directly.
  * PASTE TikZ  -> render an existing figure as-is and drop straight into the
    edit loop (tweak a figure you already have).
  * EDIT (a follow-up once a figure exists) -> the current TikZ + your
    instruction go to a frontier model, which returns the full revised figure.

Every reply states which model produced it. Non-compiling figures get one
self-repair pass (the model is shown its own error and asked to fix it).

The core is backend-agnostic: it takes an injectable
``specialist_fn(description) -> tikz_text`` so the local app and the cloud app
can each supply their own specialist. Nothing here imports torch — the heavy
model only lives behind ``specialist_fn`` (local ``serve.Specialist`` or a remote
Modal function).
"""

from __future__ import annotations

import base64
import logging
import random
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import gateway, metrics, serve
from .prompts import CONSTRUCTION_SYSTEM_PROMPT

if TYPE_CHECKING:
    import gradio as gr

logger = logging.getLogger("geotikz.copilot")


@dataclass
class RouteResult:
    """Outcome of one route. ``clarify`` => this is a short question / redirect
    (show it in chat, DON'T touch the figure). Otherwise it's a draw attempt:
    ``png`` set means a new figure; ``png`` None means the attempt failed and the
    caller preserves the previous figure while showing ``note``."""

    png: str | None
    tikz: str
    badge: str
    note: str
    clarify: bool = False

# Injected specialist: description -> raw TikZ text (may include prose we strip).
SpecialistFn = Callable[[str], str]
# Attribution shown when the specialist wins. A plain string, or a zero-arg
# callable so the cloud can report the ACTUAL adapter that loaded (e.g. after a
# 4B->1.7B->0.6B fallback) rather than a hard-coded guess.
LabelLike = str | Callable[[], str]

# Vision-capable frontier models (must accept image content via the gateway).
VISION_MODELS = ["openai-group/gpt-5.5", "gemini-group/gemini-3.1-pro", "openai-group/gpt-4o"]
FRONTIER_MODELS = ["openai-group/gpt-5.5", "claude-group/claude-opus-4-8",
                   "gemini-group/gemini-3.1-pro", "openai-group/gpt-4o"]

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "copilot"

# ── TUNABLE ROUTING KNOBS ─────────────────────────────────────────────────────
# The specialist (qwen3-illustrator-4b-v2) is routed by an OP-VOCABULARY rule, not
# a coarse difficulty score. The robust ceiling sweep (outputs/specialist_ceiling_
# robust) found the limit is the OPERATION VOCABULARY, not chain depth: it is
# reliable on any in-vocab op even in long (<=5) chains, and only breaks on
#   (a) out-of-vocab ops / generic transforms,
#   (b) more than a handful of simultaneous derived points, and
#   (c) many-vertex regular polygons.
# So _route_to_specialist() sends a scene to the specialist UNLESS one of those
# three fires. Tune the two structural limits here.
MAX_SPECIALIST_DERIVED = 5        # more than this many derived points -> frontier
MAX_SPECIALIST_POLYGON_SIDES = 6  # regular polygon with more sides -> frontier
# Cheap/fast model for the prompt normalizer + the (rare) intent fallback.
_NORMALIZER_MODEL = "openai-group/gpt-4o"

EDIT_SYSTEM = (
    "You edit TikZ/PGF geometry figures. Given the CURRENT figure and an edit "
    "instruction, return the COMPLETE revised figure. Preferences: keep everything "
    "not mentioned unchanged; apply the requested change (scale/size, color, label "
    "position, label text, add/remove elements); prefer coordinate-free constructions "
    "(tkz-euclide / PGF calc). Output ONLY one tikzpicture, starting with "
    "\\begin{tikzpicture} and ending with \\end{tikzpicture} — no prose, no markdown."
)

# CLARIFY protocol: folded into the generation prompt so we don't pay an extra LLM
# call to triage. The model outputs EITHER a figure OR one `CLARIFY:` line.
_SCENE_SYSTEM = CONSTRUCTION_SYSTEM_PROMPT + (
    "\n\nCOPILOT ADDENDUM (overrides the 'output only the figure' rule for this app): "
    "You output EITHER a figure OR one clarifying line, never both.\n"
    "  - If the scene is clear enough to draw, output the single tikzpicture as instructed "
    "above. ERR TOWARD DRAWING: make reasonable default choices for minor unspecified "
    "details rather than asking.\n"
    "  - Only if the request is GENUINELY ambiguous in a way that changes the figure "
    "(e.g. 'a triangle with a circle': incircle vs circumcircle vs arbitrary), output "
    "EXACTLY one line: 'CLARIFY: <one short question>'.\n"
    "  - If the request is NOT about geometry, output EXACTLY: "
    "'CLARIFY: I draw geometry diagrams. Describe a geometry scene or problem and I'll illustrate it.'"
)
_EDIT_SYSTEM_CLARIFY = EDIT_SYSTEM + (
    "\n\nCOPILOT ADDENDUM: If the edit instruction is clear, output the full revised "
    "tikzpicture as above. Only if it is too vague to apply (e.g. 'change it', 'fix it', "
    "'make it better' with no direction) output EXACTLY one line: "
    "'CLARIFY: <one short question asking what to change>'. Never output both."
)
_VISION_SYSTEM = CONSTRUCTION_SYSTEM_PROMPT + (
    "\n\nCOPILOT ADDENDUM: If the image shows a geometry problem/figure, output the single "
    "tikzpicture as instructed above. If the image is NOT a geometry problem (or is "
    "unreadable), output EXACTLY one line: 'CLARIFY: That image doesn't look like a geometry "
    "problem. Try a clearer screenshot, or describe the scene in words.' Never output both."
)


def _parse_clarify(text: str) -> str | None:
    """Return the clarifying question if the model chose to clarify, else None.

    A response containing a real tikzpicture is a DRAW (returns None). Otherwise,
    an explicit ``CLARIFY:`` line is honored; and any other non-figure response is
    treated as a (generic) clarify so we ask instead of erroring.
    """
    t = (text or "").strip()
    if metrics.extract_tikz(t):
        return None
    idx = t.upper().find("CLARIFY:")
    if idx != -1:
        q = t[idx + len("CLARIFY:"):].strip().strip('"').splitlines()[0].strip()
        return q or "Could you add a little more detail about the geometry?"
    return None  # no figure and no CLARIFY marker -> caller decides (soft failure)


def _retry(fn, *, tries: int = 3, base: float = 0.6):
    """Call ``fn`` with small exponential backoff; re-raise the last error."""
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - transient (network/GPU) -> retry
            last = e
            if i < tries - 1:
                time.sleep(base * (2**i) + random.random() * 0.2)
    raise last  # type: ignore[misc]


def _combine(original: str, answer: str) -> str:
    """Fold a clarifying answer back into the original request."""
    return f"{original.strip()}: {answer.strip()}"


_CLARIFY_BADGE = "*I need one detail to draw this*"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _tikz(text: str) -> str:
    return metrics.extract_tikz(text or "") or (text or "")


def _resolve_label(label: LabelLike) -> str:
    return label() if callable(label) else label


def _attr(inner: str, kind: str = "generated") -> str:
    """Attribution badge, e.g. *generated by `openai-group/gpt-5.5` (frontier)*."""
    return f"*{kind} by {inner}*"


def _append_turn(history, user_content: str, assistant_content: str) -> list[dict]:
    """Append one exchange in Gradio's ``type="messages"`` format (role/content
    dicts). Tuples are removed in Gradio 6 -> passing them makes the Chatbot
    postprocess raise, which flags every output component with an "Error" badge.
    """
    return (history or []) + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _frontier_inner(model: str, reason: str | None = None) -> str:
    """Attribution inner for a frontier reply, optionally explaining the routing,
    e.g. `openai-group/gpt-5.5` (frontier · edited existing figure)."""
    tag = "frontier" + (f" · {reason}" if reason else "")
    return f"`{model}` ({tag})"


# --------------------------------------------------------------------------- #
# intent routing: NEW SCENE vs EDIT (only matters when a figure already exists)
# --------------------------------------------------------------------------- #
_EDIT_VERBS = re.compile(
    r"\b(make|add|remove|delete|move|shift|change|rename|relabel|recolou?r|colou?r|"
    r"scale|resize|enlarge|shrink|rotate|flip|fix|adjust|increase|decrease|highlight|"
    r"erase|swap|replace|update|tweak|thicken|bold|redraw|put|set|turn|hide|show)\b", re.I)
_EDIT_START = re.compile(
    r"^\s*(make|add|remove|delete|move|shift|change|rename|relabel|recolou?r|colou?r|"
    r"scale|resize|enlarge|shrink|rotate|flip|fix|adjust|increase|decrease|highlight|"
    r"erase|swap|replace|update|tweak|thicken|bold|redraw)\b", re.I)
_EDIT_REF = re.compile(
    r"\b(it|its|them|the label|the labels|the point|the points|the line|the lines|"
    r"the circle|the triangle|the square|the polygon|the figure|this figure|current|"
    r"bigger|smaller|larger|thicker|thinner)\b", re.I)
_SHAPE = re.compile(
    r"\b(triangle|quadrilateral|square|rectangle|parallelogram|rhombus|trapezoid|"
    r"trapezium|pentagon|hexagon|heptagon|octagon|polygon|circle|ellipse|segment)\b", re.I)
_CLASSIFY_MODEL = "openai-group/gpt-4o"  # cheap+fast for the rare ambiguous fallback


def _heuristic_intent(message: str) -> str | None:
    """"scene" | "edit" | None (ambiguous). Keyword/shape heuristics, no LLM."""
    m = (message or "").strip()
    ml = m.lower()
    # strong NEW-SCENE signals
    if re.search(r"=\s*\(-?\d", m):                                  # coords A=(0,0)
        return "scene"
    if re.search(r"\b(output|produce|construct|generate)\b", ml) and ("tikz" in ml or "figure" in ml or _SHAPE.search(ml)):
        return "scene"
    if re.search(r"\b(triangle|quadrilateral|square|rectangle|parallelogram|rhombus|"
                 r"trapezoid|pentagon|hexagon|heptagon|octagon|polygon)\s+[A-Z]{2,}", m):  # "triangle ABC"
        return "scene"
    if re.search(r"\bcircle\b.{0,40}\b(cent(er|re|ered)|radius|diameter)\b", ml):
        return "scene"
    if re.search(r"\bregular\s+(pentagon|hexagon|heptagon|octagon|polygon|\w+gon)\b", ml):
        return "scene"
    # strong EDIT signals (checked before the generic shape rule so "make the
    # triangle bigger" edits rather than redraws)
    if _EDIT_START.search(m):
        return "edit"
    if _EDIT_VERBS.search(ml) and _EDIT_REF.search(ml):
        return "edit"
    if _EDIT_REF.search(ml) and not _SHAPE.search(ml):
        return "edit"
    # a shape is named and it wasn't an edit imperative -> lean NEW SCENE
    if _SHAPE.search(ml):
        return "scene"
    return None


def _classify_intent(message: str, model: str | None = None) -> str:
    """Route a message (with a figure already open) to "scene" or "edit".
    Heuristics first; a single cheap LLM call only for genuine ambiguity; biased
    to "scene" on any uncertainty (clear geometry text -> new scene)."""
    h = _heuristic_intent(message)
    if h is not None:
        return h
    try:
        r = gateway.chat(
            [{"role": "system", "content": "In a geometry-figure chat, classify the user's "
              "message. Reply with EXACTLY one word: NEW if it describes a fresh geometry scene "
              "to draw from scratch, or EDIT if it asks to modify the current figure."},
             {"role": "user", "content": message}],
            model or _CLASSIFY_MODEL, max_tokens=8, temperature=0.0, retries=2)
        ans = (r.text or "").strip().lower()
        if "edit" in ans and "new" not in ans:
            return "edit"
    except Exception:  # noqa: BLE001
        pass
    return "scene"


# --------------------------------------------------------------------------- #
# routing: OP-VOCABULARY rule + prompt normalizer (specialist coverage, no retrain)
# --------------------------------------------------------------------------- #
# In-vocab construction ops (used to estimate how many derived points a scene asks
# for). v2 added general affine transforms, nine-point centre, incircle contact,
# square/parallelogram centre, bisector-incenter, midpoint-reflect chains.
_CONSTRUCTIONS = re.compile(
    r"\b(circumcenter|circumcircle|incenter|incircle|orthocenter|centroid|median|"
    r"bisector|altitude|tangent|midpoint|midsegment|medial|orthic|nine[- ]?point|"
    r"euler line|reflection|reflect|rotation|rotate|translation|translate|"
    r"perpendicular|intersection|diagonal|antipode|diameter|contact|foot|angle)\b", re.I)

# Ops OUTSIDE the specialist's vocabulary -> always frontier. (Nine-point centre is
# now IN vocab for v2, so it is deliberately absent here.)
_OUT_OF_VOCAB = re.compile(
    r"\b("
    r"cube|sphere|pyramid|tetrahedron|prism|cylinder|cone|dihedral|octahedron|"
    r"3d|three[- ]dimensional|"                            # solids / 3D
    r"locus|loci|trajectory|envelope|region|"              # loci / regions
    r"ellipse|parabola|hyperbola|conic|"                   # non-circle conics
    r"spiral|fractal|tessellation|lattice|vector field|"   # exotic
    r"radical axis|inversion|shear|glide|homothet|dilation|dilate|"  # transforms out of range
    r"graph of|plot of|\bplot\b|inequalit|\bfunction\b"    # function plots
    r")\b", re.I)

# Concrete affine transforms v2 CAN do. A bare 'transformation' with none of these
# named is a generic/unknown transform -> frontier.
_KNOWN_TRANSFORM = re.compile(r"\b(reflect\w*|rotat\w*|translat\w*|point[- ]reflection)\b", re.I)
_GENERIC_TRANSFORM = re.compile(r"\btransformations?\b", re.I)

# Named many-vertex polygons (heptagon and up) -> frontier (still weak: 5/12).
_BIG_POLY_WORDS = re.compile(
    r"\b(heptagon|septagon|octagon|nonagon|enneagon|decagon|hendecagon|undecagon|dodecagon)\b", re.I)


def _big_polygon(dl: str) -> bool:
    """True for a regular polygon with more than MAX_SPECIALIST_POLYGON_SIDES sides."""
    if _BIG_POLY_WORDS.search(dl):
        return True
    for m in re.finditer(r"(\d+)\s*[- ]?gon\b|(\d+)\s*[- ]?sided\b|"
                         r"regular\s+polygon[^.]*?\b(\d+)\s+sides", dl):
        for g in m.groups():
            if g and int(g) > MAX_SPECIALIST_POLYGON_SIDES:
                return True
    return False


def _derived_count(description: str) -> int:
    """Estimate the number of DERIVED (constructed) points a scene asks for.

    Counts distinct point names introduced as constructions — bound by 'let X',
    'X be …', 'X the …', or listed in an enumeration — minus base points that are
    given explicit coordinates ('A=(0,0)'). Falls back to the count of distinct
    construction ops when no names are found. (A long *chain* of in-vocab ops is
    fine; this only fires when many points are requested at once.)"""
    d = description or ""
    base = set(re.findall(r"\b([A-Z]\d?)\s*(?:=|\bat\b)\s*\(", d))     # base pts w/ coords
    cand = set(re.findall(r"\b([A-Z]\d?)\s+(?:be|the)\b", d))          # "M be", "O the"
    cand |= set(re.findall(r"\blet\s+([A-Z]\d?)\b", d, re.I))          # "let O"
    cand |= set(re.findall(r"\b([A-Z]\d?)\s*,\s*(?=[A-Z])", d))        # "P,Q,R," lists
    derived = cand - base
    if derived:
        return len(derived)
    return len(set(m.lower() for m in _CONSTRUCTIONS.findall(d.lower())))


def _route_to_specialist(description: str) -> tuple[bool, str]:
    """OP-VOCABULARY routing. Returns (use_specialist, reason_for_frontier).

    Route to the specialist UNLESS the request needs something it reliably fails:
    an out-of-vocab op / generic transform, more than MAX_SPECIALIST_DERIVED
    derived points, or a many-vertex regular polygon. Chain LENGTH is NOT a reason
    to bail — the ceiling sweep found long in-vocab chains reliable. (We can't
    tell a 5-step chain from 5 *simultaneous* derived points in free text, so we
    keep <=MAX_SPECIALIST_DERIVED local and send more to the frontier.)
    """
    dl = (description or "").lower()
    if _OUT_OF_VOCAB.search(dl):
        return False, "out-of-vocab op"
    if _GENERIC_TRANSFORM.search(dl) and not _KNOWN_TRANSFORM.search(dl):
        return False, "generic transform"
    if _big_polygon(dl):
        return False, "many-vertex polygon"
    n = _derived_count(description)  # NB: original case (point-name detection)
    if n > MAX_SPECIALIST_DERIVED:
        return False, f"{n} derived points"
    return True, ""


def _looks_templated(description: str) -> bool:
    """True if the request is already in (or close to) the specialist's trained
    template (base coords + 'output a single tikz figure … define …'), so we can
    skip the normalizer call."""
    d = description or ""
    dl = d.lower()
    return bool(re.search(r"=\s*\(-?\d", d)) and (
        "output a single tikz figure" in dl or "at their correct positions" in dl)


_NORMALIZE_SYSTEM = (
    "You rewrite a user's free-form geometry request into a STRICT template for a small "
    "figure model. Assign concrete small integer coordinates to the base points, name the "
    "single derived construction, and phrase it EXACTLY like this example:\n"
    "'Triangle ABC has vertices A=(0,0), B=(6,0), C=(1,4). Let O be the circumcenter of "
    "triangle ABC. Output a single TikZ figure that draws triangle ABC and its circumcircle, "
    "and defines the named points A, B, C, O at their correct positions.'\n"
    "Rules: keep to ONE construction; use the base shape the user asked for (triangle, "
    "quadrilateral, circle, square, regular polygon, two circles, segment…) with explicit "
    "integer coordinates; end with 'Output a single TikZ figure that draws … and defines the "
    "named points … at their correct positions.' Output ONLY the rewritten description — no "
    "preamble, no code."
)


def _normalize_for_specialist(description: str, model: str = _NORMALIZER_MODEL) -> str:
    """Rewrite a free-form request into the specialist's trained template. Returns
    the original description on any failure (never raises)."""
    try:
        r = gateway.chat(
            [{"role": "system", "content": _NORMALIZE_SYSTEM},
             {"role": "user", "content": description}],
            model, max_tokens=400, temperature=0.0, retries=2)
        out = (r.text or "").strip()
        return out or description
    except Exception:  # noqa: BLE001
        return description


def _data_url(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def _render(text: str, stem: str, out_dir: str | Path) -> tuple[serve.RenderResult, str]:
    """Extract TikZ, tidy labels (all models), compile. Fall back if tidy breaks compile.

    This is the single product render gate: every chat / paste / board / repair
    figure should pass through here so label overlap is minimized automatically.
    """
    tikz = _tikz(text)
    if not tikz:
        png = Path(out_dir) / f"{stem}.png"
        return serve.compile_and_render(text, png, dpi=200), ""
    tidied = serve.tidy_labels(tikz)
    png = Path(out_dir) / f"{stem}.png"
    # Prefer compiling the tidied figure; if that fails, keep original geometry/labels.
    # paint_points_last keeps dots above strokes so lines don't poke past markers.
    raw_body = tidied if tidied != tikz else tikz
    body = serve.paint_points_last(raw_body)
    r = serve.compile_and_render(body, png, dpi=200)
    if r.ok:
        return r, body
    if tidied != tikz:
        body2 = serve.paint_points_last(tikz)
        r2 = serve.compile_and_render(body2, png, dpi=200)
        if r2.ok:
            return r2, body2
    return r, body


def _self_repair(tikz: str, reason: str, model: str) -> serve.GenResult:
    """One repair pass: show the model its figure + failure and ask for a fix."""
    msg = [
        {"role": "system", "content": EDIT_SYSTEM},
        {"role": "user", "content": (
            f"This TikZ figure failed to render (reason: {reason}). Fix it so it "
            f"compiles and draws a correct, non-degenerate figure. Return the full "
            f"revised tikzpicture only.\n\n{tikz}")},
    ]
    res = gateway.chat(msg, model, max_tokens=4096)
    return serve.GenResult(res.text, model, res.ok, res.latency_s, res.error, res.finish_reason)


def _pdf_page1_to_png(pdf_path: str, out_dir: str | Path, dpi: int = 150) -> str:
    """Rasterise page 1 of a PDF to a PNG (PyMuPDF) for the vision route."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        png = Path(out_dir) / f"pdfin_{serve.dhash(str(pdf_path) + str(time.time()))}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png))
        return str(png)
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# routes (pure functions -> (png_path|None, tikz, badge_md, note))
# --------------------------------------------------------------------------- #
def generate_text(
    description: str,
    use_specialist: bool,
    frontier_model: str,
    *,
    specialist_fn: SpecialistFn | None = None,
    specialist_label: LabelLike = "the specialist",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    allow_frontier: bool = True,
) -> RouteResult:
    """Text scene -> figure. Specialist first (if enabled), then a clarify-aware
    frontier decision (draw OR ask ONE question). Never raises.

    When ``allow_frontier`` is False, a failed/truncated specialist attempt returns
    without calling the frontier (callers can substitute a demo cache instead).
    """
    description = (description or "").strip()
    if not description:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "Describe a geometry scene (points, shapes, and how they relate) and I'll draw it.",
                           clarify=True)
    stem = serve.dhash(description + str(time.time()))
    prefix = ""
    # Why the frontier is used (shown in the badge).
    spec_reason = "specialist off"

    # 1) specialist route — gated by OP VOCABULARY (not phrasing, not chain depth).
    # If every requested op is in the specialist's range we normalize the free-form
    # request into its trained template and try it; only out-of-vocab ops / generic
    # transforms, too many derived points, or many-vertex polygons skip to frontier.
    if use_specialist and specialist_fn is not None:
        route_local, why = _route_to_specialist(description)
        if not route_local:
            spec_reason = why  # short badge tag, e.g. "out-of-vocab op"
            prefix = {
                "out-of-vocab op": "This construction is outside the specialist's vocabulary, so a frontier model drew it. ",
                "generic transform": "This general transform is outside the specialist's range, so a frontier model drew it. ",
                "many-vertex polygon": "Many-sided regular polygons are still weak for the specialist, so a frontier model drew it. ",
            }.get(why, "This has more constructed points than the specialist handles well, so a frontier model drew it. ")
            if not allow_frontier:
                return RouteResult(
                    None, "", _CLARIFY_BADGE,
                    f"Specialist skipped ({why}).",
                    clarify=False,
                )
        else:
            t0 = time.time()
            norm_desc, normalized = description, False
            if not _looks_templated(description):
                norm_desc = _normalize_for_specialist(description) or description
                normalized = norm_desc.strip() != description.strip()
            spec_tikz = ""
            try:
                spec_tikz = _retry(lambda: specialist_fn(norm_desc) or "", tries=3) or ""
            except Exception as e:  # noqa: BLE001 - specialist down -> frontier
                logger.warning("specialist failed after retries: %s", e)
            if spec_tikz:
                # Shared _render tidies labels for every model; fall back is inside _render.
                r, tikz = _render(spec_tikz, stem, out_dir)
                if r.ok:
                    lbl = _resolve_label(specialist_label)
                    if normalized and lbl.endswith(")"):
                        lbl = lbl[:-1] + " · normalized)"
                    dt = time.time() - t0
                    note = (f"Normalized your request, then the specialist drew it in {dt:.0f}s."
                            if normalized else
                            f"Specialist drew it in {dt:.0f}s (coordinate-free, compiled).")
                    return RouteResult(str(Path(r.png_path)), tikz, _attr(lbl), note)
            spec_reason = "specialist fell back"
            # Prefer an honest reason when the specialist returned incomplete TikZ.
            prefix = "The specialist couldn't draw this one, so a frontier model did. "
            if spec_tikz and "\\begin{tikzpicture}" in spec_tikz and "\\end{tikzpicture}" not in spec_tikz:
                prefix = (
                    "The specialist's figure was cut off before it finished, so a frontier "
                    "model redrew it. "
                )
                spec_reason = "specialist truncated"
            if not allow_frontier:
                why = "truncated" if spec_reason == "specialist truncated" else "failed to compile"
                return RouteResult(
                    None, "", _CLARIFY_BADGE,
                    f"Specialist {why}.",
                    clarify=False,
                )

    if not allow_frontier:
        return RouteResult(
            None, "", _CLARIFY_BADGE,
            "Specialist unavailable.",
            clarify=False,
        )

    # 2) frontier: draw OR clarify (one call).
    messages = [
        {"role": "system", "content": _SCENE_SYSTEM},
        {"role": "user", "content": f"Scene:\n{description}\n\nReturn the TikZ figure, or a CLARIFY line."},
    ]
    res = gateway.chat(messages, frontier_model, max_tokens=4096)  # gateway retries transient 429/5xx/conn
    if not res.ok:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I'm having trouble reaching the drawing model right now. Please try again in a moment.",
                           clarify=True)
    q = _parse_clarify(res.text)
    if q is not None:
        return RouteResult(None, "", _CLARIFY_BADGE, q, clarify=True)

    r, tikz = _render(res.text, stem, out_dir)
    if not r.ok and tikz:  # one self-repair pass
        rep = _self_repair(tikz, r.reason, frontier_model)
        r2, tikz2 = _render(rep.text, stem, out_dir)
        if r2.ok:
            return RouteResult(str(Path(r2.png_path)), tikz2, _attr(_frontier_inner(frontier_model, spec_reason)),
                               prefix + "Frontier drew it (self-repaired one compile error).")
    if r.ok:
        return RouteResult(str(Path(r.png_path)), tikz, _attr(_frontier_inner(frontier_model, spec_reason)),
                           prefix + f"Frontier drew it in {res.latency_s:.0f}s (coordinate-free, compiled).")
    return RouteResult(None, "", _CLARIFY_BADGE,
                       "I couldn't turn that into a clean figure. Could you add a detail: key points, a shape, "
                       "or a relationship (e.g. 'triangle ABC with its circumcircle')?",
                       clarify=True)


_VISION_READ_SYSTEM = (
    "You read geometry problem screenshots. Output ONLY a concise English description of the "
    "geometric configuration to draw (shapes, named points, lengths when given, and constructions "
    "like incenter/incircle, tangents, perpendiculars, midpoints). No TikZ, no solution, no "
    "multiple choice. If the image is not a geometry problem, output EXACTLY one line: "
    "'CLARIFY: That image doesn't look like a geometry problem. Try a clearer screenshot, or "
    "describe the scene in words.'"
)


def _vision_read_scene(image_path: str, vision_model: str, *, source_name: str = "screenshot") -> RouteResult | str:
    """Vision OCR/read -> scene description string, or a clarify RouteResult."""
    try:
        url = _data_url(image_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read %s: %s", source_name, e)
        return RouteResult(None, "", _CLARIFY_BADGE,
                           f"I couldn't read that {source_name}. Try a clear PNG/JPG, or just describe the scene in words.",
                           clarify=True)
    messages = [
        {"role": "system", "content": _VISION_READ_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "Read the geometry problem in this image and describe the "
             "configuration to illustrate."},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]
    res = gateway.chat(messages, vision_model, max_tokens=800)
    if not res.ok:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I'm having trouble reading the image right now. Please try again shortly, "
                           "or describe the scene in words.",
                           clarify=True)
    q = _parse_clarify(res.text)
    if q is not None:
        return RouteResult(None, "", _CLARIFY_BADGE, q, clarify=True)
    desc = (res.text or "").strip()
    if not desc:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           f"I couldn't extract a geometry scene from that {source_name}. "
                           "Try a clearer shot, or describe it in words.",
                           clarify=True)
    return desc


def generate_image(
    image_path: str,
    vision_model: str,
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    source_name: str = "screenshot",
    use_specialist: bool = False,
    specialist_fn: SpecialistFn | None = None,
    specialist_label: LabelLike = "the specialist",
    frontier_model: str | None = None,
    scene_text: str | None = None,
) -> RouteResult:
    """A problem image (screenshot/PDF page) -> figure.

    When ``use_specialist`` and a ``specialist_fn`` are provided, a frontier vision
    model first *reads* the scene as text, then the usual text router can send
    in-vocab constructions to the trained illustrator. If ``scene_text`` is given
    (e.g. a verified demo prompt attached with the screenshot), that text is used
    instead of vision OCR so routing/specialist see clean in-vocab input while the
    UI can still show the image. Otherwise the vision model draws TikZ directly
    (frontier vision). Clarify-aware. Never raises.
    """
    # Specialist path: known scene text OR vision-read -> text router.
    if use_specialist and specialist_fn is not None:
        known = (scene_text or "").strip()
        used_known = bool(known)
        if known:
            read = known
        else:
            read = _vision_read_scene(image_path, vision_model, source_name=source_name)
            if isinstance(read, RouteResult):
                return read
        res = generate_text(
            read,
            True,
            frontier_model or FRONTIER_MODELS[0],
            specialist_fn=specialist_fn,
            specialist_label=specialist_label,
            out_dir=out_dir,
        )
        if res.clarify:
            return res
        # Prefix note so the badge path is clear in chat.
        if res.note and not res.note.lower().startswith(("read", "used")):
            lead = (
                f"Used the verified scene text with the {source_name}, then "
                if used_known else
                f"Read the {source_name}, then "
            )
            note = res.note[0].lower() + res.note[1:] if res.note else res.note
            res = RouteResult(res.png, res.tikz, res.badge, lead + note, clarify=False)
        return res

    inner = f"`{vision_model}` (frontier vision)"
    try:
        url = _data_url(image_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read %s: %s", source_name, e)
        return RouteResult(None, "", _CLARIFY_BADGE,
                           f"I couldn't read that {source_name}. Try a clear PNG/JPG, or just describe the scene in words.",
                           clarify=True)
    messages = [
        {"role": "system", "content": _VISION_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "Read the geometry problem shown in this image and produce a "
             "single TikZ figure that illustrates its configuration, or a CLARIFY line."},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]
    res = gateway.chat(messages, vision_model, max_tokens=4096)
    if not res.ok:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I'm having trouble reading the image right now. Please try again shortly, "
                           "or describe the scene in words.",
                           clarify=True)
    q = _parse_clarify(res.text)
    if q is not None:
        return RouteResult(None, "", _CLARIFY_BADGE, q, clarify=True)
    stem = serve.dhash("img" + str(time.time()))
    r, tikz = _render(res.text, stem, out_dir)
    if not r.ok and tikz:
        rep = _self_repair(tikz, r.reason, vision_model)
        r2, tikz2 = _render(rep.text, stem, out_dir)
        if r2.ok:
            return RouteResult(str(Path(r2.png_path)), tikz2, _attr(inner),
                               f"Read the {source_name} and drew it (self-repaired one error).")
    if r.ok:
        return RouteResult(str(Path(r.png_path)), tikz, _attr(inner),
                           f"Read the {source_name} and drew it in {res.latency_s:.0f}s.")
    return RouteResult(None, "", _CLARIFY_BADGE,
                       f"I read the {source_name} but couldn't draw a clean figure from it. "
                       "Could you describe the key parts in words?",
                       clarify=True)


def generate_pdf(
    pdf_path: str,
    vision_model: str,
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    use_specialist: bool = False,
    specialist_fn: SpecialistFn | None = None,
    specialist_label: LabelLike = "the specialist",
    frontier_model: str | None = None,
    scene_text: str | None = None,
) -> RouteResult:
    """PDF upload -> rasterise page 1 (PyMuPDF) -> image route. Never raises."""
    if not pdf_path:
        return RouteResult(None, "", _CLARIFY_BADGE, "Upload a PDF and I'll read page 1.", clarify=True)
    try:
        png_in = _pdf_page1_to_png(pdf_path, out_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not rasterise pdf: %s", e)
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I couldn't read that PDF (it may be empty or not a valid PDF). "
                           "Try another file, or describe the scene in words.",
                           clarify=True)
    return generate_image(
        png_in, vision_model, out_dir=out_dir, source_name="PDF (page 1)",
        use_specialist=use_specialist, specialist_fn=specialist_fn,
        specialist_label=specialist_label, frontier_model=frontier_model,
        scene_text=scene_text,
    )


def render_pasted(
    tikz_text: str,
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    repair_model: str | None = None,
) -> RouteResult:
    """Render a pasted/imported tikzpicture as-is, then hand it to the edit loop. Never raises."""
    tikz_text = (tikz_text or "").strip()
    if not metrics.extract_tikz(tikz_text):
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "Paste a full figure from \\begin{tikzpicture} to \\end{tikzpicture} and I'll render it.",
                           clarify=True)
    stem = serve.dhash("paste" + str(time.time()))
    r, tikz = _render(tikz_text, stem, out_dir)
    if r.ok:
        return RouteResult(str(Path(r.png_path)), tikz, _attr("you (pasted TikZ)", kind="imported"),
                           "Rendered your figure. Now edit it by chat below.")
    if tikz and repair_model:  # optional: one frontier fix so a near-good paste still lands
        rep = _self_repair(tikz, r.reason, repair_model)
        r2, tikz2 = _render(rep.text, stem, out_dir)
        if r2.ok:
            return RouteResult(str(Path(r2.png_path)), tikz2, _attr(_frontier_inner(repair_model), kind="repaired import"),
                               "Your TikZ needed one fix to compile; rendered.")
    return RouteResult(None, "", _CLARIFY_BADGE,
                       f"That TikZ didn't compile (`{r.reason}`). I kept your current figure. "
                       "Check that it's a complete, self-contained tikzpicture.",
                       clarify=True)


# --------------------------------------------------------------------------- #
def edit_figure(
    current_tikz: str,
    instruction: str,
    edit_model: str,
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> RouteResult:
    """Apply a conversational edit (clarify-aware for vague edits). Never raises."""
    instruction = (instruction or "").strip()
    if not current_tikz:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "There's no figure yet. Describe a scene and I'll draw one first.", clarify=True)
    messages = [
        {"role": "system", "content": _EDIT_SYSTEM_CLARIFY},
        {"role": "user", "content": f"Current figure:\n{current_tikz}\n\n"
         f"Edit instruction: {instruction}\n\nReturn the full revised tikzpicture, or a CLARIFY line."},
    ]
    res = gateway.chat(messages, edit_model, max_tokens=4096)
    if not res.ok:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I'm having trouble reaching the model to apply that edit. Please try again in a moment.",
                           clarify=True)
    q = _parse_clarify(res.text)
    if q is not None:
        return RouteResult(None, "", _CLARIFY_BADGE, q, clarify=True)
    stem = serve.dhash("edit" + str(time.time()))
    r, tikz = _render(res.text, stem, out_dir)
    if not r.ok and tikz:
        rep = _self_repair(tikz, r.reason, edit_model)
        r2, tikz2 = _render(rep.text, stem, out_dir)
        if r2.ok:
            return RouteResult(str(Path(r2.png_path)), tikz2,
                               _attr(_frontier_inner(edit_model, "edited existing figure"), kind="edited"),
                               "Applied the edit (self-repaired one error).")
    if r.ok:
        return RouteResult(str(Path(r.png_path)), tikz,
                           _attr(_frontier_inner(edit_model, "edited existing figure"), kind="edited"),
                           f"Applied the edit in {res.latency_s:.0f}s.")
    return RouteResult(None, "", _CLARIFY_BADGE,
                       "That edit didn't come out cleanly, so I kept your previous figure. "
                       "Could you say a bit more specifically what to change?",
                       clarify=True)


# --------------------------------------------------------------------------- #
# stateless HTTP Basic auth (durable across container restarts)
# --------------------------------------------------------------------------- #
class _BasicAuthMiddleware:
    """Pure-ASGI HTTP Basic auth.

    Gradio's built-in login stores sessions in a per-process in-memory dict
    (``app.tokens``) with a per-process ``cookie_id``, so every existing browser
    cookie 401s the moment the container recycles (scaledown / redeploy / new
    pool container) — which is exactly what broke ``/gradio_api/upload``.

    Basic auth is STATELESS: the browser caches the credentials and re-sends them
    (``Authorization: Basic``) on EVERY request — page, /config, uploads, and the
    SSE queue stream — so it survives restarts with no re-login. Implemented as a
    pure-ASGI middleware (not ``BaseHTTPMiddleware``) so it never buffers Gradio's
    streaming SSE responses.
    """

    def __init__(self, app, username: str, password: str, realm: str = "Geometry Figure Copilot"):
        self.app = app
        self.username = username
        self.password = password
        self.realm = realm

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Figure PNGs use unguessable hashed names; allow unauthenticated GETs so
        # <img> tags work even when the browser omits Basic auth on subresources.
        path = scope.get("path") or ""
        if path.startswith("/api/figures/"):
            await self.app(scope, receive, send)
            return
        raw = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        if raw.startswith("Basic "):
            try:
                user, _, pwd = base64.b64decode(raw[6:]).decode("utf-8").partition(":")
                if secrets.compare_digest(user, self.username) and secrets.compare_digest(pwd, self.password):
                    await self.app(scope, receive, send)
                    return
            except Exception:  # noqa: BLE001 - malformed header -> treat as unauthenticated
                pass
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"www-authenticate", f'Basic realm="{self.realm}"'.encode()),
                (b"content-type", b"text/plain; charset=utf-8"),
            ],
        })
        await send({"type": "http.response.body", "body": b"Not authenticated"})


def add_basic_auth(app, username: str, password: str, realm: str = "Geometry Figure Copilot"):
    """Attach stateless Basic auth to a FastAPI/Starlette app (returns the app)."""
    app.add_middleware(_BasicAuthMiddleware, username=username, password=password, realm=realm)
    return app


# --------------------------------------------------------------------------- #
# Gradio UI (single source of truth for both surfaces)
# --------------------------------------------------------------------------- #
# Chat-style Enter behavior in the message box. Gradio 6.20's built-in handling
# is inconsistent for multi-line values (Enter stops submitting once the text has
# a newline, and Shift+Enter is swallowed), so we take full control with one
# capture-phase handler: plain Enter clicks Send; Shift+Enter inserts a newline
# (and fires `input` so Gradio's state syncs). stopPropagation keeps Gradio's own
# handler from double-firing.
_ENTER_JS = """() => {
  if (window.__geoEnterWired) return;
  window.__geoEnterWired = true;
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const ta = e.target;
    if (!ta || !ta.closest || !ta.closest('#msg-box')) return;
    if (e.shiftKey) {
      e.preventDefault(); e.stopPropagation();
      const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value;
      ta.value = v.slice(0, s) + '\\n' + v.slice(en);
      ta.selectionStart = ta.selectionEnd = s + 1;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
      e.preventDefault(); e.stopPropagation();
      const btn = document.querySelector('#send-btn button') || document.querySelector('#send-btn');
      if (btn) btn.click();
    }
  }, true);
  // Iframe "Apply board edits" -> click the Gradio Apply button (JS then pulls TikZ).
  if (!window.__geoBoardApplyWired) {
    window.__geoBoardApplyWired = true;
    window.addEventListener('message', (e) => {
      if (!e.data || e.data.type !== 'geotikz-click-apply') return;
      const btn = document.querySelector('#apply-board-btn button')
        || document.querySelector('#apply-board-btn');
      if (btn) btn.click();
    });
  }
}"""

_INTRO_MD = (
    "# Geometry Figure Copilot\n"
    "Describe a geometry scene, **upload a screenshot or PDF** of a problem, or "
    "**paste existing TikZ** → get a **TikZ figure** (coordinate-free constructions). "
    "Then **edit it by chat** (“make it bigger”, “add color”, “move/rename the labels”). "
    "Each reply shows **which model** produced it (local specialist vs. frontier fallback)."
)

# Representative VALIDATED prompts the local qwen3-illustrator-4b specialist draws itself
# (see EXAMPLES.md). Wired as clickable examples that populate the Message box; the user
# then presses Enter/Send (populate-then-send avoids auto-running through the specialist's
# ~55s cold start).
EXAMPLE_PROMPTS = [
    "Triangle ABC has vertices A=(0,0), B=(6,0), C=(1,4). Let O be the circumcenter of "
    "triangle ABC. Output a single TikZ figure that draws triangle ABC and its circumcircle, "
    "and defines the named points A, B, C, O at their correct positions.",
    "Triangle ABC has vertices A=(1,0), B=(1,4), C=(6,5). Let H be the orthocenter of triangle "
    "ABC (the intersection of the three altitudes). Output a single TikZ figure that draws "
    "triangle ABC and defines A, B, C, H.",
    "Triangle ABC has vertices A=(0,0), B=(6,0), C=(1,5). Let D be the point where the internal "
    "bisector of angle A meets side BC. Also draw segment AD. Output a single TikZ figure that "
    "defines A, B, C, D.",
    "Triangle ABC has vertices A=(2,6), B=(0,0), C=(7,0). Let F be the foot of the altitude from "
    "A onto line BC. Also draw segment AF. Output a single TikZ figure that draws triangle ABC "
    "and defines A, B, C, F.",
    "A circle has center O=(0,0) and radius 3. Point P=(7,0) lies outside the circle. From P "
    "there are two tangent lines to the circle; let T1 and T2 be the two points of tangency. "
    "Output a single TikZ figure that draws the circle and the two tangent segments and defines "
    "O, P, T1, T2.",
    "Two circles are given: one centered at A=(-2,0) with radius 3, the other centered at "
    "B=(2,0) with radius 3. They intersect at two points X and Y. Output a single TikZ figure "
    "that draws both circles and their intersection points, defining A, B, X, Y.",
    "Line AB passes through A=(-5,0) and B=(5,1); P=(0,5) is a point. Let Q be the reflection of "
    "P across line AB. Output a single TikZ figure that draws line AB and the reflected point Q, "
    "defining A, B, P, Q.",
    "A regular hexagon P0P1P2P3P4P5 is inscribed in a circle of radius 4 centered at O=(0,0), "
    "with P0 at angle 0 degrees (each vertex is the previous one rotated 60 degrees about O). "
    "Output a single TikZ figure that draws the hexagon and its center O, defining O, P0, P2, P4.",
]

# (label, value) choices for the example dropdown: friendly short labels, full
# prompt as the value (so the 237-char prompts don't clutter the dropdown).
_EXAMPLE_LABELS = [
    "Circumcenter + circumcircle",
    "Orthocenter",
    "Angle bisector to a side",
    "Foot of altitude",
    "Tangents from a point",
    "Two-circle intersection",
    "Reflection over a line",
    "Regular hexagon",
]
EXAMPLE_CHOICES = list(zip(_EXAMPLE_LABELS, EXAMPLE_PROMPTS))

# AIME 2001-II-7: known in-vocab success for illustrator-4b-v2 (incircle + tangents +
# perpendiculars; op-vocab routing keeps it on the specialist). Demo PNG lives under
# web/assets/demo/ — Examples menu attaches it so paste/drag demos hit the same path.
AIME_2001_II_7 = (
    "Let triangle PQR be a right triangle with PQ = 90, PR = 120, and QR = 150. "
    "Let C1 be the inscribed circle. Construct ST with S on PR and T on QR, such that "
    "ST is perpendicular to PR and tangent to C1. Construct UV with U on PQ and V on QR "
    "such that UV is perpendicular to PQ and tangent to C1. Let C2 be the inscribed circle "
    "of triangle RST and C3 the inscribed circle of triangle QUV. Draw triangle PQR, "
    "incircle C1, segments ST and UV, and incircles C2 and C3."
)
# Query param busts the 24h Cache-Control on /assets/ so demos pick up crop updates.
AIME_DEMO_IMAGE = "/assets/demo/aime_2001_II_7.png?v=20260713b"
AIME_DEMO_EXAMPLE = {
    "label": "AIME screenshot (2001-II-7)",
    "prompt": AIME_2001_II_7,
    "image_url": AIME_DEMO_IMAGE,
    "saved": False,
}

DEFAULT_EXAMPLES_STORE = DEFAULT_OUT_DIR / "user_examples.json"


def _short_label(prompt: str) -> str:
    p = (prompt or "").strip().replace("\n", " ")
    return p if len(p) <= 50 else p[:47] + "…"


def _load_saved_examples(path: str | Path) -> list[str]:
    """User-saved example prompts (persisted JSON). Never raises."""
    try:
        import json
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
    except Exception:  # noqa: BLE001
        logger.exception("load saved examples failed")
    return []


def _save_saved_examples(path: str | Path, items: list[str]) -> None:
    try:
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items, indent=2))
    except Exception:  # noqa: BLE001
        logger.exception("save saved examples failed")


def _dropdown_choices(saved: list[str]) -> list[tuple[str, str]]:
    """Dropdown = built-in validated defaults + user-saved (marked with ★)."""
    return list(EXAMPLE_CHOICES) + [(f"★ {_short_label(p)}", p) for p in saved]


# --------------------------------------------------------------------------- #
# interactive editor: TikZ figure -> structured spec -> JSXGraph board (iframe)
# --------------------------------------------------------------------------- #
def _collect_point_names(tikz: str) -> list[str]:
    names: list[str] = []
    defs = [r"\\tkzDefPoint\([^)]*\)\{([A-Za-z]\w*)\}", r"\\tkzGetPoint\{([A-Za-z]\w*)\}",
            r"\\tkzGetPoints\{([A-Za-z]\w*)\}\{([A-Za-z]\w*)\}",
            r"\\coordinate\s*\(\s*([A-Za-z]\w*)\s*\)", r"\\node\s*\(\s*([A-Za-z]\w*)\s*\)"]
    for pat in defs:
        for m in re.finditer(pat, tikz):
            for g in m.groups():
                if g and g not in names:
                    names.append(g)
    for pat in [r"\\tkzDrawPoints\(([^)]*)\)", r"\\tkzLabelPoints(?:\[[^\]]*\])?\(([^)]*)\)"]:
        for m in re.finditer(pat, tikz):
            for n in m.group(1).split(","):
                n = n.strip()
                if re.fullmatch(r"[A-Za-z]\w*", n) and n not in names:
                    names.append(n)
    return names


def _literal_coords(tikz: str) -> dict[str, tuple[float, float]]:
    num = r"(-?\d+(?:\.\d+)?)"
    out: dict[str, tuple[float, float]] = {}
    for m in re.finditer(rf"\\tkzDefPoint\(\s*{num}\s*,\s*{num}\s*\)\{{([A-Za-z]\w*)\}}", tikz):
        out[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    for m in re.finditer(rf"\\coordinate\s*\(\s*([A-Za-z]\w*)\s*\)\s*at\s*\(\s*{num}\s*,\s*{num}\s*\)", tikz):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def _free_points(tikz: str) -> set[str]:
    free = {m.group(1) for m in re.finditer(r"\\tkzDefPoint\([^)]*\)\{([A-Za-z]\w*)\}", tikz)}
    free |= {m.group(1) for m in re.finditer(r"\\coordinate\s*\(\s*([A-Za-z]\w*)\s*\)\s*at\s*\(\s*-?\d", tikz)}
    return free


_PT = r"([A-Za-z]\w*)"
# Center-kind aliases used by tkz-euclide / specialist prompts.
_CENTER_KIND = {
    "circum": "circumcenter", "circumcenter": "circumcenter",
    "in": "incenter", "incenter": "incenter",
    "ortho": "orthocenter", "orthocenter": "orthocenter",
    "centroid": "centroid", "gravity": "centroid",
}


def _drawn_labeled_names(tikz: str) -> set[str]:
    """Points that are explicitly drawn or labeled (exclude invisible helpers)."""
    names: set[str] = set()
    for pat in (r"\\tkzDrawPoints\(([^)]*)\)",
                r"\\tkzLabelPoints(?:\[[^\]]*\])?\(([^)]*)\)",
                r"\\tkzDrawPolygon\(([^)]*)\)",
                r"\\fill[^;]*\(\s*([A-Za-z]\w*)\s*\)"):
        for m in re.finditer(pat, tikz):
            body = m.group(1)
            for n in re.findall(r"[A-Za-z]\w*", body):
                names.add(n)
    return names


def _parse_constraints(tikz: str) -> dict[str, dict]:
    """Infer geometric constraints for derived points from TikZ macros / calc.

    Returns ``{name: {type, parents, ...}}``. Unknown / unparseable constructions
    are omitted (those points stay free-drag on the board). Additive only.
    """
    out: dict[str, dict] = {}
    helpers: dict[str, dict] = {}  # intermediate GetPoint names (e.g. bisector ray)

    def _set(name: str, c: dict, *, helper: bool = False) -> None:
        if not name or name in out or name in helpers:
            return
        (helpers if helper else out)[name] = c

    # --- tkzDefTriangleCenter[kind](A,B,C)\tkzGetPoint{N} ---
    for m in re.finditer(
        rf"\\tkzDefTriangleCenter\[\s*(\w+)\s*\]\s*\(\s*{_PT}\s*,\s*{_PT}\s*,\s*{_PT}\s*\)"
        rf"\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        kind = _CENTER_KIND.get(m.group(1).lower())
        if kind:
            _set(m.group(5), {"type": kind, "parents": [m.group(2), m.group(3), m.group(4)]})

    # --- bare \\tkzDefCircumCenter / InCenter / OrthoCenter / Centroid ---
    for macro, kind in (
        ("CircumCenter", "circumcenter"), ("InCenter", "incenter"),
        ("OrthoCenter", "orthocenter"), ("Centroid", "centroid"),
        ("GravityCenter", "centroid"),
    ):
        for m in re.finditer(
            rf"\\tkzDef{macro}\s*\(\s*{_PT}\s*,\s*{_PT}\s*,\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}",
            tikz,
        ):
            _set(m.group(4), {"type": kind, "parents": [m.group(1), m.group(2), m.group(3)]})

    # --- midpoint ---
    for m in re.finditer(
        rf"\\tkzDefMidPoint\s*\(\s*{_PT}\s*,\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}", tikz
    ):
        _set(m.group(3), {"type": "midpoint", "parents": [m.group(1), m.group(2)]})

    # --- projection / foot ---
    for m in re.finditer(
        rf"\\tkzDefPointBy\[\s*projection\s*=\s*onto\s+{_PT}\s*--\s*{_PT}\s*\]\s*"
        rf"\(\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        # foot of P onto A--B  -> parents [P, A, B]
        _set(m.group(4), {"type": "foot", "parents": [m.group(3), m.group(1), m.group(2)]})

    # --- reflection over a line ---
    for m in re.finditer(
        rf"\\tkzDefPointBy\[\s*reflection\s*=\s*over\s+{_PT}\s*--\s*{_PT}\s*\]\s*"
        rf"\(\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        _set(m.group(4), {"type": "reflection", "parents": [m.group(3), m.group(1), m.group(2)]})

    # --- point reflection (symmetry through a center) ---
    for m in re.finditer(
        rf"\\tkzDefPointBy\[\s*symmetry\s*=\s*center\s+{_PT}\s*\]\s*"
        rf"\(\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        _set(m.group(3), {"type": "point_reflection", "parents": [m.group(2), m.group(1)]})

    # --- rotation ---
    for m in re.finditer(
        rf"\\tkzDefPointBy\[\s*rotation\s*=\s*center\s+{_PT}\s+angle\s+(-?[\d.]+)\s*\]\s*"
        rf"\(\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        _set(m.group(4), {
            "type": "rotation",
            "parents": [m.group(3), m.group(1)],  # [src, center]
            "angle": float(m.group(2)),
        })

    # --- translation ---
    for m in re.finditer(
        rf"\\tkzDefPointBy\[\s*translation\s*=\s*from\s+{_PT}\s+to\s+{_PT}\s*\]\s*"
        rf"\(\s*{_PT}\s*\)\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        # image of P translated by (to - from) -> parents [P, from, to]
        _set(m.group(4), {
            "type": "translation",
            "parents": [m.group(3), m.group(1), m.group(2)],
        })

    # --- angle bisector ray helper: DefLine[bisector](B,A,C)\tkzGetPoint{ba} ---
    for m in re.finditer(
        rf"\\tkzDefLine\[\s*bisector\s*\]\s*\(\s*{_PT}\s*,\s*{_PT}\s*,\s*{_PT}\s*\)"
        rf"\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        _set(m.group(4), {
            "type": "bisector_ray",
            "parents": [m.group(1), m.group(2), m.group(3)],  # B,A,C with apex A
        }, helper=True)

    # --- line-line intersection ---
    for m in re.finditer(
        rf"\\tkzInterLL\s*\(\s*{_PT}\s*,\s*{_PT}\s*\)\s*\(\s*{_PT}\s*,\s*{_PT}\s*\)"
        rf"\s*\\tkzGetPoint\{{{_PT}\}}",
        tikz,
    ):
        a, b, c, d, name = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        # Promote angle-bisector ∩ side into a dedicated constraint when possible.
        for (p, q), other in (((a, b), (c, d)), ((c, d), (a, b))):
            h = helpers.get(q) or helpers.get(p)
            apex = None
            if helpers.get(q) and helpers[q]["type"] == "bisector_ray":
                apex = p  # line is (apex, ray_pt)
                ray = helpers[q]
            elif helpers.get(p) and helpers[p]["type"] == "bisector_ray":
                apex = q
                ray = helpers[p]
            else:
                continue
            if apex and apex == ray["parents"][1]:  # apex matches DefLine middle arg
                _set(name, {
                    "type": "bisector_meet",
                    "parents": ray["parents"] + [other[0], other[1]],  # B,A,C,P,Q
                })
                break
        else:
            _set(name, {"type": "intersection", "parents": [a, b, c, d]})

    # --- PGF calc: midpoint ($(A)!0.5!(B)$) / partway / projection ($(A)!(P)!(B)$) ---
    for m in re.finditer(
        rf"\\coordinate\s*\(\s*{_PT}\s*\)\s*at\s*\(\s*\$\s*\(\s*{_PT}\s*\)"
        rf"\s*!\s*([^!]+?)\s*!\s*\(\s*{_PT}\s*\)\s*\$\s*\)",
        tikz,
    ):
        name, left, mid, right = m.group(1), m.group(2), m.group(3).strip(), m.group(4)
        if name in out:
            continue
        mid_bare = mid.strip("() \t")
        if re.fullmatch(r"0\.5|1/2", mid_bare):
            _set(name, {"type": "midpoint", "parents": [left, right]})
        elif re.fullmatch(r"-?[\d.]+", mid_bare):
            _set(name, {"type": "partway", "parents": [left, right], "t": float(mid_bare)})
        elif re.fullmatch(r"[A-Za-z]\w*", mid_bare):
            _set(name, {"type": "foot", "parents": [mid_bare, left, right]})

    return out, set(helpers)


def _parse_segments(tikz: str, pts: dict) -> list[list[str]]:
    segs: list[list[str]] = []

    def add(a, b):
        if a in pts and b in pts and a != b and [a, b] not in segs and [b, a] not in segs:
            segs.append([a, b])

    for m in re.finditer(r"\\tkzDrawPolygon\(([^)]*)\)", tikz):
        ns = [n.strip() for n in m.group(1).split(",") if n.strip()]
        for i in range(len(ns)):
            add(ns[i], ns[(i + 1) % len(ns)])
    for m in re.finditer(r"\\tkzDrawSegments?\(([^)]*)\)", tikz):
        body = m.group(1).strip()
        parts = body.split() if " " in body else [body]
        for pair in parts:
            xy = [t.strip() for t in pair.split(",")]
            if len(xy) == 2:
                add(xy[0], xy[1])
    for m in re.finditer(r"\\tkzDrawLine\(([^)]*)\)", tikz):
        xy = [t.strip() for t in m.group(1).split(",")]
        if len(xy) == 2:
            add(xy[0], xy[1])
    for m in re.finditer(r"\\draw\b([^;]*);", tikz):
        body = m.group(1)
        chain = []
        for part in body.split("--"):
            mm = re.search(r"\(\s*([A-Za-z]\w*)\s*\)", part)
            chain.append(mm.group(1) if mm else None)
        for i in range(len(chain) - 1):
            if chain[i] and chain[i + 1]:
                add(chain[i], chain[i + 1])
        if "cycle" in body:  # close the polygon
            named = [c for c in chain if c]
            if len(named) >= 2:
                add(named[-1], named[0])
    return segs


def _parse_circles(tikz: str, pts: dict) -> list[dict]:
    circles: list[dict] = []
    for m in re.finditer(r"\\tkzDrawCircle(?:\[[^\]]*\])?\(([^)]*)\)", tikz):
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) == 2 and args[0] in pts and args[1] in pts:
            circles.append({"center": args[0], "through": args[1]})
    for m in re.finditer(r"\\draw[^;]*?\(\s*([A-Za-z]\w*)\s*\)\s*circle\s*\(\s*([\d.]+)\s*\)", tikz):
        if m.group(1) in pts:
            circles.append({"center": m.group(1), "r": float(m.group(2))})
    for m in re.finditer(r"\\draw[^;]*?\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*circle\s*\(\s*([\d.]+)\s*\)", tikz):
        circles.append({"cx": float(m.group(1)), "cy": float(m.group(2)), "r": float(m.group(3))})
    return circles


def _figure_spec(tikz: str, timeout: int = 45) -> dict | None:
    """Structured, interactive-ready spec from a TikZ figure. Coords come from
    literal defs first, then a single compile-extract for any derived points.
    Derived points also carry constraint metadata when the TikZ construction is
    recognizable, so the JSXGraph board can re-solve them on drag."""
    try:
        tikz = metrics.extract_tikz(tikz or "") or (tikz or "")
        if not tikz.strip():
            return None
        names = _collect_point_names(tikz)
        if not names:
            return None
        coords = {n: c for n, c in _literal_coords(tikz).items() if n in names}
        missing = [n for n in names if n not in coords]
        if missing:
            try:
                from . import extract
                ext = extract.extract_named_coords(tikz, missing, timeout=timeout)
                coords.update({n: c for n, c in ext.items() if c})
            except Exception:  # noqa: BLE001
                logger.exception("board coord extract failed")
        if not coords:
            return None
        free = _free_points(tikz)
        constraints, helper_names = _parse_constraints(tikz)
        visible = _drawn_labeled_names(tikz)
        points = []
        for n, (x, y) in coords.items():
            # Drop invisible bisector-ray helpers (e.g. Dbl) unless drawn/labeled.
            if n in helper_names and n not in visible and n not in constraints:
                continue
            is_free = n in free and n not in constraints
            entry: dict = {"name": n, "x": round(x, 4), "y": round(y, 4), "free": is_free}
            if n in constraints:
                entry["constraint"] = constraints[n]
                entry["free"] = False
            points.append(entry)
        return {
            "points": points,
            "segments": _parse_segments(tikz, coords),
            "circles": _parse_circles(tikz, coords),
        }
    except Exception:  # noqa: BLE001
        logger.exception("figure spec failed")
        return None


_JSX_CDN_JS = "https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraphcore.js"
_JSX_CDN_CSS = "https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraph.css"

_EMPTY_BOARD = (
    "<div style='padding:14px;color:#666;font-family:system-ui,sans-serif;font-size:14px'>"
    "Generate a figure and it becomes <b>editable</b> here: drag points, resize circles, "
    "add points, then <b>Apply board edits</b> to push changes into the chat figure / TikZ, "
    "or export TikZ/SVG.</div>"
)

# Gradio button JS: request current board TikZ from the sandboxed iframe via postMessage,
# then hand it to the Python apply handler as the first input. Gradio 6 awaits Promises.
_APPLY_BOARD_JS = """(tikz, history, current_tikz, current_png, pending) => {
  return new Promise((resolve) => {
    const pass = (t) => resolve([t || '', history, current_tikz, current_png, pending]);
    const iframe = document.querySelector('iframe[title="interactive geometry editor"]');
    if (!iframe || !iframe.contentWindow) { pass(''); return; }
    let done = false;
    const finish = (t) => {
      if (done) return;
      done = true;
      window.removeEventListener('message', handler);
      pass(t);
    };
    const handler = (e) => {
      if (!e.data || e.data.type !== 'geotikz-tikz') return;
      finish(e.data.tikz || '');
    };
    window.addEventListener('message', handler);
    try { iframe.contentWindow.postMessage({type: 'geotikz-request-tikz'}, '*'); }
    catch (err) { finish(''); return; }
    setTimeout(() => finish(''), 3000);
  });
}"""

_BOARD_DOC = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="__CSS__">
<script src="__JS__"></script>
<style>
 html,body{margin:0;padding:0;font-family:system-ui,-apple-system,sans-serif}
 #bar{padding:6px 8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;border-bottom:1px solid #eee}
 #bar button{font-size:13px;padding:4px 9px;border:1px solid #d0d0d0;border-radius:6px;background:#fff;cursor:pointer}
 #bar button:hover{background:#f3f4f6}
 #hint{color:#888;font-size:12px;margin-left:auto}
 #jxg{width:100%;height:430px;background:#fff}
 #out{width:100%;box-sizing:border-box;height:120px;display:none;font-family:ui-monospace,monospace;font-size:12px;border:0;border-top:1px solid #eee;padding:8px}
</style></head><body>
<div id="bar">
 <button id="addpt">+ point</button>
 <button id="apply" style="background:#ecfdf5;border-color:#6ee7b7;font-weight:600">Apply board edits</button>
 <button id="tikz">export TikZ</button>
 <button id="dltikz">download .tex</button>
 <button id="dlsvg">download SVG</button>
 <span id="hint">magenta = free · red = constrained (re-solves) · drag free points · Apply pushes into chat figure</span>
</div>
<div id="jxg" class="jxgbox"></div>
<textarea id="out" readonly></textarea>
<script>
var SPEC = __SPEC__;
function boot(){ if(!window.JXG || !JXG.JSXGraph){ return setTimeout(boot,120); }
  try{ run(); }catch(e){ document.getElementById('jxg').innerHTML =
    '<div style="padding:12px;color:#a00">interactive board error: '+e+'</div>'; } }
function run(){
  var pts=SPEC.points||[], segs=SPEC.segments||[], circ=SPEC.circles||[];
  var xs=pts.map(function(p){return p.x;}), ys=pts.map(function(p){return p.y;});
  circ.forEach(function(c){ if(c.r!==undefined){ var cx,cy;
    if(c.cx!==undefined){cx=c.cx;cy=c.cy;} else { var cp=pts.filter(function(p){return p.name===c.center;})[0]; if(cp){cx=cp.x;cy=cp.y;} }
    if(cx!==undefined){ xs.push(cx-c.r,cx+c.r); ys.push(cy-c.r,cy+c.r); } } });
  if(!xs.length){ xs=[-5,5]; ys=[-5,5]; }
  var minx=Math.min.apply(null,xs),maxx=Math.max.apply(null,xs),miny=Math.min.apply(null,ys),maxy=Math.max.apply(null,ys);
  var pad=0.25*Math.max(maxx-minx,maxy-miny,1)+1;
  var board=JXG.JSXGraph.initBoard('jxg',{boundingbox:[minx-pad,maxy+pad,maxx+pad,miny-pad],
    keepaspectratio:true,axis:false,showNavigation:true,showCopyright:false,
    pan:{enabled:true,needShift:false},zoom:{wheel:true}});
  var P={};
  function rename(pt){ var t=Date.now(); if(pt._t && t-pt._t<340){ var nn=prompt('Rename point '+pt.name+' to:',pt.name);
    if(nn){ try{pt.setName(nn);}catch(e){pt.setAttribute({name:nn});} board.update(); } } pt._t=t; }
  function xy(a){ return [a.X(), a.Y()]; }
  function dist(a,b){ var dx=a.X()-b.X(), dy=a.Y()-b.Y(); return Math.sqrt(dx*dx+dy*dy); }
  function midXY(a,b){ return [(a.X()+b.X())/2,(a.Y()+b.Y())/2]; }
  function partwayXY(a,b,t){ return [a.X()+t*(b.X()-a.X()), a.Y()+t*(b.Y()-a.Y())]; }
  function footXY(p,a,b){ var ax=a.X(),ay=a.Y(),dx=b.X()-ax,dy=b.Y()-ay,d=dx*dx+dy*dy;
    if(d<1e-12) return [ax,ay]; var t=((p.X()-ax)*dx+(p.Y()-ay)*dy)/d; return [ax+t*dx,ay+t*dy]; }
  function reflectXY(p,a,b){ var f=footXY(p,a,b); return [2*f[0]-p.X(), 2*f[1]-p.Y()]; }
  function pointReflectXY(p,m){ return [2*m.X()-p.X(), 2*m.Y()-p.Y()]; }
  function rotateXY(p,c,deg){ var r=deg*Math.PI/180, cos=Math.cos(r), sin=Math.sin(r);
    var dx=p.X()-c.X(), dy=p.Y()-c.Y(); return [c.X()+dx*cos-dy*sin, c.Y()+dx*sin+dy*cos]; }
  function translateXY(p,frm,to){ return [p.X()+(to.X()-frm.X()), p.Y()+(to.Y()-frm.Y())]; }
  function lineIntersectXY(a,b,c,d){
    var x1=a.X(),y1=a.Y(),x2=b.X(),y2=b.Y(),x3=c.X(),y3=c.Y(),x4=d.X(),y4=d.Y();
    var den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4); if(Math.abs(den)<1e-12) return [(x1+x2)/2,(y1+y2)/2];
    var t=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/den;
    return [x1+t*(x2-x1), y1+t*(y2-y1)]; }
  function circumXY(a,b,c){
    var ax=a.X(),ay=a.Y(),bx=b.X(),by=b.Y(),cx=c.X(),cy=c.Y();
    var D=2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by)); if(Math.abs(D)<1e-12) return midXY(a,b);
    var a2=ax*ax+ay*ay,b2=bx*bx+by*by,c2=cx*cx+cy*cy;
    return [((a2*(by-cy)+b2*(cy-ay)+c2*(ay-by))/D), ((a2*(cx-bx)+b2*(ax-cx)+c2*(bx-ax))/D)]; }
  function centroidXY(a,b,c){ return [(a.X()+b.X()+c.X())/3,(a.Y()+b.Y()+c.Y())/3]; }
  function orthoXY(a,b,c){ // altitude from A to BC ∩ altitude from B to AC
    var f1=footXY(a,b,c), f2=footXY(b,a,c);
    // build points as temp coords via lineIntersect of A--f1 and B--f2
    var A={X:function(){return a.X();},Y:function(){return a.Y();}};
    var F1={X:function(){return f1[0];},Y:function(){return f1[1];}};
    var B={X:function(){return b.X();},Y:function(){return b.Y();}};
    var F2={X:function(){return f2[0];},Y:function(){return f2[1];}};
    return lineIntersectXY(A,F1,B,F2); }
  function incenterXY(a,b,c){
    var aa=dist(b,c), bb=dist(a,c), cc=dist(a,b), s=aa+bb+cc; if(s<1e-12) return centroidXY(a,b,c);
    return [(aa*a.X()+bb*b.X()+cc*c.X())/s, (aa*a.Y()+bb*b.Y()+cc*c.Y())/s]; }
  function bisectorMeetXY(B,A,C,P,Q){
    // unit vectors along AB, AC; their sum is the bisector direction; meet PQ
    var bx=B.X()-A.X(), by=B.Y()-A.Y(), bl=Math.sqrt(bx*bx+by*by)||1;
    var cx=C.X()-A.X(), cy=C.Y()-A.Y(), cl=Math.sqrt(cx*cx+cy*cy)||1;
    var dx=bx/bl+cx/cl, dy=by/bl+cy/cl;
    var R={X:function(){return A.X()+dx;},Y:function(){return A.Y()+dy;}};
    return lineIntersectXY(A,R,P,Q); }
  function solveConstraint(c){
    var pr=c.parents||[];
    function g(i){ return P[pr[i]]; }
    if(pr.some(function(_,i){return !g(i);})) return null;
    switch(c.type){
      case 'midpoint': return midXY(g(0),g(1));
      case 'partway': return partwayXY(g(0),g(1),c.t!=null?c.t:0.5);
      case 'foot': return footXY(g(0),g(1),g(2));
      case 'reflection': return reflectXY(g(0),g(1),g(2));
      case 'point_reflection': return pointReflectXY(g(0),g(1));
      case 'rotation': return rotateXY(g(0),g(1),c.angle||0);
      case 'translation': return translateXY(g(0),g(1),g(2));
      case 'intersection': return lineIntersectXY(g(0),g(1),g(2),g(3));
      case 'circumcenter': return circumXY(g(0),g(1),g(2));
      case 'centroid': return centroidXY(g(0),g(1),g(2));
      case 'orthocenter': return orthoXY(g(0),g(1),g(2));
      case 'incenter': return incenterXY(g(0),g(1),g(2));
      case 'bisector_meet': return bisectorMeetXY(g(0),g(1),g(2),g(3),g(4));
      default: return null;
    }
  }
  function freeAttrs(p){ return {name:p.name,size:3,strokeColor:'#c026d3',fillColor:'#c026d3',
    label:{fontSize:15,offset:[7,7]}}; }
  function lockedAttrs(p){ return {name:p.name,size:3,strokeColor:'#e11d48',fillColor:'#e11d48',
    fixed:true,highlight:false,label:{fontSize:15,offset:[7,7]}}; }
  // Free / unconstrained first, then constrained in dependency order (multi-pass).
  var pending=pts.slice(), guard=0;
  while(pending.length && guard++<64){
    var next=[];
    pending.forEach(function(p){
      if(p.constraint){
        var parents=p.constraint.parents||[];
        if(parents.some(function(n){return !P[n];})){ next.push(p); return; }
        var coords=solveConstraint(p.constraint);
        if(!coords){ // parents exist but type unknown -> free-drag fallback
          P[p.name]=board.create('point',[p.x,p.y],freeAttrs(p));
        } else {
          // Function coords + explicit parents so JSXGraph re-solves when bases move.
          (function(spec){
            var el=board.create('point',[
              function(){ var xy=solveConstraint(spec.constraint); return xy?xy[0]:spec.x; },
              function(){ var xy=solveConstraint(spec.constraint); return xy?xy[1]:spec.y; }
            ], lockedAttrs(spec));
            (spec.constraint.parents||[]).forEach(function(n){ if(P[n]) try{el.addParents(P[n]);}catch(e){} });
            P[spec.name]=el;
          })(p);
        }
      } else {
        P[p.name]=board.create('point',[p.x,p.y], p.free===false ? lockedAttrs(p) : freeAttrs(p));
        if(p.free===false){ /* derived w/o constraint: keep red but allow drag */ P[p.name].setAttribute({fixed:false,highlight:true}); }
      }
      if(P[p.name]) P[p.name].on('up',function(){rename(P[p.name]);});
    });
    if(next.length===pending.length){ // cycle / missing parent — drop remaining as free-drag
      next.forEach(function(p){
        P[p.name]=board.create('point',[p.x,p.y], p.free===false?lockedAttrs(p):freeAttrs(p));
        if(p.free===false) P[p.name].setAttribute({fixed:false,highlight:true});
        P[p.name].on('up',function(){rename(P[p.name]);});
      });
      break;
    }
    pending=next;
  }
  segs.forEach(function(s){ if(P[s[0]]&&P[s[1]]) board.create('segment',[P[s[0]],P[s[1]]],
    {strokeColor:'#222',strokeWidth:2,fixed:true,highlight:false}); });
  circ.forEach(function(c){
    if(c.through!==undefined && P[c.center] && P[c.through]){
      board.create('circle',[P[c.center],P[c.through]],{strokeColor:'#1d4ed8',strokeWidth:2,fixed:true,highlight:false});
    } else if(c.r!==undefined){
      var cc=(c.center!==undefined&&P[c.center])?P[c.center]:board.create('point',[c.cx,c.cy],{name:'',size:1,withLabel:false,color:'#1d4ed8'});
      var ring=board.create('point',[cc.X()+c.r,cc.Y()],{name:'',size:2,face:'o',strokeColor:'#1d4ed8',fillColor:'#fff'});
      board.create('circle',[cc,ring],{strokeColor:'#1d4ed8',strokeWidth:2,fixed:true,highlight:false});
    } });
  var addN=0;
  document.getElementById('addpt').onclick=function(){ addN++; var b=board.getBoundingBox();
    var np=board.create('point',[Math.round((b[0]+b[2])/2*100)/100,Math.round((b[1]+b[3])/2*100)/100],
      {name:'N'+addN,size:3,strokeColor:'#0891b2',fillColor:'#0891b2',label:{fontSize:15,offset:[7,7]}});
    np.on('up',function(){rename(np);}); };
  function isNamed(p){ return !!(p && p.name && /^[A-Za-z]/.test(p.name)); }
  function nm(p){ return isNamed(p)?p.name:('P'+p.id); }
  function toTikz(){ var s='\\begin{tikzpicture}\n';
    // Named points only (skip circle-resize rings / unlabeled helpers).
    var pl=board.objectsList.filter(function(o){return o.elType==='point' && isNamed(o);});
    pl.forEach(function(p){ s+='  \\coordinate ('+nm(p)+') at ('+p.X().toFixed(3)+','+p.Y().toFixed(3)+');\n'; });
    board.objectsList.filter(function(o){return o.elType==='segment';}).forEach(function(g){
      if(isNamed(g.point1)&&isNamed(g.point2))
        s+='  \\draw ('+nm(g.point1)+')--('+nm(g.point2)+');\n';
    });
    board.objectsList.filter(function(o){return o.elType==='circle';}).forEach(function(ci){
      var c=ci.center, r=ci.Radius().toFixed(3);
      if(isNamed(c)) s+='  \\draw ('+nm(c)+') circle ('+r+');\n';
      else if(c) s+='  \\draw ('+c.X().toFixed(3)+','+c.Y().toFixed(3)+') circle ('+r+');\n';
    });
    pl.forEach(function(p){ s+='  \\fill ('+nm(p)+') circle (1.5pt) node[above right]{$'+p.name+'$};\n'; });
    return s+'\\end{tikzpicture}'; }
  function dl(name,text,type){ var b=new Blob([text],{type:type}); var a=document.createElement('a');
    a.href=URL.createObjectURL(b); a.download=name; document.body.appendChild(a); a.click(); a.remove(); }
  document.getElementById('tikz').onclick=function(){ var o=document.getElementById('out'); o.style.display='block'; o.value=toTikz(); o.focus(); o.select(); };
  document.getElementById('dltikz').onclick=function(){ dl('figure.tex',toTikz(),'text/plain'); };
  document.getElementById('dlsvg').onclick=function(){ var svg=document.querySelector('#jxg svg');
    if(svg) dl('figure.svg',new XMLSerializer().serializeToString(svg),'image/svg+xml'); };
  document.getElementById('apply').onclick=function(){
    try { parent.postMessage({type:'geotikz-click-apply'}, '*'); } catch(e) {}
  };
  // Parent Gradio asks for current board TikZ (Apply bridge).
  window.addEventListener('message', function(e){
    if(!e.data || e.data.type!=='geotikz-request-tikz') return;
    var t=''; try{ t=toTikz(); }catch(err){}
    try { parent.postMessage({type:'geotikz-tikz', tikz:t}, '*'); } catch(err) {}
  });
  window.__board = board;  // exposed for scripted verification / debugging
  window.__P = P;
  window.__toTikz = toTikz;
  window.__solveConstraint = solveConstraint;
}
boot();
</script></body></html>"""


def _board_html(tikz: str, height: int = 560) -> str:
    """Build the interactive JSXGraph board (sandboxed iframe) from a figure, or a
    friendly placeholder if the figure can't be made interactive. Never raises."""
    try:
        import json
        spec = _figure_spec(tikz)
        if not spec or not spec.get("points"):
            return _EMPTY_BOARD
        doc = (_BOARD_DOC.replace("__CSS__", _JSX_CDN_CSS).replace("__JS__", _JSX_CDN_JS)
               .replace("__SPEC__", json.dumps(spec)))
        srcdoc = doc.replace("&", "&amp;").replace('"', "&quot;")
        return (
            '<iframe title="interactive geometry editor" '
            'sandbox="allow-scripts allow-modals allow-downloads allow-popups allow-popups-to-escape-sandbox" '
            f'style="width:100%;height:{height}px;border:1px solid #e5e7eb;border-radius:8px" '
            f'srcdoc="{srcdoc}"></iframe>')
    except Exception:  # noqa: BLE001
        logger.exception("board html failed")
        return _EMPTY_BOARD


def build_ui(
    specialist_fn: SpecialistFn | None = None,
    *,
    specialist_label: LabelLike = "the specialist",
    auth: tuple[str, str] | list[tuple[str, str]] | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    title: str = "Geometry Figure Copilot",
    intro_md: str = _INTRO_MD,
    frontier_models: list[str] | None = None,
    vision_models: list[str] | None = None,
    specialist_default: bool = False,
    specialist_toggle_label: str = "Try the specialist first",
    examples_store_path: str | Path | None = None,
    commit_examples: Callable[[], None] | None = None,
):
    """Build the copilot Gradio app (optional / local).

    Prefer ``geotikz.webapp.create_app`` for the custom website (Modal deploy).
    This Gradio UI remains for local ``scripts/copilot.py --gradio``.

    ``specialist_fn`` is the injected backend: ``description -> tikz_text``. When
    ``None`` the specialist toggle is hidden and everything runs frontier-first.
    ``auth`` (and ``out_dir``) are stashed on the returned Blocks as
    ``._geo_auth`` / ``._geo_out_dir`` so the caller can apply them at
    ``launch()`` / ``mount_gradio_app()`` time (auth cannot be bound at build).
    """
    import gradio as gr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier_models = frontier_models or FRONTIER_MODELS
    vision_models = vision_models or VISION_MODELS
    has_specialist = specialist_fn is not None
    store_path = Path(examples_store_path) if examples_store_path else DEFAULT_EXAMPLES_STORE

    def _repair_model() -> str:
        return frontier_models[0]

    def _persist_examples(items):
        _save_saved_examples(store_path, items)
        if commit_examples is not None:
            try:
                commit_examples()  # e.g. Modal Volume .commit() so it survives restarts
            except Exception:  # noqa: BLE001
                logger.exception("commit_examples failed")

    def do_save(msg_text, history):
        try:
            prompt = (msg_text or "").strip()
            if not prompt:  # fall back to the last thing the user sent
                for m in reversed(history or []):
                    if isinstance(m, dict) and m.get("role") == "user":
                        prompt = str(m.get("content", "")).strip()
                        break
            if not prompt or prompt[0] in "🖼️📄📋":
                return gr.update(), "_Type or send a text prompt first, then Save._"
            saved = _load_saved_examples(store_path)
            if prompt in EXAMPLE_PROMPTS or prompt in saved:
                return gr.update(), "_Already in the examples list._"
            saved.append(prompt)
            _persist_examples(saved)
            return gr.update(choices=_dropdown_choices(saved), value=prompt), "_Saved; it's in the dropdown._"
        except Exception:  # noqa: BLE001
            logger.exception("do_save crashed")
            return gr.update(), "_Couldn't save that one; try again._"

    def _refresh_examples():
        return gr.update(choices=_dropdown_choices(_load_saved_examples(store_path)))

    def do_remove(selected):
        try:
            if not selected:
                return gr.update(), "_Pick a saved (★) example to remove._"
            if selected in EXAMPLE_PROMPTS:
                return gr.update(), "_That's a built-in example; can't remove it._"
            saved = _load_saved_examples(store_path)
            if selected in saved:
                saved.remove(selected)
                _persist_examples(saved)
                return gr.update(choices=_dropdown_choices(saved), value=None), "_Removed._"
            return gr.update(), "_Not a saved example._"
        except Exception:  # noqa: BLE001
            logger.exception("do_remove crashed")
            return gr.update(), "_Couldn't remove that one; try again._"

    CRASH = "Sorry, something hiccuped on my end. Your figure is safe; please try again."

    # Outputs order: [chat, cur_tikz, cur_png, fig, code, badge, pending, board].
    # board defaults to gr.update() (unchanged) unless a new figure is drawn.
    def _out(history, tikz, png, badge, pending, board=None):
        return (history, tikz, png, png, (tikz or ""), badge, pending,
                board if board is not None else gr.update())

    _THINKING = "✏️ …drawing…"

    def _finish_assistant(history, content):
        """Replace the trailing '…drawing…' placeholder with the real reply (or
        append if, defensively, no placeholder is present)."""
        h = list(history or [])
        if h and isinstance(h[-1], dict) and h[-1].get("role") == "assistant":
            return h[:-1] + [{"role": "assistant", "content": content}]
        return h + [{"role": "assistant", "content": content}]

    def _apply(res, history, current_tikz, current_png, *, pend_kind, pend_text=None):
        """RouteResult -> UI outputs. The USER turn was already echoed in step 1;
        here we only fill in the ASSISTANT reply (replacing the '…drawing…'
        placeholder), update the figure, and set/clear the clarify pending state."""
        if res.clarify:
            pending = {"kind": pend_kind, "text": pend_text} if pend_kind else None
            return _out(_finish_assistant(history, res.note), current_tikz, current_png, _CLARIFY_BADGE, pending)
        if res.png:  # a new figure was drawn -> refresh the interactive board too
            return _out(_finish_assistant(history, f"{res.badge}: {res.note}"), res.tikz, res.png,
                        res.badge, None, board=_board_html(res.tikz))
        return _out(_finish_assistant(history, f"{res.badge}: {res.note}"),  # keep last figure
                    current_tikz, current_png, res.badge, None)

    def _generate_impl(message, image, history, current_tikz, current_png, use_specialist, frontier_model, pending):
        history = history or []
        message = (message or "").strip()
        if image:  # screenshot -> vision (new figure); supersedes any pending question
            vmodel = frontier_model if frontier_model in vision_models else vision_models[0]
            res = generate_image(image, vmodel, out_dir=out_dir)
            return _apply(res, history, current_tikz, current_png, pend_kind=None)
        if not message:  # nothing submitted -> gentle hint, no chat change
            return _out(history, current_tikz, current_png,
                        "_Type a geometry scene, upload a screenshot/PDF, or paste TikZ._", pending)
        if pending and pending.get("text"):  # this message answers a clarifying question
            combined = _combine(pending["text"], message)
            if pending.get("kind") == "edit" and current_tikz:
                res = edit_figure(current_tikz, combined, frontier_model, out_dir=out_dir)
                return _apply(res, history, current_tikz, current_png, pend_kind="edit", pend_text=combined)
            res = generate_text(combined, use_specialist, frontier_model,
                                specialist_fn=specialist_fn, specialist_label=specialist_label, out_dir=out_dir)
            return _apply(res, history, current_tikz, current_png, pend_kind="scene", pend_text=combined)
        if current_tikz and _classify_intent(message, frontier_model) == "edit":
            res = edit_figure(current_tikz, message, frontier_model, out_dir=out_dir)
            return _apply(res, history, current_tikz, current_png, pend_kind="edit", pend_text=message)
        # fresh scene (or a NEW scene typed while a figure is already open)
        res = generate_text(message, use_specialist, frontier_model,
                            specialist_fn=specialist_fn, specialist_label=specialist_label, out_dir=out_dir)
        return _apply(res, history, current_tikz, current_png, pend_kind="scene", pend_text=message)

    def _is_pdf(path) -> bool:
        return bool(path) and str(path).lower().endswith(".pdf")

    # -- STEP 1 (fast): echo the user's turn instantly + a '…drawing…' placeholder --
    def echo_send(message, attachment, history):
        """Echo chat turn; ``attachment`` is an image or PDF filepath (or None)."""
        message = (message or "").strip()
        if not message and not attachment:  # nothing submitted -> no echo
            return (history or []), message, attachment, message, attachment
        if attachment and _is_pdf(attachment):
            user_turn = "📄 (PDF upload)" + (f": {message}" if message else "")
        elif attachment:
            user_turn = "🖼️ (screenshot)" + (f": {message}" if message else "")
        else:
            user_turn = message
        h = (history or []) + [{"role": "user", "content": user_turn},
                               {"role": "assistant", "content": _THINKING}]
        # Clear msg + file; stash originals for step 2.
        return h, "", None, message, attachment

    def echo_example(selected, history):
        if not selected:
            return history or []
        return (history or []) + [{"role": "user", "content": selected},
                                  {"role": "assistant", "content": _THINKING}]

    def echo_paste(tikz_text, history):
        if not (tikz_text or "").strip():
            return history or []
        return (history or []) + [{"role": "user", "content": "📋 (pasted TikZ)"},
                                  {"role": "assistant", "content": _THINKING}]

    # -- STEP 2 (generation): fill in the reply; catch-all so it never surfaces an error --
    def gen_send(history, current_tikz, current_png, use_specialist, frontier_model, pending, message, attachment):
        try:
            if attachment and _is_pdf(attachment):
                vmodel = frontier_model if frontier_model in vision_models else vision_models[0]
                res = generate_pdf(attachment, vmodel, out_dir=out_dir)
                return _apply(res, history or [], current_tikz, current_png, pend_kind=None)
            return _generate_impl(message, attachment, history, current_tikz, current_png,
                                  use_specialist, frontier_model, pending)
        except Exception:  # noqa: BLE001
            logger.exception("gen_send crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped; try again._", pending)

    def gen_example(selected, history, current_tikz, current_png, use_specialist, frontier_model, pending):
        if not selected:
            return _out(history or [], current_tikz, current_png,
                        "_Pick an example, then Generate._", pending)
        try:
            return _generate_impl(selected, None, history, current_tikz, current_png,
                                  use_specialist, frontier_model, pending)
        except Exception:  # noqa: BLE001
            logger.exception("gen_example crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped; try again._", pending)

    def gen_paste(tikz_text, history, current_tikz, current_png, pending):
        if not (tikz_text or "").strip():
            return _out(history or [], current_tikz, current_png, "_Paste a full tikzpicture first._", pending)
        try:
            res = render_pasted(tikz_text, out_dir=out_dir, repair_model=_repair_model())
            if res.png:
                return _out(_finish_assistant(history, f"{res.badge}: {res.note}"),
                            res.tikz, res.png, res.badge, None, board=_board_html(res.tikz))
            return _out(_finish_assistant(history, res.note), current_tikz, current_png, _CLARIFY_BADGE, pending)
        except Exception:  # noqa: BLE001
            logger.exception("gen_paste crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped; try again._", pending)

    def gen_pdf(pdf_path, history, current_tikz, current_png, frontier_model, pending):
        if not pdf_path:
            return _out(history or [], current_tikz, current_png, "_Upload a PDF first._", pending)
        try:
            vmodel = frontier_model if frontier_model in vision_models else vision_models[0]
            res = generate_pdf(pdf_path, vmodel, out_dir=out_dir)
            return _apply(res, history or [], current_tikz, current_png, pend_kind=None)
        except Exception:  # noqa: BLE001
            logger.exception("gen_pdf crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped; try again._", pending)

    def new_figure():
        return _out([], None, None, "_Started a new figure._", None, board=_EMPTY_BOARD)

    def apply_board_edits(board_tikz, history, current_tikz, current_png, pending):
        """Compile board-exported TikZ and sync figure / chat / cur_* state.

        Catch-all: never surfaces a Gradio Error badge. On compile failure, keep
        the previous figure. Board HTML is left unchanged so live constraints stay.
        """
        try:
            raw = (board_tikz or "").strip()
            tikz = metrics.extract_tikz(raw) or raw
            if not tikz:
                return _out(
                    history or [], current_tikz, current_png,
                    "_No board figure to apply. Generate one first, then edit the board._",
                    pending,
                )
            if "tikzpicture" not in tikz:
                tikz = "\\begin{tikzpicture}\n" + tikz + "\n\\end{tikzpicture}"
            stem = f"board_{secrets.token_hex(4)}"
            r, tikz = _render(tikz, stem, out_dir)
            user_turn = "🖐 (apply board edits)"
            if not r.ok:
                badge = _attr("you (board edits)", kind="apply failed")
                note = (
                    f"Board edits didn't compile (`{r.reason}`). Kept your current figure."
                )
                h = (history or []) + [
                    {"role": "user", "content": user_turn},
                    {"role": "assistant", "content": f"{badge}: {note}"},
                ]
                return _out(h, current_tikz, current_png, badge, pending)
            badge = _attr("you (board edits)", kind="applied")
            note = "Applied board edits to the figure."
            h = (history or []) + [
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": f"{badge}: {note}"},
            ]
            # Keep the interactive board as-is (constraints still live); only
            # sync the static figure / TikZ / chat state for follow-up edits.
            return _out(h, tikz, str(Path(r.png_path)), badge, None)
        except Exception:  # noqa: BLE001
            logger.exception("apply_board_edits crashed")
            h = (history or []) + [
                {"role": "user", "content": "🖐 (apply board edits)"},
                {"role": "assistant", "content": CRASH},
            ]
            return _out(h, current_tikz, current_png, "_Something hiccuped; try again._", pending)

    # -- clean, State-free API endpoints (for gradio_client / HTTP tests) --
    def _api_msg(res):
        return res.note if res.clarify else f"{res.badge}: {res.note}"

    def api_generate(description, frontier_model):
        try:
            model = frontier_model or frontier_models[0]
            res = generate_text(description, has_specialist, model,
                                specialist_fn=specialist_fn, specialist_label=specialist_label, out_dir=out_dir)
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_generate crashed")
            return None, "", "Sorry, something hiccuped; please try again."

    def api_edit(current_tikz, instruction, frontier_model):
        try:
            model = frontier_model or frontier_models[0]
            res = edit_figure(current_tikz, instruction, model, out_dir=out_dir)
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_edit crashed")
            return None, "", "Sorry, something hiccuped; please try again."

    def api_paste(tikz_text):
        try:
            res = render_pasted(tikz_text, out_dir=out_dir, repair_model=_repair_model())
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_paste crashed")
            return None, "", "Sorry, something hiccuped; please try again."

    def api_pdf(pdf_path, frontier_model):
        try:
            model = frontier_model or frontier_models[0]
            vmodel = model if model in vision_models else vision_models[0]
            res = generate_pdf(pdf_path, vmodel, out_dir=out_dir)
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_pdf crashed")
            return None, "", "Sorry, something hiccuped; please try again."

    # Compact chat-first CSS: tighter composer row, less panel chrome.
    _UI_CSS = """
    #msg-box textarea { min-height: 52px !important; }
    #composer-row { align-items: stretch; flex-wrap: nowrap !important; }
    #attach-file { min-height: 0 !important; max-width: 160px; }
    #attach-file .wrap, #attach-file .upload-container { min-height: 52px !important; }
    #composer-btns { display: flex; flex-direction: column; gap: 6px; min-width: 88px; }
    #fig-out img { max-height: 520px; object-fit: contain; }
    """

    with gr.Blocks(title=title) as app:
        gr.Markdown(intro_md)
        cur_tikz = gr.State(None)   # last good TikZ (also the Code box content)
        cur_png = gr.State(None)    # last good figure PNG (preserved on failure)
        pending = gr.State(None)    # {"kind": "scene"|"edit", "text": ...} clarify context
        msg_hold = gr.State("")     # stashes the submitted message across the echo->gen chain
        img_hold = gr.State(None)   # stashes the submitted attachment across the echo->gen chain
        with gr.Row():
            # ── Left: chat + compact composer ────────────────────────────────
            with gr.Column(scale=5, min_width=360):
                # Gradio 6: messages format (role/content dicts) only (see _append_turn).
                chat = gr.Chatbot(label="Conversation", height=460)
                with gr.Row(elem_id="composer-row"):
                    msg = gr.Textbox(
                        label="Message", lines=2, elem_id="msg-box", scale=5, show_label=False,
                        placeholder="Describe a scene or edit. Enter to send, Shift+Enter for a new line")
                    attach_file = gr.File(
                        label="Attach image/PDF",
                        file_types=[".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"],
                        type="filepath", elem_id="attach-file", scale=2, height=52,
                        file_count="single")
                    with gr.Column(scale=1, min_width=88, elem_id="composer-btns"):
                        send = gr.Button("Send", variant="primary", elem_id="send-btn")
                        reset = gr.Button("New")
                with gr.Accordion("Paste TikZ", open=False):
                    paste_box = gr.Textbox(
                        label="Paste a full tikzpicture", lines=5, show_label=False,
                        placeholder="\\begin{tikzpicture} … \\end{tikzpicture}")
                    paste_btn = gr.Button("Render & edit this TikZ", size="sm")
                with gr.Accordion("Examples", open=False):
                    with gr.Row():
                        example_dd = gr.Dropdown(
                            _dropdown_choices(_load_saved_examples(store_path)), value=None, scale=3,
                            label="Built-in + your saved ★", show_label=False)
                        example_btn = gr.Button("Generate", scale=1)
                    with gr.Row():
                        save_ex_btn = gr.Button("★ Save current", size="sm")
                        remove_ex_btn = gr.Button("Remove selected", size="sm")
                    ex_status = gr.Markdown("")
                with gr.Accordion("Settings", open=False):
                    use_spec = gr.Checkbox(label=specialist_toggle_label, value=specialist_default,
                                           visible=has_specialist)
                    model = gr.Dropdown(frontier_models, value=frontier_models[0],
                                        label="Frontier model")

            # ── Right: figure + interactive editor (one tab away) ────────────
            with gr.Column(scale=5, min_width=360):
                badge = gr.Markdown("_No figure yet._")
                with gr.Tabs():
                    with gr.Tab("Figure"):
                        fig = gr.Image(label="Figure", type="filepath", elem_id="fig-out",
                                       show_label=False)
                        with gr.Accordion("TikZ code", open=False):
                            code = gr.Code(label="TikZ (editable / copyable)", language="latex",
                                           show_label=False)
                    with gr.Tab("Interactive"):
                        apply_board_btn = gr.Button(
                            "Apply board edits → figure / TikZ", variant="primary",
                            elem_id="apply-board-btn")
                        gr.Markdown(
                            "_Drag free (magenta) points; constrained (red) points re-solve. "
                            "Then **Apply board edits** to update the Figure / TikZ / chat state._")
                        board = gr.HTML(_EMPTY_BOARD)
                        # Slot filled by `_APPLY_BOARD_JS` (iframe → postMessage → TikZ text).
                        board_tikz_hold = gr.Textbox(visible=False, elem_id="board-tikz-hold",
                                                     value="")

        outputs = [chat, cur_tikz, cur_png, fig, code, badge, pending, board]
        # Two-step at every entry point: (1) echo the user turn instantly + clear
        # the input, then (2) .then(...) run generation (fills the assistant reply).
        _echo_out = [chat, msg, attach_file, msg_hold, img_hold]
        _gen_in = [chat, cur_tikz, cur_png, use_spec, model, pending, msg_hold, img_hold]
        send.click(echo_send, [msg, attach_file, chat], _echo_out).then(gen_send, _gen_in, outputs)
        # Enter submits (msg.submit); Shift+Enter inserts a newline (_ENTER_JS on load).
        msg.submit(echo_send, [msg, attach_file, chat], _echo_out).then(gen_send, _gen_in, outputs)
        # On each page load: wire Enter-to-send (js) and refresh the examples
        # dropdown from the persisted store (so saves show up without a restart).
        app.load(_refresh_examples, None, [example_dd], js=_ENTER_JS)
        reset.click(new_figure, None, outputs)
        apply_board_btn.click(
            apply_board_edits,
            [board_tikz_hold, chat, cur_tikz, cur_png, pending],
            outputs,
            js=_APPLY_BOARD_JS,
        )
        example_btn.click(echo_example, [example_dd, chat], [chat]).then(
            gen_example, [example_dd, chat, cur_tikz, cur_png, use_spec, model, pending], outputs)
        save_ex_btn.click(do_save, [msg, chat], [example_dd, ex_status])
        remove_ex_btn.click(do_remove, [example_dd], [example_dd, ex_status])
        paste_btn.click(echo_paste, [paste_box, chat], [chat]).then(
            gen_paste, [paste_box, chat, cur_tikz, cur_png, pending], outputs)

        # Programmatic, State-free endpoints (kept out of the visual flow).
        api_desc = gr.Textbox(visible=False)
        api_model = gr.Textbox(visible=False)
        api_instr = gr.Textbox(visible=False)
        api_cur = gr.Textbox(visible=False)
        api_paste_box = gr.Textbox(visible=False)
        api_pdf_in = gr.File(visible=False, type="filepath")
        api_png = gr.Image(visible=False, type="filepath")
        api_tikz = gr.Textbox(visible=False)
        api_note = gr.Textbox(visible=False)
        api_gen_btn = gr.Button(visible=False)
        api_edit_btn = gr.Button(visible=False)
        api_paste_btn = gr.Button(visible=False)
        api_pdf_btn = gr.Button(visible=False)
        api_gen_btn.click(api_generate, [api_desc, api_model], [api_png, api_tikz, api_note],
                          api_name="generate")
        api_edit_btn.click(api_edit, [api_cur, api_instr, api_model], [api_png, api_tikz, api_note],
                           api_name="edit")
        api_paste_btn.click(api_paste, [api_paste_box], [api_png, api_tikz, api_note],
                            api_name="paste")
        api_pdf_btn.click(api_pdf, [api_pdf_in, api_model], [api_png, api_tikz, api_note],
                          api_name="pdf")

    app._geo_auth = auth  # applied by the caller at launch()/mount time
    app._geo_out_dir = str(out_dir)
    app._geo_css = _UI_CSS  # Gradio 6: pass to launch()/mount_gradio_app(css=...)
    return app
