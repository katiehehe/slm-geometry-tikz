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
  * IMAGE (screenshot) / PDF -> a frontier VISION model reads it and emits a
    construction figure (the narrow specialist is text-only).
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
from typing import Callable

import gradio as gr

from . import gateway, metrics, serve
from .prompts import CONSTRUCTION_SYSTEM_PROMPT

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
    "(e.g. 'a triangle with a circle' — incircle vs circumcircle vs arbitrary), output "
    "EXACTLY one line: 'CLARIFY: <one short question>'.\n"
    "  - If the request is NOT about geometry, output EXACTLY: "
    "'CLARIFY: I draw geometry diagrams — describe a geometry scene or problem and I'll illustrate it.'"
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
    "problem — try a clearer screenshot, or describe the scene in words.' Never output both."
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
    return f"{original.strip()} — {answer.strip()}"


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


def _data_url(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def _render(text: str, stem: str, out_dir: str | Path) -> tuple[serve.RenderResult, str]:
    tikz = _tikz(text)
    png = Path(out_dir) / f"{stem}.png"
    return serve.compile_and_render(text, png, dpi=200), tikz


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
) -> RouteResult:
    """Text scene -> figure. Specialist first (if enabled), then a clarify-aware
    frontier decision (draw OR ask ONE question). Never raises."""
    description = (description or "").strip()
    if not description:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "Describe a geometry scene — points, shapes, and how they relate — and I'll draw it.",
                           clarify=True)
    stem = serve.dhash(description + str(time.time()))
    prefix = ""
    # Why the frontier is used (shown in the badge): specialist off unless we
    # actually try it and it can't draw -> "specialist fell back".
    spec_reason = "specialist off"

    # 1) specialist first (if enabled) — retry transient failures, never raise.
    if use_specialist and specialist_fn is not None:
        t0 = time.time()
        spec_tikz = ""
        try:
            spec_tikz = _retry(lambda: specialist_fn(description) or "", tries=3) or ""
        except Exception as e:  # noqa: BLE001 - specialist down -> frontier
            logger.warning("specialist failed after retries: %s", e)
        if spec_tikz:
            # Tidy the specialist's labels (push outward + white halo) for legibility;
            # if the tidied figure doesn't compile, fall back to the ORIGINAL so this
            # aesthetic pass can never break a figure.
            tidied = serve.tidy_labels(spec_tikz)
            r, tikz = _render(tidied, stem, out_dir)
            if not r.ok and tidied != spec_tikz:
                r, tikz = _render(spec_tikz, stem, out_dir)
            if r.ok:
                return RouteResult(str(Path(r.png_path)), tikz, _attr(_resolve_label(specialist_label)),
                                   f"Specialist drew it in {time.time() - t0:.0f}s (coordinate-free, compiled).")
        spec_reason = "specialist fell back"
        prefix = "The specialist couldn't draw this one, so a frontier model did. "

    # 2) frontier: draw OR clarify (one call).
    messages = [
        {"role": "system", "content": _SCENE_SYSTEM},
        {"role": "user", "content": f"Scene:\n{description}\n\nReturn the TikZ figure, or a CLARIFY line."},
    ]
    res = gateway.chat(messages, frontier_model, max_tokens=4096)  # gateway retries transient 429/5xx/conn
    if not res.ok:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I'm having trouble reaching the drawing model right now — please try again in a moment.",
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
                       "I couldn't turn that into a clean figure. Could you add a detail — key points, a shape, "
                       "or a relationship (e.g. 'triangle ABC with its circumcircle')?",
                       clarify=True)


def generate_image(
    image_path: str,
    vision_model: str,
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    source_name: str = "screenshot",
) -> RouteResult:
    """A problem image (screenshot/PDF page) -> figure via a frontier vision model.
    Clarify-aware (non-geometry / unreadable image -> a friendly redirect). Never raises."""
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
                           "I'm having trouble reading the image right now — please try again shortly, "
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
) -> RouteResult:
    """PDF upload -> rasterise page 1 (PyMuPDF) -> vision route. Never raises."""
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
    return generate_image(png_in, vision_model, out_dir=out_dir, source_name="PDF (page 1)")


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
                           "Rendered your figure — now edit it by chat below.")
    if tikz and repair_model:  # optional: one frontier fix so a near-good paste still lands
        rep = _self_repair(tikz, r.reason, repair_model)
        r2, tikz2 = _render(rep.text, stem, out_dir)
        if r2.ok:
            return RouteResult(str(Path(r2.png_path)), tikz2, _attr(_frontier_inner(repair_model), kind="repaired import"),
                               "Your TikZ needed one fix to compile; rendered.")
    return RouteResult(None, "", _CLARIFY_BADGE,
                       f"That TikZ didn't compile (`{r.reason}`) — I kept your current figure. "
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
                           "There's no figure yet — describe a scene and I'll draw one first.", clarify=True)
    messages = [
        {"role": "system", "content": _EDIT_SYSTEM_CLARIFY},
        {"role": "user", "content": f"Current figure:\n{current_tikz}\n\n"
         f"Edit instruction: {instruction}\n\nReturn the full revised tikzpicture, or a CLARIFY line."},
    ]
    res = gateway.chat(messages, edit_model, max_tokens=4096)
    if not res.ok:
        return RouteResult(None, "", _CLARIFY_BADGE,
                           "I'm having trouble reaching the model to apply that edit — please try again in a moment.",
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
) -> gr.Blocks:
    """Build the copilot Gradio app.

    ``specialist_fn`` is the injected backend: ``description -> tikz_text``. When
    ``None`` the specialist toggle is hidden and everything runs frontier-first.
    ``auth`` (and ``out_dir``) are stashed on the returned Blocks as
    ``._geo_auth`` / ``._geo_out_dir`` so the caller can apply them at
    ``launch()`` / ``mount_gradio_app()`` time (auth cannot be bound at build).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier_models = frontier_models or FRONTIER_MODELS
    vision_models = vision_models or VISION_MODELS
    has_specialist = specialist_fn is not None

    def _repair_model() -> str:
        return frontier_models[0]

    CRASH = "Sorry — something hiccuped on my end. Your figure is safe; please try again."

    # Outputs order everywhere: [chat, cur_tikz, cur_png, fig, code, badge, pending].
    def _out(history, tikz, png, badge, pending):
        return history, tikz, png, png, (tikz or ""), badge, pending

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
        if res.png:  # a new figure was drawn
            return _out(_finish_assistant(history, f"{res.badge} — {res.note}"), res.tikz, res.png, res.badge, None)
        return _out(_finish_assistant(history, f"{res.badge} — {res.note}"),  # keep last figure
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

    # -- STEP 1 (fast): echo the user's turn instantly + a '…drawing…' placeholder --
    def echo_send(message, image, history):
        message = (message or "").strip()
        if not message and not image:  # nothing submitted -> no echo
            return (history or []), message, image, message, image
        user_turn = ("🖼️ (screenshot)" + (f" — {message}" if message else "")) if image else message
        h = (history or []) + [{"role": "user", "content": user_turn},
                               {"role": "assistant", "content": _THINKING}]
        return h, "", None, message, image  # clear msg+img; stash originals for step 2

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

    def echo_pdf(pdf_path, history):
        if not pdf_path:
            return history or []
        return (history or []) + [{"role": "user", "content": "📄 (PDF upload)"},
                                  {"role": "assistant", "content": _THINKING}]

    # -- STEP 2 (generation): fill in the reply; catch-all so it never surfaces an error --
    def gen_send(history, current_tikz, current_png, use_specialist, frontier_model, pending, message, image):
        try:
            return _generate_impl(message, image, history, current_tikz, current_png,
                                  use_specialist, frontier_model, pending)
        except Exception:  # noqa: BLE001
            logger.exception("gen_send crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped — try again._", pending)

    def gen_example(selected, history, current_tikz, current_png, use_specialist, frontier_model, pending):
        if not selected:
            return _out(history or [], current_tikz, current_png,
                        "_Pick an example above, then Generate._", pending)
        try:
            return _generate_impl(selected, None, history, current_tikz, current_png,
                                  use_specialist, frontier_model, pending)
        except Exception:  # noqa: BLE001
            logger.exception("gen_example crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped — try again._", pending)

    def gen_paste(tikz_text, history, current_tikz, current_png, pending):
        if not (tikz_text or "").strip():
            return _out(history or [], current_tikz, current_png, "_Paste a full tikzpicture first._", pending)
        try:
            res = render_pasted(tikz_text, out_dir=out_dir, repair_model=_repair_model())
            if res.png:
                return _out(_finish_assistant(history, f"{res.badge} — {res.note}"),
                            res.tikz, res.png, res.badge, None)
            return _out(_finish_assistant(history, res.note), current_tikz, current_png, _CLARIFY_BADGE, pending)
        except Exception:  # noqa: BLE001
            logger.exception("gen_paste crashed")
            return _out(_finish_assistant(history, CRASH), current_tikz, current_png,
                        "_Something hiccuped — try again._", pending)

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
                        "_Something hiccuped — try again._", pending)

    def new_figure():
        return _out([], None, None, "_Started a new figure._", None)

    # -- clean, State-free API endpoints (for gradio_client / HTTP tests) --
    def _api_msg(res):
        return res.note if res.clarify else f"{res.badge} — {res.note}"

    def api_generate(description, frontier_model):
        try:
            model = frontier_model or frontier_models[0]
            res = generate_text(description, has_specialist, model,
                                specialist_fn=specialist_fn, specialist_label=specialist_label, out_dir=out_dir)
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_generate crashed")
            return None, "", "Sorry — something hiccuped; please try again."

    def api_edit(current_tikz, instruction, frontier_model):
        try:
            model = frontier_model or frontier_models[0]
            res = edit_figure(current_tikz, instruction, model, out_dir=out_dir)
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_edit crashed")
            return None, "", "Sorry — something hiccuped; please try again."

    def api_paste(tikz_text):
        try:
            res = render_pasted(tikz_text, out_dir=out_dir, repair_model=_repair_model())
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_paste crashed")
            return None, "", "Sorry — something hiccuped; please try again."

    def api_pdf(pdf_path, frontier_model):
        try:
            model = frontier_model or frontier_models[0]
            vmodel = model if model in vision_models else vision_models[0]
            res = generate_pdf(pdf_path, vmodel, out_dir=out_dir)
            return res.png, (res.tikz or ""), _api_msg(res)
        except Exception:  # noqa: BLE001
            logger.exception("api_pdf crashed")
            return None, "", "Sorry — something hiccuped; please try again."

    with gr.Blocks(title=title) as app:
        gr.Markdown(intro_md)
        cur_tikz = gr.State(None)   # last good TikZ (also the Code box content)
        cur_png = gr.State(None)    # last good figure PNG (preserved on failure)
        pending = gr.State(None)    # {"kind": "scene"|"edit", "text": ...} clarify context
        msg_hold = gr.State("")     # stashes the submitted message across the echo->gen chain
        img_hold = gr.State(None)   # stashes the submitted image across the echo->gen chain
        with gr.Row():
            with gr.Column(scale=1):
                # Gradio 6 dropped the ``type`` arg: messages format (role/content
                # dicts) is the ONLY accepted shape (see _append_turn).
                chat = gr.Chatbot(label="Conversation", height=380)
                msg = gr.Textbox(label="Message", lines=2, elem_id="msg-box",
                                 placeholder="Describe a scene, or an edit — Enter to send, Shift+Enter for a new line")
                with gr.Row():
                    example_dd = gr.Dropdown(
                        EXAMPLE_CHOICES, value=None, scale=3,
                        label="Example prompts (specialist can draw these)")
                    example_btn = gr.Button("Generate", scale=1)
                img = gr.Image(label="…or a screenshot of a problem", type="filepath")
                with gr.Row():
                    send = gr.Button("Send", variant="primary", elem_id="send-btn")
                    reset = gr.Button("New figure")
                with gr.Row():
                    use_spec = gr.Checkbox(label=specialist_toggle_label, value=specialist_default,
                                           visible=has_specialist)
                    model = gr.Dropdown(frontier_models, value=frontier_models[0],
                                        label="Frontier model")
                with gr.Accordion("More input types", open=False):
                    gr.Markdown("**Paste existing TikZ** — render it as-is and jump straight into the edit loop.")
                    paste_box = gr.Textbox(
                        label="Paste a full tikzpicture", lines=6,
                        placeholder="\\begin{tikzpicture} … \\end{tikzpicture}")
                    paste_btn = gr.Button("Render & edit this TikZ")
                    gr.Markdown("**Upload a PDF** — page 1 is rasterised and read by the vision model.")
                    pdf_in = gr.File(label="PDF of a problem", file_types=[".pdf"], type="filepath")
                    pdf_btn = gr.Button("Read PDF → figure")
                    gr.Markdown(
                        "_More ideas (not built yet): multiple images / multi-part problems, "
                        "Asymptote (.asy) import._")
            with gr.Column(scale=1):
                badge = gr.Markdown("_No figure yet._")
                fig = gr.Image(label="Figure", type="filepath", elem_id="fig-out")
                code = gr.Code(label="TikZ (editable / copyable)", language="latex")

        outputs = [chat, cur_tikz, cur_png, fig, code, badge, pending]
        # Two-step at every entry point: (1) echo the user turn instantly + clear
        # the input, then (2) .then(...) run generation (fills the assistant reply).
        _echo_out = [chat, msg, img, msg_hold, img_hold]
        _gen_in = [chat, cur_tikz, cur_png, use_spec, model, pending, msg_hold, img_hold]
        send.click(echo_send, [msg, img, chat], _echo_out).then(gen_send, _gen_in, outputs)
        # Enter submits (msg.submit); Shift+Enter inserts a newline (_ENTER_JS on load).
        msg.submit(echo_send, [msg, img, chat], _echo_out).then(gen_send, _gen_in, outputs)
        app.load(None, None, None, js=_ENTER_JS)
        reset.click(new_figure, None, outputs)
        example_btn.click(echo_example, [example_dd, chat], [chat]).then(
            gen_example, [example_dd, chat, cur_tikz, cur_png, use_spec, model, pending], outputs)
        paste_btn.click(echo_paste, [paste_box, chat], [chat]).then(
            gen_paste, [paste_box, chat, cur_tikz, cur_png, pending], outputs)
        pdf_btn.click(echo_pdf, [pdf_in, chat], [chat]).then(
            gen_pdf, [pdf_in, chat, cur_tikz, cur_png, model, pending], outputs)

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
    return app
