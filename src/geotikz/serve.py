"""Shared "descriptions -> figures" serving layer.

This is the small helper module all three product surfaces call (the interactive
demo, the utility eval, and the AIME auto-illustrator) so the inference +
compile + render + cache logic lives in exactly one place.

Design (per the project's thesis + the coordinate-free-construction spec):

  * SPECIALIST  -- the fine-tuned Qwen3-0.6B + `qwen3-pgf-geotikz` LoRA. Loaded
    locally (base + adapter) and prompted with the EXACT prompt it was trained
    with (`prompts.build_messages`, `enable_thinking=False`), reusing the proven
    `infer.load_model` / `infer.generate` path. It already emits coordinate-free
    PGF constructions (polar points + `calc`/`intersections`), so it is used
    as-is.
  * FRONTIER    -- a hosted model via the gateway, prompted with
    `prompts.build_construction_messages` (`CONSTRUCTION_SYSTEM_PROMPT`) so it
    also returns coordinate-free tkz-euclide / `calc` constructions, never
    hardcoded numeric coordinate lists.

Rendering wraps the figure in a tkz-euclide-capable standalone doc (a superset
preamble that compiles BOTH the specialist's `calc` figures and the frontier's
tkz-euclide figures), compiles with tectonic, and rasterises the first page to a
colour PNG. A blank/degenerate page is detected via ink ratio so "it compiled"
never silently means "it drew nothing".

Nothing here has side effects at import time beyond cheap stdlib/numpy imports;
torch is only imported when a local specialist is actually loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import gateway, metrics, tex
from .prompts import build_construction_messages, build_messages

DEFAULT_BASE = "Qwen/Qwen3-0.6B"
DEFAULT_ADAPTER = "outputs/qwen3-pgf-geotikz"

# A single rich preamble used for ALL rendering. It loads tkz-euclide plus the
# usual tikz libraries, so it is a superset that compiles the specialist's
# coordinate-free `calc` figures AND frontier tkz-euclide constructions. Kept in
# sync (deliberately) with extract.EXTRACT_TEMPLATE's preamble so "renders here"
# matches "grades there". `%` is literal TeX; body is injected via __BODY__.
RICH_TEMPLATE = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{tkz-euclide}
\usetikzlibrary{calc,angles,quotes,intersections,through,positioning,arrows,arrows.meta,%
decorations.markings,decorations.pathreplacing,shapes.geometric,shapes.misc,patterns,%
patterns.meta,backgrounds,fit,math,3d,perspective}
\begin{document}
__BODY__
\end{document}
"""


def dhash(text: str) -> str:
    """Stable short hash of a scene description (cache key material)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# specialist (local base + LoRA)
# --------------------------------------------------------------------------- #
class Specialist:
    """The fine-tuned specialist, loaded once and reused.

    Thin wrapper over `infer.load_model` / `infer.generate` (the proven path).
    Loading is lazy so importing this module never pulls in torch.
    """

    def __init__(self, base: str = DEFAULT_BASE, adapter: str | None = DEFAULT_ADAPTER):
        self.base = base
        self.adapter = adapter
        self._model = None
        self._tok = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "Specialist":
        if self._model is None:
            from . import infer  # local import: keeps torch out of module import

            self._model, self._tok, self._device = infer.load_model(self.base, self.adapter)
        return self

    def generate(self, description: str, max_new_tokens: int = 512) -> str:
        from . import infer

        self.load()
        return infer.generate(
            self._model, self._tok, self._device, description, max_new_tokens=max_new_tokens
        )

    def generate_batch(
        self, descriptions: list[str], max_new_tokens: int = 512, batch_size: int = 8
    ) -> list[str]:
        """Batched local generation (left-padded), mirroring the Modal path.

        Same system prompt + `enable_thinking=False` as `infer.generate`, so the
        outputs match single-shot generation.
        """
        import torch

        self.load()
        tok, model, device = self._tok, self._model, self._device
        prev_side = tok.padding_side
        tok.padding_side = "left"  # decoder-only: left-pad so gen tokens align
        outs: list[str] = []
        try:
            for start in range(0, len(descriptions), batch_size):
                batch = descriptions[start : start + batch_size]
                prompts = [
                    tok.apply_chat_template(
                        build_messages(d),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for d in batch
                ]
                inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    g = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                gen = g[:, inputs["input_ids"].shape[1] :]
                outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        finally:
            tok.padding_side = prev_side
        return outs


# --------------------------------------------------------------------------- #
# frontier (gateway, construction prompt)
# --------------------------------------------------------------------------- #
@dataclass
class GenResult:
    """One model generation attempt (specialist or frontier)."""

    text: str
    source: str  # e.g. "specialist" or the gateway model id
    ok: bool
    latency_s: float
    error: str | None = None
    finish_reason: str | None = None


def frontier_generate(
    description: str,
    model: str,
    *,
    construction: bool = True,
    client=None,
    max_tokens: int = 4096,
) -> GenResult:
    """Generate a figure from a hosted frontier model via the gateway.

    ``construction=True`` (the default, per the coordinate-free spec) uses
    ``CONSTRUCTION_SYSTEM_PROMPT`` so the model returns tkz-euclide / ``calc``
    constructions rather than hardcoded coordinates.
    """
    messages = build_construction_messages(description) if construction else build_messages(description)
    res = gateway.chat(messages, model, client=client, max_tokens=max_tokens)
    return GenResult(
        text=res.text,
        source=model,
        ok=res.ok,
        latency_s=res.latency_s,
        error=res.error,
        finish_reason=res.finish_reason,
    )


def specialist_latency_generate(spec: Specialist, description: str, max_new_tokens: int = 512) -> GenResult:
    """Single-shot specialist generation with wall-clock latency measured."""
    t0 = time.time()
    text = spec.generate(description, max_new_tokens=max_new_tokens)
    return GenResult(text=text, source="specialist", ok=bool(text.strip()),
                     latency_s=round(time.time() - t0, 3))


# --------------------------------------------------------------------------- #
# compile + render
# --------------------------------------------------------------------------- #
@dataclass
class RenderResult:
    """Outcome of compiling a figure and rasterising it to PNG."""

    compiles: bool
    degenerate: bool
    png_path: str | None
    reason: str
    ink_ratio: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def ok(self) -> bool:
        """A genuinely usable figure: compiled AND drew something."""
        return self.compiles and not self.degenerate and self.png_path is not None


def wrap_construction(tikz: str) -> str:
    return RICH_TEMPLATE.replace("__BODY__", tikz)


# A figure whose page exceeds this (in TeX points; 1pt = 1/72") is treated as
# degenerate: OOD inputs make the specialist blow the canvas up to tens of inches
# (a real failure, and rasterising it would be a decompression bomb).
MAX_PAGE_PT = 2200.0


def _rasterise(pdf_path: Path, png_path: Path, dpi: int) -> tuple[float, float, int, int, str]:
    """Render page 1 to a colour PNG.

    Returns (ink_ratio, dark_ratio, width_px, height_px, note). ``note`` is
    "oversized" (page too big -> not rendered) or "" (rendered ok). ink_ratio is
    the fraction of non-white pixels (any mark); dark_ratio is the fraction of
    near-black pixels (heavy fill), used to catch solid-blob "figures".
    """
    import fitz  # pymupdf
    import numpy as np

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        rect = page.rect
        if rect.width > MAX_PAGE_PT or rect.height > MAX_PAGE_PT or rect.width < 1 or rect.height < 1:
            return 0.0, 0.0, int(rect.width), int(rect.height), "oversized"
        pix = page.get_pixmap(dpi=dpi)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
        gray = page.get_pixmap(dpi=72, colorspace=fitz.csGRAY)
        arr = np.frombuffer(gray.samples, dtype=np.uint8).reshape(gray.height, gray.width)
        ink_ratio = float((arr < 250).mean())   # any mark (vs near-white)
        dark_ratio = float((arr < 80).mean())    # heavy near-black fill
        return ink_ratio, dark_ratio, pix.width, pix.height, ""
    finally:
        doc.close()


def compile_and_render(
    figure: str,
    png_path: str | Path,
    *,
    dpi: int = 150,
    timeout: int = 60,
    min_ink: float = 0.0015,
    max_dark: float = 0.5,
) -> RenderResult:
    """Extract the tikzpicture, compile it (tkz-euclide preamble), render to PNG.

    ``degenerate`` is True when the page compiled but is not a usable figure:
      * blank      -- ink_ratio < ``min_ink`` ("compiled" but drew nothing),
      * solid blob -- dark_ratio > ``max_dark`` (a giant filled shape, e.g. the
                      specialist collapsing an OOD scene to a black disk),
      * oversized  -- page larger than MAX_PAGE_PT (OOD scenes blow up the canvas).
    These guards keep "it compiled" from silently meaning "usable illustration".
    """
    png_path = Path(png_path)
    tikz = metrics.extract_tikz(figure or "")
    if not tikz:
        return RenderResult(False, True, None, "no-figure")
    if not tex.has_tectonic():
        return RenderResult(False, True, None, "tectonic-not-installed")

    tmp = Path(tempfile.mkdtemp(prefix="geoserve_"))
    try:
        tex_path = tmp / "fig.tex"
        tex_path.write_text(wrap_construction(tikz))
        try:
            proc = subprocess.run(
                ["tectonic", "-X", "compile", "--outfmt", "pdf", "-o", str(tmp), str(tex_path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RenderResult(False, True, None, "timeout")
        except FileNotFoundError:
            return RenderResult(False, True, None, "tectonic-not-installed")

        pdf_path = tmp / "fig.pdf"
        if proc.returncode != 0 or not pdf_path.exists():
            return RenderResult(False, True, None, f"exit={proc.returncode}")
        try:
            ink, dark, w, h, note = _rasterise(pdf_path, png_path, dpi)
        except Exception as e:  # noqa: BLE001 - render is best-effort
            return RenderResult(True, True, None, f"raster-error: {type(e).__name__}")
        if note == "oversized":
            return RenderResult(True, True, None, "oversized", width=w, height=h)
        if ink < min_ink:
            reason = "blank-page"
        elif dark > max_dark:
            reason = "solid-blob"
        else:
            reason = "ok"
        degenerate = reason != "ok"
        # The PNG is written either way (it exists on disk); callers gate on
        # `.ok`, which is False for a degenerate page.
        return RenderResult(True, degenerate, str(png_path), reason,
                            ink_ratio=round(ink, 5), width=w, height=h)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# label tidying (legibility) — used on every render path
# --------------------------------------------------------------------------- #
_LABELPOINTS_RE = re.compile(r"\\tkzLabelPoints(\[[^\]]*\])?\(([^()]*)\)")
# Point-name nodes that duplicate \tkzLabelPoints (same glyph, same anchor).
_POINT_NODE_RE = re.compile(
    r"\\node\s*\[[^\]]*\]\s*at\s*\(\s*([A-Za-z][A-Za-z0-9]*)\s*\)\s*\{\s*\$?\1\$?\s*\}\s*;",
    re.I,
)
# Midway / pos length labels that often collide with endpoint point labels.
_MIDWAY_NODE_RE = re.compile(
    r"((?:\\(?:draw|path|filldraw)\b[^\n]*?)|)"
    r"\bnode\s*\[([^\]]*?\b(?:midway|pos\s*=\s*0\.5)[^\]]*?)\]\s*(\{[^}]*\})",
    re.I,
)
# Inline point / annotation labels: \fill (P) ... node[below left] {$P$};
_INLINE_LABEL_RE = re.compile(
    r"(\\(?:fill|filldraw|draw)\b(?:\s*\[[^\]]*\])?\s*"
    r"\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)[^;]*?)"
    r"(\bnode\s*(?:\[([^\]]*)\])?\s*(\{[^}]*\}))",
    re.I,
)
# Extra separation so labels sit clear of strokes (no fill/halo behind text).
_LABEL_SEP = "4pt"
_LABEL_SEP_FAR = "8pt"
_LABEL_SEP_MAX = "14pt"
_DIRS = (
    "above", "above right", "right", "below right",
    "below", "below left", "left", "above left",
)
# Unit offsets for approximate label-box centers (scene units).
_DIR_OFFSETS = (
    (0.0, 1.0), (0.7, 0.7), (1.0, 0.0), (0.7, -0.7),
    (0.0, -1.0), (-0.7, -0.7), (-1.0, 0.0), (-0.7, 0.7),
)
_PLACE_OPT_RE = re.compile(
    r"^(?:anchor\s*=\s*.+|(?:above|below)(?:\s+(?:left|right))?|left|right)"
    r"(?:\s*=\s*.+)?$",
    re.I,
)


def _label_direction(name: str, pts: dict, cx: float, cy: float, *, sep: str = _LABEL_SEP) -> str:
    """8-way tikz placement pointing radially OUTWARD from the figure centroid."""
    import math

    p = pts.get(name)
    if not p:
        return f"above right={sep}"
    dx, dy = p[0] - cx, p[1] - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return f"above right={sep}"
    a = math.degrees(math.atan2(dy, dx))  # -180..180
    for lo, d in [(-157.5, "left"), (-112.5, "below left"), (-67.5, "below"),
                  (-22.5, "below right"), (22.5, "right"), (67.5, "above right"),
                  (112.5, "above"), (157.5, "above left")]:
        if a < lo:
            return f"{d}={sep}"
    return f"left={sep}"


def _dir_index(a: float) -> int:
    """Map atan2 degrees to one of 8 compass slots (see _DIRS)."""
    for i, lo in enumerate([-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]):
        if a < lo:
            # bins -> left, below left, below, below right, right, above right, above, above left
            mapping = [6, 5, 4, 3, 2, 1, 0, 7]
            return mapping[i]
    return 6  # left


def _assign_noncolliding_dirs(pts: dict, cx: float, cy: float) -> dict[str, str]:
    """Outward dirs with collision avoidance for nearby points."""
    import math

    names = list(pts.keys())
    preferred: dict[str, int] = {}
    for n, (x, y) in pts.items():
        dx, dy = x - cx, y - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            preferred[n] = 1  # above right
        else:
            preferred[n] = _dir_index(math.degrees(math.atan2(dy, dx)))

    if len(pts) >= 2:
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    else:
        span = 1.0
    near = 0.18 * span

    assigned: dict[str, int] = {}
    seps: dict[str, str] = {}
    order = sorted(names, key=lambda n: -((pts[n][0] - cx) ** 2 + (pts[n][1] - cy) ** 2))
    for n in order:
        slot = preferred[n]
        sep = _LABEL_SEP
        for _try in range(8):
            conflict = False
            for m, mslot in assigned.items():
                dist = math.hypot(pts[n][0] - pts[m][0], pts[n][1] - pts[m][1])
                if dist > near:
                    continue
                dslot = min((slot - mslot) % 8, (mslot - slot) % 8)
                if dslot <= 1:
                    conflict = True
                    break
            if not conflict:
                break
            slot = (slot + 1) % 8
            sep = _LABEL_SEP_FAR
        assigned[n] = slot
        seps[n] = sep

    return {n: f"{_DIRS[assigned[n]]}={seps[n]}" for n in names}


def _nudge_midway_labels(tikz: str) -> str:
    """Push length/midway nodes off sides that clash with endpoint point labels."""
    flip = {
        "left": "above",
        "right": "below",
        "above": "right",
        "below": "left",
    }

    def _repl(m: re.Match) -> str:
        prefix, opts, body = m.group(1) or "", m.group(2) or "", m.group(3) or ""
        opts_l = opts.lower()
        if re.search(r"(above|below|left|right)\s*=\s*\d", opts_l):
            return m.group(0)
        new_opts = opts
        moved = False
        for src, dst in flip.items():
            if re.search(rf"\b{src}\b", opts_l) and not re.search(rf"\b{dst}\b", opts_l):
                new_opts = re.sub(rf"\b{src}\b", f"{dst}={_LABEL_SEP_FAR}", opts, count=1, flags=re.I)
                moved = True
                break
        if not moved:
            if not any(s in opts_l for s in ("above", "below", "left", "right")):
                new_opts = f"{opts}, above={_LABEL_SEP_FAR}" if opts.strip() else f"above={_LABEL_SEP_FAR}"
            else:
                new_opts = f"{opts}, above={_LABEL_SEP_FAR}"
        return f"{prefix}node[{new_opts}] {body}"

    return _MIDWAY_NODE_RE.sub(_repl, tikz)


def _find_balanced(s: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """Index of matching closer for s[start]==open_ch, or None."""
    if start >= len(s) or s[start] != open_ch:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def _plain_label_text(body: str) -> str:
    """Strip TikZ/math wrappers for length estimates (C_1, r_1=30, sqrt...)."""
    t = (body or "").strip()
    if t.startswith("{") and t.endswith("}"):
        t = t[1:-1].strip()
    if t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()
    t = re.sub(r"\\(?:sqrt|frac|mathrm|text|textbf|textit)\b", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?|[{}^_]", "", t)
    return re.sub(r"\s+", "", t)


def _set_placement_opts(opts: str, dir_spec: str) -> str:
    """Replace compass/anchor placement; keep colors and other styles."""
    kept = [p.strip() for p in (opts or "").split(",") if p.strip() and not _PLACE_OPT_RE.match(p.strip())]
    return ", ".join([dir_spec] + kept)


def _sep_for_text(text: str, *, far: bool = False) -> str:
    if far or len(text) >= 6:
        return _LABEL_SEP_MAX
    if len(text) >= 3:
        return _LABEL_SEP_FAR
    return _LABEL_SEP


def _label_box(
    x: float, y: float, slot: int, text: str, span: float, *, far: bool = False
) -> tuple[float, float, float, float]:
    """Axis-aligned box for a label placed at (x,y) toward compass slot."""
    # Inflated vs rendered text: short scene spans (scaled TikZ) still need
    # generous boxes so formula labels like r_1=30 / w_2=45 collide.
    char_w = max(span * 0.045, 3.0)
    h = max(span * 0.08, 6.0)
    w = max(len(text) * char_w * 0.7, char_w * 2.5)
    ox, oy = _DIR_OFFSETS[slot]
    dist = span * (0.07 if far else 0.045) + (0.35 * w if abs(ox) > 0.5 else 0.28 * h)
    cx = x + ox * dist
    cy = y + oy * dist
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _boxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float
) -> bool:
    return not (
        a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]
    )


def _iter_standalone_nodes(tikz: str):
    """Yield (start, end, opts, at_expr, body) for ``\\node[...] at (...) {...};``."""
    i = 0
    n = len(tikz)
    while i < n:
        j = tikz.find(r"\node", i)
        if j < 0:
            return
        # Word-boundary: reject \nodepart / \newcommand etc.
        after = j + 5
        if after < n and (tikz[after].isalnum() or tikz[after] == "@"):
            i = after
            continue
        k = after
        while k < n and tikz[k].isspace():
            k += 1
        opts = ""
        if k < n and tikz[k] == "[":
            end = _find_balanced(tikz, k, "[", "]")
            if end is None:
                i = after
                continue
            opts = tikz[k + 1 : end]
            k = end + 1
            while k < n and tikz[k].isspace():
                k += 1
        # Optional node name: \node (foo) at ...
        if k < n and tikz[k] == "(":
            end = _find_balanced(tikz, k, "(", ")")
            if end is None:
                i = after
                continue
            maybe = tikz[k + 1 : end].strip()
            k2 = end + 1
            while k2 < n and tikz[k2].isspace():
                k2 += 1
            if k2 < n and tikz.startswith("at", k2):
                # named node with at — skip the name group
                k = k2
            elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", maybe or ""):
                # bare named node without at — not a floating label
                i = end + 1
                continue
            else:
                # Unusual; bail on this match
                i = after
                continue
        if k >= n or not tikz.startswith("at", k):
            i = after
            continue
        k += 2
        while k < n and tikz[k].isspace():
            k += 1
        if k >= n or tikz[k] != "(":
            i = after
            continue
        end = _find_balanced(tikz, k, "(", ")")
        if end is None:
            i = after
            continue
        at_expr = tikz[k + 1 : end]
        k = end + 1
        while k < n and tikz[k].isspace():
            k += 1
        if k >= n or tikz[k] != "{":
            i = after
            continue
        end = _find_balanced(tikz, k, "{", "}")
        if end is None:
            i = after
            continue
        body = tikz[k : end + 1]
        k = end + 1
        while k < n and tikz[k].isspace():
            k += 1
        if k < n and tikz[k] == ";":
            k += 1
        yield j, k, opts, at_expr, body
        i = k


def _resolve_label_xy(at_expr: str, registry: dict) -> tuple[float, float] | None:
    """Resolve a TikZ ``at (...)`` expression via metrics point parsing."""
    try:
        pt = metrics._parse_point(at_expr.strip(), registry)
    except Exception:  # noqa: BLE001
        return None
    return pt


def _spread_annotation_labels(tikz: str) -> str:
    """Spread letter + formula node labels so nearby text does not overlap.

    Handles ``\\node[...] at (...) {text}`` (including ``r_1=30``, ``\\sqrt{...}``)
    and inline ``\\fill (P) ... node[...] {text}``. Prefers outward compass
    slots from the figure centroid; uses approximate boxes for collisions.
    """
    import math

    registry = metrics.parse_named_coords(tikz)
    if not registry:
        return tikz

    sites: list[dict] = []

    for start, end, opts, at_expr, body in _iter_standalone_nodes(tikz):
        # Skip empty / purely decorative nodes.
        text = _plain_label_text(body)
        if not text:
            continue
        # Skip nodes that only draw a mark with no visible text-ish content.
        if re.fullmatch(r"[.,;:|]+", text):
            continue
        xy = _resolve_label_xy(at_expr, registry)
        if xy is None:
            continue
        sites.append({
            "kind": "standalone",
            "start": start,
            "end": end,
            "opts": opts,
            "body": body,
            "at_expr": at_expr,
            "x": xy[0],
            "y": xy[1],
            "text": text,
        })

    for m in _INLINE_LABEL_RE.finditer(tikz):
        name, opts, body = m.group(2), m.group(4) or "", m.group(5)
        # Skip midway/path labels (handled elsewhere).
        if re.search(r"\b(?:midway|pos\s*=)", opts, re.I):
            continue
        text = _plain_label_text(body)
        if not text:
            continue
        xy = registry.get(name)
        if xy is None:
            continue
        # Avoid double-counting if this span is already a standalone \node.
        if any(s["start"] <= m.start(3) < s["end"] for s in sites):
            continue
        sites.append({
            "kind": "inline",
            "start": m.start(3),
            "end": m.end(3),
            "opts": opts,
            "body": body,
            "prefix": m.group(1),
            "node_prefix": "node",
            "x": xy[0],
            "y": xy[1],
            "text": text,
            "name": name,
        })

    if not sites:
        return tikz

    xs = [s["x"] for s in sites] + [p[0] for p in registry.values()]
    ys = [s["y"] for s in sites] + [p[1] for p in registry.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    pad = 0.01 * span

    # Outermost first so corner letters keep outward slots; then denser /
    # longer labels claim remaining free compass room.
    def _priority(s: dict) -> tuple:
        dist2 = (s["x"] - cx) ** 2 + (s["y"] - cy) ** 2
        nearby = sum(
            1 for t in sites
            if math.hypot(s["x"] - t["x"], s["y"] - t["y"]) < 0.22 * span
        )
        short_outer = 0 if (len(s["text"]) <= 2 and dist2 > (0.28 * span) ** 2) else 1
        return (short_outer, -dist2, -nearby, -len(s["text"]))

    order = sorted(range(len(sites)), key=lambda i: _priority(sites[i]))
    placed_boxes: list[tuple[float, float, float, float]] = []
    placed_meta: list[tuple[float, float, int]] = []  # x, y, slot
    choices: dict[int, tuple[int, bool]] = {}
    near = 0.20 * span

    def _inward_penalty(slot: int, x: float, y: float) -> int:
        """Penalize compass dirs that point back toward the figure centroid."""
        ox, oy = _DIR_OFFSETS[slot]
        tx, ty = cx - x, cy - y
        norm = math.hypot(tx, ty)
        if norm < 1e-9:
            return 0
        # Cosine of angle between label offset and vector-to-centroid.
        cos = (ox * tx + oy * ty) / (math.hypot(ox, oy) * norm)
        if cos > 0.55:
            return 12
        if cos > 0.15:
            return 5
        return 0

    for idx in order:
        s = sites[idx]
        dx, dy = s["x"] - cx, s["y"] - cy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            preferred = 1  # above right
        else:
            preferred = _dir_index(math.degrees(math.atan2(dy, dx)))

        best: tuple[int, int, bool, tuple[float, float, float, float]] | None = None
        for far in (False, True):
            for step in range(8):
                slot = (preferred + step) % 8
                # Nearby anchors must not share the same / adjacent compass slot.
                slot_clash = False
                for px, py, pslot in placed_meta:
                    if math.hypot(s["x"] - px, s["y"] - py) > near:
                        continue
                    dslot = min((slot - pslot) % 8, (pslot - slot) % 8)
                    if dslot <= 1:
                        slot_clash = True
                        break
                if slot_clash:
                    continue
                box = _label_box(s["x"], s["y"], slot, s["text"], span, far=far)
                if any(_boxes_overlap(box, pb, pad) for pb in placed_boxes):
                    continue
                score = (
                    step
                    + (2 if far else 0)
                    + _inward_penalty(slot, s["x"], s["y"])
                )
                if best is None or score < best[0]:
                    best = (score, slot, far, box)
            # Prefer resolving with far sep before accepting a very inward slot.
            if best is not None and best[0] <= 4:
                break
        if best is None:
            # Fall back: ignore soft slot adjacency, keep box + inward checks.
            for far in (True, False):
                for step in range(8):
                    slot = (preferred + step) % 8
                    box = _label_box(s["x"], s["y"], slot, s["text"], span, far=far)
                    if any(_boxes_overlap(box, pb, pad) for pb in placed_boxes):
                        continue
                    score = step + _inward_penalty(slot, s["x"], s["y"]) + (1 if not far else 0)
                    if best is None or score < best[0]:
                        best = (score, slot, far, box)
                if best is not None and best[0] <= 6:
                    break
        if best is None:
            slot, far = preferred, True
            for step in range(8):
                cand = (preferred + step * 2) % 8  # skip every other slot
                if _inward_penalty(cand, s["x"], s["y"]) >= 12:
                    continue
                if all(
                    math.hypot(s["x"] - px, s["y"] - py) > near
                    or min((cand - pslot) % 8, (pslot - cand) % 8) > 1
                    for px, py, pslot in placed_meta
                ):
                    slot = cand
                    break
            box = _label_box(s["x"], s["y"], slot, s["text"], span, far=True)
        else:
            _, slot, far, box = best
        choices[idx] = (slot, far)
        placed_boxes.append(box)
        placed_meta.append((s["x"], s["y"], slot))

    # Rewrite from the end so earlier spans stay valid.
    out = tikz
    for idx in sorted(choices.keys(), key=lambda i: sites[i]["start"], reverse=True):
        s = sites[idx]
        slot, far = choices[idx]
        sep = _sep_for_text(s["text"], far=far)
        dir_spec = f"{_DIRS[slot]}={sep}"
        new_opts = _set_placement_opts(s["opts"], dir_spec)
        if s["kind"] == "standalone":
            piece = f"\\node[{new_opts}] at ({s['at_expr']}) {s['body']};"
            out = out[: s["start"]] + piece + out[s["end"] :]
        else:
            piece = f"node[{new_opts}] {s['body']}"
            out = out[: s["start"]] + piece + out[s["end"] :]
    return out


def tidy_labels(tikz: str, *, timeout: int = 60) -> str:
    """Make point / length / annotation labels legible without changing geometry.

    Rewrites ``\\tkzLabelPoints``, drops duplicate point-name nodes, spreads nearby
    point and formula labels onto different compass slots (outward, collision-aware),
    and nudges midway length labels. No white halo. Best-effort: returns the
    original on failure.
    """
    try:
        src = tikz or ""
        if not src.strip():
            return tikz

        names: list[str] = []
        for m in _LABELPOINTS_RE.finditer(src):
            for n in m.group(2).split(","):
                n = n.strip()
                if n and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", n) and n not in names:
                    names.append(n)
        for m in _POINT_NODE_RE.finditer(src):
            n = m.group(1)
            if n not in names:
                names.append(n)

        out = src
        pts: dict = {}
        if names:
            try:
                from . import extract

                coords = extract.extract_named_coords(src, names, timeout=timeout)
                pts = {n: c for n, c in coords.items() if c is not None}
            except Exception:  # noqa: BLE001
                pts = {}

        if len(pts) >= 2:
            cx = sum(p[0] for p in pts.values()) / len(pts)
            cy = sum(p[1] for p in pts.values()) / len(pts)
            dirs = _assign_noncolliding_dirs(pts, cx, cy)

            def _repl_lp(m):
                ns = [n.strip() for n in m.group(2).split(",") if n.strip()]
                return "\n  ".join(
                    f"\\node[{dirs.get(n, _label_direction(n, pts, cx, cy))}] at ({n}) {{${n}$}};"
                    for n in ns)

            had_lp = bool(_LABELPOINTS_RE.search(out))
            # Drop pre-existing duplicate point-name nodes BEFORE rewriting
            # \tkzLabelPoints, so the new nodes are not immediately deleted.
            if had_lp:
                out = _POINT_NODE_RE.sub("", out)
                out = _LABELPOINTS_RE.sub(_repl_lp, out)
            else:
                def _repl_pn(m):
                    n = m.group(1)
                    if n in dirs:
                        return f"\\node[{dirs[n]}] at ({n}) {{${n}$}};"
                    return m.group(0)

                out = _POINT_NODE_RE.sub(_repl_pn, out)
            out = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", out)
        elif _LABELPOINTS_RE.search(out):
            cx = cy = 0.0
            if pts:
                cx = sum(p[0] for p in pts.values()) / len(pts)
                cy = sum(p[1] for p in pts.values()) / len(pts)

            def _repl(m):
                ns = [n.strip() for n in m.group(2).split(",") if n.strip()]
                return "\n  ".join(
                    f"\\node[{_label_direction(n, pts, cx, cy)}] at ({n}) {{${n}$}};"
                    for n in ns)

            out = _LABELPOINTS_RE.sub(_repl, out)

        out = _spread_annotation_labels(out)
        out = _nudge_midway_labels(out)
        return out
    except Exception:  # noqa: BLE001 - never let tidying break the render path
        return tikz


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
class Cache:
    """Append-only JSONL cache of raw model outputs, keyed by a string key.

    Used so re-running a surface never re-spends gateway budget or re-runs local
    inference for descriptions already seen. Thread-safe for concurrent writers.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line
                if "key" in rec:
                    self._mem[rec["key"]] = rec  # later lines win

    def get(self, key: str) -> dict | None:
        return self._mem.get(key)

    def put(self, key: str, record: dict) -> dict:
        rec = {"key": key, **record}
        with self._lock:
            self._mem[key] = rec
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        return rec

    def __contains__(self, key: str) -> bool:
        return key in self._mem


# --------------------------------------------------------------------------- #
# high-level: illustrate one scene (specialist first, frontier fallback)
# --------------------------------------------------------------------------- #
@dataclass
class Illustration:
    """Full record of illustrating one scene end to end."""

    description: str
    route: str  # "specialist" | "frontier" | "none"
    png_path: str | None
    tikz: str | None
    specialist: dict = field(default_factory=dict)
    frontier: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def illustrate(
    description: str,
    png_path: str | Path,
    *,
    specialist: Specialist | None = None,
    frontier_model: str | None = None,
    dpi: int = 150,
    max_new_tokens: int = 512,
) -> Illustration:
    """Try the specialist, then (optionally) a frontier fallback.

    Returns which route produced a usable (compiling, non-degenerate)
    coordinate-free figure, the winning TikZ, and the PNG path.
    """
    png_path = Path(png_path)
    spec = specialist or Specialist()

    # 1) specialist
    sres = specialist_latency_generate(spec, description, max_new_tokens=max_new_tokens)
    s_tikz = metrics.extract_tikz(sres.text or "") or (sres.text or "")
    s_body = tidy_labels(s_tikz) if s_tikz.strip() else s_tikz
    srender = compile_and_render(s_body or sres.text, png_path, dpi=dpi)
    if not srender.ok and s_body != s_tikz:
        s_body = s_tikz
        srender = compile_and_render(s_body or sres.text, png_path, dpi=dpi)
    spec_info = {"ok": srender.ok, "compiles": srender.compiles,
                 "degenerate": srender.degenerate, "reason": srender.reason,
                 "latency_s": sres.latency_s, "tikz": metrics.extract_tikz(s_body or "")}
    if srender.ok:
        return Illustration(description, "specialist", str(png_path),
                            spec_info["tikz"], specialist=spec_info)

    # 2) frontier fallback (construction prompt)
    if frontier_model:
        fres = frontier_generate(description, frontier_model, construction=True)
        f_tikz = metrics.extract_tikz(fres.text or "") or (fres.text or "")
        f_body = tidy_labels(f_tikz) if f_tikz.strip() else f_tikz
        frender = compile_and_render(f_body or fres.text, png_path, dpi=dpi)
        if not frender.ok and f_body != f_tikz:
            f_body = f_tikz
            frender = compile_and_render(f_body or fres.text, png_path, dpi=dpi)
        front_info = {"ok": frender.ok, "compiles": frender.compiles,
                      "degenerate": frender.degenerate, "reason": frender.reason,
                      "latency_s": fres.latency_s, "model": frontier_model,
                      "tikz": metrics.extract_tikz(f_body or "")}
        if frender.ok:
            return Illustration(description, "frontier", str(png_path),
                                front_info["tikz"], specialist=spec_info, frontier=front_info)
        return Illustration(description, "none", None, None,
                            specialist=spec_info, frontier=front_info)

    return Illustration(description, "none", None, None, specialist=spec_info)
