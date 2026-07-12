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
# label tidying (legibility) — used on the specialist route
# --------------------------------------------------------------------------- #
_LABELPOINTS_RE = re.compile(r"\\tkzLabelPoints(\[[^\]]*\])?\(([^()]*)\)")
# Extra separation so labels sit clear of strokes (no fill/halo behind text).
_LABEL_SEP = "3pt"


def _label_direction(name: str, pts: dict, cx: float, cy: float) -> str:
    """8-way tikz placement pointing radially OUTWARD from the figure centroid."""
    import math

    p = pts.get(name)
    if not p:
        return f"above right={_LABEL_SEP}"
    dx, dy = p[0] - cx, p[1] - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return f"above right={_LABEL_SEP}"
    a = math.degrees(math.atan2(dy, dx))  # -180..180
    for lo, d in [(-157.5, "left"), (-112.5, "below left"), (-67.5, "below"),
                  (-22.5, "below right"), (22.5, "right"), (67.5, "above right"),
                  (112.5, "above"), (157.5, "above left")]:
        if a < lo:
            return f"{d}={_LABEL_SEP}"
    return f"left={_LABEL_SEP}"


def tidy_labels(tikz: str, *, timeout: int = 60) -> str:
    """Make tkz-euclide point labels legible: push each ``\\tkzLabelPoints`` label
    radially OUTWARD (away from the figure centroid) so labels avoid strokes.

    No white halo / background fill — text only, offset outside the shape.

    Best-effort and non-destructive: this ONLY rewrites label placement,
    never geometry, and returns the ORIGINAL tikz unchanged if it can't safely
    rewrite (including when coords can't be extracted). Callers must still
    compile-check the result and fall back to the original on any failure.
    """
    try:
        if not _LABELPOINTS_RE.search(tikz or ""):
            return tikz
        names: list[str] = []
        for m in _LABELPOINTS_RE.finditer(tikz):
            for n in m.group(2).split(","):
                n = n.strip()
                if n and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", n) and n not in names:
                    names.append(n)
        if not names:
            return tikz

        # True coords of every labeled point (handles tkzDef* derived points via a
        # compile-extract read-back). If it can't, leave labels untouched.
        pts: dict = {}
        try:
            from . import extract

            coords = extract.extract_named_coords(tikz, names, timeout=timeout)
            pts = {n: c for n, c in coords.items() if c is not None}
        except Exception:  # noqa: BLE001
            pts = {}

        if len(pts) < 2:
            return tikz

        cx = sum(p[0] for p in pts.values()) / len(pts)
        cy = sum(p[1] for p in pts.values()) / len(pts)

        def _repl(m):
            ns = [n.strip() for n in m.group(2).split(",") if n.strip()]
            return "\n  ".join(
                f"\\node[{_label_direction(n, pts, cx, cy)}] at ({n}) {{${n}$}};"
                for n in ns)

        return _LABELPOINTS_RE.sub(_repl, tikz)
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
    srender = compile_and_render(sres.text, png_path, dpi=dpi)
    spec_info = {"ok": srender.ok, "compiles": srender.compiles,
                 "degenerate": srender.degenerate, "reason": srender.reason,
                 "latency_s": sres.latency_s, "tikz": metrics.extract_tikz(sres.text or "")}
    if srender.ok:
        return Illustration(description, "specialist", str(png_path),
                            spec_info["tikz"], specialist=spec_info)

    # 2) frontier fallback (construction prompt)
    if frontier_model:
        fres = frontier_generate(description, frontier_model, construction=True)
        frender = compile_and_render(fres.text, png_path, dpi=dpi)
        front_info = {"ok": frender.ok, "compiles": frender.compiles,
                      "degenerate": frender.degenerate, "reason": frender.reason,
                      "latency_s": fres.latency_s, "model": frontier_model,
                      "tikz": metrics.extract_tikz(fres.text or "")}
        if frender.ok:
            return Illustration(description, "frontier", str(png_path),
                                front_info["tikz"], specialist=spec_info, frontier=front_info)
        return Illustration(description, "none", None, None,
                            specialist=spec_info, frontier=front_info)

    return Illustration(description, "none", None, None, specialist=spec_info)
