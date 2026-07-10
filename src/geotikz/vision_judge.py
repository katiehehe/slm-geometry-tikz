"""Vision LLM-as-judge — does a RENDERED figure faithfully depict the problem?

Coordinate matching (``extract.grade``) is the gold correctness signal, but it
only works when a problem has clean ground-truth named-point coordinates. Many
real competition figures do NOT: 3D perspective solids, combinatorial / lattice
/ tiling diagrams, shaded-region and area problems, configurations with no clean
named points. Discarding all of those (as a coordinate-only or 3D-rejecting text
gate would) throws away exactly the figures that widen coverage.

Instead, for a figure that COMPILES to a non-degenerate picture, we render it to
PNG and show a capable VISION model the problem TEXT alongside the rendered
FIGURE, asking whether the drawing faithfully depicts the described
configuration. This KEEPS verified 3D/combinatorial figures.

This is a SOFTER signal than coordinate verification — a judge can be fooled, and
it certifies "looks right", not "provably right". Callers should report
judge-verified coverage SEPARATELY from coordinate-verified coverage (see
``VisionVerdict.mode``). If the gateway rejects image inputs, we fall back to a
TEXT judge over the TikZ source and flag the weaker signal via ``mode="text"``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from . import gateway

VISION_RUBRIC = (
    "You validate auto-generated diagrams for competition geometry problems. You "
    "are given the PROBLEM text and a rendered FIGURE (an image). Decide whether "
    "the figure FAITHFULLY DEPICTS the configuration the problem describes: the "
    "correct objects (triangles, circles, polygons, 3D solids, lattices, regions, "
    "...), their key relationships and incidences, and the overall structure.\n"
    "- 3D solids drawn in perspective, combinatorial / lattice / tiling diagrams, "
    "and shaded-region figures ARE acceptable when they match the problem. Do NOT "
    "reject a figure merely for being 3D, combinatorial, or schematic.\n"
    "- Reject ONLY if the figure is blank/garbled, depicts a DIFFERENT "
    "configuration, is a generic placeholder, or omits the problem's central "
    "objects.\n"
    "- Ignore exact measurements, missing labels, colors, and styling.\n"
    'Return STRICT JSON only: {"approved": true|false, "reason": "<short>"}'
)

TEXT_RUBRIC = (
    "You validate auto-generated diagrams for competition geometry problems. You "
    "are given the PROBLEM text and the FIGURE's TikZ/tkz-euclide SOURCE (it "
    "compiles to a non-blank picture, but you cannot see the rendering). Decide "
    "whether the construction faithfully depicts the configuration the problem "
    "describes: the correct objects and their relationships/incidences.\n"
    "- 3D, combinatorial, lattice, and region constructions ARE acceptable when "
    "they match the problem. Do NOT reject merely for being 3D or combinatorial.\n"
    "- Reject ONLY if the construction clearly depicts a DIFFERENT configuration, "
    "is a generic placeholder, or omits the problem's central objects.\n"
    'Return STRICT JSON only: {"approved": true|false, "reason": "<short>"}'
)

_IMG_UNSUPPORTED = (
    "image", "multimodal", "vision", "content must be a string",
    "invalid type for 'content'", "does not support", "image_url",
)


@dataclass
class VisionVerdict:
    approved: bool
    reason: str
    mode: str  # "vision" | "text" | "error"
    raw: str = ""


def _data_uri(png_path: str | Path) -> str:
    b = Path(png_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def _parse(txt: str) -> tuple[bool | None, str]:
    a, b = txt.find("{"), txt.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            o = json.loads(txt[a:b + 1])
            if "approved" in o:
                return bool(o["approved"]), str(o.get("reason", ""))[:220]
        except Exception:  # noqa: BLE001
            pass
    low = txt.strip().lower()
    if low.startswith("yes"):
        return True, txt[:220]
    if low.startswith("no"):
        return False, txt[:220]
    return None, txt[:220]


def _looks_image_unsupported(err: str) -> bool:
    e = (err or "").lower()
    return any(s in e for s in _IMG_UNSUPPORTED)


def judge_vision(problem: str, png_path: str | Path, model: str,
                 max_tokens: int = 1024) -> VisionVerdict:
    msgs = [
        {"role": "system", "content": VISION_RUBRIC},
        {"role": "user", "content": [
            {"type": "text", "text": f"PROBLEM:\n{problem}\n\nDoes the FIGURE "
                                     "faithfully depict this configuration?"},
            {"type": "image_url", "image_url": {"url": _data_uri(png_path)}},
        ]},
    ]
    res = gateway.chat(msgs, model, max_tokens=max_tokens)
    if not res.ok:
        return VisionVerdict(False, f"vision-error: {res.error}", "error", "")
    ap, reason = _parse(res.text or "")
    if ap is None:
        return VisionVerdict(False, "unparseable", "vision", (res.text or "")[:220])
    return VisionVerdict(ap, reason, "vision", (res.text or "")[:220])


def judge_text(problem: str, tikz: str, model: str,
               max_tokens: int = 2048) -> VisionVerdict:
    msgs = [
        {"role": "system", "content": TEXT_RUBRIC},
        {"role": "user", "content": f"PROBLEM:\n{problem}\n\nFIGURE (TikZ):\n{tikz}"},
    ]
    res = gateway.chat(msgs, model, max_tokens=max_tokens)
    if not res.ok:
        return VisionVerdict(False, f"text-error: {res.error}", "error", "")
    ap, reason = _parse(res.text or "")
    if ap is None:
        return VisionVerdict(False, "unparseable", "text", (res.text or "")[:220])
    return VisionVerdict(ap, reason, "text", (res.text or "")[:220])


def judge(problem: str, png_path: str | Path | None, tikz: str, model: str,
          prefer_vision: bool = True) -> VisionVerdict:
    """Vision judge over the rendered PNG, with a text-over-source fallback.

    Falls back to the text judge when there is no PNG, or when the provider
    rejects image inputs (so a gateway without vision still yields a verdict,
    flagged ``mode="text"`` for the weaker signal).
    """
    if prefer_vision and png_path and Path(png_path).exists():
        v = judge_vision(problem, png_path, model)
        if not (v.mode == "error" and _looks_image_unsupported(v.reason)):
            return v  # a real yes/no (or a non-image error we won't paper over)
    return judge_text(problem, tikz, model)
