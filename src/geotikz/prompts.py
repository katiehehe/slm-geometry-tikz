"""Prompt templates. The system prompt encodes the Behavior Spec."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler. Given a geometry scene described only "
    "through relationships and constraints (no explicit coordinates), you must "
    "derive the exact coordinates yourself and output a single valid TikZ/PGF "
    "figure that compiles and renders the described geometry. "
    "Output ONLY the TikZ code, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. No prose, no explanations, no markdown fences."
)

USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."

# Construction mode (v3 olympiad litmus). The grader compiles the figure with
# tkz-euclide + the calc/intersections/through/... tikz libraries ALREADY loaded,
# then reads back the true coordinate of each requested named point. So the model
# only needs to emit one tikzpicture whose named points land in the right place —
# by any means (direct coordinates OR coordinate-free constructions).
CONSTRUCTION_SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler for olympiad constructions. You are given "
    "a geometry scene: some base points are given by coordinates, and one or more "
    "further points are described only by their geometric construction (e.g. the "
    "circumcenter, incenter, orthocenter, centroid, the foot of an altitude, where "
    "an angle bisector meets a side, a midpoint, or a point of tangency).\n\n"
    "Output ONE TikZ figure that realizes the scene and defines every requested "
    "named point at its correct location. You may work either way:\n"
    "  - compute the coordinates yourself and place them, e.g. "
    "\\coordinate (O) at (2.5,1.375); , or\n"
    "  - use coordinate-free constructions. The full tkz-euclide package and the "
    "tikz libraries calc, intersections, through, angles, positioning are ALREADY "
    "loaded, so macros like \\tkzDefPoint(0,0){A}, "
    "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}, "
    "\\tkzDefTriangleCenter[in]/[ortho]/[centroid], "
    "\\tkzDefPointBy[projection=onto B--C](A)\\tkzGetPoint{F}, "
    "\\tkzDefMidPoint(B,C)\\tkzGetPoint{M}, \\tkzDefLine[bisector](B,A,C), "
    "\\tkzDefTangent[from = P](O,W)\\tkzGetPoints{T1}{T2}, and PGF calc "
    "($(a)!(c)!(b)$) are all available.\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "  1. Every requested point MUST be a referenceable named coordinate/node using "
    "the EXACT name requested (case-sensitive), created by any of: "
    "\\coordinate (NAME) at (...);  \\tkzDefPoint(...){NAME}  \\tkzGetPoint{NAME}  "
    "or \\node (NAME) at (...) {}. Do not rename points.\n"
    "  2. Output ONLY the figure, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. Do NOT include \\documentclass, \\usepackage, or "
    "\\begin{document} — only the tikzpicture. No prose, no explanations, no "
    "markdown fences."
)

CONSTRUCTION_USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."


def build_messages(description: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(description=description)},
    ]


def build_construction_messages(description: str) -> list[dict]:
    """Messages for the v3 olympiad construction litmus (tkz-euclide available)."""
    return [
        {"role": "system", "content": CONSTRUCTION_SYSTEM_PROMPT},
        {"role": "user", "content": CONSTRUCTION_USER_TEMPLATE.format(description=description)},
    ]


def to_chat_example(ex: dict) -> dict:
    """Format one generated example as a chat SFT record (prompt + completion)."""
    return {
        "messages": build_messages(ex["description"])
        + [{"role": "assistant", "content": ex["tikz"]}]
    }
