"""Objective metrics for the Behavior Spec.

- extract_tikz / is_figure_only  -> spec-adherence (figure only, no prose)
- mse / ssim                     -> render-and-diff
- parse_coords / coord_match     -> coordinate assertion against ground truth
"""

from __future__ import annotations

import math
import re

import numpy as np

from .scene import line_intersection as _line_intersection

_BEGIN = r"\begin{tikzpicture}"
_END = r"\end{tikzpicture}"


def extract_tikz(text: str) -> str | None:
    """Pull the first \\begin{tikzpicture}...\\end{tikzpicture} block."""
    i = text.find(_BEGIN)
    j = text.find(_END)
    if i == -1 or j == -1 or j < i:
        return None
    return text[i : j + len(_END)]


def is_figure_only(text: str) -> bool:
    """True iff the output is essentially just the figure (allow code fences/whitespace)."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", stripped).strip()
    return stripped.startswith(_BEGIN) and stripped.endswith(_END)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on two [0,1] grayscale images of equal shape."""
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return float(num / den) if den else 0.0


# Trig in TikZ is in degrees; expose a safe whitelist for coordinate expressions.
_ALLOWED_FUNCS = {"cos", "sin", "tan", "sqrt", "abs", "atan", "acos", "asin", "pi"}
_EVAL_ENV = {
    "cos": lambda d: math.cos(math.radians(d)),
    "sin": lambda d: math.sin(math.radians(d)),
    "tan": lambda d: math.tan(math.radians(d)),
    "sqrt": math.sqrt,
    "abs": abs,
    "atan": lambda x: math.degrees(math.atan(x)),
    "acos": lambda x: math.degrees(math.acos(x)),
    "asin": lambda x: math.degrees(math.asin(x)),
    "pi": math.pi,
}
_NODE_LABEL_RE = re.compile(r"node[^;]*?\{\s*\$?([A-Za-z]\w*)\$?\s*\}")
_COORD_DEF_RE = re.compile(r"\\coordinate\s*\(\s*([A-Za-z]\w*)\s*\)\s*at\s*")

_MACRO_DEFS = (
    re.compile(r"\\def\\([A-Za-z]+)\s*\{([^{}]*)\}"),
    re.compile(r"\\newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*\{([^{}]*)\}"),
    re.compile(r"\\pgfmath\w*macro\s*\{?\\([A-Za-z]+)\}?\s*\{([^{}]*)\}"),
)


def _expand_macros(tikz: str) -> str:
    """Inline user macros (\\def / \\newcommand / \\pgfmathsetmacro) so scalar
    expressions like ``(199:\\r)`` become numerically evaluable."""
    macros: dict[str, str] = {}
    for rgx in _MACRO_DEFS:
        for m in rgx.finditer(tikz):
            macros[m.group(1)] = m.group(2)
    if not macros:
        return tikz
    try:
        for _ in range(2):  # two passes resolve macros defined in terms of macros
            for name in sorted(macros, key=len, reverse=True):
                repl = f"({macros[name]})"
                # function replacement -> the value is inserted verbatim (no
                # backslash-escape interpretation, which \draw etc. would break).
                tikz = re.sub(rf"\\{re.escape(name)}(?![A-Za-z])", lambda _m, r=repl: r, tikz)
    except re.error:  # never let a pathological macro abort grading
        return tikz
    return tikz


def _split_rel(s: str) -> tuple[str, str, str] | None:
    """Split ``a |- b`` / ``a -| b`` at the top-level relative operator, if any."""
    depth = 0
    for i in range(len(s) - 1):
        ch = s[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and s[i : i + 2] in ("|-", "-|"):
            return s[:i].strip(), s[i : i + 2], s[i + 2 :].strip()
    return None


def _eval_scalar(expr: str) -> float | None:
    """Safely evaluate a numeric TikZ scalar (supports cos/sin/... in degrees)."""
    expr = expr.strip().strip("{}").strip()
    if not expr:
        return None
    if any(tok not in _ALLOWED_FUNCS for tok in re.findall(r"[A-Za-z_]+", expr)):
        return None
    if not re.fullmatch(r"[0-9A-Za-z_+\-*/().,\s]*", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, _EVAL_ENV))  # noqa: S307 - whitelisted
    except Exception:  # noqa: BLE001 - grading is best-effort
        return None


def _paren_groups(s: str) -> list[str]:
    """Contents of each top-level (...) group, respecting nested parens."""
    groups, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
            if depth == 0 and start is not None:
                groups.append(s[start:i])
                start = None
    return groups


def _split_top_comma(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _split_top(s: str, sep: str) -> list[str]:
    """Split on top-level (depth-0) single-char sep, respecting nested brackets."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _split_terms(s: str) -> list[tuple[float, str]]:
    """Split a calc sum into (sign, term) at top-level + / - (leading sign ok)."""
    terms, depth, cur, sign = [], 0, "", 1.0
    for ch in s:
        if ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch in "+-" and depth == 0:
            if cur.strip():
                terms.append((sign, cur))
                cur = ""
                sign = 1.0 if ch == "+" else -1.0
            else:
                sign *= 1.0 if ch == "+" else -1.0
        else:
            cur += ch
    if cur.strip():
        terms.append((sign, cur))
    return terms


def _parse_operand(s: str, registry: dict[str, tuple[float, float]]):
    """Parse one calc operand, stripping a $...$ wrapper and one paren layer."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return _parse_point(s, registry)


def _project(p, a, b):
    """Orthogonal projection of point p onto line a-b (PGF's (a)!(p)!(b))."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = dx * dx + dy * dy
    if d < 1e-9:
        return None
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / d
    return (a[0] + t * dx, a[1] + t * dy)


def _parse_point(inner: str, registry: dict[str, tuple[float, float]]):
    """Resolve one (...) coordinate to (x,y): calc, relative, reference, polar, cartesian."""
    inner = inner.strip()
    if inner.startswith("$") and inner.endswith("$"):  # calc: $ ... $
        inner = inner[1:-1].strip()
    if "!" in inner:  # calc partway (A!t!B) or projection (A!<point>!B)
        parts = _split_top(inner, "!")
        if len(parts) == 3:
            left = _parse_operand(parts[0], registry)
            right = _parse_operand(parts[2], registry)
            if left is None or right is None:
                return None
            t = _eval_scalar(parts[1].strip())
            if t is not None:  # partway
                return (left[0] + t * (right[0] - left[0]),
                        left[1] + t * (right[1] - left[1]))
            mid = _parse_operand(parts[1], registry)  # projection of mid onto left-right
            return _project(mid, left, right) if mid is not None else None
    if len(_split_top(inner, ",")) == 1:  # no top-level comma -> maybe a calc vector sum
        terms = _split_terms(inner)
        if len(terms) > 1:
            acc = [0.0, 0.0]
            for sign, term in terms:
                pt = _parse_operand(term, registry)
                if pt is None:
                    break
                acc[0] += sign * pt[0]
                acc[1] += sign * pt[1]
            else:
                return (acc[0], acc[1])
    rel = _split_rel(inner)  # (a |- b) = (a_x, b_y);  (a -| b) = (b_x, a_y)
    if rel is not None:
        left, op, right = rel
        lp, rp = _parse_point(left, registry), _parse_point(right, registry)
        if lp is None or rp is None:
            return None
        return (lp[0], rp[1]) if op == "|-" else (rp[0], lp[1])
    if inner in registry:  # reference to a named \coordinate
        return registry[inner]
    if ":" in inner and "," not in inner:  # polar: angle:radius
        a, _, r = inner.partition(":")
        av, rv = _eval_scalar(a), _eval_scalar(r)
        if av is None or rv is None:
            return None
        return (rv * math.cos(math.radians(av)), rv * math.sin(math.radians(av)))
    parts = _split_top_comma(inner)  # cartesian x,y (may be expressions)
    if len(parts) != 2:
        return None
    x, y = _eval_scalar(parts[0]), _eval_scalar(parts[1])
    if x is None or y is None:
        return None
    return (x, y)


def _statement_point(stmt: str, registry: dict[str, tuple[float, float]]):
    """The anchor coordinate a statement draws at (after `at` if present)."""
    m = re.search(r"\bat\b", stmt)
    region = stmt[m.end():] if m else stmt
    for g in _paren_groups(region):
        pt = _parse_point(g, registry)
        if pt is not None:
            return pt
    return None


def parse_named_coords(tikz: str) -> dict[str, tuple[float, float]]:
    """Recover {name: (x,y)} for named/labeled points in a TikZ figure.

    Handles literal coords, `\\coordinate (N) at (...)` definitions and later
    `(N)` references, polar `(angle:r)`, and expressions like `{4.5*cos(161)}`.
    """
    tikz = _expand_macros(tikz)
    # Resolve \coordinate defs + PGF name-path intersections to a fixed point: a
    # `\coordinate` may reference an intersection point (`name intersections={by=X}`)
    # and vice-versa, so we iterate until the registry stops growing.
    registry: dict[str, tuple[float, float]] = {}
    for _ in range(6):
        before = len(registry)
        for m in _COORD_DEF_RE.finditer(tikz):
            name = m.group(1)
            if name in registry:
                continue
            groups = _paren_groups(tikz[m.end():])
            if groups:
                pt = _parse_point(groups[0], registry)
                if pt is not None:
                    registry[name] = pt
        named_paths: dict[str, tuple] = {}
        for m in re.finditer(r"name path=(\w+)\]([^;]*);", tikz):
            pts = [p for p in (_parse_point(g, registry) for g in _paren_groups(m.group(2)))
                   if p is not None]
            if len(pts) >= 2:
                named_paths[m.group(1)] = (pts[0], pts[-1])
        for m in re.finditer(r"name intersections=\{\s*of=(\w+)\s+and\s+(\w+),\s*by=(\w+)", tikz):
            a, b, x = m.groups()
            if x in registry or a not in named_paths or b not in named_paths:
                continue
            ip = _line_intersection(named_paths[a][0], named_paths[a][1],
                                    named_paths[b][0], named_paths[b][1])
            if ip is not None:
                registry[x] = ip
        if len(registry) == before:
            break

    out: dict[str, tuple[float, float]] = dict(registry)
    for stmt in tikz.split(";"):
        labels = _NODE_LABEL_RE.findall(stmt)
        if not labels:
            continue
        pt = _statement_point(stmt, registry)
        if pt is None:
            continue
        for name in labels:
            out[name] = pt
    return out


def coord_match(
    pred_tikz: str, gt_points: dict[str, list[float]], atol: float = 0.05
) -> dict:
    """Compare predicted named coords to ground truth within tolerance."""
    pred = parse_named_coords(pred_tikz)
    total = len(gt_points)
    hits = 0
    per_point = {}
    for name, (gx, gy) in gt_points.items():
        if name in pred:
            px, py = pred[name]
            err = max(abs(px - gx), abs(py - gy))
            ok = err <= atol
            per_point[name] = {"ok": ok, "err": round(err, 4)}
            hits += int(ok)
        else:
            per_point[name] = {"ok": False, "err": None}
    return {
        "matched": hits,
        "total": total,
        "accuracy": hits / total if total else 0.0,
        "all_correct": hits == total and total > 0,
        "per_point": per_point,
    }
