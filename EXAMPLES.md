# Specialist example prompts

These are copy‑pasteable, coordinate‑free geometry scene descriptions that the **local
specialist** (`Qwen/Qwen3-4B` + LoRA `qwen3-illustrator-4b`, the adapter the deployed
[Geometry Figure Copilot](scripts/copilot_modal.py) uses) can draw **itself** — i.e. it
produces a compiling, non‑degenerate TikZ figure and the copilot routes to the specialist
instead of falling back to a frontier model.

Marks (validated in one warm batched GPU run through the 4B specialist, then compiled +
degeneracy‑checked with `serve.compile_and_render`):

- ✅ validated — the specialist produced a compiling, non‑degenerate figure (routes to specialist)
- ✗ validated — the specialist output failed/degenerate (would fall back to the frontier)
- ○ not in the validation batch, but the same construction family as a ✅ (expected in‑distribution)

Batch result: **11/12 ✅** (one warm A100 container, one batch, ~1.5 min of GPU). Typical
warm single‑request specialist latency on the deployed app is ~8–15s per figure (the first
call after idle pays a ~55s model‑load cold start).

> Tip: on the hosted app the specialist is tried first by default, so these draw via
> `qwen3-illustrator-4b (specialist · Modal GPU)`. Locally, tick "Try the specialist first"
> (or run with `--modal-specialist`) to route to it.

---

## A. Triangle centers

1. ✅ `Triangle ABC has vertices A=(0,0), B=(6,0), C=(1,4). Let O be the circumcenter of triangle ABC. Output a single TikZ figure that draws triangle ABC and its circumcircle, and defines the named points A, B, C, O at their correct positions.`
2. ○ `Triangle ABC has vertices A=(0,0), B=(7,0), C=(2,5). Let I be the incenter of triangle ABC and draw its incircle. Output a single TikZ figure that defines A, B, C, I at their correct positions.`
3. ✅ `Triangle ABC has vertices A=(1,0), B=(1,4), C=(6,5). Let H be the orthocenter of triangle ABC (the intersection of the three altitudes). Output a single TikZ figure that draws triangle ABC and defines A, B, C, H.`
4. ✗ `Triangle ABC has vertices A=(1,4), B=(4,4), C=(2,0). Let G be the centroid of triangle ABC (the intersection of the three medians). Output a single TikZ figure that draws triangle ABC with its three medians and defines A, B, C, G.` — the specialist defined `G` correctly but drew the medians via undefined midpoints → compile error → frontier fallback. Dropping "with its three medians" (just mark `G`) makes it pass.

## B. Cevians, feet, bisectors

5. ✅ `Triangle ABC has vertices A=(0,0), B=(6,0), C=(1,5). Let D be the point where the internal bisector of angle A meets side BC. Also draw segment AD. Output a single TikZ figure that defines A, B, C, D.`
6. ✅ `Triangle ABC has vertices A=(2,6), B=(0,0), C=(7,0). Let F be the foot of the altitude from A onto line BC. Also draw segment AF. Output a single TikZ figure that draws triangle ABC and defines A, B, C, F.`
7. ○ `Triangle ABC has vertices A=(5,5), B=(8,0), C=(2,1). Let M be the midpoint of side BC, so AM is the median from A; also draw segment AM. Define A, B, C, M.`
8. ○ `In triangle ABC with A=(6,4), B=(2,3), C=(6,0), let AF be the altitude from A (F on BC) and BM the median from B (M the midpoint of CA). The altitude AF and median BM meet at X. Draw triangle ABC with AF and BM and define A, B, C, X.`

## C. Circles & tangents

9. ✅ `A circle has center O=(0,0) and radius 3. Point P=(7,0) lies outside the circle. From P there are two tangent lines to the circle; let T1 and T2 be the two points of tangency. Output a single TikZ figure that draws the circle and the two tangent segments and defines O, P, T1, T2.`
10. ✅ `Two circles are given: one centered at A=(-2,0) with radius 3, the other centered at B=(2,0) with radius 3. They intersect at two points X and Y. Output a single TikZ figure that draws both circles and their intersection points, defining A, B, X, Y.`
11. ○ `A circle has center O=(1,1) and radius 4; A=(5,1) lies on it. Let B be the point diametrically opposite A (so AB is a diameter). Draw the circle and diameter AB, defining O, A, B.`

## D. Midpoints & midsegments

12. ✅ `Triangle ABC has vertices A=(0,5), B=(-4,-1), C=(6,-1). Let M be the midpoint of AB and N the midpoint of AC, so MN is the midsegment parallel to BC. Output a single TikZ figure that draws triangle ABC with midsegment MN and defines A, B, C, M, N.`
13. ○ `Segment AB has endpoints A=(-4,-1) and B=(4,3). Let M be the midpoint of AB. Draw segment AB with its midpoint, defining A, B, M.`

## E. Quadrilaterals & diagonals

14. ✅ `Convex quadrilateral ABCD has vertices A=(0,0), B=(5,1), C=(6,5), D=(1,4). Its diagonals AC and BD intersect at point X. Output a single TikZ figure that draws quadrilateral ABCD with its two diagonals and defines A, B, C, D, X.`
15. ○ `Three vertices of parallelogram ABCD are A=(6,1), B=(1,2), C=(0,5). Let D be the fourth vertex so that ABCD is a parallelogram. Draw parallelogram ABCD, defining A, B, C, D.`
16. ○ `ABCD is a square (vertices in order) with side AB from A=(0,0) to B=(4,1). Draw square ABCD, defining A, B, C, D.`

## F. Transformations

17. ✅ `Line AB passes through A=(-5,0) and B=(5,1); P=(0,5) is a point. Let Q be the reflection of P across line AB. Output a single TikZ figure that draws line AB and the reflected point Q, defining A, B, P, Q.`
18. ○ `Point P=(4,1) is rotated about center O=(0,0) by 90 degrees to give point Q. Draw the rotation taking P to Q about O, defining O, P, Q.`
19. ○ `Points A=(-5,0), B=(6,0) define a line and P=(1,5) is a point off it. Let F be the foot of the perpendicular from P to line AB; also draw segment PF. Define A, B, P, F.`

## G. Regular polygons

20. ✅ `A regular hexagon P0P1P2P3P4P5 is inscribed in a circle of radius 4 centered at O=(0,0), with P0 at angle 0 degrees (each vertex is the previous one rotated 60 degrees about O). Output a single TikZ figure that draws the hexagon and its center O, defining O, P0, P2, P4.`
21. ○ `A regular pentagon P0P1P2P3P4 is inscribed in a circle of radius 4 centered at O=(0,0), with P0 at angle 90 degrees (each vertex the previous one rotated 72 degrees about O). Draw the pentagon and its center O.`

## H. Canonical (lightly coordinate‑free) scenes

22. ✅ `An equilateral triangle ABC with its inscribed circle (incircle). Output a single TikZ figure that draws the triangle and its incircle and labels A, B, C.`
23. ○ `A right triangle ABC with the right angle at B; mark the right angle and label A, B, C.`
24. ○ `Triangle ABC with its circumscribed circle (circumcircle) and circumcenter O; label A, B, C, O.`
25. ○ `A circle with center O and a chord AB, plus the perpendicular from O to the chord meeting it at its midpoint M; label O, A, B, M.`

---

## What keeps a prompt on the specialist vs. pushes it to the frontier

The specialist was fine‑tuned on ~4k CONSTRUCTION‑prompt records: ~45% round‑trip‑validated
synthetic constructions (base points with coordinates + one named derived construction) and
~55% distilled real AIME/MATH problem text. Its comfort zone is the synthetic template.

**Stays with the specialist (high hit rate):**

- **Match the training template:** name a base shape with explicit (small, integer‑ish)
  coordinates, then **one named derived point/construction**, then "Output a single TikZ
  figure that draws … and defines the named points … at their correct positions."
- **In‑vocabulary constructions:** circumcenter / incenter / orthocenter (+ their circles),
  angle bisector → side, foot of altitude / perpendicular, median (as `AM`), tangents from an
  external point, line / diagonal intersection, midpoint & midsegment, parallelogram 4th
  vertex, square, reflection / rotation, regular pentagon / hexagon / octagon, two‑circle
  intersection, antipode / diameter.
- **One derived construction per prompt**, small named‑point sets (A, B, C + 1–2 derived), a
  single figure.

**Pushes it to the frontier (out of its comfort zone):**

- **Asking it to also draw multi‑part sub‑constructions** it must first define (e.g. "draw the
  three medians", "draw all three altitudes") — it tends to reference undefined
  midpoints/feet and fail to compile (this is exactly why #4 fell back). Ask for the
  *center/point*, not the full pencil of cevians.
- **Fully abstract, no anchor** ("some triangle with a special point") — coordinates or a
  canonical shape anchor it; pure vagueness is where it drifts.
- **Out‑of‑vocabulary / non‑plane construction:** 3D (pyramids, spheres), inequalities / loci,
  mixed algebra‑geometry problem text, multi‑figure or many‑labeled configurations, or long
  free‑form statements — these are the noisy part of the distilled data, where frontier
  fallback (and the copilot's clarify path) takes over.
