# Capacity probe: pushing the geometry-illustrator from 1.7B → 4B

**Question.** The distilled 1.7B illustrator (`qwen3-illustrator`, base Qwen3-1.7B)
faithfully illustrates **11.3%** of held-out AIME geometry *locally*. Does raising
model capacity — same data, same recipe, bigger base — move that FAITHFUL coverage
toward the frontier teacher's ~64% ceiling, or does it plateau?

**Setup (additive, one variable changed).** A new LoRA, `qwen3-illustrator-4b`
(base **Qwen/Qwen3-4B**), trained on the **exact same** `data/illustrator_train_chat.jsonl`
(3,996 records) with the **exact same** learning hyper-parameters as the 1.7B run
(LoRA r=32/α=64/dropout=0.05/all-linear, lr 2e-4 cosine, 2 epochs, effective batch
16, max_len 2560, bf16). Only the base model (and, as infra, the GPU: A100-80GB)
changed. Evaluated on the **same 150 held-out AIME problems** (seed `20260709`),
with the **same vision judge** (`gemini-3.1-pro`, vision mode), the **same frontier
fallback** (`gpt-5.5`), and the **same coordinate grader** on the **same 240
held-out synthetic** problems. The 1.7B adapter, its outputs, and all existing
artifacts are untouched; everything here is new (`*-4b` scripts / outputs / adapter).

---

## Headline: 1.7B → 4B (before → after)

### AIME — 150 held-out geometry problems (seed 20260709)

| Signal | 1.7B `qwen3-illustrator` | 4B `qwen3-illustrator-4b` | Δ |
| :-- | --: | --: | --: |
| **compile + non-degenerate** (local) | 69.3% (104/150) | **70.0%** (105/150) | +0.7 pt |
| **judge-verified / FAITHFUL** (local) | 11.3% (17/150) | **24.0%** (36/150) | **+12.7 pt (2.1×)** |
| union faithful (local + judge-gated frontier) | 64.0% (96/150) | **68.0%** (102/150) | +4.0 pt |
| compile coverage, any route | 87.3% (131/150) | 86.0% (129/150) | −1.3 pt\* |
| frontier-fallback faithful (on the remainder) | 52.7% (79/150) | 44.0% (66/150) | see note |

\* Not a regression: judge-gated routing sent **fewer** problems to the frontier
(114 vs 133) precisely because the local 4B *faithfully* covered more, so slightly
fewer distinct problems reach the frontier's compile path. The meaningful signal —
faithful union — **rose**.

### Held-out synthetic — 240 problems, coordinate-verified (atol 0.05)

| | base | tuned |
| :-- | --: | --: |
| **1.7B** | 7.9% (19/240) · compile 98.3% | 93.8% (225/240) · compile 98.3% |
| **4B** | 9.2% (22/240) · compile 14.6% | **97.1%** (233/240) · compile **100.0%** |

Tuned-4B is 12/12 coordinate-exact on **19 of 20** construction families; the lone
laggard is `regular_polygon` at **5/12** (up from 1/12) — rotation-built many-vertex
figures still accumulate small errors / label-order mismatches.

---

## The honest read: capacity helped, but did **not** reach the frontier

**Capacity doubled *faithfulness* without changing *drawing rate*.** Local compile
coverage barely moved (69.3% → 70.0%): both models "draw a plausible figure" for
~70% of AIME geometry. What the extra parameters bought is drawing the **right**
figure more often — the local **compile→faithful conversion** went from
**16.3%** (17/104) to **34.3%** (36/105), i.e. it **more than doubled**. That is
exactly the signature you want from capacity: not more ink, but better *reasoning
about the configuration*.

**But 24% is still a long way from ~64%.** The frontier teacher's faithful ceiling
on this sample is ~64%. Going 11.3% → 24.0% closes only about **a quarter of the
gap** between the 1.7B's local faithfulness and that ceiling (12.7 / 52.7 ≈ 24%),
and in absolute terms the 4B is at ~**37%** of the teacher's number. A 4B does
**not** reason like gpt-5.5 on novel hard problems; it just fails less often. This
is the expected, honestly-scoped result — capacity is a real lever, not a
substitute for frontier-scale reasoning.

**The *system* now clears the old ceiling — because the local share grew.** The
faithful union rose 64.0% → **68.0%**. This is not the frontier getting better (it
is byte-identical, reused from cache); it is the stronger local model faithfully
handling **36** problems for free (up from 17), so union = local-faithful +
frontier-faithful-on-remainder now exceeds the previous frontier-anchored 64%.
More of the coverage is now free and local (24% vs 11.3%), which is the point of a
*local* specialist.

**Where the gains landed (local faithful, by decade):**

| decade | 1.7B | 4B |
| :-- | --: | --: |
| 1980s | 2 | 2 |
| 1990s | 3 | 4 |
| 2000s | 6 | 10 |
| 2010s | 5 | 14 |
| 2020s | 1 | 6 |

The jump concentrates in 2000s–2020s problems, which lean on coordinate/analytic
setups the 4B can now carry through; pre-2000 synthetic-geometry phrasing stays hard.

---

## Remaining failure modes

- **No faithful figure at all: 48/150 = 32%** (down from 54/150 = 36%). These are
  the structurally hard tail — 3D solids, heavy combinatorics, non-planar or
  region-counting configs — where neither the local 4B nor the frontier reliably
  produces a faithful diagram. Matches the ~31–36% "not cleanly planar" estimate.
- **Compile-but-unfaithful: 69 local figures** (105 compile − 36 faithful). The 4B
  sets up a geometric-looking scene but misreads a constraint (wrong incidence,
  swapped side, a length/ratio not honored). This is the residual reasoning gap
  the vision judge exposes and refuses to count.
- **`regular_polygon` (synthetic): 5/12 coordinate-exact.** Improved but still the
  weakest family; rotation-built vertices drift past atol / get labeled out of order.
- **Base-4B is not inherently better at raw TikZ.** Untuned Qwen3-4B compiles
  *fewer* synthetic figures than untuned 1.7B (14.6% vs 17.1%); the lift is entirely
  from the LoRA + distilled data. Data remains the lever — capacity *amplifies* it.

---

## How to run the 4B illustrator

The adapter lives on the Modal Volume `geotikz-outputs` as **`qwen3-illustrator-4b`**
(base `Qwen/Qwen3-4B`; also downloaded to `./outputs/qwen3-illustrator-4b`).

```bash
# Train (detached; new adapter, 1.7B run untouched):
modal run --detach scripts/train_illustrator_4b_modal.py --epochs 2

# AIME coverage (before→after), two signals + judge-gated union:
uv run python scripts/illustrate_aime.py --n 150 --backend modal \
    --specialist-script scripts/infer_illustrator_4b_modal.py \
    --out-dir outputs/aime_gallery_illustrator_4b --max-new-tokens 1536 \
    --fallback-model openai-group/gpt-5.5

# Coordinate-verified synthetic pass (base vs tuned 4B):
uv run python scripts/eval_syn_illustrator.py --also-base \
    --script scripts/infer_illustrator_4b_modal.py \
    --tuned-adapter qwen3-illustrator-4b \
    --out-dir outputs/syn_eval_illustrator_4b

# Batch inference on any problems file (id/description JSONL):
modal run scripts/infer_illustrator_4b_modal.py --input in.jsonl --output out.jsonl
```

**New artifacts (all additive):**
`scripts/train_illustrator_4b_modal.py`, `scripts/infer_illustrator_4b_modal.py`,
`outputs/aime_gallery_illustrator_4b/` (gallery + `coverage_stats.json` +
`coverage_report.md`), `outputs/syn_eval_illustrator_4b/report.md`, adapter at
`outputs/qwen3-illustrator-4b/` and on the Volume. `scripts/eval_syn_illustrator.py`
gained optional `--script/--tuned-adapter/--tuned-label/--base-label` flags
(defaults reproduce the 1.7B run exactly).

## Verification (it actually ran)

- **Training:** exit 0, `train_runtime` 3307 s on A100-80GB, mean `train_loss`
  **0.148** (final step **0.072**; the 1.7B's final was 0.19 — the 4B fits the data
  tighter). Adapter committed to the Volume (`adapter_model.safetensors` 264 MB +
  `adapter_config.json` + tokenizer) — `modal volume ls geotikz-outputs qwen3-illustrator-4b`.
- **AIME eval:** exit 0; **105 specialist + 67 frontier PNGs** rendered;
  `coverage_stats.json` / `coverage_report.md` / `index.html` non-empty. The 4B
  regenerated all 150 of its own outputs (no stale reuse); frontier outputs (150)
  and judge verdicts (225) were reused from the 1.7B run via content-addressed caches.
- **Synthetic eval:** exit 0; `report.md` non-empty; tuned-4B **240/240 compiled**.
- **Gateway spend kept modest:** only **106 new vision-judge calls** (105 new 4B
  figures + 1 frontier) and **0 new gpt-5.5 completions** — the fair-comparison
  frontier + judge were reused, not re-billed.

**Bottom line.** Capacity is a real but sub-frontier lever: 1.7B→4B **roughly
doubled** local faithful AIME coverage (11.3% → **24.0%**) and lifted the full
system to **68%** faithful, while synthetic coordinate-verified accuracy rose
93.8% → **97.1%**. It closed ~a quarter of the gap to the frontier's ~64% faithful
ceiling — meaningful, but a 4B still cannot reason like gpt-5.5 on the hard,
non-planar tail. Measured, not assumed.
