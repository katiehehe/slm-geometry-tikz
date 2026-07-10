#!/usr/bin/env bash
# Provider-grouped parallel difficulty sweep.
#
# Each provider (OpenAI, Anthropic, Google, xAI, DeepSeek) has its OWN rate
# limits, so we fetch them CONCURRENTLY (one process per provider) while models
# WITHIN a provider run sequentially. Per-provider concurrency is unchanged vs a
# sequential run (16 workers each) — we just stop leaving four providers idle.
#
# Every raw result is cached to disk as it completes, so this is safe to Ctrl-C
# or close your laptop: just re-run this script when you're back and it resumes
# from cache — finished models are skipped, zero API calls wasted.
#
#   Phase 1: parallel gather-only  (API-bound; ~1h cold, instant for cached)
#   Phase 2: one scoring pass over all models -> results.json + recommendation
#
# Usage:  bash scripts/run_sweep_parallel.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# MUST match the grid the run was built with (outputs/sweep/meta.json), or the
# script will refuse to reuse the cache. Do NOT add --rebuild-grid here.
GRID=(--chains 2 3 4 5 6 7 --k 40 --op-dial --out outputs/sweep)

OPENAI=(openai-group/gpt-5.5 openai-group/gpt-5.4 openai-group/gpt-5-mini openai-group/gpt-4.1 openai-group/gpt-4o)
ANTHROPIC=(claude-group/claude-opus-4-8 claude-group/claude-sonnet-5 claude-group/claude-haiku-4-5)
GOOGLE=(gemini-group/gemini-3.1-pro gemini-group/gemini-3.5-flash)
XAI=(xai-group/grok-4.5)
DEEPSEEK=(deepseek-v3.2)
ALL=("${OPENAI[@]}" "${ANTHROPIC[@]}" "${GOOGLE[@]}" "${XAI[@]}" "${DEEPSEEK[@]}")

gather () {  # $1=tag ; rest=model ids
  local tag="$1"; shift
  echo "[$(date +%H:%M:%S)] gather START $tag: $*"
  uv run python -u scripts/difficulty_sweep.py --models "$@" \
      "${GRID[@]}" --workers 16 --gather-only > "logs/gather_$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] gather DONE  $tag"
}

echo "=== Phase 1: parallel gather by provider ==="
gather openai    "${OPENAI[@]}"    &
gather anthropic "${ANTHROPIC[@]}" &
gather google    "${GOOGLE[@]}"    &
gather xai       "${XAI[@]}"       &
gather deepseek  "${DEEPSEEK[@]}"  &
wait
echo "=== Phase 1 complete: all raw outputs cached ==="

echo "=== Phase 2: unified scoring + recommendation ==="
uv run python -u scripts/difficulty_sweep.py --models "${ALL[@]}" \
    "${GRID[@]}" --workers 16 --score-workers 4 2>&1 | tee logs/score_final.log
echo "=== ALL DONE -> outputs/sweep/results.json ==="
