#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Tri-SDK LongMemEval orchestrator.
#
# Phase 1: Extract once (Go) into a persistent brain under $JB_HOME.
# Phase 2: Spawn TS, Go, Py `memory serve` daemons attached to the same brain.
# Phase 3: Run the Go LME runner once per SDK with `--actor-endpoint` pointed
#          at each daemon. Retrieval + reading happens in the daemon, the judge
#          stays in-process so every SDK scores against the same judge config.
# Phase 4: Tear the daemons down.
# Phase 5: Print a per-SDK pass-rate summary and the output directory.

set -euo pipefail

DATASET="${DATASET:-$HOME/code/jeffs-brain/memory/eval/datasets/longmemeval_s.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SEED="${SEED:-42}"
BRAIN_ID="${BRAIN_ID:-eval-lme}"
JB_HOME="${JB_HOME:-/tmp/jb-lme-shared}"
CONCURRENCY="${CONCURRENCY:-128}"
REPLAY_CONCURRENCY="${REPLAY_CONCURRENCY:-256}"
EXTRACT_MODEL="${EXTRACT_MODEL:-gpt-5-mini}"
ACTOR_MODEL="${ACTOR_MODEL:-gpt-4o}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
# Retrieval knobs. topK 20 keeps multi-session recall healthy because
# the 2nd/3rd truth session often sits at rank 10-18 without reranker.
# The LLM reranker uses the same actor model to score candidate chunks.
TOP_K="${TOP_K:-20}"
RERANK_PROVIDER="${RERANK_PROVIDER:-llm}"
RERANK_MODEL="${RERANK_MODEL:-$ACTOR_MODEL}"
MAX_COST="${MAX_COST:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/code/jeffs-brain/memory/eval/results/tri-lme-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found at $DATASET" >&2
  echo "       set DATASET=... or fetch longmemeval_s.json into eval/datasets/" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

MEMORY_GO="${MEMORY_GO:-/tmp/memory-go}"
if [[ ! -x "$MEMORY_GO" ]]; then
  (cd "$HOME/code/jeffs-brain/memory/sdks/go" && go build -o "$MEMORY_GO" ./cmd/memory)
fi

echo "== Phase 1: extract-only (shared brain at $JB_HOME) =="
rm -rf "$JB_HOME"
mkdir -p "$JB_HOME/brains"
JB_HOME="$JB_HOME" "$MEMORY_GO" eval lme run \
  --dataset "$DATASET" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --ingest-mode replay \
  --extract-only \
  --brain-cache "$JB_HOME/brains/$BRAIN_ID" \
  --replay-concurrency "$REPLAY_CONCURRENCY" \
  --extract-model "$EXTRACT_MODEL" \
  --judge "" \
  --no-reader \
  --output "$OUTPUT_ROOT/extract.json" \
  --manifest "$OUTPUT_ROOT/manifest.json" \
  --max-cost-usd "$MAX_COST"

echo "== Phase 2: spawn tri-SDK daemons =="
declare -A PORTS=([ts]=18850 [go]=18851 [py]=18852)
declare -A PIDS

# The TS SDK reads JB_LLM_API_KEY / JB_LLM_BASE_URL as the provider-agnostic
# way to inject credentials. Mirror the ANTHROPIC_* env so the TS daemon
# authenticates against the same proxy as Go and Py.
TS_JB_LLM_API_KEY="${JB_LLM_API_KEY:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}}"
TS_JB_LLM_BASE_URL="${JB_LLM_BASE_URL:-${ANTHROPIC_BASE_URL:-${OPENAI_BASE_URL:-}}}"

# Keep the TS dist fresh so `node dist/cli.js serve` picks up latest code.
(cd "$HOME/code/jeffs-brain/memory/sdks/ts/memory" && bun run build > /dev/null)

for sdk in ts go py; do
  port="${PORTS[$sdk]}"
  case "$sdk" in
    ts)
      setsid env JB_HOME="$JB_HOME" \
        JB_LLM_PROVIDER="${JB_LLM_PROVIDER:-}" JB_LLM_MODEL="$ACTOR_MODEL" \
        ${TS_JB_LLM_BASE_URL:+JB_LLM_BASE_URL="$TS_JB_LLM_BASE_URL"} \
        ${TS_JB_LLM_API_KEY:+JB_LLM_API_KEY="$TS_JB_LLM_API_KEY"} \
        ${OPENAI_API_KEY:+OPENAI_API_KEY="$OPENAI_API_KEY"} \
        ${OPENAI_BASE_URL:+OPENAI_BASE_URL="$OPENAI_BASE_URL"} \
        ${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
        ${ANTHROPIC_BASE_URL:+ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"} \
        node "$HOME/code/jeffs-brain/memory/sdks/ts/memory/dist/cli.js" serve --addr "127.0.0.1:$port" \
        > "$OUTPUT_ROOT/daemon-ts.log" 2>&1 < /dev/null &
      ;;
    go)
      setsid env JB_HOME="$JB_HOME" \
        JB_LLM_PROVIDER="${JB_LLM_PROVIDER:-}" JB_LLM_MODEL="$ACTOR_MODEL" \
        JB_RERANK_PROVIDER="$RERANK_PROVIDER" JB_RERANK_MODEL="$RERANK_MODEL" \
        ${OPENAI_API_KEY:+OPENAI_API_KEY="$OPENAI_API_KEY"} \
        ${OPENAI_BASE_URL:+OPENAI_BASE_URL="$OPENAI_BASE_URL"} \
        ${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
        ${ANTHROPIC_BASE_URL:+ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"} \
        "$MEMORY_GO" serve --addr "127.0.0.1:$port" \
        > "$OUTPUT_ROOT/daemon-go.log" 2>&1 < /dev/null &
      ;;
    py)
      setsid env JB_HOME="$JB_HOME" \
        JB_LLM_PROVIDER="${JB_LLM_PROVIDER:-}" JB_LLM_MODEL="$ACTOR_MODEL" \
        JB_RERANK_PROVIDER="$RERANK_PROVIDER" JB_RERANK_MODEL="$RERANK_MODEL" \
        ${OPENAI_API_KEY:+OPENAI_API_KEY="$OPENAI_API_KEY"} \
        ${OPENAI_BASE_URL:+OPENAI_BASE_URL="$OPENAI_BASE_URL"} \
        ${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
        ${ANTHROPIC_BASE_URL:+ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"} \
        uv --project "$HOME/code/jeffs-brain/memory/sdks/py" run memory serve --addr "127.0.0.1:$port" \
        > "$OUTPUT_ROOT/daemon-py.log" 2>&1 < /dev/null &
      ;;
  esac
  PIDS[$sdk]=$!
  disown || true
done

cleanup() {
  local code=$?
  echo "== Phase 4: cleanup daemons =="
  for sdk in ts go py; do
    pid="${PIDS[$sdk]:-}"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      # Also clean up the session group so child processes go with it.
      kill -- "-$pid" 2>/dev/null || true
    fi
  done
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "Waiting up to 20s for daemons to start..."
deadline=$((SECONDS + 20))
for sdk in ts go py; do
  port="${PORTS[$sdk]}"
  while ! curl -s -f "http://127.0.0.1:$port/healthz" > /dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "ERROR: $sdk daemon at $port not healthy after 20s" >&2
      tail -n 40 "$OUTPUT_ROOT/daemon-$sdk.log" >&2 || true
      exit 1
    fi
    sleep 0.5
  done
  # Create the brain in-daemon (no-op if already exists).
  curl -s -X POST "http://127.0.0.1:$port/v1/brains" \
    -H 'Content-Type: application/json' \
    -d "{\"brainId\":\"$BRAIN_ID\"}" > /dev/null || true
done

# Give each daemon time to finish its initial FTS scan + vector backfill
# before we fire the question load. On a cold brain cache the Go daemon
# embeds ~10k chunks in ~60-90s; we conservatively wait longer so
# hybrid search is genuinely populated when the runner hits /search.
WARMUP_SECONDS="${WARMUP_SECONDS:-15}"
if [[ "$WARMUP_SECONDS" -gt 0 ]]; then
  echo "Warmup: sleeping ${WARMUP_SECONDS}s for index + vector backfill..."
  sleep "$WARMUP_SECONDS"
fi

echo "== Phase 3: parallel runs against each daemon =="
# retrieve-only: daemon acts as pure retrieval substrate, the runner
# applies the augmented CoT reader prompt + judge in-process. This
# isolates retrieval quality as the only variable across SDKs so the
# scores are apples-to-apples.
for sdk in ts go py; do
  port="${PORTS[$sdk]}"
  "$MEMORY_GO" eval lme run \
    --dataset "$DATASET" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --ingest-mode none \
    --actor-endpoint "http://127.0.0.1:$port" \
    --actor-endpoint-style retrieve-only \
    --actor-brain "$BRAIN_ID" \
    --actor-topk "$TOP_K" \
    --concurrency "$CONCURRENCY" \
    --actor "$ACTOR_MODEL" \
    --judge "$JUDGE_MODEL" \
    --max-cost-usd "$MAX_COST" \
    --output "$OUTPUT_ROOT/result-$sdk.json" \
    --manifest "$OUTPUT_ROOT/manifest-$sdk.json" \
    > "$OUTPUT_ROOT/runner-$sdk.log" 2>&1 &
done
wait

echo "== Phase 5: summary =="
{
  echo "# Tri-SDK LongMemEval run"
  echo ""
  echo "- Dataset: $DATASET"
  echo "- Sample: $SAMPLE_SIZE questions, seed $SEED"
  echo "- Brain id: $BRAIN_ID (shared at $JB_HOME)"
  echo "- Actor model: $ACTOR_MODEL"
  echo "- Judge model: $JUDGE_MODEL"
  echo "- Extract model: $EXTRACT_MODEL"
  echo "- Concurrency: questions=$CONCURRENCY replay=$REPLAY_CONCURRENCY"
  echo ""
  echo "## Results"
  echo ""
  echo "| SDK | Pass | Total | Pass rate | Cost USD |"
  echo "|-----|------|-------|-----------|----------|"
} > "$OUTPUT_ROOT/README.md"

for sdk in ts go py; do
  result="$OUTPUT_ROOT/result-$sdk.json"
  if [[ -f "$result" ]]; then
    python3 - "$result" "$sdk" <<'PY' | tee -a "$OUTPUT_ROOT/README.md"
import json, sys
path, sdk = sys.argv[1], sys.argv[2]
d = json.load(open(path))
questions = d.get('questions') or []
total = d.get('questions_run') or len(questions)
passed = sum(
    1 for q in questions
    if q.get('judge_verdict') in ('correct', 'abstain_correct')
)
rate = passed / total if total else 0.0
cost = (d.get('cost_accounting') or {}).get('total_usd', 0.0)
print(f"| {sdk} | {passed} | {total} | {rate:.1%} | ${cost:.4f} |")
PY
  else
    echo "| $sdk | n/a | n/a | n/a | n/a |" | tee -a "$OUTPUT_ROOT/README.md"
  fi
done

echo ""
echo "Full outputs at: $OUTPUT_ROOT"
