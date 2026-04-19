#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Tri-SDK LongMemEval orchestrator.
#
# Phase 1: Extract once (Go) into a persistent brain under $JB_HOME.
# Phase 2: Spawn TS, Go, Py `memory serve` daemons attached to the same brain.
# Phase 3: Run the Go LME runner once per SDK with `--actor-endpoint` pointed
#          at each daemon. This benchmarks the shared daemon
#          `search-retrieve-only` scenario only: retrieval happens in the
#          daemon via `/search`, while the shared augmented reader + judge stay
#          in-process so every SDK scores against the same reader and judge.
# Phase 4: Tear the daemons down.
# Phase 5: Write `tri-lme-<timestamp>/README.md` plus per-SDK result JSON,
#          manifests, and daemon or runner logs, then print the output
#          directory.

set -euo pipefail

DATASET="${DATASET:-$HOME/code/jeffs-brain/memory/eval/datasets/longmemeval_s.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SEED="${SEED:-42}"
SAMPLE_IDS_FILE="${SAMPLE_IDS_FILE:-}"
BRAIN_ID="${BRAIN_ID:-eval-lme}"
JB_HOME="${JB_HOME:-/tmp/jb-lme-shared}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
CONCURRENCY="${CONCURRENCY:-16}"
REPLAY_CONCURRENCY="${REPLAY_CONCURRENCY:-16}"
EXTRACT_MODEL="${EXTRACT_MODEL:-gpt-5}"
ACTOR_MODEL="${ACTOR_MODEL:-gpt-4o}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
CONTEXTUALISE="${CONTEXTUALISE:-0}"
CONTEXTUALISE_CACHE_DIR="${CONTEXTUALISE_CACHE_DIR:-$HOME/.local/state/jeffs-brain/evals/contextualise-cache}"
# Retrieval knobs. topK 20 keeps multi-session recall healthy because
# the 2nd/3rd truth session often sits at rank 10-18 without reranker.
# Default to the explicit post-parity retrieval breadth used by the
# reference runs so manifests stay self-describing instead of relying on
# daemon defaults.
TOP_K="${TOP_K:-20}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid-rerank}"
CANDIDATE_K="${CANDIDATE_K:-60}"
RERANK_TOP_N="${RERANK_TOP_N:-20}"
RERANK_PROVIDER="${RERANK_PROVIDER:-llm}"
RERANK_MODEL="${RERANK_MODEL:-$ACTOR_MODEL}"
ACTOR_SCOPE="${ACTOR_SCOPE-memory}"
ACTOR_PROJECT="${ACTOR_PROJECT-$BRAIN_ID}"
ACTOR_PATH_PREFIX="${ACTOR_PATH_PREFIX-}"
MAX_COST="${MAX_COST:-20}"
# Replay-backed tri-SDK runs always write to their own timestamped
# directory under eval/results/.
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/code/jeffs-brain/memory/eval/results/tri-lme-$(date +%Y%m%d-%H%M%S)}"
READER_CACHE_DIR="${READER_CACHE_DIR:-$HOME/.local/state/jeffs-brain/evals/reader-cache}"
JUDGE_CACHE_DIR="${JUDGE_CACHE_DIR:-$HOME/.local/state/jeffs-brain/evals/judge-cache}"
VECTOR_READY_TIMEOUT_SECONDS="${VECTOR_READY_TIMEOUT_SECONDS:-600}"
TS_PORT="${TS_PORT:-18850}"
GO_PORT="${GO_PORT:-18851}"
PY_PORT="${PY_PORT:-18852}"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found at $DATASET" >&2
  echo "       set DATASET=... or fetch longmemeval_s.json into eval/datasets/" >&2
  exit 2
fi
if [[ -n "$SAMPLE_IDS_FILE" && ! -f "$SAMPLE_IDS_FILE" ]]; then
  echo "ERROR: sample ids file not found at $SAMPLE_IDS_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
SAMPLE_IDS_ARGS=()
if [[ -n "$SAMPLE_IDS_FILE" ]]; then
  SAMPLE_IDS_ARGS=(--sample-ids-file "$SAMPLE_IDS_FILE")
fi

run_logged_step() {
  local log_path=$1
  local label=$2
  shift 2

  if ! "$@" >"$log_path" 2>&1; then
    echo "ERROR: $label failed. See $log_path" >&2
    tail -n 40 "$log_path" >&2 || true
    exit 1
  fi
}

infer_llm_provider() {
  if [[ -n "${JB_LLM_PROVIDER:-}" ]]; then
    printf '%s\n' "$JB_LLM_PROVIDER"
    return 0
  fi
  if [[ -n "${OPENAI_API_KEY:-}" || -n "${OPENAI_BASE_URL:-}" ]]; then
    printf 'openai\n'
    return 0
  fi
  if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${ANTHROPIC_BASE_URL:-}" ]]; then
    printf 'anthropic\n'
    return 0
  fi
  if [[ -n "${OLLAMA_HOST:-}" ]]; then
    printf 'ollama\n'
    return 0
  fi
  return 1
}

ollama_model_names() {
  local host="${OLLAMA_HOST:-http://127.0.0.1:11434}"
  if [[ "$host" != *"://"* ]]; then
    host="http://$host"
  fi
  local bare_host="${host#*://}"
  if [[ "$bare_host" != *:* ]]; then
    host="${host}:11434"
  fi
  python3 - "$host" <<'PY'
import json
import sys
import urllib.request

host = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{host}/api/tags", timeout=30) as resp:
    data = json.load(resp)
for model in data.get("models", []):
    name = str(model.get("name", "")).strip()
    if name:
        print(name)
PY
}

require_ollama_model() {
  local model=$1
  local label=$2

  if [[ -z "$model" ]]; then
    return 0
  fi
  if ollama_model_names | grep -Fx -- "$model" >/dev/null 2>&1; then
    return 0
  fi

  echo "ERROR: $label model '$model' is not available on Ollama at ${OLLAMA_HOST:-http://127.0.0.1:11434}" >&2
  echo "Available Ollama models:" >&2
  ollama_model_names | sed 's/^/  - /' >&2 || true
  exit 2
}

if ! JB_LLM_PROVIDER="$(infer_llm_provider)"; then
  echo "ERROR: set JB_LLM_PROVIDER explicitly or export OPENAI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_HOST before running tri-SDK evals" >&2
  exit 2
fi
export JB_LLM_PROVIDER

if [[ "$JB_LLM_PROVIDER" == "ollama" ]]; then
  require_ollama_model "$ACTOR_MODEL" "actor"
  require_ollama_model "$JUDGE_MODEL" "judge"
  require_ollama_model "$EXTRACT_MODEL" "extract"
  if [[ "$RERANK_PROVIDER" == "llm" ]]; then
    require_ollama_model "$RERANK_MODEL" "rerank"
  fi
fi

CONTEXTUALISE_ARGS=()
if [[ "$CONTEXTUALISE" != "0" && "$CONTEXTUALISE" != "false" ]]; then
  CONTEXTUALISE_ARGS+=(--contextualise)
  if [[ -n "$CONTEXTUALISE_CACHE_DIR" ]]; then
    CONTEXTUALISE_ARGS+=(--contextualise-cache-dir "$CONTEXTUALISE_CACHE_DIR")
  fi
fi

ACTOR_FILTER_ARGS=()
if [[ -n "$ACTOR_SCOPE" ]]; then
  ACTOR_FILTER_ARGS+=(--actor-scope "$ACTOR_SCOPE")
fi
if [[ -n "$ACTOR_PROJECT" ]]; then
  ACTOR_FILTER_ARGS+=(--actor-project "$ACTOR_PROJECT")
fi
if [[ -n "$ACTOR_PATH_PREFIX" ]]; then
  ACTOR_FILTER_ARGS+=(--actor-path-prefix "$ACTOR_PATH_PREFIX")
fi

MEMORY_GO="${MEMORY_GO:-/tmp/memory-go}"
# Always rebuild so benchmark runs never accidentally reuse a stale Go
# daemon / runner binary from a previous iteration.
run_logged_step \
  "$OUTPUT_ROOT/build-go.log" \
  "Go CLI build before extract" \
  bash -lc "cd \"$HOME/code/jeffs-brain/memory/sdks/go\" && go build -o \"$MEMORY_GO\" ./cmd/memory"

echo "== Phase 1: extract-only (shared brain at $JB_HOME) =="
mkdir -p "$JB_HOME/brains"
if [[ "$SKIP_EXTRACT" == "1" || "$SKIP_EXTRACT" == "true" ]]; then
  if [[ ! -d "$JB_HOME/brains/$BRAIN_ID" ]]; then
    echo "ERROR: SKIP_EXTRACT is set but no existing brain cache was found at $JB_HOME/brains/$BRAIN_ID" >&2
    exit 2
  fi
  echo "Skipping extract and reusing existing brain cache at $JB_HOME/brains/$BRAIN_ID"
else
  rm -rf "$JB_HOME/brains/$BRAIN_ID"
  JB_HOME="$JB_HOME" "$MEMORY_GO" eval lme run \
    --dataset "$DATASET" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    "${SAMPLE_IDS_ARGS[@]}" \
    --ingest-mode replay \
    --extract-only \
    --brain-cache "$JB_HOME/brains/$BRAIN_ID" \
    --replay-concurrency "$REPLAY_CONCURRENCY" \
    --extract-model "$EXTRACT_MODEL" \
    "${CONTEXTUALISE_ARGS[@]}" \
    --judge "" \
    --no-reader \
    --output "$OUTPUT_ROOT/extract.json" \
    --manifest "$OUTPUT_ROOT/manifest.json" \
    --max-cost-usd "$MAX_COST"
fi

echo "== Phase 2: spawn tri-SDK daemons =="
declare -A PORTS=([ts]="$TS_PORT" [go]="$GO_PORT" [py]="$PY_PORT")
declare -A PIDS

# The TS SDK reads JB_LLM_API_KEY / JB_LLM_BASE_URL as the provider-agnostic
# way to inject credentials. Mirror the ANTHROPIC_* env so the TS daemon
# authenticates against the same proxy as Go and Py.
TS_JB_LLM_API_KEY="${JB_LLM_API_KEY:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}}"
TS_JB_LLM_BASE_URL="${JB_LLM_BASE_URL:-${ANTHROPIC_BASE_URL:-${OPENAI_BASE_URL:-}}}"

# Keep the TS dist fresh so `node dist/cli.js serve` picks up latest code.
run_logged_step \
  "$OUTPUT_ROOT/build-ts.log" \
  "TypeScript CLI build before daemon spawn" \
  bash -lc "cd \"$HOME/code/jeffs-brain/memory/sdks/ts/memory\" && bun run build"

for sdk in ts go py; do
  port="${PORTS[$sdk]}"
  case "$sdk" in
    ts)
      setsid env JB_HOME="$JB_HOME" \
        JB_LLM_PROVIDER="$JB_LLM_PROVIDER" JB_LLM_MODEL="$ACTOR_MODEL" \
        JB_RERANK_PROVIDER="$RERANK_PROVIDER" JB_RERANK_MODEL="$RERANK_MODEL" \
        ${JB_RERANK_URL:+JB_RERANK_URL="$JB_RERANK_URL"} \
        ${JB_RERANK_API_KEY:+JB_RERANK_API_KEY="$JB_RERANK_API_KEY"} \
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
        JB_LLM_PROVIDER="$JB_LLM_PROVIDER" JB_LLM_MODEL="$ACTOR_MODEL" \
        JB_RERANK_PROVIDER="$RERANK_PROVIDER" JB_RERANK_MODEL="$RERANK_MODEL" \
        ${JB_RERANK_URL:+JB_RERANK_URL="$JB_RERANK_URL"} \
        ${JB_RERANK_API_KEY:+JB_RERANK_API_KEY="$JB_RERANK_API_KEY"} \
        ${OPENAI_API_KEY:+OPENAI_API_KEY="$OPENAI_API_KEY"} \
        ${OPENAI_BASE_URL:+OPENAI_BASE_URL="$OPENAI_BASE_URL"} \
        ${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
        ${ANTHROPIC_BASE_URL:+ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"} \
        "$MEMORY_GO" serve --addr "127.0.0.1:$port" \
        > "$OUTPUT_ROOT/daemon-go.log" 2>&1 < /dev/null &
      ;;
    py)
      setsid env JB_HOME="$JB_HOME" \
        JB_LLM_PROVIDER="$JB_LLM_PROVIDER" JB_LLM_MODEL="$ACTOR_MODEL" \
        JB_RERANK_PROVIDER="$RERANK_PROVIDER" JB_RERANK_MODEL="$RERANK_MODEL" \
        ${JB_RERANK_URL:+JB_RERANK_URL="$JB_RERANK_URL"} \
        ${JB_RERANK_API_KEY:+JB_RERANK_API_KEY="$JB_RERANK_API_KEY"} \
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

warm_daemon() {
  local port=$1
  python3 - "$port" "$BRAIN_ID" "$ACTOR_SCOPE" "$ACTOR_PROJECT" "$ACTOR_PATH_PREFIX" <<'PY'
import json
import sys
import urllib.request

port, brain_id, actor_scope, actor_project, actor_path_prefix = sys.argv[1:]
payload = {
    "query": "warmup",
    "topK": 1,
    "mode": "hybrid",
}
filters = {}
if actor_scope:
    filters["scope"] = actor_scope
if actor_project:
    filters["project"] = actor_project
if actor_path_prefix:
    filters["pathPrefix"] = actor_path_prefix
if filters:
    payload["filters"] = filters
body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/brains/{brain_id}/search",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    resp.read()
PY
}

build_vector_probe_query() {
  python3 - "$JB_HOME/brains/$BRAIN_ID" "$ACTOR_SCOPE" "$ACTOR_PROJECT" "$ACTOR_PATH_PREFIX" <<'PY'
import pathlib
import re
import sys

brain_root, actor_scope, actor_project, actor_path_prefix = sys.argv[1:]
root = pathlib.Path(brain_root)

def candidates() -> list[pathlib.Path]:
    scoped: list[pathlib.Path] = []
    if actor_path_prefix:
        prefix = root / actor_path_prefix
        if prefix.is_file():
            return [prefix]
        if prefix.is_dir():
            scoped.extend(sorted(p for p in prefix.rglob("*.md") if p.is_file()))
    if actor_scope == "raw_lme":
        raw = root / "raw" / "lme"
        if raw.is_dir():
            scoped.extend(sorted(p for p in raw.rglob("*.md") if p.is_file()))
    elif actor_scope == "memory":
        if actor_project:
            project = root / "memory" / "project" / actor_project
            if project.is_dir():
                scoped.extend(sorted(p for p in project.rglob("*.md") if p.is_file()))
        global_dir = root / "memory" / "global"
        if global_dir.is_dir():
            scoped.extend(sorted(p for p in global_dir.rglob("*.md") if p.is_file()))
    if scoped:
        return scoped
    return sorted(p for p in root.rglob("*.md") if p.is_file())

for path in candidates():
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            text = parts[1]
    words = re.findall(r"[A-Za-z][A-Za-z0-9'_-]{3,}", text)
    if len(words) >= 3:
        print(" ".join(words[:6]))
        raise SystemExit(0)
print("memory preference project update")
PY
}

wait_for_vector_readiness() {
  local sdk=$1
  local port=$2
  local timeout_seconds=$3
  local probe_query=$4
  python3 - "$sdk" "$port" "$BRAIN_ID" "$ACTOR_SCOPE" "$ACTOR_PROJECT" "$ACTOR_PATH_PREFIX" "$timeout_seconds" "$probe_query" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

sdk, port, brain_id, actor_scope, actor_project, actor_path_prefix, timeout_raw, probe_query = sys.argv[1:]
deadline = time.time() + int(timeout_raw)

payload = {
    "query": probe_query,
    "topK": 5,
    "mode": "hybrid",
}
filters = {}
if actor_scope:
    filters["scope"] = actor_scope
if actor_project:
    filters["project"] = actor_project
if actor_path_prefix:
    filters["pathPrefix"] = actor_path_prefix
if filters:
    payload["filters"] = filters
body = json.dumps(payload).encode("utf-8")
url = f"http://127.0.0.1:{port}/v1/brains/{brain_id}/search"

while time.time() < deadline:
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
    except urllib.error.URLError:
        time.sleep(1)
        continue

    items = data.get("chunks") or data.get("results") or data.get("items") or []
    for item in items:
        value = item.get("vectorSimilarity")
        if isinstance(value, (int, float)) and value != 0:
            raise SystemExit(0)
    time.sleep(1)

print(
    f"ERROR: {sdk} daemon did not return vector-backed hits within {int(timeout_raw)}s",
    file=sys.stderr,
)
raise SystemExit(1)
PY
}

echo "Priming daemons against the extracted brain..."
for sdk in ts go py; do
  warm_daemon "${PORTS[$sdk]}" &
done
wait

# Give each daemon time to finish its initial FTS scan + vector backfill
# before we fire the question load. On a cold brain cache the Go daemon
# embeds ~10k chunks in ~60-90s; we conservatively wait longer so
# hybrid search is genuinely populated when the runner hits /search.
WARMUP_SECONDS="${WARMUP_SECONDS:-90}"
if [[ "$WARMUP_SECONDS" -gt 0 ]]; then
  echo "Warmup: sleeping ${WARMUP_SECONDS}s for index + vector backfill..."
  sleep "$WARMUP_SECONDS"
fi

if [[ "$RETRIEVAL_MODE" != "bm25" ]]; then
  PROBE_QUERY="$(build_vector_probe_query)"
  echo "Waiting for daemon vector backfill to return vector-backed hits..."
  for sdk in ts go py; do
    wait_for_vector_readiness "$sdk" "${PORTS[$sdk]}" "$VECTOR_READY_TIMEOUT_SECONDS" "$PROBE_QUERY"
  done
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
    "${SAMPLE_IDS_ARGS[@]}" \
    --ingest-mode none \
    --actor-endpoint "http://127.0.0.1:$port" \
    --actor-endpoint-style retrieve-only \
    --actor-brain "$BRAIN_ID" \
    --retrieval-mode "$RETRIEVAL_MODE" \
    --actor-topk "$TOP_K" \
    --actor-candidatek "$CANDIDATE_K" \
    --actor-rerank-topn "$RERANK_TOP_N" \
    "${ACTOR_FILTER_ARGS[@]}" \
    --concurrency "$CONCURRENCY" \
    --actor "$ACTOR_MODEL" \
    --judge "$JUDGE_MODEL" \
    --reader-cache-dir "$READER_CACHE_DIR" \
    --judge-cache-dir "$JUDGE_CACHE_DIR" \
    --max-cost-usd "$MAX_COST" \
    --output "$OUTPUT_ROOT/result-$sdk.json" \
    --manifest "$OUTPUT_ROOT/manifest-$sdk.json" \
    > "$OUTPUT_ROOT/runner-$sdk.log" 2>&1 &
done
wait

# Stamp SDK, scenario, and retrieve-only provenance onto each per-SDK
# manifest so the tri-run artefacts remain self-describing on disk.
for sdk in ts go py; do
  manifest="$OUTPUT_ROOT/manifest-$sdk.json"
  if [[ -f "$manifest" ]]; then
    python3 - \
      "$manifest" \
      "$sdk" \
      "$BRAIN_ID" \
      "$RETRIEVAL_MODE" \
      "$RERANK_PROVIDER" \
      "$RERANK_MODEL" \
      "$TOP_K" \
      "$CANDIDATE_K" \
      "$RERANK_TOP_N" \
      "$ACTOR_SCOPE" \
      "$ACTOR_PROJECT" \
      "$ACTOR_PATH_PREFIX" \
      <<'PY'
import json
import sys
from pathlib import Path

path, sdk, actor_brain, actor_retrieval_mode, rerank_provider, rerank_model, actor_topk, actor_candidatek, actor_rerank_topn, actor_scope, actor_project, actor_path_prefix = sys.argv[1:]
manifest_path = Path(path)
data = json.loads(manifest_path.read_text(encoding="utf-8"))


def blank_to_none(value):
    return value if value != "" else None


data.update(
    {
        "sdk": sdk,
        "scenario": "search-retrieve-only",
        "actor_endpoint_style": "retrieve-only",
        "actor_brain": actor_brain,
        "actor_retrieval_mode": actor_retrieval_mode,
        "actor_rerank_provider": rerank_provider,
        "actor_rerank_model": rerank_model,
        "actor_topk": int(actor_topk),
        "actor_candidatek": int(actor_candidatek),
        "actor_rerank_topn": int(actor_rerank_topn),
        "actor_scope": blank_to_none(actor_scope),
        "actor_project": blank_to_none(actor_project),
        "actor_path_prefix": blank_to_none(actor_path_prefix),
        "shared_extract_output": "extract.json",
        "shared_extract_manifest": "manifest.json",
    }
)
manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
  fi
done

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
  echo "- Scenario: search-retrieve-only via actor-endpoint-style=retrieve-only"
  echo "- Contextualise replay: $CONTEXTUALISE"
  echo "- Skip extract: $SKIP_EXTRACT"
  echo "- Concurrency: questions=$CONCURRENCY replay=$REPLAY_CONCURRENCY"
  echo "- Retrieval mode: $RETRIEVAL_MODE"
  echo "- Retrieval knobs: topK=$TOP_K candidateK=$CANDIDATE_K rerankTopN=$RERANK_TOP_N"
  echo "- Rerank backend: provider=$RERANK_PROVIDER model=$RERANK_MODEL"
  echo "- Shared reader cache: $READER_CACHE_DIR"
  echo "- Shared judge cache: $JUDGE_CACHE_DIR"
  echo "- Actor filters: scope=${ACTOR_SCOPE:-<none>} project=${ACTOR_PROJECT:-<none>} pathPrefix=${ACTOR_PATH_PREFIX:-<none>}"
  echo ""
  echo "## Artefacts"
  echo ""
  echo "- extract.json: shared extract-only replay result"
  echo "- manifest.json: shared extract-only replay manifest"
  echo "- result-<sdk>.json: per-SDK scored retrieve-only benchmark result"
  echo "- manifest-<sdk>.json: per-SDK retrieve-only benchmark manifest with sdk, scenario, actor endpoint style, retrieval knobs, actor filters, and shared extract references"
  echo "- build-go.log / build-ts.log: pre-daemon build logs"
  echo "- daemon-<sdk>.log / runner-<sdk>.log: daemon and runner logs"
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
