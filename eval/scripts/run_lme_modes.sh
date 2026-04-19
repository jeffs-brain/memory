#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/code/jeffs-brain}"
MEMORY_DIR="${MEMORY_DIR:-$ROOT_DIR/memory}"
GO_SDK_DIR="${GO_SDK_DIR:-$MEMORY_DIR/sdks/go}"
RESULTS_DIR="${RESULTS_DIR:-$MEMORY_DIR/eval/results}"

S_DATASET="${S_DATASET:-$MEMORY_DIR/eval/datasets/longmemeval_s.json}"
ORACLE_DATASET="${ORACLE_DATASET:-$MEMORY_DIR/eval/datasets/longmemeval_oracle.json}"
ORACLE_URL="${ORACLE_URL:-https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json}"
ORACLE_SHA256="${ORACLE_SHA256:-821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c}"

SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SEED="${SEED:-42}"
PINNED_SAMPLE_IDS_FILE="${SAMPLE_IDS_FILE:-}"
CONCURRENCY="${CONCURRENCY:-16}"
REPLAY_CONCURRENCY="${REPLAY_CONCURRENCY:-16}"
ACTOR_MODEL="${ACTOR_MODEL:-gpt-4o}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
EXTRACT_MODEL="${EXTRACT_MODEL:-gpt-5}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid-rerank}"
MAX_COST="${MAX_COST:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_DIR/lme-modes-$(date +%Y%m%d-%H%M%S)}"
MEMORY_GO="${MEMORY_GO:-/tmp/memory-go}"
RUN_VERBATIM_RETRIEVAL="${RUN_VERBATIM_RETRIEVAL:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SHARED_JB_HOME="${SHARED_JB_HOME:-$OUTPUT_ROOT/shared-brain}"
READER_CACHE_DIR="${READER_CACHE_DIR:-$HOME/.local/state/jeffs-brain/evals/reader-cache}"
JUDGE_CACHE_DIR="${JUDGE_CACHE_DIR:-$HOME/.local/state/jeffs-brain/evals/judge-cache}"

mkdir -p "$OUTPUT_ROOT"

if [[ -n "$PINNED_SAMPLE_IDS_FILE" && ! -f "$PINNED_SAMPLE_IDS_FILE" ]]; then
  echo "ERROR: sample ids file not found at $PINNED_SAMPLE_IDS_FILE" >&2
  exit 2
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

require_dataset() {
  local path=$1
  local label=$2
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $label dataset not found at $path" >&2
    exit 2
  fi
}

ensure_oracle_dataset() {
  if [[ -f "$ORACLE_DATASET" ]]; then
    local existing_sha
    existing_sha="$(sha256sum "$ORACLE_DATASET" | awk '{print $1}')"
    if [[ "$existing_sha" == "$ORACLE_SHA256" ]]; then
      return 0
    fi
  fi

  mkdir -p "$(dirname "$ORACLE_DATASET")"
  local tmp_path="${ORACLE_DATASET}.tmp"
  rm -f "$tmp_path"
  curl -fsSL "$ORACLE_URL" -o "$tmp_path"
  local downloaded_sha
  downloaded_sha="$(sha256sum "$tmp_path" | awk '{print $1}')"
  if [[ "$downloaded_sha" != "$ORACLE_SHA256" ]]; then
    rm -f "$tmp_path"
    echo "ERROR: oracle dataset SHA mismatch: got $downloaded_sha expected $ORACLE_SHA256" >&2
    exit 2
  fi
  mv "$tmp_path" "$ORACLE_DATASET"
}

materialise_sample_ids() {
  local source_dataset=$1
  local output_path=$2
  python3 - "$source_dataset" "$SAMPLE_SIZE" "$SEED" "$output_path" <<'PY'
import json
import pathlib
import sys
from collections import defaultdict

dataset_path, sample_size_raw, seed_raw, output_path = sys.argv[1:]
questions = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8"))
sample_size = int(sample_size_raw)
seed = int(seed_raw)

def deterministic_select(items, count, seed):
    if count >= len(items):
        return list(items)
    indices = list(range(len(items)))
    state = seed & ((1 << 64) - 1)
    for index in range(len(indices) - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        swap = (state >> 33) % (index + 1)
        indices[index], indices[swap] = indices[swap], indices[index]
    return [items[indices[index]] for index in range(count)]

if sample_size <= 0 or sample_size >= len(questions):
    selected = questions
else:
    by_category = defaultdict(list)
    for question in questions:
      by_category[question["question_type"]].append(question)

    total = len(questions)
    allocations = []
    allocated = 0
    for category in sorted(by_category):
        category_questions = by_category[category]
        numerator = len(category_questions) * sample_size
        allocation = min(len(category_questions), numerator // total)
        allocations.append({
            "category": category,
            "questions": category_questions,
            "alloc": allocation,
            "remainder": numerator % total,
        })
        allocated += allocation

    remaining = sample_size - allocated
    allocations.sort(key=lambda item: (-item["remainder"], item["category"]))
    for item in allocations:
        if remaining == 0:
            break
        if item["alloc"] >= len(item["questions"]):
            continue
        item["alloc"] += 1
        remaining -= 1
    allocations.sort(key=lambda item: item["category"])

    selected = []
    for item in allocations:
        selected.extend(deterministic_select(item["questions"], item["alloc"], seed))

ids = []
seen = set()
for question in selected:
    qid = str(question.get("question_id", "")).strip()
    if not qid or qid in seen:
        continue
    seen.add(qid)
    ids.append(qid)

pathlib.Path(output_path).write_text("\n".join(ids) + "\n", encoding="utf-8")
PY
}

materialise_aligned_oracle_sample_dataset() {
  local source_dataset=$1
  local oracle_dataset=$2
  local sample_ids_path=$3
  local output_path=$4
  python3 - "$source_dataset" "$oracle_dataset" "$sample_ids_path" "$output_path" <<'PY'
import json
import pathlib
import sys

source_dataset, oracle_dataset, sample_ids_path, output_path = map(pathlib.Path, sys.argv[1:])
source_rows = {
    str(row.get("question_id", "")).strip(): row
    for row in json.loads(source_dataset.read_text(encoding="utf-8"))
}
oracle_rows = {
    str(row.get("question_id", "")).strip(): row
    for row in json.loads(oracle_dataset.read_text(encoding="utf-8"))
}

sample_ids = []
seen = set()
for line in sample_ids_path.read_text(encoding="utf-8").splitlines():
    question_id = line.strip()
    if question_id == "":
        continue
    if question_id in seen:
        raise SystemExit(f"duplicate sample id in {sample_ids_path}: {question_id}")
    seen.add(question_id)
    sample_ids.append(question_id)

aligned = []
missing = []
for question_id in sample_ids:
    source_row = source_rows.get(question_id)
    oracle_row = oracle_rows.get(question_id)
    if source_row is None or oracle_row is None:
        missing.append(question_id)
        continue
    aligned_row = dict(oracle_row)
    aligned_row["question_date"] = source_row.get("question_date", oracle_row.get("question_date", ""))
    aligned.append(aligned_row)

if missing:
    raise SystemExit(
        "question ids missing while aligning oracle sample: " + ", ".join(missing)
    )

pathlib.Path(output_path).write_text(
    json.dumps(aligned, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

write_mode_summary() {
  local mode_dir=$1
  local mode_label=$2
  local result_path=$3
  local manifest_path=$4
  python3 - "$mode_dir" "$mode_label" "$result_path" "$manifest_path" <<'PY'
import json
import pathlib
import sys

mode_dir, mode_label, result_path, manifest_path = sys.argv[1:]
out_path = pathlib.Path(mode_dir) / "README.md"

result = json.loads(pathlib.Path(result_path).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))

lines = [
    f"# {mode_label}",
    "",
    f"- Benchmark mode: {manifest.get('benchmark_mode', '<unknown>')}",
    f"- Context source: {manifest.get('context_source', '<unknown>')}",
    f"- Questions run: {result.get('questions_run', 0)}",
    f"- Overall score: {result.get('overall_score', 0):.3f}",
    f"- Task average: {result.get('task_avg_score', 0):.3f}",
    f"- Exact match: {result.get('exact_match_score', 0):.3f}",
]

cost = result.get("cost_accounting") or {}
lines.append(
    f"- Cost USD total: {cost.get('total_usd', 0):.4f} "
    f"(ingest {cost.get('ingest_usd', 0):.4f}, agent {cost.get('agent_usd', 0):.4f}, judge {cost.get('judge_usd', 0):.4f})"
)
lines.append("")
lines.append("## Artefacts")
lines.append("")
lines.append(f"- result-go.json")
lines.append(f"- manifest-go.json")
lines.append(f"- run.log")
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

write_top_level_summary() {
  python3 - "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []

def add_row(label: str, result_path: pathlib.Path, manifest_path: pathlib.Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "label": label,
            "benchmark_mode": manifest.get("benchmark_mode", "<unknown>"),
            "context_source": manifest.get("context_source", "<unknown>"),
            "questions_run": result.get("questions_run", 0),
            "overall_score": result.get("overall_score", 0),
        }
    )

add_row(
    "oracle",
    root / "oracle" / "result-go.json",
    root / "oracle" / "manifest-go.json",
)
add_row(
    "full-context",
    root / "full-context" / "result-go.json",
    root / "full-context" / "manifest-go.json",
)

tri_dir = root / "real-retrieval"
for sdk in ("go", "ts", "py"):
    result_path = tri_dir / f"result-{sdk}.json"
    manifest_path = tri_dir / f"manifest-{sdk}.json"
    if result_path.is_file() and manifest_path.is_file():
        add_row(f"real-retrieval ({sdk})", result_path, manifest_path)

verbatim_dir = root / "verbatim-retrieval"
for sdk in ("go", "ts", "py"):
    result_path = verbatim_dir / f"result-{sdk}.json"
    manifest_path = verbatim_dir / f"manifest-{sdk}.json"
    if result_path.is_file() and manifest_path.is_file():
        add_row(f"verbatim-retrieval ({sdk})", result_path, manifest_path)

lines = [
    "# LME benchmark modes",
    "",
    "This directory contains the primary LongMemEval benchmark scenarios, plus an optional verbatim transcript diagnostic.",
    "",
    "| Run | Benchmark mode | Context source | Questions | Overall score |",
    "|-----|----------------|----------------|-----------|---------------|",
]
for row in rows:
    lines.append(
        f"| {row['label']} | {row['benchmark_mode']} | {row['context_source']} | {row['questions_run']} | {row['overall_score']:.3f} |"
    )

lines.extend(
    [
        "",
        "## Subdirectories",
        "",
        "- oracle/",
        "- full-context/",
        "- real-retrieval/",
    ]
)

if verbatim_dir.is_dir():
    lines.append("- verbatim-retrieval/")

(root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

require_dataset "$S_DATASET" "full-context"
ensure_oracle_dataset
SAMPLE_IDS_FILE="$OUTPUT_ROOT/sample-ids.txt"
ALIGNED_ORACLE_DATASET="$OUTPUT_ROOT/oracle-sample-aligned.json"
if [[ -n "$PINNED_SAMPLE_IDS_FILE" ]]; then
  cp "$PINNED_SAMPLE_IDS_FILE" "$SAMPLE_IDS_FILE"
else
  materialise_sample_ids "$S_DATASET" "$SAMPLE_IDS_FILE"
fi
materialise_aligned_oracle_sample_dataset "$S_DATASET" "$ORACLE_DATASET" "$SAMPLE_IDS_FILE" "$ALIGNED_ORACLE_DATASET"

if LLM_PROVIDER="$(infer_llm_provider)"; then
  if [[ "$LLM_PROVIDER" == "ollama" ]]; then
    require_ollama_model "$ACTOR_MODEL" "actor"
    require_ollama_model "$JUDGE_MODEL" "judge"
    require_ollama_model "$EXTRACT_MODEL" "extract"
  fi
fi

run_logged_step \
  "$OUTPUT_ROOT/build-go.log" \
  "Go CLI build" \
  bash -lc "cd \"$GO_SDK_DIR\" && go build -o \"$MEMORY_GO\" ./cmd/memory"

mkdir -p "$OUTPUT_ROOT/oracle" "$OUTPUT_ROOT/full-context"

run_logged_step \
  "$OUTPUT_ROOT/oracle/run.log" \
  "oracle benchmark" \
  env JB_LLM_MODEL="$ACTOR_MODEL" "$MEMORY_GO" eval lme run \
    --dataset "$ALIGNED_ORACLE_DATASET" \
    --benchmark-mode oracle \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --sample-ids-file "$SAMPLE_IDS_FILE" \
    --ingest-mode bulk \
    --concurrency "$CONCURRENCY" \
    --actor "$ACTOR_MODEL" \
    --judge "$JUDGE_MODEL" \
    --reader-cache-dir "$READER_CACHE_DIR" \
    --judge-cache-dir "$JUDGE_CACHE_DIR" \
    --max-cost-usd "$MAX_COST" \
    --output "$OUTPUT_ROOT/oracle/result-go.json" \
    --manifest "$OUTPUT_ROOT/oracle/manifest-go.json"

write_mode_summary \
  "$OUTPUT_ROOT/oracle" \
  "Oracle" \
  "$OUTPUT_ROOT/oracle/result-go.json" \
  "$OUTPUT_ROOT/oracle/manifest-go.json"

run_logged_step \
  "$OUTPUT_ROOT/full-context/run.log" \
  "full-context benchmark" \
  env JB_LLM_MODEL="$ACTOR_MODEL" "$MEMORY_GO" eval lme run \
    --dataset "$S_DATASET" \
    --benchmark-mode full-context \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --sample-ids-file "$SAMPLE_IDS_FILE" \
    --ingest-mode bulk \
    --concurrency "$CONCURRENCY" \
    --actor "$ACTOR_MODEL" \
    --judge "$JUDGE_MODEL" \
    --reader-cache-dir "$READER_CACHE_DIR" \
    --judge-cache-dir "$JUDGE_CACHE_DIR" \
    --max-cost-usd "$MAX_COST" \
    --output "$OUTPUT_ROOT/full-context/result-go.json" \
    --manifest "$OUTPUT_ROOT/full-context/manifest-go.json"

write_mode_summary \
  "$OUTPUT_ROOT/full-context" \
  "Full context" \
  "$OUTPUT_ROOT/full-context/result-go.json" \
  "$OUTPUT_ROOT/full-context/manifest-go.json"

run_logged_step \
  "$OUTPUT_ROOT/real-retrieval-launch.log" \
  "real-retrieval benchmark" \
  env \
    DATASET="$S_DATASET" \
    SAMPLE_SIZE="$SAMPLE_SIZE" \
    SEED="$SEED" \
    CONCURRENCY="$CONCURRENCY" \
    REPLAY_CONCURRENCY="$REPLAY_CONCURRENCY" \
    SAMPLE_IDS_FILE="$SAMPLE_IDS_FILE" \
    ACTOR_MODEL="$ACTOR_MODEL" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    EXTRACT_MODEL="$EXTRACT_MODEL" \
    RETRIEVAL_MODE="$RETRIEVAL_MODE" \
    READER_CACHE_DIR="$READER_CACHE_DIR" \
    JUDGE_CACHE_DIR="$JUDGE_CACHE_DIR" \
    MAX_COST="$MAX_COST" \
    JB_HOME="$SHARED_JB_HOME" \
    SKIP_EXTRACT="$SKIP_EXTRACT" \
    OUTPUT_ROOT="$OUTPUT_ROOT/real-retrieval" \
    "$MEMORY_DIR/eval/scripts/run_tri_lme.sh"

if [[ "$RUN_VERBATIM_RETRIEVAL" == "1" || "$RUN_VERBATIM_RETRIEVAL" == "true" ]]; then
  run_logged_step \
    "$OUTPUT_ROOT/verbatim-retrieval-launch.log" \
    "verbatim-retrieval benchmark" \
    env \
      DATASET="$S_DATASET" \
      SAMPLE_SIZE="$SAMPLE_SIZE" \
      SEED="$SEED" \
      CONCURRENCY="$CONCURRENCY" \
      REPLAY_CONCURRENCY="$REPLAY_CONCURRENCY" \
      SAMPLE_IDS_FILE="$SAMPLE_IDS_FILE" \
      ACTOR_MODEL="$ACTOR_MODEL" \
      JUDGE_MODEL="$JUDGE_MODEL" \
      EXTRACT_MODEL="$EXTRACT_MODEL" \
      RETRIEVAL_MODE="$RETRIEVAL_MODE" \
      READER_CACHE_DIR="$READER_CACHE_DIR" \
      JUDGE_CACHE_DIR="$JUDGE_CACHE_DIR" \
      MAX_COST="$MAX_COST" \
      JB_HOME="$SHARED_JB_HOME" \
      SKIP_EXTRACT=1 \
      ACTOR_SCOPE="raw_lme" \
      ACTOR_PROJECT="" \
      ACTOR_PATH_PREFIX="raw/lme/" \
      OUTPUT_ROOT="$OUTPUT_ROOT/verbatim-retrieval" \
      "$MEMORY_DIR/eval/scripts/run_tri_lme.sh"
fi

write_top_level_summary

echo "$OUTPUT_ROOT"
