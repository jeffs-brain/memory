# eval

Cross-SDK evaluation runner for `jeffs-brain/memory`. Drives the TypeScript, Go, and Python SDKs through the same HTTP ask contract, scores their answers, and publishes a matrix under `results/`.

## What it does

1. Spawns the chosen SDK's `memory serve` daemon on a random port (or one you supply).
2. Loads a JSONL dataset of questions.
3. For each question, POSTs `/v1/brains/{brain}/ask` with `{"question", "topK", "mode"}` and consumes the SSE response, folding `answer_delta` (or `token`) frames into the final answer and collecting `citation` frames.
4. Scores with either the deterministic `exact` scorer (case-insensitive substring match) or the `judge` scorer (LLM-as-judge via OpenAI `gpt-4o-mini` by default).
5. Writes `results/<date>/<sdk>.json`.
6. Fails loudly when the pass rate falls below a configurable floor (default 0.90; target greater than or equal to 0.934).

## Current results

- Tri-SDK smoke (20 questions, Ollama `gemma3:latest`, `exact` scorer, 2026-04-18): TypeScript, Go, and Python all at 19/20 (95%). Write-up: [`results/cross-sdk/cross-sdk-smoke-tri-fix-2026-04-18.md`](./results/cross-sdk/cross-sdk-smoke-tri-fix-2026-04-18.md). Per-SDK JSON: `results/cross-sdk/{ts,go,py}-smoke-tri-fix-2026-04-18.json`.
- Full LongMemEval replay lives in the Go SDK (`memory eval lme run --ingest-mode=replay`) and targets the 93.4% parity benchmark.

## Pre-ingestion

The runner assumes a brain has already been populated against the target SDK. Before running a corpus-grounded dataset, create and populate a brain named `eval` via the SDK's CLI, for example:

```bash
memory ingest ./corpus --brain eval
```

Override the brain via `--brain <id>` on the runner when you want to target something else. The packaged `smoke.jsonl` and `lme.jsonl` are provider-agnostic and do not require a specific corpus.

## Local run

```bash
cd ~/code/jeffs-brain/memory/eval
uv sync

# Smoke (exact scorer, no API cost)
uv run python runner.py --sdk ts --dataset datasets/smoke.jsonl --scorer exact
uv run python runner.py --sdk go --dataset datasets/smoke.jsonl --scorer exact
uv run python runner.py --sdk py --dataset datasets/smoke.jsonl --scorer exact

# Full (LLM judge, needs OPENAI_API_KEY)
OPENAI_API_KEY=sk-... uv run python runner.py --sdk ts --dataset datasets/lme.jsonl --scorer judge
```

### CLI flags

| Flag         | Default                 | Notes                                                                 |
| ------------ | ----------------------- | --------------------------------------------------------------------- |
| `--sdk`      | required                | `ts`, `go`, or `py`.                                                  |
| `--mode`     | `direct`                | `direct` or `agentic`.                                                |
| `--dataset`  | `datasets/lme.jsonl`    | JSONL file.                                                           |
| `--scorer`   | `judge`                 | `exact` or `judge`.                                                   |
| `--limit`    | none                    | Cap question count.                                                   |
| `--output`   | `results/`              | Where to write `<date>/<sdk>.json`.                                   |
| `--port`     | `0`                     | `0` means random free port.                                           |
| `--floor`    | `0.90`                  | Below this, the runner exits non-zero.                                |
| `--brain`    | `eval`                  | brainId passed into `POST /v1/brains/{brain}/ask`; pre-populate it before running. |
| `--top-k`    | `5`                     | Forwarded as `topK` on each ask payload.                              |

## Environment

- `JB_LLM_PROVIDER` / `JB_LLM_MODEL` - pin the SDK daemon's provider and model (`openai`, `anthropic`, `ollama`, or `fake`).
- `OPENAI_API_KEY` - required for OpenAI readers and the default `judge` scorer.
- `OPENAI_BASE_URL` - override the OpenAI endpoint (Azure, local shim, etc.).
- `JB_LLM_BASE_URL` - TypeScript SDK: alternate LLM endpoint (e.g. an OpenAI-compatible proxy).
- `JB_LLM_API_KEY` - TypeScript SDK: explicit API key for the configured provider.
- `JB_EVAL_JUDGE_MODEL` - override the judge model (default OpenAI `gpt-4o`).
- `JB_EVAL_BUDGET_USD` - fail-fast when accumulated judge spend exceeds the threshold.
- `OLLAMA_HOST` - Ollama endpoint for local Ollama-backed runs (default `http://localhost:11434`).
- `ANTHROPIC_API_KEY` - required when `JB_LLM_PROVIDER=anthropic`.

For the Go LME runner (`memory eval lme run`) the additional knobs are `JB_LME_JUDGE_MODEL` and `JB_LME_ACTOR_MODEL`.

## Dataset contract

JSONL, one question per line.

| key                   | type     | notes                                                 |
| --------------------- | -------- | ----------------------------------------------------- |
| `id`                  | string   | unique per dataset                                    |
| `question`            | string   | the prompt                                            |
| `expected_substrings` | string[] | required for `exact` scorer                           |
| `reference_answer`    | string   | required for `judge` scorer                           |
| `tags`                | string[] | optional, for slicing results                         |

See `datasets/README.md` for details and how to populate `lme.jsonl`.

## Scorers

- `exact` - deterministic. `answer` must contain at least one `expected_substring` (case-insensitive). Returns 1.0 or 0.0. No network.
- `judge` - asks an LLM to rate the answer against the `reference_answer` on a 0..1 scale (rubric: faithfulness, citation presence, semantic match). Defaults to OpenAI `gpt-4o` (paper-faithful); override with `JB_EVAL_JUDGE_MODEL`. Set `JB_EVAL_BUDGET_USD` to cap spend.

## Cross-SDK results layout

```
results/
  <date>/              # single-SDK runs written by runner.py
    ts.json
    go.json
    py.json
  cross-sdk/           # multi-SDK benchmark reports
    cross-sdk-smoke-tri-fix-2026-04-18.md
    ts-smoke-tri-fix-2026-04-18.json
    go-smoke-tri-2026-04-18.json
    py-smoke-tri-2026-04-18.json
```

## LongMemEval replay

Full replay lives in the Go SDK. Recommended configuration for local runs:

- **Extract model**: `gpt-5-mini` — cheap, reasoning-capable, no temperature knob.
- **Actor model**: `gpt-4o` — paper's recommended reader.
- **Judge model**: `gpt-4o` — paper's recommended judge.
- **Dataset**: `longmemeval_s.json` (500 questions, SHA-pinned). Fetch via `scripts/fetch-lme.sh` from the repo root.
- **Concurrency**: 16 replay workers, 16 question workers locally. OpenAI tier-5 handles this comfortably on gpt-4o + gpt-5-mini.

### Single-SDK run

```bash
set -a && source ~/code/jeffs-brain/memory/.env && set +a

memory eval lme run \
  --dataset eval/datasets/longmemeval_s.json \
  --sample-size 50 \
  --seed 42 \
  --ingest-mode replay \
  --extract-model gpt-5-mini \
  --replay-concurrency 16 \
  --concurrency 16 \
  --actor gpt-4o \
  --judge gpt-4o \
  --max-cost-usd 25 \
  --brain-cache /tmp/jb-lme/eval-lme \
  --output /tmp/lme-go.json
```

### Tri-SDK run (recommended)

The orchestrator extracts once and benchmarks all three daemons from the shared brain:

```bash
set -a && source ~/code/jeffs-brain/memory/.env && set +a

JB_LLM_PROVIDER=openai \
ACTOR_MODEL=gpt-4o \
JUDGE_MODEL=gpt-4o \
EXTRACT_MODEL=gpt-5-mini \
SAMPLE_SIZE=50 \
CONCURRENCY=16 \
REPLAY_CONCURRENCY=16 \
MAX_COST=25 \
bash eval/scripts/run_tri_lme.sh
```

Phases:
1. Go extracts into `$JB_HOME/brains/$BRAIN_ID` (`/tmp/jb-lme-shared/brains/eval-lme` by default).
2. TS, Go, Py `memory serve` daemons spawn against the shared brain.
3. Go LME runner fires three times in parallel with `--actor-endpoint` pointed at each daemon — retrieval + reader happen in the daemon, judge happens in-process so every SDK scores against the same gpt-4o judge config.
4. Tear daemons down; emit `tri-lme-<timestamp>/README.md` with the pass-rate table plus per-SDK result JSON + manifest.

The 93.4% parity target is tracked against the jeff baseline. Every run writes a `RunManifest` (dataset SHA, judge model, prompt version, seed, sample size, ingest mode) so scores are only comparable when all four key fields match.

## Adding a new SDK

1. Drop a subclass of `sdks/base.py:SdkRunner` into `sdks/<name>.py` that knows:
   - the command to start the daemon,
   - how to poll `/healthz` until 200,
   - how to stop cleanly.
2. Register it in `sdks/__init__.py:get_runner`.
3. Add the new value to the `--sdk` click Choice in `runner.py`.
4. Add a matrix entry to `.github/workflows/eval-nightly.yml`.

## CI

- `eval-smoke` runs on every PR across `{ts, go, py}`: ~15 questions, `exact` scorer, no LLM cost.
- `eval-nightly` runs at 03:00 UTC across `{ts, go, py}` with the `judge` scorer and the full LME set. Floor 0.90, target 0.934.

## Cost model

Per the restructure plan: roughly $3 to $5 per day with `gpt-4o-mini` as both reader and judge across three SDKs and 500 questions, so $100 to $150 per month. Paid from the Erys AI card. `JB_EVAL_BUDGET_USD` caps spend per run.

## Release-candidate runs

Tag a release candidate (e.g. `memory-v0.2.0-rc.1`). The nightly workflow's `workflow_dispatch` trigger accepts a `ref` input, so run it manually against the rc tag to get a green signal before cutting the final release.

## TODO for the next pass

Search the tree for `TODO(eval)`:

- `sdks/*.py` - prebuild Go and TS binaries in CI to avoid paying the compile cost per nightly invocation.
- `scorer/judge.py` - optional prompt cache so re-runs against identical (model, prompt) pairs skip the API call.
