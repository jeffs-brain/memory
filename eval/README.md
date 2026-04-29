# eval

Cross-SDK evaluation runner for `jeffs-brain/memory`. Drives the TypeScript, Go, and Python SDKs through the shared HTTP daemon surface, scores the resulting answers or retrieval blobs, and writes per-run artefacts under `results/`.

## What it does

1. Spawns the chosen SDK's `memory serve` daemon on a random port (or one you supply).
2. Loads a JSONL dataset of questions.
3. For each question, runs one explicit cross-SDK scenario:
   - `ask-basic`: `POST /v1/brains/{brain}/ask` with `{"question", "topK", "mode"}`.
   - `ask-augmented`: `POST /v1/brains/{brain}/ask` with `{"question", "topK", "mode", "readerMode": "augmented", "questionDate"?}`.
   - `search-retrieve-only`: `POST /v1/brains/{brain}/search` with `{"query", "topK", "mode", "questionDate"?, "candidateK"?, "rerankTopN"?}` and folds the returned chunk text into a retrieval-only answer blob for scoring.
4. Forwards the retrieval `mode` unchanged to the daemon. The runner default is `auto`, so each SDK resolves it locally to `hybrid` when embeddings are available or `bm25` otherwise.
5. Scores with either the deterministic `exact` scorer (case-insensitive substring match) or the `judge` scorer (LLM-as-judge via OpenAI `gpt-4o` by default).
6. Writes `<output>/<YYYY-MM-DD>/<sdk>.json`.
7. Fails loudly when the pass rate falls below a configurable floor (default `0.90`).

## Parity expectations

- The runner verifies the same three daemon scenarios across `ts`, `go`, and `py`: `ask-basic`, `ask-augmented`, and `search-retrieve-only`.
- Parity means the same request shape, transport shape, retrieval-mode semantics, and temporal forwarding rules across all three daemons.
- `ask-basic` and `ask-augmented` are answer-scoring scenarios. `search-retrieve-only` is a retrieval-scoring scenario built from returned chunks only. It does not score a daemon-generated answer.
- The runner is for pass/fail comparison and artefact inspection. We are not treating this README as a published benchmark page.
- Native LongMemEval tooling is intentionally uneven today. Go remains the reference replay runner and tri-SDK orchestrator, TypeScript has native `memory eval lme` commands for single-SDK runs, and Python participates in LME parity through `memory serve` only.

## Native LME status today

| SDK | Status |
| --- | ------ |
| Go | Full native `memory eval lme run` surface, including replay ingest, extract-only runs, `actor-endpoint-style=retrieve-only`, and the shared tri-SDK orchestration in `eval/scripts/run_tri_lme.sh`. This is the reference path. |
| TypeScript | Native `memory eval lme` commands exist for single-SDK fetch, run, score, compare, and doctor flows. They are not the coordinator for the replay-backed tri-SDK benchmark. |
| Python | No native `memory eval lme` CLI today. Python joins parity runs through `memory serve`, either under this shared runner or under the Go tri-SDK orchestration. |

## Pre-ingestion

The runner assumes a brain has already been populated against the target SDK when you are doing corpus-grounded verification. Before running `ask-augmented` or `search-retrieve-only` against a memory corpus, create and populate a brain named `eval` via the SDK's CLI, for example:

```bash
memory ingest ./corpus --brain eval
```

Override the brain via `--brain <id>` on the runner when you want to target
something else. `smoke.jsonl` is for fast offline checks and is typically run
with `--seed-reference-brain`, which deletes and recreates the target brain
then ingests one markdown file per dataset row using that row's
`reference_answer`. `lme.jsonl` is the shared corpus-grounded daemon dataset
used for `ask-augmented` and `search-retrieve-only` once the `eval` brain has
been populated. The tri-SDK replay script does its own shared extract step and
does not require a pre-populated daemon brain.

## Local run

```bash
cd ~/code/jeffs-brain/memory/eval
uv sync

# Fast offline parity check using a seeded reference brain
for sdk in ts go py; do
  uv run python runner.py \
    --sdk "$sdk" \
    --dataset datasets/smoke.jsonl \
    --scorer exact \
    --scenario search-retrieve-only \
    --mode bm25 \
    --brain eval \
    --seed-reference-brain \
    --output results/smoke-search
done

# Corpus-grounded parity checks, against a populated `eval` brain
for sdk in ts go py; do
  OPENAI_API_KEY=sk-... uv run python runner.py \
    --sdk "$sdk" \
    --dataset datasets/lme.jsonl \
    --scorer judge \
    --scenario ask-augmented \
    --brain eval \
    --output results/ask-augmented
done

for sdk in ts go py; do
  OPENAI_API_KEY=sk-... uv run python runner.py \
    --sdk "$sdk" \
    --dataset datasets/lme.jsonl \
    --scorer judge \
    --scenario search-retrieve-only \
    --brain eval \
    --output results/search-retrieve-only
done
```

Use a separate `--output` root per scenario when comparing SDKs on the same day. The runner writes one `<sdk>.json` under `<output>/<YYYY-MM-DD>/`, so in practice you want roots such as `results/smoke-search`, `results/ask-augmented`, and `results/search-retrieve-only`.

### CLI flags

| Flag         | Default                 | Notes                                                                 |
| ------------ | ----------------------- | --------------------------------------------------------------------- |
| `--sdk`      | required                | `ts`, `go`, or `py`.                                                  |
| `--scenario` | `ask-basic`             | `ask-basic`, `ask-augmented`, or `search-retrieve-only`.              |
| `--mode`     | `auto`                  | Daemon retrieval mode: `auto`, `hybrid`, `hybrid-rerank`, `bm25`, or `semantic`. Forwarded unchanged to `/ask` or `/search`. |
| `--dataset`  | `datasets/lme.jsonl`    | JSONL file.                                                           |
| `--scorer`   | `judge`                 | `exact` or `judge`.                                                   |
| `--limit`    | none                    | Cap question count.                                                   |
| `--output`   | `results/`              | Scenario-specific output root. The runner writes `<output>/<YYYY-MM-DD>/<sdk>.json`. |
| `--port`     | `0`                     | `0` means random free port.                                           |
| `--floor`    | `0.90`                  | Below this, the runner exits non-zero.                                |
| `--brain`    | `eval`                  | brainId passed into `POST /v1/brains/{brain}/ask` or `/search`; pre-populate it before running. |
| `--top-k`    | `5`                     | Forwarded as `topK` on `/ask` or `/search`.                           |
| `--candidate-k` | `0`                  | Forwarded as `candidateK` on `search-retrieve-only`. `0` defers to the daemon default. |
| `--rerank-top-n` | `0`                 | Forwarded as `rerankTopN` on `search-retrieve-only`. `0` defers to the daemon default. |
| `--seed-reference-brain` | off         | Deletes and recreates `--brain`, then ingests one markdown document per dataset row using `reference_answer`. Intended for offline retrieval smoke checks. |

### What each scenario exercises

| Scenario | Request | Transport | Scored artefact | What it exercises |
| -------- | ------- | --------- | --------------- | ----------------- |
| `ask-basic` | `POST /ask` with `question`, `topK`, `mode` | SSE | concatenated `answer_delta` stream, or `done.answer` when present | Standard `/ask` retrieval-to-reader path, plus `retrieve`, `citation`, and `done` event shape. |
| `ask-augmented` | `POST /ask` with `question`, `topK`, `mode`, `readerMode=augmented`, optional `questionDate` | SSE | concatenated `answer_delta` stream, or `done.answer` when present | Augmented reader prompt path, temporal anchor forwarding, and the same SSE contract. |
| `search-retrieve-only` | `POST /search` with `query`, `topK`, `mode`, optional `questionDate`, `candidateK`, and `rerankTopN` | JSON | concatenated returned chunk `text`, falling back to `summary` | Pure retrieval behaviour only: chunk selection, optional rerank knobs, and returned chunk payload parity. No daemon-generated answer path is used. |

### How we test it

- `ask-basic` and `ask-augmented` both consume SSE, ignore `retrieve` frames for scoring, collect `citation` frames for the result JSON, and score only the final answer text.
- `search-retrieve-only` calls `/search` directly, records chunk metadata as citations, and scores the merged retrieval blob the runner builds from returned chunks only.
- `question_date` is forwarded in `ask-augmented` and `search-retrieve-only`, where the runner sends it as `questionDate`.
- `candidateK` and `rerankTopN` are forwarded only for `search-retrieve-only`, and only when the CLI flags are set above zero.
- `--mode auto` is the shared default because it matches daemon semantics. The harness does not resolve `auto` itself.

### Verification workflow

1. Pick the scenario you want to verify.
2. Run the SDK-local regression tests that pin that scenario's behaviour.
3. Run the shared runner across `ts`, `go`, and `py` for that same scenario.
4. Compare `results/<scenario>/<YYYY-MM-DD>/ts.json`, `go.json`, and `py.json` for request, citation, and answer or retrieval-blob shape. For replay-backed tri-SDK runs, compare `tri-lme-<timestamp>/result-*.json` alongside `manifest.json` and `manifest-*.json`. Treat the output as verification artefacts rather than a public benchmark.

### SDK-local regression commands

The runner compares SDKs against each other. Scenario regressions inside one SDK are pinned by its own test suite:

| SDK | Command |
| --- | ------- |
| TypeScript | `cd sdks/ts/memory && bun x vitest run src/http/handlers.test.ts src/http/daemon.test.ts` |
| Go | `cd go && go test ./cmd/memory ./eval/lme` |
| Python | `cd sdks/py && uv run pytest tests/test_serve_ask_augmented.py tests/test_serve_handlers_real.py tests/test_retrieval_temporal.py` |

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

For the native Go LME runner (`memory eval lme run`) the additional knobs are `JB_LME_JUDGE_MODEL` and `JB_LME_ACTOR_MODEL`.

## Dataset contract

The shared daemon runner consumes JSONL, one question per line. Native LongMemEval replay uses the upstream `longmemeval_s.json` JSON array directly and does not go through `runner.py`.

| key                   | type     | notes                                                 |
| --------------------- | -------- | ----------------------------------------------------- |
| `id`                  | string   | unique per dataset                                    |
| `question`            | string   | the prompt                                            |
| `question_date`       | string   | optional. Forwarded as `questionDate` in `ask-augmented` and `search-retrieve-only`. |
| `expected_substrings` | string[] | required for `exact` scorer                           |
| `reference_answer`    | string   | required for `judge` scorer                           |
| `tags`                | string[] | optional, for slicing results                         |

See `datasets/README.md` for details and how to populate `lme.jsonl`.

## Scorers

- `exact` - deterministic. `answer` must contain at least one `expected_substring` (case-insensitive). Returns 1.0 or 0.0. No network.
- `judge` - asks an LLM to rate the answer against the `reference_answer` on a 0..1 scale (rubric: faithfulness, citation presence, semantic match). Defaults to OpenAI `gpt-4o` (paper-faithful); override with `JB_EVAL_JUDGE_MODEL`. Set `JB_EVAL_BUDGET_USD` to cap spend.

## Results layout

```
results/
  ask-basic/           # shared runner output root for one scenario
    <YYYY-MM-DD>/
      ts.json
      go.json
      py.json
  ask-augmented/
    <YYYY-MM-DD>/
      ts.json
      go.json
      py.json
  search-retrieve-only/
    <YYYY-MM-DD>/
      ts.json
      go.json
      py.json
  tri-lme-<timestamp>/ # replay-backed tri-SDK benchmark runs
    README.md
    build-go.log       # Go CLI build log before extract
    build-ts.log       # TS CLI build log before daemon spawn
    extract.json
    manifest.json      # shared extract-only replay manifest
    result-ts.json
    result-go.json
    result-py.json
    manifest-ts.json   # per-SDK retrieve-only benchmark manifest
    manifest-go.json
    manifest-py.json
    daemon-ts.log
    daemon-go.log
    daemon-py.log
    runner-ts.log
    runner-go.log
    runner-py.log
```

## Native LongMemEval replay

The cross-SDK runner does not perform replay ingest or agentic LongMemEval runs. Use the native SDK runners for those workflows. Recommended Go configuration for local runs:

- **Extract model**: `gpt-5`. Use the stronger extractor by default for replay fidelity.
- **Actor model**: `gpt-4o`. Paper's recommended reader.
- **Judge model**: `gpt-4o`. Paper's recommended judge.
- **Dataset**: `longmemeval_s.json` (500 questions, SHA-pinned). Fetch via `scripts/fetch-lme.sh` from the repo root.
- **Concurrency**: 16 replay workers, 16 question workers locally. OpenAI tier-5 handles this comfortably on gpt-4o + gpt-5.
- **Sampling**: the Go sampler now fills the requested `--sample-size` exactly. `50` means 50 questions, not a floored per-category subset.
- **Reproducibility**: use the shared reader cache plus the shared judge cache when you care about score stability across reruns.

### Single-SDK run

```bash
set -a && source ~/code/jeffs-brain/memory/.env && set +a

memory eval lme run \
  --dataset eval/datasets/longmemeval_s.json \
  --sample-size 50 \
  --seed 42 \
  --ingest-mode replay \
  --extract-model gpt-5 \
  --replay-concurrency 16 \
  --concurrency 16 \
  --actor gpt-4o \
  --judge gpt-4o \
  --max-cost-usd 25 \
  --brain-cache /tmp/jb-lme/eval-lme \
  --output /tmp/lme-go.json
```

### Tri-SDK run (recommended)

The orchestrator extracts once and benchmarks all three daemons from the shared brain. It exercises the shared daemon `search-retrieve-only` scenario only, using actor-endpoint `retrieve-only` mode so retrieval happens in each daemon while extraction, evidence rendering, the shared augmented reader, judging, and manifest writing stay in the Go runner process:

```bash
set -a && source ~/code/jeffs-brain/memory/.env && set +a

JB_LLM_PROVIDER=openai \
ACTOR_MODEL=gpt-4o \
JUDGE_MODEL=gpt-4o \
EXTRACT_MODEL=gpt-5 \
SAMPLE_SIZE=50 \
CONCURRENCY=16 \
REPLAY_CONCURRENCY=16 \
CANDIDATE_K=60 \
RERANK_TOP_N=20 \
MAX_COST=25 \
JB_HOME=/tmp/jb-lme-s50 \
bash eval/scripts/run_tri_lme.sh
```

Phases:
1. Go extracts into `$JB_HOME/brains/$BRAIN_ID` (`/tmp/jb-lme-s50/brains/eval-lme` in the example above).
2. TS, Go, Py `memory serve` daemons spawn against the shared brain.
3. Go LME runner fires three times in parallel with `--actor-endpoint` pointed at each daemon. In `retrieve-only` mode the daemon stays retrieval-only, returning `/search` payloads only. The scored answer still comes from the shared Go-side evidence renderer, augmented reader, and judge, which keeps the cross-SDK comparison aligned.
   For replay-backed tri-SDK runs we pin actor retrieval to replay memory only via `--actor-scope memory --actor-project <brain-id>`, which keeps global memory plus the eval brain's project memory in scope while excluding raw transcript rows.
4. Tear daemons down; emit `tri-lme-<timestamp>/README.md` with the run
summary plus `extract.json`, `manifest.json`, per-SDK result JSON, per-SDK
manifests, and daemon or runner logs.

The script now cleans only `$JB_HOME/brains/$BRAIN_ID` before the extract
step rather than deleting the whole `JB_HOME` tree.

When extraction is already known-good and you only need to re-measure daemon
retrieval or reader changes, set `SKIP_EXTRACT=1` and point `JB_HOME` at an
existing replay cache:

```bash
set -a && source ~/code/jeffs-brain/memory/.env && set +a

JB_LLM_PROVIDER=openai \
ACTOR_MODEL=gpt-4o \
JUDGE_MODEL=gpt-4o \
SAMPLE_SIZE=50 \
CONCURRENCY=16 \
REPLAY_CONCURRENCY=16 \
CANDIDATE_K=60 \
RERANK_TOP_N=20 \
MAX_COST=25 \
JB_HOME=/tmp/jb-lme-shared-20260419-poststabilise-openai \
SKIP_EXTRACT=1 \
bash eval/scripts/run_tri_lme.sh
```

Every run writes a `RunManifest` with the base LME fields. In the replay-backed tri-SDK flow:

- `build-go.log` and `build-ts.log` capture the pre-daemon build steps. If the harness stops before daemon spawn, those logs are the first place to check.
- `manifest.json` is the shared extract-only replay manifest.
- Each `manifest-<sdk>.json` is a per-SDK retrieve-only benchmark manifest. The script stamps `sdk`, `scenario`, `actor_endpoint_style`, `actor_brain`, `actor_topk`, `actor_candidatek`, `actor_rerank_topn`, `actor_scope`, `actor_project`, `actor_path_prefix`, `shared_extract_output`, and `shared_extract_manifest` so the retrieve-only comparison is reproducible from disk alone.

Benchmark notes:

- `run_lme_modes.sh` and `run_tri_lme.sh` both default `READER_CACHE_DIR=~/.local/state/jeffs-brain/evals/reader-cache`.
- Both now also default `JUDGE_CACHE_DIR=~/.local/state/jeffs-brain/evals/judge-cache`.
- `run_lme_modes.sh` now forwards `SKIP_EXTRACT` into the real-retrieval leg, so a sample-50 replay can populate the shared brain once and the later 500-question run can reuse it.
- If you need an exact dev or held-out subset, pass `SAMPLE_IDS_FILE=/abs/path/to/ids.txt`. The top-level mode runner copies that file into its output root and reuses it across all benchmark modes.

## Adding a new SDK

1. Drop a subclass of `sdks/base.py:SdkRunner` into `sdks/<name>.py` that knows:
   - the command to start the daemon,
   - how to poll `/healthz` until 200,
   - how to stop cleanly.
2. Register it in `sdks/__init__.py:get_runner`.
3. Add the new value to the `--sdk` click Choice in `runner.py`.
4. Add a matrix entry to `.github/workflows/eval-nightly.yml`.

## CI

- `eval-smoke` runs on every PR across `{ts, go, py}` as an offline retrieval check: it seeds a small reference brain from `smoke.jsonl`, then scores `search-retrieve-only` in `bm25` mode with the `exact` scorer.
- `eval-nightly` is reserved for broader benchmark coverage; native LongMemEval parity still lives with the SDK-specific runners.

## Cost model

Per the restructure plan: roughly $3 to $5 per day on a cheaper `gpt-4o-mini` reader and judge profile across three SDKs and 500 questions, so $100 to $150 per month. Paid from the Erys AI card. `JB_EVAL_BUDGET_USD` caps spend per run.

## Release-candidate runs

Tag a release candidate (e.g. `memory-v0.2.0-rc.1`). The nightly workflow's `workflow_dispatch` trigger accepts a `ref` input, so run it manually against the rc tag to get a green signal before cutting the final release.

## TODO for the next pass

Search the tree for `TODO(eval)`:

- `sdks/*.py` - prebuild Go and TS binaries in CI to avoid paying the compile cost per nightly invocation.
- `scorer/judge.py` - optional prompt cache so re-runs against identical (model, prompt) pairs skip the API call.
