# eval

Cross-SDK evaluation runner for `jeffs-brain/memory`. Drives the TS, Go, and Python SDKs through an identical HTTP contract, scores their answers, and publishes a nightly matrix.

## What it does

1. Spawns the chosen SDK's `memory serve` daemon on a random port.
2. Loads a JSONL dataset of questions.
3. For each question, POSTs `/v1/brains/{brain}/ask` to the daemon with `{"question", "topK", "mode"}` and consumes the SSE response, folding `answer_delta` (or `token`) frames into the final answer and collecting `citation` frames.
4. Scores with either the deterministic `exact` judge (substring match) or the `judge` scorer (LLM-as-judge via OpenAI `gpt-4o-mini` by default).
5. Writes `results/<date>/<sdk>.json`.
6. Fails loudly if the pass rate falls below a configurable floor (default 0.90; target greater than or equal to 0.934).

## Pre-ingestion

The runner assumes a brain has already been populated against the target SDK. Before running the LME dataset, create and populate a brain named `eval` via the SDK's CLI, for example:

```bash
memory ingest ./corpus --brain eval
```

Override the brain via `--brain <id>` on the runner if you want to target something else.

## Local run

```bash
cd ~/code/jeffs-brain/memory/eval
uv sync

# Smoke (exact scorer, no API cost)
uv run python runner.py --sdk ts --dataset datasets/smoke.jsonl --scorer exact

# Full (LLM judge, needs OPENAI_API_KEY)
OPENAI_API_KEY=sk-... uv run python runner.py --sdk ts --dataset datasets/lme.jsonl --scorer judge
```

CLI flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--sdk` | required | `ts`, `go`, or `py` |
| `--mode` | `direct` | `direct` or `agentic` |
| `--dataset` | `datasets/lme.jsonl` | JSONL file |
| `--scorer` | `judge` | `exact` or `judge` |
| `--limit` | none | cap question count |
| `--output` | `results/` | where to write `<date>/<sdk>.json` |
| `--port` | `0` | `0` means random free port |
| `--floor` | `0.90` | below this, the runner exits non-zero |
| `--brain` | `eval` | brainId passed into `POST /v1/brains/{brain}/ask`; pre-populate it before running |
| `--top-k` | `5` | forwarded as `topK` on each ask payload |

## Dataset contract

JSONL, one question per line. Keys:

| key | type | notes |
| --- | --- | --- |
| `id` | string | unique per dataset |
| `question` | string | the prompt |
| `expected_substrings` | string[] | required for `exact` scorer |
| `reference_answer` | string | required for `judge` scorer |
| `tags` | string[] | optional, for slicing results |

See `datasets/README.md` for details and how to populate `lme.jsonl`.

## Scorers

- `exact` — deterministic. `answer` must contain at least one `expected_substring`. Returns 1.0 or 0.0. No network.
- `judge` — asks an LLM to rate the answer against the `reference_answer` on a 0..1 scale (rubric: faithfulness, citation presence, semantic match). Defaults to OpenAI `gpt-4o-mini`; override with `JB_EVAL_JUDGE_MODEL`. Set `JB_EVAL_BUDGET_USD` to fail-fast when accumulated spend exceeds the threshold.

## Adding a new SDK

1. Drop a subclass of `sdks/base.py:SdkRunner` into `sdks/<name>.py` that knows:
   - the command to start the daemon,
   - how to poll `/healthz` until 200,
   - how to stop cleanly.
2. Register it in `sdks/__init__.py:get_runner`.
3. Add the new value to the `--sdk` click Choice in `runner.py`.
4. Add a matrix entry to `.github/workflows/eval-nightly.yml`.

## CI

- `eval-smoke` runs on every PR. 15-ish questions, `exact` scorer, no LLM cost.
- `eval-nightly` runs at 03:00 UTC across `{ts, go, py}`. Uses the `judge` scorer with the full 500-question LME set. Floor 0.90, target 0.934 (TS current baseline).

## Cost model

Per the restructure plan section 6: roughly $3 to $5 per day with `gpt-4o-mini` as both reader and judge across three SDKs and 500 questions, so $100 to $150 per month. Paid from the Erys AI card. `JB_EVAL_BUDGET_USD` caps spend per run.

## Release-candidate runs

Tag a release candidate (e.g. `memory-v0.2.0-rc.1`). The nightly workflow's `workflow_dispatch` trigger accepts a `ref` input, so run it manually against the rc tag to get a green signal before cutting the final release.

## TODO for the next pass

Search the tree for `TODO(eval)` — these are the spots stubbed during scaffolding:

- `sdks/*.py` — actual daemon commands depend on each SDK's binary being present. Integration tests are skipped until they are.
- `scorer/judge.py` — optional prompt cache so re-runs against identical (model, prompt) pairs skip the API call.
