# datasets

The shared daemon runner reads JSONL datasets from this directory. Those
files are one question per line and drive one explicit daemon scenario at a
time.

## Scenario use

| Scenario | Request field used | What gets scored |
| -------- | ------------------ | ---------------- |
| `ask-basic` | `question` -> `POST /v1/brains/{brain}/ask` | The answer reconstructed from the SSE stream. Exercises the standard `/ask` path. |
| `ask-augmented` | `question` and optional `question_date` -> `POST /v1/brains/{brain}/ask` with `readerMode=augmented` | The augmented answer reconstructed from the SSE stream. Exercises temporal forwarding and the augmented reader prompt. |
| `search-retrieve-only` | `question` forwarded as `query` and optional `question_date` forwarded as `questionDate` -> `POST /v1/brains/{brain}/search` | The retrieval blob built from returned chunk `text`, falling back to `summary`. Exercises retrieval parity only. |

When comparing SDKs, run one scenario at a time and give each scenario its own `--output` root, for example `results/ask-basic`, `results/ask-augmented`, and `results/search-retrieve-only`. The runner writes one `<sdk>.json` per output root and day.

## Contract

Every line is a single JSON object. Required and optional keys:

| key | type | required | notes |
| --- | --- | --- | --- |
| `id` | string | yes | unique per dataset |
| `question` | string | yes | the user prompt |
| `question_date` | string | no | forwarded as `questionDate` in `ask-augmented` and `search-retrieve-only` |
| `expected_substrings` | string[] | required for `--scorer exact` | case-insensitive substring match against the returned `answer` |
| `reference_answer` | string | required for `--scorer judge` | gold answer the judge compares against |
| `tags` | string[] | no | optional, useful for slicing results |

Lines that are blank or start with `#` are ignored by the loader, handy for
keeping human notes at the top of a file.

Scenario parity expects the same dataset row to work across all three SDKs. The only scenario-specific input field today is `question_date`, which is forwarded as `questionDate` for `ask-augmented` and `search-retrieve-only`.

## Files

- `smoke.jsonl` - 20 provider-agnostic factual questions. Cheap, fast,
  deterministic (`exact` scorer). Runs on every PR; used for fast
  `ask-basic` parity checks.
- `lme.jsonl` - 100-question benchmark spanning facts, definitional,
  temporal, procedural, comparison, and memory-retrieval-specific prompts
  (SQLite FTS5, BM25, embeddings, RAG, vector search). Used for the
  broader `ask-augmented` and `search-retrieve-only` verification flow
  once an `eval` brain has been populated.
- `longmemeval_s.json` - upstream LongMemEval-S JSON array. This is for the
  native Go replay runner and the replay-backed tri-SDK script, not for the
  shared JSONL daemon runner.

## Adding a new dataset

Drop a new `<name>.jsonl` file here and point `--dataset` at it. No code
changes required.

## Corpus-grounded datasets

`lme.jsonl` is already the shared corpus-grounded daemon dataset. It becomes
meaningful once the target SDK has a populated `eval` brain, or once the
replay-backed tri-SDK flow has extracted the shared brain cache first.

`smoke.jsonl` is the provider-agnostic empty-brain dataset. `longmemeval_s.json`
is the upstream 500-question replay dataset used by the native Go LME runner
and the replay-backed tri-SDK script. It is a JSON array, not JSONL.
