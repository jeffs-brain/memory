# datasets

Eval datasets live here as JSONL. One question per line. The runner reads
`datasets/<file>.jsonl` and sends each `question` to the SDK's ask endpoint
(`POST /v1/brains/{brain}/ask`).

## Contract

Every line is a single JSON object. Required and optional keys:

| key | type | required | notes |
| --- | --- | --- | --- |
| `id` | string | yes | unique per dataset |
| `question` | string | yes | the user prompt |
| `expected_substrings` | string[] | required for `--scorer exact` | case-insensitive substring match against the returned `answer` |
| `reference_answer` | string | required for `--scorer judge` | gold answer the judge compares against |
| `tags` | string[] | no | optional, useful for slicing results |

Lines that are blank or start with `#` are ignored by the loader, handy for
keeping human notes at the top of a file.

## Files

- `smoke.jsonl` — 20 provider-agnostic factual questions. Cheap, fast,
  deterministic (`exact` scorer). Runs on every PR.
- `lme.jsonl` — 100-question benchmark spanning facts, definitional,
  temporal, procedural, comparison, and memory-retrieval-specific prompts
  (SQLite FTS5, BM25, embeddings, RAG, vector search). Answerable from
  general LLM knowledge so it can run against any pre-ingested brain.

## Adding a new dataset

Drop a new `<name>.jsonl` file here and point `--dataset` at it. No code
changes required.

## Corpus-grounded datasets

The current `lme.jsonl` is provider-agnostic: questions are scored against
the LLM's direct answer quality via `/ask`. A later pass should add
corpus-grounded fixtures that depend on a specific set of ingested
documents; those will ship as their own `<name>.jsonl` plus a sibling
`<name>.corpus/` directory and are out of scope here.
