# jeffs-brain/memory

[![TS CI](https://github.com/jeffs-brain/memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jeffs-brain/memory/actions/workflows/ci.yml)
[![Go CI](https://github.com/jeffs-brain/memory/actions/workflows/go.yml/badge.svg)](https://github.com/jeffs-brain/memory/actions/workflows/go.yml)
[![Eval Smoke](https://github.com/jeffs-brain/memory/actions/workflows/eval-smoke.yml/badge.svg)](https://github.com/jeffs-brain/memory/actions/workflows/eval-smoke.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Cross-language memory library for LLM agents. Three SDKs (TypeScript, Go, Python) that all implement the same [spec](./spec) so a brain created by one SDK reads and writes cleanly through another.

Local-first, hosted-optional. Apache-2.0.

## SDK matrix

| Language   | Package                        | Version | Install                                |
| ---------- | ------------------------------ | ------- | -------------------------------------- |
| TypeScript | `@jeffs-brain/memory`          | 0.1.0   | `npm i -g @jeffs-brain/memory`         |
| TypeScript | `@jeffs-brain/memory-postgres` | 0.1.0   | `npm i @jeffs-brain/memory-postgres`   |
| TypeScript | `@jeffs-brain/memory-openfga`  | 0.1.0   | `npm i @jeffs-brain/memory-openfga`    |
| TypeScript | `@jeffs-brain/memory-mcp`      | 0.1.0   | `npx -y @jeffs-brain/memory-mcp`       |
| TypeScript | `@jeffs-brain/install`         | 0.1.0   | `npx @jeffs-brain/install`             |
| Go         | `github.com/jeffs-brain/memory/go` | pre-release | `go install github.com/jeffs-brain/memory/go/cmd/memory@latest` |
| Go         | `cmd/memory-mcp`               | pre-release | `go install github.com/jeffs-brain/memory/go/cmd/memory-mcp@latest` |
| Python     | `jeffs-brain-memory`           | 0.0.1 (pre-publish) | `pip install jeffs-brain-memory` or `uv add jeffs-brain-memory` |
| Python     | `jeffs-brain-memory-mcp`       | 0.1.0   | `uvx jeffs-brain-memory-mcp`           |

All three SDKs implement the full spec wire surface. The shared HTTP conformance suite sits at 28/29 green across SDKs. Cross-SDK evaluation parity today is the shared daemon surface in exactly three scenarios: `ask-basic`, `ask-augmented`, and `search-retrieve-only`.

### Parity caveats

- The Postgres + pgvector adapter (`@jeffs-brain/memory-postgres`) is TypeScript-only today. Go and Python use SQLite-backed stores.
- Python has no native LongMemEval runner. Full LME replay, judge wiring, and agentic loops live in the Go and TypeScript SDKs; Python is exercised through the shared daemon scenarios only.

## Evaluation parity

The shared runner covers those daemon scenarios only. It forwards retrieval mode unchanged and defaults to `auto`, so each SDK daemon resolves `auto` to `hybrid` when embeddings are available or `bm25` otherwise. Full LongMemEval replay, replay ingest, and agentic loops stay in the native SDK runners; Go provides the reference `memory eval lme run --ingest-mode=replay` path. See [`eval/README.md`](./eval/README.md) for the shared verification flow and artefact layout.

### Shared daemon scenarios

These are the only cross-SDK daemon scenarios the shared harness verifies:

| Scenario | Request shape | Main check |
| -------- | ------------- | ---------- |
| `ask-basic` | `POST /ask` with `question`, `topK`, `mode` | Standard `/ask` path: retrieval, basic reader prompt, and SSE `retrieve`, `answer_delta`, `citation`, and `done` events. |
| `ask-augmented` | `POST /ask` with `question`, `topK`, `mode`, `readerMode=augmented`, optional `questionDate` | Augmented `/ask` path: temporal anchor forwarding plus the same SSE contract. |
| `search-retrieve-only` | `POST /search` with `query`, `topK`, `mode`, optional `questionDate`, `candidateK`, and `rerankTopN` | Retrieval path only: chunk selection, optional rerank knobs, and returned JSON chunk payloads. |

Parity expectation is explicit:

- The same scenario request body is accepted by the TypeScript, Go, and Python daemons.
- `ask-basic` and `ask-augmented` expose the same SSE contract: `retrieve`, `answer_delta*`, `citation*`, `done`.
- `search-retrieve-only` exposes the same JSON retrieval contract and is scored from returned retrieval content only.
- `questionDate` is forwarded for `ask-augmented` and `search-retrieve-only` when the dataset row provides it.
- `candidateK` and `rerankTopN` are forwarded on `search-retrieve-only` when the runner flags are non-zero.
- `mode` is forwarded unchanged by the runner; each daemon resolves `auto` locally.
- The native tri-SDK LongMemEval replay run pins `search-retrieve-only` to replay memory only via actor-side retrieval filters: `--actor-scope memory --actor-project <brain-id>`. That keeps global memory plus the eval brain's project memory in scope while excluding raw transcript rows.
- Model wording can differ. Parity is about protocol, retrieval, and prompt-path behaviour, not byte-identical answers.

### What we test

- `ask-basic`: the basic `/ask` request and SSE response path, including `retrieve`, `answer_delta`, `citation`, and `done`.
- `ask-augmented`: the augmented `/ask` prompt path, including `readerMode=augmented`, temporal anchor forwarding, and the same SSE contract.
- `search-retrieve-only`: the `/search` JSON contract only. The scorer consumes the returned retrieval content, not a daemon-generated answer.

### How we run and compare it

- SDK-local tests pin the scenario contract inside each implementation.
- The shared runner in [`eval/runner.py`](./eval/runner.py) starts one SDK daemon at a time, runs one scenario per invocation, and writes `<output>/<YYYY-MM-DD>/<sdk>.json`.
- The replay-backed tri-SDK workflow in [`eval/scripts/run_tri_lme.sh`](./eval/scripts/run_tri_lme.sh) extracts once with Go, starts all three daemons against the same brain, and benchmarks `search-retrieve-only` only. Retrieval happens inside each daemon via `/search`; the shared augmented reader and judge stay in the Go runner process.

Use a separate `--output` root per scenario when comparing SDKs on the same day. The runner writes one `<sdk>.json` per output root and date, so reusing the same output root for a second scenario overwrites the earlier JSON for that SDK.

## Verification workflow

1. Run the SDK-local regression tests that pin the scenario contract in each implementation.
2. Run the shared runner against `ts`, `go`, and `py` for the same scenario.
3. Inspect the generated run artefacts under `eval/results/` for transport, citations, and answer or retrieval-blob shape.

To compare one scenario across all three SDKs, run the shared runner from [`eval/`](./eval) once per SDK:

```bash
cd eval
uv sync

# Fast parity check for the basic ask path
for sdk in ts go py; do
  uv run python runner.py \
    --sdk "$sdk" \
    --dataset datasets/smoke.jsonl \
    --scorer exact \
    --scenario ask-basic \
    --output results/ask-basic
done

# Corpus-grounded parity checks, using a populated `eval` brain
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

Compare `eval/results/<scenario>/<YYYY-MM-DD>/ts.json`, `go.json`, and `py.json` for the shared runner, or `eval/results/tri-lme-<timestamp>/result-*.json` for replay-backed tri-SDK retrieval parity.

Per-SDK regression commands for those scenarios live in the SDK READMEs below.

## Quick start

### TypeScript

```bash
npm i -g @jeffs-brain/memory
memory init
memory ingest ./docs
memory search "question"
```

Full walkthrough: [docs.jeffsbrain.com/getting-started/typescript/](https://docs.jeffsbrain.com/getting-started/typescript/).

### Go

```bash
go install github.com/jeffs-brain/memory/go/cmd/memory@latest
memory init
memory ingest ./docs
memory search "question"
```

Full walkthrough: [docs.jeffsbrain.com/getting-started/go/](https://docs.jeffsbrain.com/getting-started/go/).

### Python

```bash
pip install jeffs-brain-memory
# or: uv add jeffs-brain-memory
memory init
memory ingest ./docs
memory search "question"
```

Full walkthrough: [docs.jeffsbrain.com/getting-started/python/](https://docs.jeffsbrain.com/getting-started/python/).

## MCP quick start

One command wires `jeffs-brain` into Claude Code, Claude Desktop, Cursor, Windsurf, and Zed:

```bash
npx @jeffs-brain/install
```

Or register a single host manually against any SDK's MCP wrapper:

```bash
# TypeScript
claude mcp add jeffs-brain -- npx -y @jeffs-brain/memory-mcp

# Go
claude mcp add jeffs-brain -- memory-mcp

# Python
claude mcp add jeffs-brain -- uvx jeffs-brain-memory-mcp
```

Each wrapper exposes the same 11 `memory_*` tools defined in [`spec/MCP-TOOLS.md`](./spec/MCP-TOOLS.md).

## Algorithm surface

Every SDK implements the full pipeline:

- Hybrid retrieval: BM25 (SQLite FTS5 or Postgres tsvector) plus vector (sqlite-vec or pgvector) fused with Reciprocal Rank Fusion at `k=60`.
- Retry ladder: five-rung fallback from quoted to bare-word to single-strongest-term to trigram fuzzy to refresh-and-retry.
- Intent reweight: detected query intent shifts RRF weights between lexical and semantic.
- Cross-encoder rerank: LLM-backed or HTTP-backed reranker over the top candidates (Go ships both modes; TS ships the LLM path).
- Query distill: optional LLM rewrite of noisy input to clean retrieval query. Shipping in Go and TS (opt-in).
- Memory stages: extract, reflect, consolidate, recall with global and project-scoped buffers.
- Knowledge base: markdown, URL, file, and PDF ingest with frontmatter, wikilinks, and compile passes.
- Cross-SDK eval parity: `memory serve` is exercised in `ask-basic`, `ask-augmented`, and `search-retrieve-only`.
- Native LongMemEval runners: full replay ingest, judge wiring, and agentic loops are outside the cross-SDK harness.
- Authorisation: every SDK ships a `Provider` contract, an in-process RBAC adapter (workspace -> brain -> collection -> document hierarchy with `admin`/`writer`/`reader` roles and `deny:<role>` overrides), a `Store` wrapper that runs a check on every read/write/delete, and a sibling OpenFGA HTTP adapter. The shared FGA model lives at [`spec/openfga/`](./spec/openfga).

## Repo layout

```
spec/
  openfga/      Canonical OpenFGA authorisation model shared by every SDK adapter
  ...           Language-neutral protocol, storage, algorithms, query DSL, MCP tool contract
sdks/
  ts/
    memory/              Core TS SDK: store, retrieval, ingestion, query DSL, CLI, acl primitives
    memory-postgres/     Postgres + pgvector adapter
    memory-openfga/      OpenFGA authorisation adapter
  go/
    acl/                 Authorisation contract, in-process RBAC, Store wrapper
    aclopenfga/          OpenFGA HTTP adapter
    ...           Full Go SDK: store/fs|git|mem|http, search, query, retrieval, memory, knowledge, eval/lme, cmd/memory, cmd/memory-mcp
  py/
    src/jeffs_brain_memory/
      acl/             Authorisation contract, in-process RBAC, Store wrapper
      acl_openfga/     OpenFGA HTTP adapter
      ...              Full Python SDK: store, search, query, retrieval, rerank, memory, knowledge, ingest, llm, http daemon
mcp/
  ts/           @jeffs-brain/memory-mcp (npm)
  py/           jeffs-brain-memory-mcp (PyPI)
install/        @jeffs-brain/install, one-shot multi-agent installer
examples/       ts/hello-world and go/hello-world ready-to-run
docs/           Astro Starlight documentation site, Cloudflare Pages ready
eval/           Cross-SDK conformance and evaluation runner (--sdk ts|go|py)
```

The Go MCP wrapper ships from `sdks/go/cmd/memory-mcp/`. The `mcp/go/` folder is reserved for future extraction.

## Documentation

Full docs site: [docs.jeffsbrain.com](https://docs.jeffsbrain.com).

Topic guides:

- [Stores](https://docs.jeffsbrain.com/guides/stores/) - filesystem, git, in-memory, HTTP, and Postgres backends
- [Knowledge](https://docs.jeffsbrain.com/guides/knowledge/) - markdown, URL, file, and PDF ingest with frontmatter and wikilinks
- [Retrieval](https://docs.jeffsbrain.com/guides/retrieval/) - hybrid BM25 plus vector, RRF, retry ladder, intent reweight, rerank
- [Memory lifecycle](https://docs.jeffsbrain.com/guides/memory-lifecycle/) - extract, reflect, consolidate, recall
- [Authorisation](https://docs.jeffsbrain.com/guides/authorization/) - in-process RBAC and OpenFGA adapter
- [Configuration reference](https://docs.jeffsbrain.com/reference/configuration/) - `memory.config.json`, env vars, CLI flags

## Links

- [`spec/`](./spec) - protocol, storage, algorithms, MCP tools, query DSL, conformance harness
- [`docs/`](./docs) - public documentation site (Astro Starlight, Cloudflare Pages)
- [`examples/`](./examples) - runnable hello-world per language
- [`eval/`](./eval) - cross-SDK evaluation runner, datasets, and verification workflow
- [`install/`](./install) - `@jeffs-brain/install` orchestrator
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) - dev setup, commit style, DCO, PR process
- [`SECURITY.md`](./SECURITY.md) - vulnerability reporting
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) - community standards
- [`NOTICE`](./NOTICE) - bundled third-party components

## Licence

Apache-2.0. See [`LICENSE`](./LICENSE).
