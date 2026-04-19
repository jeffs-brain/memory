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

All three SDKs implement the full spec wire surface. The shared HTTP conformance suite sits at 28/29 green across SDKs and the tri-SDK smoke benchmark scores identically on every SDK.

## Tri-SDK benchmark

20-question smoke dataset, `exact` scorer, local Ollama `gemma3:latest` actor. Captured 2026-04-18 from `eval/results/cross-sdk/`.

| SDK        | Pass  | Rate | p50 latency | p95 latency |
| ---------- | ----- | ---- | ----------- | ----------- |
| TypeScript | 19/20 | 95%  | 446 ms      | 836 ms      |
| Go         | 19/20 | 95%  | 471 ms      | 717 ms      |
| Python     | 19/20 | 95%  | 407 ms      | 630 ms      |

All three SDKs pass exactly the same 19 questions. The single failure is a shared gemma3 knowledge miss, not a daemon regression. See `eval/results/cross-sdk/cross-sdk-smoke-tri-fix-2026-04-18.md` for the full write-up.

Full LongMemEval replay (93.4% parity target) runs through `memory eval lme run --ingest-mode=replay` on Go; see `eval/README.md` for the cross-SDK runner.

## Quick start

### TypeScript

```bash
npm i -g @jeffs-brain/memory
memory init
memory ingest ./docs
memory search "question"
```

### Go

```bash
go install github.com/jeffs-brain/memory/go/cmd/memory@latest
memory init
memory ingest ./docs
memory search "question"
```

### Python

```bash
pip install jeffs-brain-memory
# or: uv add jeffs-brain-memory
memory init
memory ingest ./docs
memory search "question"
```

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
- LongMemEval runner: Go ships the full replay + agentic modes; cross-SDK eval harness drives all three through the same HTTP ask contract.
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

## Links

- [`spec/`](./spec) - protocol, storage, algorithms, MCP tools, query DSL, conformance harness
- [`docs/`](./docs) - public documentation site (Astro Starlight, Cloudflare Pages)
- [`examples/`](./examples) - runnable hello-world per language
- [`eval/`](./eval) - cross-SDK eval runner and benchmark results
- [`install/`](./install) - `@jeffs-brain/install` orchestrator
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) - dev setup, commit style, DCO, PR process
- [`SECURITY.md`](./SECURITY.md) - vulnerability reporting
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) - community standards
- [`LICENSE`](./LICENSE) - Apache-2.0
- [`NOTICE`](./NOTICE) - bundled third-party components

Docs site: https://docs.jeffsbrain.com
