# jeffs-brain/memory

[![TS CI](https://github.com/jeffs-brain/memory/actions/workflows/ci.yml/badge.svg)](https://github.com/jeffs-brain/memory/actions/workflows/ci.yml)
[![Go CI](https://github.com/jeffs-brain/memory/actions/workflows/go.yml/badge.svg)](https://github.com/jeffs-brain/memory/actions/workflows/go.yml)
[![Eval Smoke](https://github.com/jeffs-brain/memory/actions/workflows/eval-smoke.yml/badge.svg)](https://github.com/jeffs-brain/memory/actions/workflows/eval-smoke.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Cross-language memory library for LLM agents, driven by a shared HTTP and storage specification so every SDK behaves the same.

Local-first, hosted-optional. Apache-2.0.

## Status

| Language   | Package                      | Status                           | Install                                   |
| ---------- | ---------------------------- | -------------------------------- | ----------------------------------------- |
| TypeScript | `@jeffs-brain/memory`        | v0.0.1 (pre-publish)             | `npm i -g @jeffs-brain/memory`            |
| TypeScript | `@jeffs-brain/memory-mcp`    | v0.0.1 (pre-publish)             | `npx -y @jeffs-brain/memory-mcp`          |
| TypeScript | `@jeffs-brain/memory-postgres` | v0.0.1 (pre-publish)           | `npm i @jeffs-brain/memory-postgres`      |
| TypeScript | `@jeffs-brain/memory-openfga` | v0.0.1 (pre-publish)            | `npm i @jeffs-brain/memory-openfga`       |
| Go         | `github.com/jeffs-brain/memory/sdks/go` | Coming, Phase 3       | placeholder                               |
| Python     | `jeffs_brain_memory`         | Coming, Phase 4                  | placeholder                               |

The behaviour contract lives in [`spec/`](./spec) so every SDK and MCP wrapper stays interoperable.

## Quick start (TypeScript)

```bash
npm i -g @jeffs-brain/memory
memory init
memory ingest ./docs
memory search "question"
```

## MCP quick start

Register the MCP server with Claude Code:

```bash
claude mcp add jeffs-brain -- npx -y @jeffs-brain/memory-mcp
```

Any MCP-capable client that can spawn a stdio server works the same way.

## Repo layout

```
spec/           Language-neutral protocol, storage, algorithms, query DSL, MCP tool contract (substantive)
sdks/
  ts/
    memory/              Core TS SDK: store, retrieval, ingestion, query DSL, CLI (substantive)
    memory-postgres/     Postgres + pgvector adapter (substantive)
    memory-openfga/      OpenFGA authorisation adapter (substantive)
  go/           Planned, Phase 3 (placeholder)
  py/           Planned, Phase 4 (placeholder)
mcp/
  ts/           TS MCP wrapper, @jeffs-brain/memory-mcp (substantive)
  go/           Planned (placeholder)
  py/           Planned (placeholder)
examples/       Sample apps per language (ts substantive, go and py placeholder)
docs/           Public documentation site source
install/        Bootstrap scripts (placeholder)
eval/           Cross-language conformance and evaluation harnesses
```

## Links

- [`spec/`](./spec) - protocol, storage, algorithms, MCP tools, query DSL
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) - dev setup, commit style, DCO, PR process
- [`SECURITY.md`](./SECURITY.md) - vulnerability reporting
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) - community standards
- [`LICENSE`](./LICENSE) - Apache-2.0
- [`NOTICE`](./NOTICE) - bundled third-party components

Docs site: https://docs.jeffsbrain.com (placeholder, coming soon).
