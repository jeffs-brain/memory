# jeffs-brain-memory

Python SDK for Jeffs Brain — the cross-language memory library for LLM agents.

Part of the polyglot [`jeffs-brain/memory`](https://github.com/jeffs-brain/memory) monorepo. This SDK tracks the same [spec](../../spec/) and conformance fixtures as the TypeScript and Go SDKs.

## Status

Phase 4 scaffold. Module layout, CLI, and HTTP daemon skeleton are in place. Real implementations land incrementally, in the dependency order laid out in the restructure plan: `path`, `store`, `query`, `search`, `retrieval`, `rerank`, `memory`, `knowledge`, `ingest`, `eval`.

Until then, CLI subcommands print a stub message and HTTP endpoints return `501 Not Implemented` via Problem+JSON. `GET /healthz` returns `200`.

## Install

```bash
pip install jeffs-brain-memory
# or
uv add jeffs-brain-memory
```

## Usage

```bash
memory init
memory ingest ./docs
memory search "question"
memory serve --host 127.0.0.1 --port 8080
```

Environment variables:

- `JB_HOME` — storage root (default `~/.jeffs-brain/`)
- `JB_TOKEN` — hosted bearer token; when set the SDK drives the platform backend via `HttpStore`
- `JB_ENDPOINT` — hosted endpoint URL when `JB_TOKEN` is set

## Development

```bash
uv sync
uv run memory --version
uv run pytest -q
```

## Spec

Protocol, query DSL, algorithms, storage: see [`../../spec/`](../../spec/).

## Licence

Apache-2.0. See [LICENSE](./LICENSE).
