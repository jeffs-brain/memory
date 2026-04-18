# jeffs-brain/memory

Open source memory library for AI agents. Local-first, hosted-optional. Apache-2.0.

Polyglot by design: TypeScript today, Go and Python coming. The behaviour contract lives in [`spec/`](./spec) so every SDK and MCP wrapper stays interoperable.

## Layout

```
spec/         Language-neutral protocol, storage, algorithms and MCP tool contract
sdks/
  ts/         Published TypeScript SDK (memory + adapters)
  go/         Planned Go SDK
  py/         Planned Python SDK
mcp/          MCP wrappers per language
examples/     Sample apps per language
docs/         Public documentation site source
install/      Bootstrapper scripts
eval/         Cross-language conformance and evaluation harnesses
```

## TypeScript packages

- `@jeffs-brain/memory` — core store, retrieval, ingestion, query DSL, CLI
- `@jeffs-brain/memory-postgres` — Postgres + pgvector adapter
- `@jeffs-brain/memory-openfga` — OpenFGA authorisation adapter

## Quick start

```bash
bun install
bun run typecheck
bun run test
```

The headline install flow once the MCP wrapper ships will be:

```bash
npx @jeffs-brain/memory-mcp
```

Docs: https://docs.jeffsbrain.com (placeholder, coming soon).

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). Issues and PRs welcome.
