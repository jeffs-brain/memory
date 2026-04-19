# @jeffs-brain/memory

Local-first, pluggable memory and retrieval library for LLM agents. Ships a `Store` abstraction over filesystem, Git, in-memory, and HTTP backends, hybrid BM25 plus pure-JS vector search, an extract, recall, reflect, and consolidate memory pipeline, RBAC plus an OpenFGA adapter, cross-encoder rerank, opt-in LLM query distillation, and a slim `memory` CLI that speaks the shared HTTP protocol. Runs entirely offline using an Ollama provider and the built-in hash embedder, or against OpenAI, Anthropic, or TEI when you want quality.

Part of the polyglot [`jeffs-brain/memory`](https://github.com/jeffs-brain/memory) repo. This SDK tracks the same [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec) and conformance fixtures as the Go and Python SDKs.

## Install

```bash
npm i @jeffs-brain/memory
# or
bun add @jeffs-brain/memory
```

The published `memory` binary runs on Node 20+ (its shebang is `#!/usr/bin/env node`). Bun is the preferred local development runtime for this package, but it is not required at install or runtime for end users.

## Feature support

- Stores: `FsStore`, `MemStore`, `GitStore`, `HttpStore` (spec/PROTOCOL.md wire client).
- Search: SQLite FTS5 BM25, pure-JS vector search, Reciprocal Rank Fusion (`k=60`).
- Query DSL: tokenisation, stopword filtering (en and nl), alias expansion, FTS5 compilation.
- Retrieval: hybrid BM25 + vector, five-rung retry ladder, intent reweight, cross-encoder rerank, opt-in query distill.
- Memory stages: extract, reflect, consolidate, recall, session buffers, episode recorder.
- Knowledge: markdown chunker, URL/file/PDF ingest, wikilinks, compile passes.
- Authorisation: pluggable `AccessControlProvider` contract (`@jeffs-brain/memory/acl`), in-process RBAC (workspace -> brain -> collection -> document hierarchy, `admin`/`writer`/`reader` roles, `deny:<role>` overrides), `withAccessControl(store, provider, subject, ...)` Store wrapper, optional `close()` lifecycle hook. Pair with [`@jeffs-brain/memory-openfga`](https://www.npmjs.com/package/@jeffs-brain/memory-openfga) for production tuple-store backed checks.
- Conformance: 28/29 cases green against `spec/conformance/http-contract.json`.
- CLI: `memory init|ingest|search|ask|serve|remember|recall|reflect|consolidate|create-brain|list-brains|eval`.

## Embedded usage

```ts
import { createMemStore, createMemory, createHashEmbedder } from '@jeffs-brain/memory'

const store = createMemStore()
const embedder = createHashEmbedder()
const mem = createMemory({ store, provider, embedder, cursorStore, scope: 'project', actorId: 'me' })

await mem.extract({ messages })
const hits = await mem.recall({ query: 'what did we decide about auth?' })
console.log(hits)
```

Swap `createHashEmbedder()` for `OllamaEmbedder` or `TEIEmbedder` when you need real retrieval quality. The hash embedder is deterministic, zero-network, and intended for dev and CI only.

For a single orchestration surface covering pre-turn, post-turn, and session-end work:

```ts
import { createMemoryLifecycle } from '@jeffs-brain/memory'

const lifecycle = createMemoryLifecycle({ memory: mem })
const promptContext = await lifecycle.beforeTurn({ message: 'How should we handle auth?' })
const extracted = await lifecycle.afterTurn({ messages, sessionId: 'session-1' })
const ended = await lifecycle.endSession({ messages, sessionId: 'session-1', consolidate: true })
```

## CLI quickstart

```bash
memory init ./brain
memory ingest ./brain --path notes/meeting.md --content "we picked Postgres"
memory search ./brain --query "which database did we pick?"
memory serve --addr 127.0.0.1:18844
```

`memory serve` speaks the wire protocol documented at [`spec/PROTOCOL.md`](https://github.com/jeffs-brain/memory/blob/main/spec/PROTOCOL.md) so any language SDK or the cross-SDK eval runner can drive it.

## MCP server

To expose a brain to Claude Code, Claude Desktop, Cursor, Windsurf, or Zed, install [`@jeffs-brain/memory-mcp`](https://www.npmjs.com/package/@jeffs-brain/memory-mcp) (stdio server, 11 canonical tools). The [`@jeffs-brain/install`](https://www.npmjs.com/package/@jeffs-brain/install) orchestrator wires every host in one command:

```bash
npx @jeffs-brain/install
```

## Examples and docs

- [`examples/ts/hello-world`](https://github.com/jeffs-brain/memory/tree/main/examples/ts/hello-world) - BM25 search over a markdown corpus.
- [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec) - protocol, storage, algorithms, query DSL, MCP tool contract.
- Docs site: https://docs.jeffsbrain.com

## Companion packages

- `@jeffs-brain/memory-postgres` - Postgres + pgvector adapter.
- `@jeffs-brain/memory-openfga` - OpenFGA authorisation adapter.
- `@jeffs-brain/memory-mcp` - Model Context Protocol stdio server.
- `@jeffs-brain/install` - multi-agent installer.

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
