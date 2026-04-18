# @jeffs-brain/memory

Local-first, pluggable memory and retrieval library for LLM agents. Ships a `Store` abstraction over filesystem, Git, in-memory and HTTP backends, hybrid BM25 plus pure-JS vector search, an extract, recall, reflect and consolidate memory pipeline, RBAC plus an OpenFGA adapter, a rerank layer, and a slim `memory` CLI. Runs entirely offline using an Ollama provider and the built-in hash embedder, or against OpenAI, Anthropic, or TEI when you want quality.

## Install

```bash
npm i @jeffs-brain/memory
# or
bun add @jeffs-brain/memory
```

The published `memory` binary runs on Node 20+ (its shebang is `#!/usr/bin/env node`). Bun is the preferred local development runtime for this package, but it is not required at install or runtime for end users.

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

For a single orchestration surface covering pre-turn, post-turn and session-end work:

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
```

## MCP server

To expose a brain to Claude, Cursor, Windsurf or any MCP-aware client, install [`@jeffs-brain/memory-mcp`](https://www.npmjs.com/package/@jeffs-brain/memory-mcp) and see the repo docs.

## Docs

- Repo README and links: https://github.com/jeffs-brain/memory#readme
- Protocol, storage, algorithms, query DSL, MCP tool contract: [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec)
- Companion packages: `@jeffs-brain/memory-postgres`, `@jeffs-brain/memory-openfga`, `@jeffs-brain/memory-mcp`

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
