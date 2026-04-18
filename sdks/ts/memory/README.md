# @jeffs-brain/memory

A local-first, pluggable memory and retrieval library for AI agents. Ships a Store abstraction over filesystem/Git/Postgres, hybrid BM25 + vector search, an extract/recall/reflect/consolidate memory pipeline, RBAC plus an OpenFGA adapter, a rerank layer, and a slim `jbmem` CLI. Runs entirely offline using an Ollama provider and the built-in hash embedder, or against OpenAI/Anthropic/TEI when you want quality.

## Install

```bash
bun add @jeffs-brain/memory
```

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

If you want one reusable orchestration surface for pre-turn, post-turn, and session-end work:

```ts
import { createMemoryLifecycle } from '@jeffs-brain/memory'

const lifecycle = createMemoryLifecycle({ memory: mem })
const promptContext = await lifecycle.beforeTurn({ message: 'How should we handle auth?' })
const extracted = await lifecycle.afterTurn({ messages, sessionId: 'session-1' })
const ended = await lifecycle.endSession({ messages, sessionId: 'session-1', consolidate: true })
```

## CLI quickstart

```bash
jbmem init ./brain
jbmem ingest ./brain --path notes/meeting.md --content "we picked Postgres"
jbmem search ./brain --query "which database did we pick?"
```

## MCP server

To expose a brain to Claude, Cursor, or any MCP-aware client, see [`docs/getting-started-mcp.md`](../../docs/getting-started-mcp.md).

## Architecture

The full porting and design notes live in [`PORTING-SPEC.md`](../../PORTING-SPEC.md) at the repo root.

## License

Apache-2.0. See the [`LICENSE`](./LICENSE) file for the full text.
