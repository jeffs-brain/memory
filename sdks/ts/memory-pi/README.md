# @jeffs-brain/memory-pi

A pi extension that wires the [`@jeffs-brain/memory`](../memory) pipeline
into the pi coding agent so every session gains an active long-term
memory layer.

The extension exposes:

- **Eleven `memory_*` tools** the LLM can call directly: `remember`,
  `recall`, `search`, `ask`, `ingest_file`, `ingest_url`, `extract`,
  `reflect`, `consolidate`, `create_brain`, `list_brains`.
- **Four pi lifecycle hooks**: `before_agent_start` (recall injection
  with optional cache-friendly diffing), `context` (per-turn recall
  injection below the prompt-cache boundary), `turn_end` (non-blocking
  extract queue), `session_shutdown` (queue drain, optional reflect /
  consolidate, store close).
- **Auto-detection** for local Ollama (`bge-m3` embeddings, `gemma3`
  chat) and explicit `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` providers,
  with `autodetectStore` picking `fs` or `git` based on the brain root.

## Install

```bash
npm install @jeffs-brain/memory-pi
```

The extension assumes the parent `@jeffs-brain/memory` SDK is also
available; both packages ship from the same monorepo.

## Quick start (standalone pi user)

Drop `memory-pi` into the pi extensions directory:

```typescript
// ~/.pi/agent/extensions/memory-pi.ts
import { createMemoryExtension } from '@jeffs-brain/memory-pi'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'

export default async function (pi: ExtensionAPI) {
  const ext = createMemoryExtension(pi, {
    brainRoot: process.env.MEMORY_PI_BRAIN_ROOT,
    brainId: 'default',
    store: { kind: 'auto' },
    embedder: { kind: 'auto' },
    provider: { kind: 'auto' },
    recall: {
      onPrompt: true,
      cacheFriendly: true,
      topK: 5,
      scope: 'global',
    },
    extract: { onTurnEnd: true, minMessages: 6, contextualise: true },
    reflect: { onSessionEnd: true },
    consolidate: { schedule: 'manual' },
  })
  await ext.ready
}
```

`MEMORY_PI_CONFIG` (JSON) is honoured by the default export so
`pi -e ./node_modules/@jeffs-brain/memory-pi/dist/index.js` works without
recompiling the extension.

## Quick start (Jeff / Jill consumer)

The Jeff and Jill runtimes link the same package via the W1 / W2 work
items. They construct the extension explicitly so the runtime can hold
a handle:

```typescript
import { createMemoryExtension, type MemoryExtension } from '@jeffs-brain/memory-pi'

const memory: MemoryExtension = createMemoryExtension(pi, {
  brainRoot: '/home/jeff/.local/share/jeff/brain',
  brainId: 'jeff',
  store: { kind: 'git', remote: 'git@github.com:lleverage-ai/jeffs-brain.git' },
  embedder: { kind: 'ollama', baseUrl: 'http://localhost:11434', model: 'bge-m3' },
  provider: { kind: 'anthropic', apiKey: process.env.ANTHROPIC_API_KEY!, model: 'claude-opus-4-6' },
  acl: { actorId: 'jeff' },
})

await memory.ready

process.on('beforeExit', () => memory.close())
```

## Quick start (MCP fallback)

When pi is not available but you still want the same tool surface, run
the parent `@jeffs-brain/memory` MCP server: it exposes the identical
eleven tools over MCP. The pi extension and the MCP server share the
on-disk brain layout so both can read the same data.

## Configuration

All keys are optional; sensible defaults are auto-detected.

```typescript
type MemoryExtensionConfig = {
  brainRoot?: string  // default: ~/.config/memory-pi/brains
  brainId?: string    // default: 'default'

  store?: { kind: 'auto' | 'fs' | 'git' | 'http'; remote?: string; endpoint?: string; token?: string }
  embedder?: { kind: 'auto' | 'ollama' | 'openai' | 'tei' | 'off'; baseUrl?: string; model?: string; apiKey?: string; endpoint?: string }
  provider?: { kind: 'auto' | 'openai' | 'anthropic' | 'ollama'; apiKey?: string; baseUrl?: string; model?: string }
  reranker?: { kind: 'auto' | 'llm' | 'tei' | 'off'; endpoint?: string }

  recall?: {
    onPrompt?: boolean        // default: true
    topK?: number             // default: 5
    minScore?: number         // default: 0
    scope?: 'global' | 'project' | 'agent'  // default: 'global'
    fallbackScopes?: ('global' | 'project' | 'agent')[]
    cacheFriendly?: boolean   // default: true
  }

  extract?: {
    onTurnEnd?: boolean       // default: true
    minMessages?: number      // default: 6
    contextualise?: boolean
  }

  reflect?: { onSessionEnd?: boolean }
  consolidate?: { schedule?: 'manual' | 'session' | '@daily' }

  tools?: {
    expose?: ('remember' | 'recall' | 'search' | 'ask' | 'ingest_file' | 'ingest_url' | 'extract' | 'reflect' | 'consolidate' | 'create_brain' | 'list_brains')[]
  }

  acl?: {
    actorId?: string
    provider?: 'rbac' | { kind: 'openfga'; endpoint: string }
  }
}
```

## Cache-friendly recall injection

When `recall.cacheFriendly: true` (default), the extension only rewrites
the system prompt when the recall set changes. Identical recall hits
across turns leave the prompt alone, so Anthropic / OpenAI prompt
caching stays warm. Flip `recall.onPrompt: false` to move injection to
the per-turn `context` boundary instead (below the cache).

## Single-brain hosts (flatLayout)

Some hosts manage exactly one brain per identity at a well-known path
with content (`wiki/`, `memory/`, `raw/`, ...) sitting directly under the
brain root. The default multi-brain layout, which nests
`<root>/<brainId>/<content>`, does not fit that shape.

Pass `flatLayout: true` to tell `createMemoryExtension` that `brainRoot`
already IS the brain. Combined with `searchIndexPath`, this also lets
hosts that keep the brain in a git working tree redirect the FTS sqlite
to a machine-local state directory so it never enters the tree.

```typescript
const ext = createMemoryExtension(pi, {
  brainRoot: '/var/lib/myagent/brain',          // single-brain root
  brainId: 'myagent',                            // logical label only
  flatLayout: true,
  searchIndexPath: '/var/state/myagent/search.sqlite',
})
```

When `flatLayout` is on, the runtime also runs a one-shot indexer on
first boot: it walks the configured `bootstrapScanDirs` (default
`['wiki', 'memory', 'raw']`), chunks every markdown file, and upserts
the chunks directly into the FTS index via `SearchIndex.upsertChunks`.
The Store is bypassed entirely so the source `.md` files are never
re-written. Re-entries are no-ops once `knowledge_chunks` is populated.

Environment overrides:

| Var | Effect |
|---|---|
| `MEMORY_PI_FLAT_LAYOUT=true` | Enable flat layout without touching the config object. |
| `MEMORY_PI_SEARCH_INDEX_PATH=/path/...` | Override the FTS sqlite path. |
| `MEMORY_PI_BRAIN_ROOT=/path/...` | Override `brainRoot`. |
| `MEMORY_PI_BRAIN_ID=...` | Override `brainId`. |

## Tools

| Tool name              | Operation                                          |
|------------------------|----------------------------------------------------|
| `memory_remember`      | Persist a markdown note (frontmatter included).    |
| `memory_recall`        | Five-stage recall over the brain.                  |
| `memory_search`        | Hybrid BM25 + vector retrieval.                    |
| `memory_ask`           | Grounded answer with citations.                    |
| `memory_ingest_file`   | Ingest a local file (<= 25 MiB).                   |
| `memory_ingest_url`    | Fetch + ingest a URL (<= 5 MiB fallback).          |
| `memory_extract`       | Extract memorable facts from a transcript.         |
| `memory_reflect`       | Run reflection over a session.                     |
| `memory_consolidate`   | Compile summaries, prune episodic memory.          |
| `memory_create_brain`  | Create a new brain under the configured root.      |
| `memory_list_brains`   | List the brains the host can see.                  |

## Status

Active. W3 of the pi-Jeff migration. The runtime is feature-complete and
covered by smoke tests; subsequent work items wire Jeff and Jill on top
of it.

## Licence

Apache-2.0.
