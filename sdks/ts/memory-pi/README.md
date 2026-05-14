# @jeffs-brain/memory-pi

A pi extension that wires `@jeffs-brain/memory` into the pi agent harness.

- Eleven LLM-callable `memory_*` tools: remember, recall, search, ask,
  ingest_file, ingest_url, extract, reflect, consolidate, create_brain,
  list_brains.
- Four lifecycle hooks: `before_agent_start` (recall injection, cache-friendly),
  `context` (re-inject on diff), `turn_end` (extract), `session_shutdown`
  (reflect + close).
- Auto-detects local Ollama on `localhost:11434` for `bge-m3` embeddings and
  `gemma3` chat.
- `autodetectStore` picks `fs` vs `git` based on the brain root.
- Optional MCP fallback for portability via a config flag.

## Status

Scaffolded. W3 in the pi-Jeff migration. See `~/code/me/jeff-on-pi/PLAN.md`
section 5 for the design.

## Quick start

```typescript
import { createMemoryExtension } from "@jeffs-brain/memory-pi"
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"

export default function (pi: ExtensionAPI) {
  createMemoryExtension(pi, {
    store: { kind: "auto" },
    embedder: { kind: "auto" },
    provider: { kind: "auto" },
    recall: { onPrompt: true, scope: "global", topK: 5 },
    extract: { onTurnEnd: true, contextualise: true },
  })
}
```
