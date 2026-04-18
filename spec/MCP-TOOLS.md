# MCP tool surface

Every Jeffs Brain SDK ships an MCP wrapper that exposes the same `memory_*` tool surface. The tool names, descriptions, and input schemas are canonical and cross-language. The reference implementation lives in `apps/mcp/src/tools/`.

All schemas are described below in Zod-like shorthand. Optional arguments are marked `?`. `brain` on every tool that takes it defaults to the `JBMCP_DEFAULT_BRAIN` environment variable (or the first accessible brain when that is unset) and is resolved server-side via `resolveBrain`.

Each tool returns an MCP response with:

- `content[0].text`: a short human-readable summary.
- `structuredContent`: the raw JSON payload for programmatic consumers.

Errors from any tool surface as a standard MCP error response via `toToolError`. The `no_brain` code is returned by every brain-scoped tool when brain resolution fails.

---

## `memory_search`

Search memory notes in a brain and return matching note content with citations.

**Description**: "Search memory notes in a brain and return matching note content with citations. `scope` selects the memory namespace, and `sort` controls whether relevance or recency wins."

**Input schema**

```
{
  query:  string (1..4096 chars)
  brain?: string
  top_k?: integer (1..100, default 10)
  scope?: 'all' | 'global' | 'project' | 'agent'
  sort?:  'relevance' | 'recency' | 'relevance_then_recency'
}
```

**Output shape**

```
{
  query:    string,
  brain_id: string,
  hits:     Array<{ score: number, path: string, content: string, ... }>,
  took_ms:  number
}
```

Text summary lists the top five hits formatted as `#N score=S path\n<first 320 chars>`.

---

## `memory_recall`

Recall memories for a query. With a `session_id`, the backend's session recall endpoint weights recent conversation context; without it, falls back to `memory_search`.

**Description**: "Recall memories for a query. Pass session_id to weight recent session context; otherwise uses the dedicated memory-search surface. `scope` selects the memory namespace rather than a generic metadata filter."

**Input schema**

```
{
  query:       string (1..4096 chars)
  brain?:      string
  scope?:      'global' | 'project' | 'agent'
  session_id?: string
  top_k?:      integer (1..50)
}
```

**Output shape** (session mode)

```
{ chunks: Array<{ ... }> }   # session recall payload
```

**Output shape** (fallback mode): same as `memory_search`.

---

## `memory_remember`

Store a new markdown document in the brain.

**Description**: "Store a new memory (markdown document) in the brain. Returns the created document id and path."

**Input schema**

```
{
  content: string (1..5_000_000 chars)
  title?:  string (1..512 chars)      # derived from first heading if omitted
  brain?:  string
  tags?:   Array<string (1..64 chars)> (max 64)
  path?:   string (1..1024 chars)
}
```

**Output shape**

```
{ id: string, path: string, byte_size: number, ... }  # full document record
```

Tags are forwarded as comma-joined `metadata.tags`.

---

## `memory_ingest_file`

Ingest a local file (<= 25 MB) into the brain via the file-ingest endpoint.

**Description**: "Ingest a local file (<= 25 MB) into the brain. Returns the ingest result."

**Input schema**

```
{
  path:   string                       # absolute or relative local path
  brain?: string
  as?:    'markdown' | 'text' | 'pdf' | 'json'
}
```

Content type is inferred from the extension when `as` is omitted. Files over 25 MiB are rejected with code `file_too_large`.

**Output shape**

```
{
  status:      'queued' | 'ingested',
  path:        string,
  job_id?:     string,
  chunk_count?: number,
  reused?:     boolean,
  ...
}
```

---

## `memory_ingest_url`

Fetch a URL and ingest its content. Uses the server-side `/ingest/url` when the SDK build exposes it; otherwise falls back to a local fetch + document create.

**Description**: "Fetch a URL and ingest its contents into the brain. Uses the server-side /ingest/url endpoint when available; otherwise fetches locally and creates a document."

**Input schema**

```
{
  url:    string (must be a valid URL)
  brain?: string
}
```

Local fallback caps the fetched body at 5 MiB and records `path: 'fallback'` in the structured content.

**Output shape**

```
# server path
{ path: 'server', result: IngestResponse }

# fallback path
{ path: 'fallback', document: DocumentRecord }
```

---

## `memory_extract`

Submit a transcript so the server can asynchronously derive memorable facts. When `session_id` is supplied, messages are appended to that session with `skip_extract` set on all but the final message; otherwise, a transcript document is created.

**Description**: "Submit a conversation transcript so the server can asynchronously extract memorable facts. If session_id is provided the messages are appended to that session; otherwise a transcript document is created."

**Input schema**

```
{
  messages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool',
    content: string (>= 1 char)
  }> (1..500 entries)
  brain?:      string
  actor_id?:   string
  session_id?: string
}
```

**Output shape**

```
# session mode
{ mode: 'session', messages: Array<SessionMessageRecord> }

# transcript mode
{ mode: 'transcript', document: DocumentRecord }
```

---

## `memory_reflect`

Close a session and trigger server-side reflection over its messages.

**Description**: "Close a session and trigger server-side reflection over its messages."

**Input schema**

```
{
  session_id: string (>= 1 char)
  brain?:     string
}
```

**Output shape**

```
{
  reflection_status: 'completed' | 'no_result' | ...,
  reflection?: {
    outcome: string,
    should_record_episode: boolean,
    path: string,
    retry_feedback?: string
  },
  reflection_attempted: boolean,
  ended_at?: string,
  ...
}
```

The text summary embeds outcome, record-episode flag, path, and ended-at timestamp for fast human scanning.

---

## `memory_consolidate`

Trigger a consolidation pass on the brain (compile summaries, promote stable notes, prune stale episodic memory).

**Description**: "Trigger a consolidation pass on the brain (compile summaries, promote stable notes, prune stale episodic memory)."

**Input schema**

```
{
  brain?: string
}
```

Routing: invokes `brains.consolidate(id)` when the SDK build exposes it; otherwise falls back to `brains.compile(id)`. Returns `not_implemented` when neither exists.

**Output shape**

```
{ result: unknown }
```

---

## `memory_create_brain`

Provision a new brain. Generates a slug from the name if one is not provided.

**Description**: "Create a new brain. Generates a slug from the name if one is not provided."

**Input schema**

```
{
  name:        string (1..128 chars)
  slug?:       string (1..64 chars, /^[a-z0-9][a-z0-9-]*$/)
  visibility?: 'private' | 'tenant' | 'public'       # default 'private'
}
```

**Output shape**

```
{ id: string, slug: string, name: string, visibility: string, ... }
```

---

## `memory_list_brains`

List all brains the caller has access to.

**Description**: "List all brains the caller has access to."

**Input schema**: `{}` (no arguments).

**Output shape**

```
{
  items: Array<{ id, slug, name, visibility, ... }>,
  ...
}
```

---

## `memory_ask`

Ask a question grounded in the brain. Streams answer tokens as MCP progress notifications and returns the final answer with citations.

**Description**: "Ask a question grounded in the brain. Streams answer tokens as MCP progress notifications and returns the final answer with citations."

**Input schema**

```
{
  query:  string (1..8192 chars)
  brain?: string
  top_k?: integer (1..50)
}
```

Streaming behaviour: each `answer_delta` SSE frame triggers a `notifications/progress` message when the caller supplied a `progressToken`. The final `done` event resolves the call with the full answer text.

**Output shape**

```
{
  answer:    string,
  citations: Array<AskCitationEvent>,
  retrieved: Array<RetrievedChunk>
}
```

Error events from the stream surface as `{ code, message, retryable }` in the text response.

## TODOs / ambiguities

- **TODO**: Nail down the exact shape of `IngestResponse`, `DocumentRecord`, `SessionMessageRecord`, `AskCitationEvent`, and `RetrievedChunk` in this spec rather than deferring to `@jeffs-brain/shared`. Until then SDK implementers should treat those types as the canonical source.
- **TODO**: Define a cross-language convention for MCP progress tokens so the Go and Python SDK ports can share behaviour tests with TypeScript.
