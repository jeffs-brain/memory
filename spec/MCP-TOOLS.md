# MCP tool surface

Every Jeffs Brain SDK ships an MCP wrapper that exposes the same `memory_*` tool surface. The tool names, descriptions, and input schemas are canonical and cross-language. The reference implementation lives in `apps/mcp/src/tools/`.

All schemas are described below in Zod-like shorthand. Optional arguments are marked `?`. `brain` on every tool that takes it defaults to the `JBMCP_DEFAULT_BRAIN` environment variable (or the first accessible brain when that is unset) and is resolved server-side via `resolveBrain`.

Each tool returns an MCP response with:

- `content[0].text`: a short human-readable summary.
- `structuredContent`: the raw JSON payload for programmatic consumers.

Errors from any tool surface as a standard MCP error response via `toToolError`. The `no_brain` code is returned by every brain-scoped tool when brain resolution fails.

## Namespace URIs

MCP tool payloads carry canonical relative `path` values. When callers need URI-shaped references inside chat content or markdown, the following URI convention is canonical:

- `memory://global/<topic-stem>` -> `memory/global/<topic-stem>.md`
- `memory://project/<actorId>/<topic-stem>` -> `memory/project/<actorId>/<topic-stem>.md`
- `memory://agent/<actorId>/<topic-stem>` -> `memory/agent/<actorId>/<topic-stem>.md`
- `wiki://<article-stem>` -> `wiki/<article-stem>.md`
- `wiki://architecture/events` -> `wiki/architecture/events.md`

Rules:

- Document `path` values stay relative and POSIX-normalised. They never carry a leading slash.
- `memory://` uses the URI authority as the namespace selector. Valid authorities are `global`, `project`, and `agent`.
- `project` and `agent` URIs require the first path segment after the authority to be the actor or project slug.
- `wiki://` resolves the full article stem from `authority + path`. A single-segment article therefore looks like `wiki://release-checklist`, while a nested article can look like `wiki://architecture/events`.
- URI stems omit the `.md` suffix.
- Path segments use standard percent-encoding. Callers MUST decode before resolution and MUST re-encode reserved characters when serialising.
- Resolution is exact. Consumers MUST NOT invent cross-scope fallbacks when resolving a URI.

---

## `memory_search`

Search memory notes in a brain and return matching note content with citations.

**Description**: "Search memory notes in a brain and return matching note content with citations. `scope` is an exact namespace filter, and `sort` controls whether relevance or recency wins."

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

**Description**: "Recall memories for a query. Pass session_id to weight recent session context; otherwise uses the dedicated memory-search surface. `scope` is an exact namespace filter rather than a generic metadata filter."

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

## Response type shapes

The wire contract for every `structuredContent` payload is pinned here. Types are lifted from `packages/shared/src/schemas/` in the platform monorepo (`ask.ts`, `document.ts`, `knowledge.ts`, `search.ts`, `session.ts`). Timestamps are RFC 3339 with offset. IDs are opaque strings. Keys are `snake_case` on the wire (mirroring the Zod schemas); SDK ports MUST preserve this even when the idiomatic in-memory naming convention differs.

### `IngestResponse`

Returned by `memory_ingest_file` (always, from `POST /v1/brains/{id}/documents/ingest/file`) and by `memory_ingest_url` on the server path (`POST /v1/brains/{id}/documents/ingest/url`). It is a discriminated union on `status`. Source: `schemas/knowledge.ts`.

```ts
type IngestCompletedResponse = {
  status?: 'completed'          // omitted by the completed-path for back-compat
  document_id: string
  path: string
  hash: string
  chunk_count: number           // non-negative integer
  embedded_count: number        // non-negative integer
  duration_ms: number           // non-negative integer
  reused: boolean               // default false
  // Additional keys allowed (schema is `.loose()`); SDKs MUST preserve unknown fields verbatim.
}

type IngestQueuedResponse = IngestCompletedResponse & {
  status: 'queued'
  job_id: string
}

type IngestResponse = IngestCompletedResponse | IngestQueuedResponse
```

Example (`completed`):

```json
{
  "status": "completed",
  "document_id": "doc_01hxyz...",
  "path": "/ingest/readme.md",
  "hash": "2fbe9c...",
  "chunk_count": 18,
  "embedded_count": 18,
  "duration_ms": 412,
  "reused": false
}
```

Example (`queued`):

```json
{
  "status": "queued",
  "job_id": "job_01hxyz...",
  "document_id": "doc_01hxyz...",
  "path": "/ingest/big-manual.pdf",
  "hash": "b3cafe...",
  "chunk_count": 0,
  "embedded_count": 0,
  "duration_ms": 7,
  "reused": false
}
```

### `DocumentRecord`

The full persisted document as returned by `POST /v1/brains/{id}/documents` and surfaced in `memory_remember`, the `memory_ingest_url` fallback path, and `memory_extract` transcript mode. Source: `schemas/document.ts`.

```ts
type DocumentMetadata = Record<string, string | number | boolean | null>   // keys 1..128 chars

type DocumentRecord = {
  id: string                     // opaque document id
  brain_id: string
  title: string                  // 1..512 chars
  path: string                   // 1..1024 chars, NUL-free
  source: 'ingest' | 'extract' | 'compile' | 'reflect'
  content_type: string           // default 'text/markdown'
  byte_size: number              // non-negative integer, canonical text size
  checksum_sha256: string        // 64-char lowercase hex
  metadata: DocumentMetadata     // defaults to {}
  commit_sha: string             // 7..64 char lowercase hex
  created_at: string             // RFC 3339 with offset
  updated_at: string
  deleted_at: string | null
}
```

Example:

```json
{
  "id": "doc_01hxyz...",
  "brain_id": "brn_01hxyz...",
  "title": "Saturday run notes",
  "path": "memory/global/user-preference-running.md",
  "source": "ingest",
  "content_type": "text/markdown",
  "byte_size": 1842,
  "checksum_sha256": "2fbe9c34a1...b7",
  "metadata": { "tags": "running,health" },
  "commit_sha": "a1b2c3d4e5f6",
  "created_at": "2026-04-18T08:12:00+00:00",
  "updated_at": "2026-04-18T08:12:00+00:00",
  "deleted_at": null
}
```

`DocumentRecord` is **not** included in `structuredContent` as a standalone key; it is always nested under `document` (extract transcript mode and ingest-url fallback) or inlined at the top level (remember) via spread.

### `SessionMessageRecord`

Returned by `memory_extract` session mode, one entry per message created via `POST /v1/brains/{id}/sessions/{sessionId}/messages`. Source: `schemas/session.ts` (`MessageSchema`).

```ts
type SessionMessageRecord = {
  id: string                     // opaque message id
  session_id: string
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string                // up to 1_000_000 chars
  name?: string                  // tool name when role is 'tool'; 1..128 chars
  created_at: string             // RFC 3339 with offset
}
```

Note: the `metadata` bag (including `actor_id` and the `skip_extract` flag used by `memory_extract`) is accepted on the **create** request (`CreateMessageSchema`) but is not echoed back on the returned `MessageSchema`. SDK ports MUST NOT surface `metadata` on the read shape even when the server happens to include it.

Example:

```json
{
  "id": "msg_01hxyz...",
  "session_id": "ses_01hxyz...",
  "role": "user",
  "content": "what did I watch last Friday?",
  "created_at": "2026-04-18T08:12:00+00:00"
}
```

### `AskCitationEvent`

SSE event streamed during `memory_ask` alongside `answer_delta` frames. Collected into the `citations` array on the final tool response. Source: `schemas/ask.ts` (`AskCitationEventSchema`).

```ts
type AskCitationEvent = {
  type: 'citation'
  chunk_id: string
  document_id: string
  answer_start: number           // non-negative integer, char offset into the accumulated answer
  answer_end: number             // non-negative integer
  quote: string                  // the supporting text lifted from the chunk
}
```

Example:

```json
{
  "type": "citation",
  "chunk_id": "chk_01hxyz...",
  "document_id": "doc_01hxyz...",
  "answer_start": 184,
  "answer_end": 231,
  "quote": "I finished the 10k route in under 55 minutes."
}
```

Other SSE event types that flow alongside citations during `memory_ask` (`retrieve`, `answer_delta`, `done`, `error`) are defined in `schemas/ask.ts` under the `AskEvent` discriminated union and are out of scope for this spec section; only `AskCitationEvent` surfaces in the tool's `structuredContent`.

### `RetrievedChunk`

The shape of each entry in `memory_ask`'s `structuredContent.retrieved` array, copied from the `chunks` field of the first `retrieve` SSE event. Source: `schemas/ask.ts` (`AskRetrieveEventSchema.chunks[*]`).

```ts
type RetrievedChunk = {
  chunk_id: string
  document_id: string
  score: number
  preview: string                // up to 512 chars
}
```

Example:

```json
{
  "chunk_id": "chk_01hxyz...",
  "document_id": "doc_01hxyz...",
  "score": 0.874,
  "preview": "## Saturday\n\nI finished the 10k route in under 55 minutes..."
}
```

Note: this is intentionally thinner than the `SearchResultChunk` shape returned by `memory_search` (which carries `brain_id`, `metadata`, `highlights`, and optional `component_scores`). `memory_ask` trades detail for streaming throughput. SDK ports MUST NOT substitute the richer `SearchResultChunk` here; the two endpoints speak different wire shapes by design, and cross-mixing will break conformance.

## Progress tokens

Long-running tools emit MCP progress notifications using the standard `notifications/progress` method defined in the Model Context Protocol spec. The reference implementation in `memory_ask` (see `apps/mcp/src/tools/ask.ts`) is the canonical pattern; every SDK port MUST follow the same conventions.

### Token lifecycle

1. The **client** generates a `progressToken` and includes it in the tool call's `_meta.progressToken` field. The token is an opaque string (typically a UUIDv4, but any non-empty string is accepted; the server never parses it). A tool call without `_meta.progressToken` MUST NOT emit any `notifications/progress` messages.
2. The **server** reads `extra._meta?.progressToken` at the top of the tool handler. If absent, streaming updates are suppressed and only the final response is returned.
3. The **server** sends progress notifications via `extra.sendNotification({ method: 'notifications/progress', params: { progressToken, progress, message? } })`. The `progressToken` MUST match what the client provided, verbatim.
4. On completion (normal or error), the final tool response is returned via the usual `content[0].text` + `structuredContent` path. There is no explicit "progress done" notification: the tool reply itself terminates the progress stream.

### Notification payload

`params.progress` is a **monotonically increasing counter** (an integer in TS, starting at 0 and incremented by 1 per emitted event). It is **not** a percentage and it is **not** paired with a `total` field in the v1.0 SDKs. The counter is semantically "events emitted so far"; clients that want a percentage should scale it against their own expectations or wait for `total` to be populated in a future spec version.

`params.message` is an optional free-form string. `memory_ask` uses it to carry the incremental `answer_delta.delta` string so clients can render tokens as they stream. Other tools either omit `message` or supply a short human-readable stage label.

### Per-tool progress semantics

| Tool | Emits progress? | Counter unit | `message` contents |
| --- | --- | --- | --- |
| `memory_ask` | Yes, when `progressToken` present. | Number of `answer_delta` SSE frames received. | The delta text for that frame. |
| `memory_ingest_file` | **Reserved.** The v1.0 TS implementation does not emit `notifications/progress` frames; the tool runs synchronously end-to-end and the final response carries a `status` of `queued` or `completed`. | n/a | n/a |
| `memory_ingest_url` | **Reserved.** Same as `memory_ingest_file`. | n/a | n/a |
| `memory_consolidate` | **Reserved.** Fire-and-forget; the tool returns whatever `brains.consolidate`/`brains.compile` resolves with. | n/a | n/a |
| `memory_reflect` | No. Synchronous close + reflect; no progress stream. | n/a | n/a |
| All other tools | No. | n/a | n/a |

SDK ports that add progress support to a tool marked **Reserved** MUST land a spec update first. Until then, their absence of progress emission is part of the wire contract: clients MUST tolerate tools that honour `progressToken` with zero notifications before the final reply.

### Example notification (from `memory_ask`)

```json
{
  "method": "notifications/progress",
  "params": {
    "progressToken": "b7f2a5d2-9b1c-4ed9-9a1d-0a7f3c1c9e4c",
    "progress": 42,
    "message": " cor"
  }
}
```

The client concatenates `message` values in order to reconstruct the streaming answer. The final `done` SSE event on the underlying stream still populates `structuredContent.answer` with the full assembled text, so clients that miss a frame can fall back to the final payload.

### Error handling during streaming

If the underlying SSE stream emits an `error` event (see `schemas/ask.ts` `AskErrorEventSchema`), the server stops emitting `notifications/progress` and returns a text-only tool response describing the error with `{ code, message, retryable }`. Clients MUST treat the absence of a final reply accompanied by no terminal `notifications/progress` as an abort and surface the error code verbatim.
