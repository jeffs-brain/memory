# HTTP store protocol

This document describes the wire surface consumed by an HTTP-backed Jeffs Brain `Store` implementation. The TypeScript reference lives in `packages/memory/src/store/http.ts` and the reference server routes live in `apps/backend/src/routes/documents-fs.ts` and `apps/backend/src/routes/events.ts`. Any SDK that ships an HTTP store must drive these endpoints with byte-equivalent behaviour.

## Base URL and brain scoping

All endpoints are rooted under a per-brain prefix:

```
/v1/brains/{brainId}
```

`brainId` is URL-encoded by the client (`encodeURIComponent` semantics). The base URL itself is supplied by the SDK caller at store construction time and may point at any origin (localhost, a tenant backend, a tunnel). Trailing slashes on the base URL are tolerated and trimmed by the client before joining.

## Authentication

Authentication is optional at the protocol level but required by the reference backend's middleware. When the SDK is configured with an `apiKey` or a generic `token`, every request carries:

```
Authorization: Bearer <apiKey or token>
```

`apiKey` wins over `token` when both are set. Requests without an authentication header hit unauthenticated routes only (used by in-process test harnesses). The server enforces API key scopes such as `documents:read` and `documents:write` on top of tenant-level RBAC.

## Common headers

- `User-Agent: @jeffs-brain/memory (+HttpStore)` (or the equivalent SDK-specific identifier)
- `Accept: application/json` on JSON-returning endpoints, `application/octet-stream` on byte-returning endpoints, `text/event-stream` on the SSE stream
- `Content-Type: application/octet-stream` on write/append bodies, `application/json` on rename and batch-ops bodies

## Timeouts

The reference client defaults to a 30 second per-request timeout implemented via `AbortController`. Callers can pass a `signal` that composes with this internal timeout.

## Error shape

Non-2xx responses carry a Problem+JSON body (RFC 7807-adjacent) with at least the following optional fields:

```json
{
  "status": 404,
  "title": "Not Found",
  "detail": "brain: not found: memory/a.md",
  "code": "not_found"
}
```

Status mapping in the client:

- `404` → `ErrNotFound`
- `400` → `ErrInvalidPath`
- Anything else → generic `StoreError` carrying the parsed problem

## Path encoding

Paths are POSIX-style, relative, and validated client-side before reaching the wire. The rules are:

- No leading slash
- No trailing slash
- No `..` or `.` segments
- No empty segments
- No backslashes
- No NUL bytes
- Must already be canonical (i.e. `cleanPosix(p) === p`)

Paths travel as the `path` query parameter (or `dir` for listing) and are URL-encoded using standard `URLSearchParams` semantics. See `spec/STORAGE.md` for the full validation rules.

## Endpoints

### `GET /v1/brains/{brainId}/documents/read`

Read the raw content of a single document.

**Query**

- `path` (required): canonical path

**Response** `200 OK`

```
Content-Type: application/octet-stream
<raw bytes>
```

**Errors**

- `404` when the path does not exist
- `400` when the path is invalid

### `HEAD /v1/brains/{brainId}/documents`

Existence check for a single path.

**Query**

- `path` (required)

**Response**

- `200 OK` with empty body when the path exists
- `404` when the path does not exist

### `GET /v1/brains/{brainId}/documents/stat`

Metadata for a single path.

**Query**

- `path` (required)

**Response** `200 OK`

```json
{
  "path": "memory/a.md",
  "size": 12,
  "mtime": "2025-01-02T03:04:05.678Z",
  "is_dir": false
}
```

### `GET /v1/brains/{brainId}/documents`

List entries under a directory. This same URL serves the legacy id-indexed document listing when none of the listing-specific query parameters are present; the Store uses the listing-specific params to force the path-scoped variant.

**Query**

- `dir` (optional, empty string means the brain root)
- `recursive` (`true` | `false`)
- `include_generated` (`true` | `false`)
- `glob` (optional shell-style pattern applied to base names)

**Response** `200 OK`

```json
{
  "items": [
    {
      "path": "memory/a.md",
      "size": 12,
      "mtime": "2025-01-02T03:04:05.678Z",
      "is_dir": false
    }
  ]
}
```

Entries are sorted ascending by path. Files whose base name begins with `_` are treated as generated and hidden unless `include_generated=true`.

### `PUT /v1/brains/{brainId}/documents`

Replace the content at a path. Creates parent structure implicitly.

**Query**

- `path` (required)

**Body**: raw bytes (`application/octet-stream`). Reference server caps at 2 MiB per request; configurable.

**Response** `204 No Content`.

### `POST /v1/brains/{brainId}/documents/append`

Append bytes to a path. Creates the file when missing.

**Query**

- `path` (required)

**Body**: raw bytes (`application/octet-stream`).

**Response** `204 No Content`.

### `DELETE /v1/brains/{brainId}/documents`

Delete a single path.

**Query**

- `path` (required). Note: calls without `path` fall through to the legacy id-indexed delete handler.

**Response** `204 No Content`. `404` when the path does not exist.

### `POST /v1/brains/{brainId}/documents/rename`

Move content from one path to another. Overwrites the destination.

**Body**

```json
{ "from": "raw/old.md", "to": "raw/new.md" }
```

**Response** `204 No Content`. `404` when `from` does not exist.

### `POST /v1/brains/{brainId}/documents/batch-ops`

Atomic multi-operation batch.

**Body**

```json
{
  "reason": "ingest",
  "message": "optional commit message",
  "author": "optional",
  "email": "optional",
  "ops": [
    { "type": "write",  "path": "memory/a.md", "content_base64": "..." },
    { "type": "append", "path": "memory/log.md", "content_base64": "..." },
    { "type": "delete", "path": "memory/old.md" },
    { "type": "rename", "path": "raw/src.md", "to": "raw/dst.md" }
  ]
}
```

Reference server caps the decoded payload at 8 MiB; enforced as ops are decoded one at a time so offending batches fail fast.

**Response** `200 OK`

```json
{ "committed": 4 }
```

All ops commit atomically. Failure of any single op rolls back the entire batch and surfaces the failing error through Problem+JSON.

### `GET /v1/brains/{brainId}/events` (SSE)

Change event stream for the brain.

**Accept**: `text/event-stream`

**Frames**

- `event: ready` emitted once after the stream attaches. `data: ok`.
- `event: ping` emitted every `pingIntervalMs` (default 25s) to keep proxies from closing idle streams. `data: keepalive`.
- `event: change` emitted for every committed mutation. Payload:

```json
{
  "kind": "created|updated|deleted|renamed",
  "path": "memory/a.md",
  "old_path": "memory/original.md",
  "reason": "ingest",
  "when": "2025-01-02T03:04:05.678Z"
}
```

Each frame carries a monotonically increasing `id`. Clients parse `event:` and `data:` lines manually per the SSE spec and ignore comment lines (leading `:`). `old_path` and `reason` are optional.

Closing the stream happens when the client aborts the underlying request. Subscribe/unsubscribe semantics on the SDK side multiplex one SSE connection across multiple local sinks.

## Batch semantics summary

See `spec/STORAGE.md` for the full Batch contract. Highlights:

- Buffered locally on the client; `commit()` is a single POST to `/documents/batch-ops`.
- Writes see their own pending mutations via an in-memory journal replay.
- `write` followed by `delete` on the same path cancels both.
- `write` followed by `write` on the same path keeps only the latter.
- `rename` is materialised client-side as a write-to-destination plus delete-of-source so the server sees a flat sequence.
- `append` is materialised to `write` after replaying any buffered base content so that server-side append ordering is irrelevant.

## TODOs / ambiguities

- **TODO**: Formally specify the Problem+JSON `code` vocabulary. The reference currently maps specific HTTP statuses to exceptions without reading `code`.
- **TODO**: Specify permitted body size limits as part of the contract (reference uses 2 MiB write / 8 MiB batch but these are configurable server-side).
- **TODO**: Capture the workspace / tenant scoping story. The reference resolves tenant from the auth principal; a multi-tenant SDK will need a documented tenant header or subpath.
- **TODO**: Document expected idempotency keys (none today) and retry safety for each verb.
- **TODO**: Describe caching semantics. The reference sends no `Cache-Control` and does not honour `If-None-Match`; a spec-level answer is needed.
