---
title: "Protocol"
description: "HTTP store and SSE wire contract every SDK implements."
sidebar:
  order: 1
---


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

## Error codes

Non-2xx responses MUST carry a Problem+JSON body. The `code` field is a machine-readable identifier; SDKs MAY branch on it when the HTTP status alone is insufficient. Clients that do not recognise a `code` value MUST fall back to status-based dispatch.

The TypeScript client in `http.ts` maps on status only: `404` becomes `ErrNotFound`, `400` becomes `ErrInvalidPath`, and anything else bubbles up as a generic `StoreError` wrapping the parsed Problem+JSON payload. The vocabulary below is the closed set emitted by a compliant reference server.

| `code` | HTTP status | Emitted by | Meaning |
| --- | --- | --- | --- |
| `validation_error` | `400` | `PUT /documents`, `POST /documents/append`, `POST /documents/rename`, `POST /documents/batch-ops` | Request violates schema or path validation (leading slash, traversal, NUL byte, non-canonical, empty segment, invalid JSON body, missing required field). Maps to `ErrInvalidPath` client-side. |
| `not_found` | `404` | `GET /documents/read`, `HEAD /documents`, `GET /documents/stat`, `DELETE /documents`, `POST /documents/rename` | Target path does not exist. Maps to `ErrNotFound` client-side. |
| `conflict` | `409` | `POST /documents/batch-ops` backed by a concurrency-aware store (e.g. `PostgresStore`, `GitStore` with push contention) | Optimistic concurrency or git push/rebase rejection. Maps to `ErrConflict` client-side. See `spec/STORAGE.md`. |
| `unauthorized` | `401` | Any endpoint when authentication is required but absent or malformed | Missing or unparseable `Authorization` header. |
| `forbidden` | `403` | Any endpoint when the authenticated principal lacks the scope or RBAC right (`documents:read`, `documents:write`, etc.) | Scope or role does not permit the requested action. |
| `payload_too_large` | `413` | `PUT /documents`, `POST /documents/append`, `POST /documents/batch-ops` | Body or decoded batch payload exceeds the limits under "Body size limits". |
| `unsupported_media_type` | `415` | `PUT /documents`, `POST /documents/append`, `POST /documents/rename`, `POST /documents/batch-ops` | `Content-Type` header does not match the expected value for the endpoint. |
| `rate_limited` | `429` | Any endpoint | Per-principal or per-tenant quota exceeded. The response SHOULD include `Retry-After`. Reserved: the reference client does not implement automatic retry. |
| `internal_error` | `500` | Any endpoint | Unhandled server-side failure. |
| `bad_gateway` | `502` | Any endpoint backed by an upstream store that returned an unusable response | Upstream failure distinct from `internal_error`. |
| `timeout` | `504` | Long-running operations (batch, SSE attach) | Server-side deadline exceeded. Clients retry with their own back-off. |

Servers MAY omit `code` when `status` and `title` unambiguously identify the failure, but SHOULD populate it for every response listed above. Clients MUST NOT reject an unknown `code`.

Example:

```json
{
  "status": 413,
  "title": "Payload Too Large",
  "detail": "batch payload exceeds 8 MiB after base64 decode",
  "code": "payload_too_large"
}
```

## Body size limits

The following limits are part of the wire contract. Servers MAY enforce stricter limits and surface `413 payload_too_large`; clients MUST NOT send requests that exceed the values below.

| Endpoint | Limit | Field |
| --- | --- | --- |
| `PUT /documents` | `2 MiB` (`2097152` bytes) | Raw request body |
| `POST /documents/append` | `2 MiB` (`2097152` bytes) | Raw request body |
| `POST /documents/batch-ops` | `8 MiB` (`8388608` bytes) | Sum of decoded `content_base64` bytes across all ops |
| `POST /documents/batch-ops` | `1024` ops | `ops` array length |
| `GET /documents` listing | `10000` items | `items` array length in the response; callers that need more MUST paginate with `glob` + recursive flags |
| `GET /documents/read` | No client-enforced upper bound | Reference server streams arbitrarily large documents back |

Servers that choose to honour larger limits MUST still produce `413 payload_too_large` at their own ceiling. Servers that choose stricter limits MUST produce `413 payload_too_large` rather than silently truncating. Byte counts are measured after decompression when a `Content-Encoding` is applied.

## Tenant and workspace scoping

Tenant and workspace identifiers are NEVER conveyed as URL path segments on the document endpoints. The `brainId` in `/v1/brains/{brainId}` is the only scope on the wire. Servers resolve the owning tenant from the authenticated principal:

- When the request is authenticated (`Authorization: Bearer <apiKey or token>`) the server looks up the principal's tenant and verifies that `brainId` belongs to it. Cross-tenant access MUST produce `403 forbidden`.
- When the request is unauthenticated and the server permits unauthenticated routes (in-process test harnesses, single-tenant deployments) the server MUST treat `brainId` as authoritative.

Multi-tenant SDKs SHOULD issue one `apiKey`/`token` per tenant and let the server resolve scope. Tenant headers are reserved for future use:

- `X-Tenant-Id` and `X-Workspace-Id` are reserved header names. A v1.0 server MUST ignore them. Servers MUST NOT reject a request that carries either header with an unknown value. A future v1.1 MAY define normative semantics (for example, selecting among tenants owned by a single principal) at which point this section will be updated.

There is no default tenant. A server that cannot resolve the owning tenant MUST respond `401 unauthorized` (principal absent) or `403 forbidden` (principal present but mismatched), never `400`.

## Idempotency

Reserved. The reference client does not send `Idempotency-Key` and the reference server does not inspect it. The contract is:

- Servers MUST NOT reject a request that carries an `Idempotency-Key` header. The header MUST be silently ignored in v1.0.
- Servers MUST NOT persist idempotency records in v1.0.
- Reserved for v1.1. When implemented, `Idempotency-Key` will be honoured on `PUT /documents`, `POST /documents/append`, `DELETE /documents`, `POST /documents/rename`, and `POST /documents/batch-ops`. `GET` and `HEAD` are naturally idempotent and will not accept the header.

Retry safety for v1.0:

| Verb | Endpoint | Retry-safe without `Idempotency-Key`? | Notes |
| --- | --- | --- | --- |
| `GET` | `/documents/read`, `/documents/stat`, `/documents` | Yes | Pure reads. |
| `HEAD` | `/documents` | Yes | Pure read. |
| `PUT` | `/documents` | Yes | Last-writer-wins; replaying a `PUT` with the same body is observationally equivalent. |
| `POST` | `/documents/append` | No | Duplicate appends duplicate bytes. Callers SHOULD use `POST /documents/batch-ops` with a `write` of the concatenated content for retry-safe growth. |
| `DELETE` | `/documents` | Partial | A second `DELETE` of the same path returns `404 not_found`. Callers treating `404` as success on a retry are safe. |
| `POST` | `/documents/rename` | Partial | A second `rename` with the same `from`/`to` returns `404 not_found` once the source no longer exists. |
| `POST` | `/documents/batch-ops` | No | Replaying a batch re-applies every op. Callers MUST wrap batches in transport-level single-delivery. |

## Caching

The reference server emits no `Cache-Control` and honours no `If-None-Match` / `If-Modified-Since` in v1.0. Clients MUST NOT cache response bodies beyond the scope of the in-flight request unless they add their own invalidation via the SSE event stream.

Per-endpoint expectations:

| Endpoint | `Cache-Control` on response | `ETag` on response | `If-None-Match` honoured? |
| --- | --- | --- | --- |
| `GET /documents/read` | `no-store` | Reserved; servers MAY emit a strong ETag derived from the content SHA-256, clients MUST ignore in v1.0 | No |
| `HEAD /documents` | `no-store` | Same as above | No |
| `GET /documents/stat` | `no-store` | Same as above | No |
| `GET /documents` listing | `no-store` | Not emitted | No |
| `GET /events` (SSE) | `no-cache` | Not applicable | Not applicable |

Mutating endpoints (`PUT`, `POST`, `DELETE`) MUST NOT emit `Cache-Control` or `ETag`.

Reserved for v1.1:
- `If-None-Match` short-circuit returning `304 Not Modified` on `GET /documents/read` and `GET /documents/stat`.
- `If-Match` preconditions on `PUT /documents` and `DELETE /documents` enabling optimistic concurrency via ETag.

Clients MAY already send `If-None-Match` / `If-Match`; v1.0 servers MUST ignore them and respond as if they were absent.
