---
title: "Storage"
description: "Store interface, path validation, batch semantics, error taxonomy."
sidebar:
  order: 2
---


This document describes the `Store` interface, the `Batch` contract, path validation rules, error taxonomy, and the `ChangeEvent` shape every Jeffs Brain SDK must implement. The TypeScript reference lives in `packages/memory/src/store/`.

## Path validation

Paths are brand-typed (`Path = string & { __brand: 'BrainPath' }`) so that any function accepting a `Path` is guaranteed to have seen the validator. The validation rules, enforced by `validatePath` / `toPath`:

- Non-empty string.
- No NUL byte (`\0`).
- No backslash (`\`).
- No leading slash (`/foo/bar`).
- No trailing slash (`foo/bar/`).
- No `.` or `..` path segments.
- No empty path segments (i.e. no `//`).
- Must already be canonical: `cleanPosix(p) === p` where `cleanPosix` mirrors Go's `path.Clean`.

Generated files are identified by a leading underscore on the base name (`_index.md` → generated). Generated files are hidden from listings by default; pass `includeGenerated: true` to surface them.

Shell-glob matching over base names follows the subset Go's `path.Match` supports:

- `*` matches any run of non-separator characters.
- `?` matches a single character.
- `[...]` matches one of the enclosed characters; leading `!` or `^` negates; `a-z` is a range.

## Store interface

```
type Store = {
    read(path: Path): Promise<Buffer>
    write(path: Path, content: Buffer): Promise<void>
    append(path: Path, content: Buffer): Promise<void>
    delete(path: Path): Promise<void>
    rename(src: Path, dst: Path): Promise<void>
    exists(path: Path): Promise<boolean>
    stat(path: Path): Promise<FileInfo>
    list(dir: Path | '', opts?: ListOpts): Promise<FileInfo[]>
    batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void>
    subscribe(sink: EventSink): Unsubscribe
    localPath(path: Path): string | undefined
    close(): Promise<void>
}
```

Type shapes:

```
type FileInfo   = { path: Path, size: number, modTime: Date, isDir: boolean }
type ListOpts   = { recursive?: boolean, glob?: string, includeGenerated?: boolean }
type BatchOptions = { reason: string, message?: string, author?: string, email?: string }
type ChangeKind = 'created' | 'updated' | 'deleted' | 'renamed'
type ChangeEvent = {
    kind: ChangeKind,
    path: Path,
    oldPath?: Path,
    reason?: string,
    when: Date,
}
type EventSink   = (event: ChangeEvent) => void
type Unsubscribe = () => void
```

Semantics:

- `read` returns the full byte content or throws `ErrNotFound`.
- `write` replaces content. Creates parent structure implicitly.
- `append` creates-then-extends; equivalent to `write(concat(prev, content))` with no prior read if the file does not exist.
- `delete` throws `ErrNotFound` for missing paths.
- `rename` moves content atomically and overwrites the destination. `ErrNotFound` when the source is missing.
- `exists` never throws on "missing"; only on invalid paths or backend errors.
- `stat` returns `{ path, size, modTime, isDir }`; throws `ErrNotFound` for missing paths.
- `list` returns entries sorted ascending by path. Non-recursive listings include one entry per immediate child (directory entries have `isDir: true`). Recursive listings walk the subtree and omit directory entries.
- `localPath` returns a filesystem path when the backend has one (FsStore, GitStore checkout) and `undefined` otherwise (HttpStore, MemStore).
- `close` makes the store unusable; further mutations throw `ErrReadOnly`. Idempotent.

## Batch semantics

`batch(opts, fn)` opens a write journal, runs `fn(batch)`, and commits on successful return. The `Batch` handle offers the same read/write surface as `Store` minus `batch`, `subscribe`, `localPath`, and `close`.

### Merging rules

The journal is replayed whenever a pending read happens and whenever the commit flushes to the backend. The reference `replay(journal, path)` resolves a single path into one of three states: `present(content)`, `deleted`, or `untouched`.

- **Write + Delete on the same path**: the two cancel; the final state is `deleted` if the file did not exist before the batch, or `deleted` relative to the batch's committed state.
- **Write + Write on the same path**: the later write wins.
- **Append after Write**: materialises as a single `write` of the concatenated content.
- **Rename**: the `HttpBatch` reference materialises rename as `write(dst, content)` followed by `delete(src)` so the server sees a flat sequence. In-memory backends may implement rename natively, but the observable outcome is identical.
- **Read** inside the batch returns the result of replaying the journal up to that point: `present` buffers are served without a backend hit; `deleted` states throw `ErrNotFound`; `untouched` defers to the backend.
- **Exists** and **stat** follow the same three-state replay before falling back to the backend.
- **List** fetches the backend listing, overlays pending creations, drops pending deletions, and re-applies the glob/generated/recursive filters over the merged result.

### Atomicity guarantees per backend

- **MemStore**: in-process map. Atomic per JavaScript microtask.
- **FsStore**: POSIX filesystem backend. Atomic per `rename(2)` swap for writes; batches buffer all mutations and apply them in sequence, rolling back on error.
- **GitStore**: commit-per-batch. Produces one commit with the batch `reason` as the message subject. Atomic at the commit boundary.
- **HttpStore**: journals locally, sends one `POST /documents/batch-ops` on commit. Atomic at the server's transaction boundary. An error before the commit request means nothing reached the wire; a non-2xx response means the server rolled back.

All four backends share the invariant that a thrown error from `fn` leaves the brain untouched.

## Error taxonomy

```
class StoreError       extends Error       # base class, name = 'StoreError'
class ErrNotFound      extends StoreError  # missing path
class ErrConflict      extends StoreError  # concurrent modification
class ErrReadOnly      extends StoreError  # store closed or backend is read-only
class ErrInvalidPath   extends StoreError  # path validation failed
class ErrSchemaVersion extends StoreError  # on-disk schema version unsupported
```

Helpers: `isNotFound`, `isInvalidPath`, `isReadOnly`. Every SDK should expose equivalent sentinel types so cross-language tests can map them through the HTTP Problem+JSON error shape.

HTTP store error mapping:

- HTTP `404` → `ErrNotFound`
- HTTP `400` → `ErrInvalidPath`
- Anything else → generic `StoreError` wrapping the parsed Problem+JSON payload

## ChangeEvent shape and ordering

`subscribe(sink)` registers a sink that receives a `ChangeEvent` per committed mutation. Ordering guarantees:

- For single mutations, exactly one event fires after the backend commits.
- For batches, one event fires per materialised op, in the order `fn` issued them.
- Events carry the op reason when provided in `BatchOptions.reason` or directly on single-mutation calls that forward a reason (batch today).
- HTTP stores deliver events asynchronously over SSE; subscribers may need to await one tick after a mutation before the sink fires.
- `subscribe` returns an `Unsubscribe` function. When the last subscriber unsubscribes on an HTTP store, the underlying SSE connection is torn down; re-subscribing opens a new connection.

`oldPath` is set on `renamed` events only. `reason` is optional on all events.

## ErrConflict

`ErrConflict` signals that a mutation lost a race with a concurrent writer or with an upstream that has moved on. It is the SDK-level sentinel for optimistic concurrency failures. SDKs in other languages MUST expose an equivalent sentinel so cross-language tests can map through the HTTP Problem+JSON `code`.

Throw sites in the TypeScript reference (`gitstore.ts`):

- `GitStore` push rejection: when `git push` fails as a non-fast-forward, the store raises `ErrConflict` with detail `git push rejected by <remote>/<branch>`. Triggered by a remote that advanced since the last pull.
- `GitStore` rebase conflict: when the auto-rebase step during pull hits a merge conflict the store aborts the rebase and raises `ErrConflict` with detail `git pull conflicted while rebasing on <remote>/<branch>`.
- `GitStore` stash-pop conflict: when a pull succeeded but re-applying the auto-stash conflicts the store raises `ErrConflict` with detail `git pull restored remote changes but local stash pop conflicted`.

`MemStore` and `FsStore` do not raise `ErrConflict`; they serialise mutations in-process and never observe contention. Adapters that bolt on optimistic concurrency (the Postgres adapter at `sdks/ts/memory-postgres/src/store.ts`, `memory-openfga` wrappers that enforce row-version checks) raise `ErrConflict` from `write`, `append`, `delete`, `rename`, and `batch` when the on-disk version does not match the expected version.

Wire representation over HTTP (see `spec/PROTOCOL.md` error codes table):

- HTTP status `409 Conflict`.
- Problem+JSON `code`: `"conflict"`.
- `detail` SHOULD include the contending resource (path, branch, or batch description) so clients can surface it to humans.

```json
{
  "status": 409,
  "title": "Conflict",
  "detail": "brain: concurrent modification: git push rejected by origin/main",
  "code": "conflict"
}
```

Clients SHOULD NOT retry `ErrConflict` automatically. Callers that want last-writer-wins semantics MUST refetch current state, rebase their intended mutation on top, and retry explicitly.

## Event ordering

`subscribe(sink)` returns an `Unsubscribe` handle. The backend publishes to every live sink; the rules that follow apply identically across `MemStore`, `FsStore`, `GitStore`, and `HttpStore`.

Per-store total order. A single store instance publishes events in commit order. For a batch of N operations the store emits N events, one per materialised op, in the order they were journalled.

Per-sink delivery. Every registered sink receives the complete stream. Fan-out is synchronous inside the emitter: the store iterates its sink set and invokes each sink before returning from the commit path. Sinks observe events in the same order as the store's emit order; there is no per-sink reordering.

Concurrency and reentrancy. Sinks are invoked serially. A sink that throws does not short-circuit the fan-out; emitters MUST catch and log, never propagate. A sink that itself mutates the store observes its own events; callers responsible for avoiding infinite loops.

Delivery guarantees:

- `MemStore`, `FsStore`, `GitStore`: at-most-once. Events fire exactly once per committed mutation to every sink registered at the moment `emit` runs. Sinks subscribed after the commit completes MISS the event; there is no replay buffer.
- `HttpStore`: at-most-once over SSE. The server emits one `change` frame per committed mutation. A client that disconnects MISSES every frame produced while it was absent. Reconnecting opens a fresh stream with no replay. `Last-Event-ID` is NOT honoured in v1.0 (reserved).

Client cursor semantics:

- v1.0 does not expose a replay cursor. Clients that need durable change tracking MUST layer their own journal over the store (for example, by writing an append-only log on every mutation).
- The `id:` field on SSE frames is monotonically increasing per stream attach (see `spec/PROTOCOL.md`) but MUST NOT be interpreted as a cross-attach cursor. Reserved for v1.1.

Ordering across subscribers to the same store is guaranteed: if sink A observes event E1 before E2, sink B observes the same ordering. There is no causal-only mode; per-store total order is the contract.

## GitStore signing

`GitStore` supports a pluggable commit signing callback so deployments can integrate with SSH agents, KMS, GPG, or any other key custodian without baking cryptography into the SDK. The helpers live in `sdks/ts/memory/src/store/gitstore.ts`.

Types:

```
type GitSignPayload = { payload: string }
type GitSignFn      = (payload: GitSignPayload) => Promise<string>
```

`GitSignPayload.payload` is the UTF-8 serialised commit object that would be hashed to produce the commit SHA. `GitSignFn` returns the detached signature as an ASCII-armored string (for example, a PGP `-----BEGIN PGP SIGNATURE-----` block or an SSH `-----BEGIN SSH SIGNATURE-----` block). The signature is embedded verbatim in the commit header; callers own the key material.

Invocation points. The callback is invoked by `isomorphic-git` whenever a commit object is about to be written:

- Every batch commit in `commitTouched` (the single commit produced by each call to `Store.batch`).
- The `[init]` bootstrap commit produced by `initialiseRepository` when the store creates a brand-new repository.

The callback is NOT invoked on:

- Tag creation (v1.0 does not tag).
- `git push` (transport authentication is handled by the OS git binary and is out of scope for the SDK).
- `FsStore` or `MemStore` mutations.

Wiring. `GitStoreOptions.sign` is optional. When present, `GitStore` passes an internal `signingKey` placeholder to `isomorphic-git` purely to arm the signing path; the placeholder is never exposed to the callback. When absent, commits are unsigned and the placeholder is never set.

```
const store = await createGitStore({
  dir: '/var/brain/main',
  init: true,
  sign: async ({ payload }) => signWithSshAgent(payload),
})
```

Language SDKs. Go and Python SDKs MUST expose the same two-type shape (`GitSignPayload` with a single `payload: string` field; `GitSignFn` taking the payload and returning a future-of-string). The callback MUST receive the pre-hashed commit object verbatim and MUST return the detached armored signature. Implementations MUST invoke the callback at every batch commit and the init commit, and MUST NOT invoke it for tags or pushes in v1.0.
