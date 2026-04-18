# Storage

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

- **MemStore** — in-process map. Atomic per JavaScript microtask.
- **FsStore** — POSIX filesystem backend. Atomic per `rename(2)` swap for writes; batches buffer all mutations and apply them in sequence, rolling back on error.
- **GitStore** — commit-per-batch. Produces one commit with the batch `reason` as the message subject. Atomic at the commit boundary.
- **HttpStore** — journals locally, sends one `POST /documents/batch-ops` on commit. Atomic at the server's transaction boundary. An error before the commit request means nothing reached the wire; a non-2xx response means the server rolled back.

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

## TODOs / ambiguities

- **TODO**: The `ErrConflict` type is defined but never thrown by the reference today. Pin down when optimistic concurrency kicks in (probably the Postgres backend) and specify the wire representation.
- **TODO**: Specify observable event ordering across multiple subscribers: is a single commit fan-out ordered across sinks, or per-sink only?
- **TODO**: The reference `GitStore` signing helpers (`GitSignFn`, `GitSignPayload`) are not yet part of the language-neutral contract. Decide whether Go/Python SDKs need them.
