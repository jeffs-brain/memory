export type { Path } from './path.js'
export {
  validatePath,
  toPath,
  isGenerated,
  lastSegment,
  joinPath,
  pathUnder,
  matchGlob,
} from './path.js'

export {
  StoreError,
  ErrNotFound,
  ErrConflict,
  ErrReadOnly,
  ErrInvalidPath,
  ErrSchemaVersion,
  isNotFound,
  isInvalidPath,
  isReadOnly,
} from './errors.js'

import type { Path } from './path.js'

export type FileInfo = {
  readonly path: Path
  readonly size: number
  readonly modTime: Date
  readonly isDir: boolean
}

export type ListOpts = {
  readonly recursive?: boolean
  readonly glob?: string
  readonly includeGenerated?: boolean
}

export type BatchOptions = {
  readonly reason: string
  readonly message?: string
  readonly author?: string
  readonly email?: string
}

export type ChangeKind = 'created' | 'updated' | 'deleted' | 'renamed'

export type ChangeEvent = {
  readonly kind: ChangeKind
  readonly path: Path
  readonly oldPath?: Path
  readonly reason?: string
  readonly when: Date
}

export type EventSink = (event: ChangeEvent) => void

export type Unsubscribe = () => void

export type OpKind = 'write' | 'append' | 'delete' | 'rename'

export type Op =
  | { readonly kind: 'write'; readonly path: Path; readonly content: Buffer }
  | { readonly kind: 'append'; readonly path: Path; readonly content: Buffer }
  | { readonly kind: 'delete'; readonly path: Path }
  | { readonly kind: 'rename'; readonly src: Path; readonly dst: Path }

export type Batch = {
  read(path: Path): Promise<Buffer>
  write(path: Path, content: Buffer): Promise<void>
  append(path: Path, content: Buffer): Promise<void>
  delete(path: Path): Promise<void>
  rename(src: Path, dst: Path): Promise<void>
  exists(path: Path): Promise<boolean>
  stat(path: Path): Promise<FileInfo>
  list(dir: Path | '', opts?: ListOpts): Promise<FileInfo[]>
}

export type Store = {
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

export { FsStore, createFsStore } from './fsstore.js'
export type { FsStoreOptions } from './fsstore.js'
export { MemStore, createMemStore } from './memstore.js'
export { GitStore, createGitStore, readGitLog, listCommitFiles } from './gitstore.js'
export type {
  CommitSummary,
  GitStoreOptions,
  GitSignature,
  GitSignFn,
  GitSignPayload,
} from './gitstore.js'
export { autodetectStore } from './autodetect.js'
export { HttpStore, createHttpStore } from './http.js'
export type { HttpStoreOptions, HttpStoreCapabilities } from './http.js'
