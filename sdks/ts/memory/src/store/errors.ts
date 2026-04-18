export class StoreError extends Error {
  override readonly name: string = 'StoreError'
  constructor(
    message: string,
    override readonly cause?: unknown,
  ) {
    super(message)
  }
}

export class ErrNotFound extends StoreError {
  override readonly name = 'ErrNotFound'
  constructor(path: string, cause?: unknown) {
    super(`brain: not found: ${path}`, cause)
  }
}

export class ErrConflict extends StoreError {
  override readonly name = 'ErrConflict'
  constructor(message: string, cause?: unknown) {
    super(`brain: concurrent modification: ${message}`, cause)
  }
}

export class ErrReadOnly extends StoreError {
  override readonly name = 'ErrReadOnly'
  constructor() {
    super('brain: read-only')
  }
}

export class ErrInvalidPath extends StoreError {
  override readonly name = 'ErrInvalidPath'
  constructor(reason: string) {
    super(`brain: invalid path: ${reason}`)
  }
}

export class ErrSchemaVersion extends StoreError {
  override readonly name = 'ErrSchemaVersion'
  constructor(message: string) {
    super(`brain: unsupported schema version: ${message}`)
  }
}

export const isNotFound = (err: unknown): err is ErrNotFound => err instanceof ErrNotFound
export const isInvalidPath = (err: unknown): err is ErrInvalidPath => err instanceof ErrInvalidPath
export const isReadOnly = (err: unknown): err is ErrReadOnly => err instanceof ErrReadOnly
