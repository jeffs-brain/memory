import { createHash } from 'node:crypto'
import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { openDatabase, type DriverKind, type SqlDb } from '../search/driver.js'

const EMBED_CACHE_SCHEMA = `
CREATE TABLE IF NOT EXISTS lme_embed_cache (
  model TEXT NOT NULL,
  checksum TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (model, checksum)
);
`

export type LMEEmbedCacheOptions = {
  readonly path: string
  readonly driver?: DriverKind
}

export class LMEEmbedCache {
  private constructor(private readonly db: SqlDb) {}

  static async open(opts: LMEEmbedCacheOptions): Promise<LMEEmbedCache> {
    await mkdir(dirname(opts.path), { recursive: true })
    const db = await openDatabase({
      path: opts.path,
      ...(opts.driver !== undefined ? { driver: opts.driver } : {}),
    })
    db.exec(EMBED_CACHE_SCHEMA)
    return new LMEEmbedCache(db)
  }

  get(model: string, text: string): number[] | undefined {
    const checksum = embedChecksum(model, text)
    const row = this.db
      .prepare(
        `SELECT dim, vector
           FROM lme_embed_cache
          WHERE model = ? AND checksum = ?`,
      )
      .get(model, checksum) as
      | { readonly dim: number | bigint; readonly vector: Uint8Array | Buffer }
      | null
      | undefined
    if (row === undefined || row === null) return undefined
    if (row.dim === null || row.vector === null) return undefined
    const dim =
      typeof row.dim === 'bigint' ? Number(row.dim) : row.dim
    const vector = unpackVector(Buffer.from(row.vector), dim)
    return vector
  }

  put(model: string, text: string, vector: readonly number[]): void {
    if (vector.length === 0) return
    const checksum = embedChecksum(model, text)
    this.db
      .prepare(
        `INSERT INTO lme_embed_cache (model, checksum, dim, vector, updated_at)
         VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(model, checksum) DO UPDATE SET
           dim = excluded.dim,
           vector = excluded.vector,
           updated_at = CURRENT_TIMESTAMP`,
      )
      .run(model, checksum, vector.length, packVector(vector))
  }

  close(): void {
    this.db.close()
  }
}

export const embedChecksum = (model: string, text: string): string =>
  createHash('sha256')
    .update(model)
    .update('\x1f')
    .update(text)
    .digest('hex')

const packVector = (vector: readonly number[]): Buffer => {
  const floats = Float32Array.from(vector)
  return Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength)
}

const unpackVector = (buffer: Buffer, dim: number): number[] => {
  const expected = dim * Float32Array.BYTES_PER_ELEMENT
  if (buffer.byteLength !== expected) return []
  const floats = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
  )
  return [...floats]
}
