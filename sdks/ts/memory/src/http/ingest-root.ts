// SPDX-License-Identifier: Apache-2.0

/**
 * Confinement for `POST /ingest/file` requests that name a server-side
 * path instead of sending `contentBase64`. Path ingest is disabled
 * unless the daemon was started with an ingest root, and a permitted
 * path must resolve, after symlinks, inside that root. Mirrors
 * `go/cmd/memory/handler_ingest.go`.
 */

import { realpath, stat } from 'node:fs/promises'
import { isAbsolute, relative, resolve, sep } from 'node:path'

/** The request named a path the daemon refuses to read. Maps to 403. */
export class PathIngestRefusedError extends Error {
  override readonly name = 'PathIngestRefusedError'
}

/** The named path is missing or is not a regular file. Maps to 400. */
export class IngestPathInvalidError extends Error {
  override readonly name = 'IngestPathInvalidError'
}

const errorCode = (err: unknown): string | undefined =>
  typeof err === 'object' && err !== null && 'code' in err && typeof err.code === 'string'
    ? err.code
    : undefined

/**
 * Validate the configured ingest root and return its absolute,
 * symlink-free form. Blank input disables path ingest and yields
 * `undefined`. Throws when the root is missing or not a directory.
 */
export const resolveIngestRoot = async (root: string | undefined): Promise<string | undefined> => {
  const trimmed = root?.trim() ?? ''
  if (trimmed === '') return undefined
  const abs = resolve(trimmed)
  const info = await stat(await realpath(abs))
  if (!info.isDirectory()) throw new Error(`ingest root: ${abs} is not a directory`)
  // Keep the configured spelling: resolveIngestPath accepts requests through
  // it as well as through the resolved form, so a root reached via a symlink
  // (macOS /var, for one) still matches absolute paths.
  return abs
}

/** True when `path` is `root` or lies beneath it. Both must be absolute. */
const withinRoot = (root: string, path: string): boolean => {
  const rel = relative(root, path)
  return rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel)
}

const outsideRoot = (): PathIngestRefusedError =>
  new PathIngestRefusedError('path resolves outside the configured ingest root')

/**
 * Resolve `requested` inside `root`. A relative path is taken relative
 * to the root. Containment is checked on the normalised path first, so
 * a path outside the root is refused without revealing whether it
 * exists, and again after resolving symlinks, so a link inside the root
 * cannot point the read elsewhere. Returns the real path of a regular
 * file inside the root.
 */
export const resolveIngestPath = async (
  root: string | undefined,
  requested: string,
): Promise<string> => {
  if (root === undefined || root === '') {
    throw new PathIngestRefusedError(
      'server-side path ingest is disabled; send contentBase64 or start the daemon with an ingest root',
    )
  }
  const absRoot = resolve(root)
  const realRoot = await realpath(absRoot)
  const candidate = resolve(absRoot, requested)
  if (!withinRoot(absRoot, candidate) && !withinRoot(realRoot, candidate)) throw outsideRoot()
  let real: string
  try {
    real = await realpath(candidate)
  } catch (err) {
    if (errorCode(err) === 'ENOENT') throw new IngestPathInvalidError('file not found')
    throw outsideRoot()
  }
  if (!withinRoot(realRoot, real)) throw outsideRoot()
  const info = await stat(real)
  if (!info.isFile()) throw new IngestPathInvalidError('path is not a regular file')
  return real
}
