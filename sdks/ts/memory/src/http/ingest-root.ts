// SPDX-License-Identifier: Apache-2.0

/**
 * Confinement for `POST /ingest/file` requests that name a server-side
 * path instead of sending `contentBase64`. Path ingest is disabled
 * unless the daemon was started with an ingest root, and a permitted
 * path must resolve, after symlinks, inside that root. Mirrors
 * `go/cmd/memory/handler_ingest.go`.
 */

import { realpath, stat } from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'

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
 * Validate the configured ingest root and return it as an absolute
 * path. Blank input disables path ingest and yields `undefined`. Throws
 * when the root is missing or not a directory.
 */
export const resolveIngestRoot = async (root: string | undefined): Promise<string | undefined> => {
  const trimmed = root?.trim() ?? ''
  if (trimmed === '') return undefined
  const abs = resolve(trimmed)
  const info = await stat(await realpath(abs))
  if (!info.isDirectory()) throw new Error(`ingest root: ${abs} is not a directory`)
  return abs
}

/** True when `path` is `root` or lies beneath it. Both must be absolute. */
const withinRoot = (root: string, path: string): boolean => {
  const rel = relative(root, path)
  return rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel)
}

const outsideRoot = (): PathIngestRefusedError =>
  new PathIngestRefusedError('path resolves outside the configured ingest root')

/** The resolved closest directory above `path` that exists. */
const nearestExistingAncestor = async (path: string): Promise<string | undefined> => {
  for (let dir = dirname(path); ; dir = dirname(dir)) {
    try {
      return await realpath(dir)
    } catch (err) {
      if (errorCode(err) !== 'ENOENT' || dirname(dir) === dir) return undefined
    }
  }
}

/**
 * Resolve `requested` inside `root`. A relative path is taken relative
 * to the root. Containment is decided on the fully resolved path, so a
 * symlink cannot point the read elsewhere and any spelling of a path
 * inside the root is accepted. A path that does not exist is judged by
 * its nearest existing ancestor, so a path outside the root is refused
 * the same way whether or not it exists. Returns the real path of a
 * regular file inside the root.
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
  const realRoot = await realpath(resolve(root))
  const candidate = resolve(root, requested)
  let real: string
  try {
    real = await realpath(candidate)
  } catch (err) {
    if (errorCode(err) !== 'ENOENT') throw outsideRoot()
    const ancestor = await nearestExistingAncestor(candidate)
    if (ancestor === undefined || !withinRoot(realRoot, ancestor)) throw outsideRoot()
    throw new IngestPathInvalidError('file not found')
  }
  if (!withinRoot(realRoot, real)) throw outsideRoot()
  const info = await stat(real)
  if (!info.isFile()) throw new IngestPathInvalidError('path is not a regular file')
  return real
}
