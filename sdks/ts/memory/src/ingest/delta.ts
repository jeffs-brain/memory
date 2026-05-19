// SPDX-License-Identifier: Apache-2.0

/**
 * Chunk-level change detection via content-hash delta comparison.
 * Computes a BLAKE3 hash per chunk during chunking and stores chunk
 * hashes in a manifest alongside the document metadata. On re-ingest,
 * compares new chunk hashes against the stored manifest to determine
 * which chunks need re-embedding (new/changed), which can be skipped
 * (unchanged), and which must be removed from the index (deleted).
 *
 * Matching is by CONTENT HASH regardless of position: a chunk that moves
 * to a different ordinal but retains identical text is classified as
 * unchanged and does not trigger re-embedding.
 *
 * Port of go/knowledge/delta.go.
 */

import { hashChunk as blake3HashChunk } from './hash.js'

import type { Chunk } from './chunker.js'
import type { Store } from '../store/index.js'
import { isNotFound, toPath } from '../store/index.js'

const CHUNK_MANIFESTS_PREFIX = 'raw/.chunk-manifests'

/**
 * Validates that a document hash is a lowercase hex string. Prevents
 * path traversal attacks via crafted hash values.
 */
const VALID_DOCUMENT_HASH_RE = /^[a-f0-9]+$/

const validateDocumentHash = (documentHash: string): void => {
  if (!VALID_DOCUMENT_HASH_RE.test(documentHash)) {
    throw new Error(`ingest: invalid document hash "${documentHash}": must match ^[a-f0-9]+$`)
  }
}

export type ChunkManifestEntry = {
  readonly hash: string
  readonly chunkId: string
}

export type ChunkManifest = {
  readonly documentHash: string
  readonly generation: number
  readonly chunks: readonly ChunkManifestEntry[]
  readonly updatedAt: string
}

export type DeltaCategory = 'unchanged' | 'new' | 'removed'

export type ChunkDelta = {
  readonly chunk: Chunk
  readonly category: DeltaCategory
  readonly hash: string
}

/**
 * Computes a BLAKE3 content hash of a chunk's text. This is the
 * canonical hashing function for chunk-level change detection.
 * Delegates to the shared BLAKE3 hasher in hash.ts for cross-SDK
 * conformance with the Go implementation.
 */
export const hashChunk = (text: string): string =>
  blake3HashChunk(Buffer.from(text, 'utf8'))

/**
 * Compares new chunks against a stored manifest and returns the delta
 * set. Matching is by content hash regardless of position.
 *
 * When manifest is undefined (first ingest), all chunks are categorized
 * as 'new'. Removed chunks carry a synthetic Chunk with ordinal from the
 * prior manifest entry.
 *
 * Time: O(N + M) where N = newChunks.length, M = manifest.chunks.length.
 * Space: O(N + M) for the hash lookup maps.
 */
export const computeChunkDeltas = (
  newChunks: readonly Chunk[],
  manifest: ChunkManifest | undefined,
): readonly ChunkDelta[] => {
  if (newChunks.length === 0 && manifest === undefined) {
    return []
  }

  const newHashMap = new Map<string, number[]>()
  const newHashes: string[] = []
  for (let i = 0; i < newChunks.length; i++) {
    const chunk = newChunks[i]
    if (chunk === undefined) continue
    const h = hashChunk(chunk.content)
    newHashes.push(h)
    const existing = newHashMap.get(h)
    if (existing !== undefined) {
      existing.push(i)
    } else {
      newHashMap.set(h, [i])
    }
  }

  if (manifest === undefined) {
    const out: ChunkDelta[] = []
    for (let i = 0; i < newChunks.length; i++) {
      const chunk = newChunks[i]
      if (chunk === undefined) continue
      const h = newHashes[i]
      if (h === undefined) continue
      out.push({ chunk, category: 'new', hash: h })
    }
    return out
  }

  const oldHashMap = new Map<string, number[]>()
  for (let i = 0; i < manifest.chunks.length; i++) {
    const entry = manifest.chunks[i]
    if (entry === undefined) continue
    const existing = oldHashMap.get(entry.hash)
    if (existing !== undefined) {
      existing.push(i)
    } else {
      oldHashMap.set(entry.hash, [i])
    }
  }

  const matchedOld = new Set<number>()
  const out: ChunkDelta[] = []

  for (let i = 0; i < newChunks.length; i++) {
    const chunk = newChunks[i]
    if (chunk === undefined) continue
    const h = newHashes[i]
    if (h === undefined) continue
    const oldIndices = oldHashMap.get(h)
    let matched = false
    if (oldIndices !== undefined) {
      for (const oldIdx of oldIndices) {
        if (!matchedOld.has(oldIdx)) {
          matchedOld.add(oldIdx)
          matched = true
          break
        }
      }
    }
    const category: DeltaCategory = matched ? 'unchanged' : 'new'
    out.push({ chunk, category, hash: h })
  }

  // Identify removed chunks: old entries with no match in new set.
  for (let i = 0; i < manifest.chunks.length; i++) {
    if (matchedOld.has(i)) continue
    const entry = manifest.chunks[i]
    if (entry === undefined) continue
    const syntheticChunk: Chunk = {
      content: '',
      ordinal: i,
      headingPath: [],
      startLine: 0,
      endLine: 0,
      tokens: 0,
    }
    out.push({ chunk: syntheticChunk, category: 'removed', hash: entry.hash })
  }

  return out
}

/**
 * Constructs a ChunkManifest from a set of chunks and their parent
 * document hash.
 */
export const buildChunkManifest = (
  documentHash: string,
  chunks: readonly Chunk[],
): ChunkManifest => {
  const entries: ChunkManifestEntry[] = []
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]
    if (chunk === undefined) continue
    entries.push({
      hash: hashChunk(chunk.content),
      chunkId: `${documentHash}_${String(i)}`,
    })
  }
  return {
    documentHash,
    generation: 0,
    chunks: entries,
    updatedAt: '',
  }
}

const manifestPath = (documentHash: string): string => {
  validateDocumentHash(documentHash)
  return `${CHUNK_MANIFESTS_PREFIX}/${documentHash}.json`
}

/**
 * Loads the chunk manifest for a document from the store. Returns
 * undefined when the manifest does not exist (first ingest scenario).
 */
/**
 * Validates that a parsed JSON value has the required shape of a
 * ChunkManifest. Returns undefined if invalid (treats as missing
 * manifest, triggering first-ingest behaviour).
 */
const validateChunkManifest = (parsed: unknown): ChunkManifest | undefined => {
  if (parsed === null || typeof parsed !== 'object') return undefined
  const obj = parsed as { documentHash?: unknown; generation?: unknown; chunks?: unknown; updatedAt?: unknown }
  if (typeof obj.documentHash !== 'string') return undefined
  if (typeof obj.generation !== 'number') return undefined
  if (!Array.isArray(obj.chunks)) return undefined
  return {
    documentHash: obj.documentHash,
    generation: obj.generation,
    chunks: obj.chunks as readonly ChunkManifestEntry[],
    updatedAt: typeof obj.updatedAt === 'string' ? obj.updatedAt : '',
  }
}

export const readChunkManifest = async (
  store: Store,
  documentHash: string,
): Promise<ChunkManifest | undefined> => {
  const p = toPath(manifestPath(documentHash))
  try {
    const data = await store.read(p)
    const parsed: unknown = JSON.parse(data.toString('utf8'))
    return validateChunkManifest(parsed)
  } catch (err: unknown) {
    if (isNotFound(err)) {
      return undefined
    }
    throw err
  }
}

/**
 * Persists the chunk manifest for a document. Reads the prior manifest
 * to increment the generation counter. When no prior manifest exists,
 * generation starts at 1.
 */
export const writeChunkManifest = async (
  store: Store,
  manifest: ChunkManifest,
): Promise<void> => {
  const p = toPath(manifestPath(manifest.documentHash))
  const prior = await readChunkManifest(store, manifest.documentHash)
  const generation = prior !== undefined ? prior.generation + 1 : 1
  const updatedAt = new Date().toISOString()
  const toWrite: ChunkManifest = {
    ...manifest,
    generation,
    updatedAt,
  }
  const data = Buffer.from(JSON.stringify(toWrite), 'utf8')
  await store.write(p, data)
}
