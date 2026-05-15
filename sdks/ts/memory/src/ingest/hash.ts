// SPDX-License-Identifier: Apache-2.0

/**
 * BLAKE3 and SHA-256 hashing utilities for the ingest pipeline.
 * BLAKE3 is used for all new documents. SHA-256 is retained for
 * dual-read compatibility during the migration period.
 *
 * Provides deterministic 256-bit BLAKE3 hashes hex-encoded to 64-character
 * strings. Used for document deduplication and chunk-level change detection.
 * Cross-SDK conformant: identical inputs produce identical outputs in both
 * the Go and TypeScript SDKs.
 */

import { createHash } from 'node:crypto'
import { blake3 } from '@noble/hashes/blake3.js'
import { bytesToHex } from '@noble/hashes/utils.js'

/**
 * Compute the BLAKE3 hash of a buffer. Returns the full hex digest
 * (64 characters).
 */
export const hashContent = (buf: Buffer): string => bytesToHex(blake3(buf))

/**
 * Compute the SHA-256 hash of a buffer. Returns the full hex digest.
 * Used for dual-read fallback during the BLAKE3 migration period.
 */
export const hashContentSHA256 = (buf: Buffer): string =>
  createHash('sha256').update(buf).digest('hex')

/**
 * Compute a truncated SHA-256 hash (12 hex chars). Matches the legacy
 * hashSlug used prior to the BLAKE3 migration.
 */
export const hashSlugSHA256 = (data: Buffer): string => hashContentSHA256(data).slice(0, 12)

/**
 * Hasher abstracts a content-hashing algorithm so callers can swap
 * implementations (BLAKE3, SHA-256, etc.) without changing call sites.
 */
export type Hasher = {
  /** Compute a hex-encoded digest of content. */
  hash(content: Buffer): string
  /** Short identifier for the algorithm (e.g. "blake3"). */
  name: string
}

/**
 * Default BLAKE3 Hasher implementation. 256-bit output, hex-encoded to
 * 64 lowercase characters.
 */
export const blake3Hasher: Hasher = {
  hash(content: Buffer): string {
    const digest = blake3(content)
    return bytesToHex(digest)
  },
  name: 'blake3',
}

/**
 * Compute the BLAKE3 256-bit hash of document content.
 * Returns a 64-character lowercase hex string.
 *
 * Time: O(n) where n = content.length.
 * Space: O(1) beyond the 32-byte digest.
 */
export const hashDocument = (content: Buffer): string => {
  const digest = blake3(content)
  return bytesToHex(digest)
}

/**
 * Compute the BLAKE3 256-bit hash of chunk content.
 * Semantically identical to hashDocument but named separately for clarity
 * at call sites performing chunk-level change detection.
 *
 * Time: O(n) where n = content.length.
 * Space: O(1) beyond the 32-byte digest.
 */
export const hashChunk = (content: Buffer): string => {
  const digest = blake3(content)
  return bytesToHex(digest)
}

/**
 * Compute the BLAKE3 256-bit hash of a string input.
 * Convenience wrapper that encodes the string as UTF-8 before hashing.
 * Returns a 64-character lowercase hex string.
 *
 * Time: O(n) where n = byte length of the UTF-8 encoding.
 * Space: O(n) for the intermediate buffer.
 */
export const hashString = (s: string): string => {
  const buf = Buffer.from(s, 'utf8')
  const digest = blake3(buf)
  return bytesToHex(digest)
}

/**
 * Compute a truncated BLAKE3 hash (12 hex chars) suitable for use as
 * a slug fallback. Matches the Go SDK's HashSlug function.
 *
 * Time: O(n) where n = data.length.
 * Space: O(1) beyond the 32-byte digest.
 */
export const hashSlug = (data: Buffer): string => hashDocument(data).slice(0, 12)

/**
 * Compute a stable document identifier from brain ID and content hash.
 * Returns 16 hex characters of BLAKE3(brainId + ":" + contentHash).
 * Matches the Go SDK's HashDocumentID function.
 *
 * Time: O(n) where n = brainId.length + contentHash.length.
 * Space: O(n) for the intermediate buffer.
 */
export const hashDocumentId = (brainId: string, contentHash: string): string => {
  const input = Buffer.from(`${brainId}:${contentHash}`, 'utf8')
  return hashDocument(input).slice(0, 16)
}
