// SPDX-License-Identifier: Apache-2.0

/**
 * BLAKE3 and SHA-256 hashing utilities for the ingest pipeline.
 * BLAKE3 is used for all new documents. SHA-256 is retained for
 * dual-read compatibility during the migration period.
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
 * Compute a truncated BLAKE3 hash (12 hex chars) suitable for use as
 * a slug fallback.
 */
export const hashSlug = (data: Buffer): string => hashContent(data).slice(0, 12)

/**
 * Compute a stable document identifier from brain ID and content hash.
 * Returns 16 hex characters of BLAKE3(brainId + ":" + contentHash).
 */
export const hashDocumentId = (brainId: string, contentHash: string): string => {
  const combined = Buffer.from(`${brainId}:${contentHash}`, 'utf8')
  return hashContent(combined).slice(0, 16)
}

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
