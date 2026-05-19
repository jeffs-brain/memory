// SPDX-License-Identifier: Apache-2.0

/**
 * Pure helper functions for the Google Drive connector. Extracted to
 * keep the main connector file under the 700-line limit.
 */

import type { GDriveConnectorConfig } from './gdrive.js'

// -- Constants ----------------------------------------------------------------

/** Maximum response body length for error message truncation. */
const MAX_ERROR_BODY_LENGTH = 200

/** Maximum number of retry attempts on rate-limit errors. */
export const RATE_LIMIT_MAX_RETRIES = 5

/** Base backoff duration in milliseconds for rate-limit retries. */
export const RATE_LIMIT_BASE_BACKOFF_MS = 1_000

/** Maximum backoff duration in milliseconds for rate-limit retries. */
export const RATE_LIMIT_MAX_BACKOFF_MS = 60_000

/** Pattern that valid Google Drive folder IDs must match. */
const FOLDER_ID_PATTERN = /^[a-zA-Z0-9_-]+$/

/** Pattern that valid MIME types must match. */
const MIME_TYPE_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9!#$&\-.^_+]+\/[a-zA-Z0-9][a-zA-Z0-9!#$&\-.^_+]+$/

const MIME_GOOGLE_FOLDER = 'application/vnd.google-apps.folder'

// -- Types --------------------------------------------------------------------

type DriveAPIError = {
  readonly error: {
    readonly message: string
    readonly errors?: readonly { readonly reason: string }[] | undefined
  }
}

type DriveFile = {
  readonly id: string
  readonly name: string
  readonly mimeType: string
  readonly modifiedTime: string
  readonly size?: string | undefined
  readonly parents?: readonly string[] | undefined
  readonly webViewLink?: string | undefined
}

export type RateLimitParseResult = {
  readonly isRateLimit: boolean
  readonly message: string
}

// -- Functions ----------------------------------------------------------------

export const parseFileSize = (sizeStr: string | undefined): number => {
  if (sizeStr === undefined || sizeStr === '') return 0
  const parsed = Number(sizeStr)
  return Number.isNaN(parsed) ? 0 : parsed
}

export const buildFileQuery = (cfg: GDriveConnectorConfig): string => {
  const parts: string[] = []

  parts.push(`mimeType != '${MIME_GOOGLE_FOLDER}'`)

  if (cfg.folderId) {
    validateFolderId(cfg.folderId)
    parts.push(`'${cfg.folderId}' in parents`)
  }

  if (cfg.mimeTypeFilter !== undefined && cfg.mimeTypeFilter.length > 0) {
    const mimeConditions = cfg.mimeTypeFilter.map((mime) => {
      validateMimeType(mime)
      return `mimeType = '${mime}'`
    })
    parts.push(`(${mimeConditions.join(' or ')})`)
  }

  parts.push('trashed = false')

  return parts.join(' and ')
}

/**
 * Validate that a folder ID matches the expected pattern to prevent
 * query injection via the Drive API query string.
 */
const validateFolderId = (folderId: string): void => {
  if (!FOLDER_ID_PATTERN.test(folderId)) {
    throw new Error(`gdrive: invalid folder ID "${folderId}" — must be alphanumeric with hyphens/underscores`)
  }
}

/**
 * Validate that a MIME type matches the expected format to prevent
 * query injection via the Drive API query string.
 */
const validateMimeType = (mimeType: string): void => {
  if (!MIME_TYPE_PATTERN.test(mimeType)) {
    throw new Error(`gdrive: invalid MIME type "${mimeType}"`)
  }
}

export const buildFileMetadata = (file: DriveFile): Readonly<Record<string, string>> => {
  const meta: Record<string, string> = {
    source: 'gdrive',
    mime_type: file.mimeType,
    file_id: file.id,
  }
  if (file.parents !== undefined && file.parents.length > 0) {
    const firstParent = file.parents[0]
    if (firstParent !== undefined) {
      meta['parent_id'] = firstParent
    }
  }
  if (file.size !== undefined && file.size !== '') {
    meta['size'] = file.size
  }
  return meta
}

export const truncateBody = (body: string): string => {
  if (body.length <= MAX_ERROR_BODY_LENGTH) return body
  return body.slice(0, MAX_ERROR_BODY_LENGTH) + '...'
}

/**
 * Parse a rate limit error response body and determine whether it is
 * a retryable rate limit error.
 */
export const parseRateLimitError = (body: string, statusCode: number): RateLimitParseResult => {
  try {
    const apiErr = JSON.parse(body) as DriveAPIError
    const reasons = apiErr.error.errors ?? []

    const rateLimitReasons: ReadonlySet<string> = new Set([
      'rateLimitExceeded',
      'userRateLimitExceeded',
    ])

    for (const entry of reasons) {
      if (rateLimitReasons.has(entry.reason)) {
        return {
          isRateLimit: true,
          message: `gdrive: rate limit exceeded: ${apiErr.error.message}`,
        }
      }
    }

    return {
      isRateLimit: statusCode === 429,
      message: `gdrive: API error (status ${statusCode}): ${apiErr.error.message}`,
    }
  } catch {
    return {
      isRateLimit: statusCode === 429,
      message: `gdrive: rate limit error (status ${statusCode}): ${truncateBody(body)}`,
    }
  }
}

/**
 * Calculate backoff duration with exponential increase and jitter.
 * Respects the Retry-After header when present.
 */
export const calculateBackoff = (attempt: number, retryAfterHeader: string | null | undefined): number => {
  if (retryAfterHeader !== null && retryAfterHeader !== undefined) {
    const retryAfterSeconds = Number(retryAfterHeader)
    if (!Number.isNaN(retryAfterSeconds) && retryAfterSeconds > 0) {
      return Math.min(retryAfterSeconds * 1_000, RATE_LIMIT_MAX_BACKOFF_MS)
    }
  }

  const exponential = RATE_LIMIT_BASE_BACKOFF_MS * Math.pow(2, attempt)
  const jitter = Math.random() * RATE_LIMIT_BASE_BACKOFF_MS
  return Math.min(exponential + jitter, RATE_LIMIT_MAX_BACKOFF_MS)
}

/**
 * Read a response body with a size limit to prevent unbounded memory
 * allocation. Reads incrementally and aborts if the limit is exceeded.
 */
export const readResponseWithLimit = async (resp: Response, maxBytes: number): Promise<string> => {
  const reader = resp.body?.getReader()
  if (reader === undefined) {
    return resp.text()
  }

  const chunks: Uint8Array[] = []
  let totalBytes = 0
  const decoder = new TextDecoder()

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break

    totalBytes += value.byteLength
    if (totalBytes > maxBytes) {
      reader.cancel().catch(() => {})
      throw new Error(
        `gdrive: response body exceeds ${maxBytes} byte limit (received ${totalBytes} bytes so far)`,
      )
    }
    chunks.push(value)
  }

  const combined = new Uint8Array(totalBytes)
  let offset = 0
  for (const chunk of chunks) {
    combined.set(chunk, offset)
    offset += chunk.byteLength
  }

  return decoder.decode(combined)
}

/**
 * Strip access tokens and authorisation headers from error messages
 * to prevent credential leakage through error wrapping.
 */
export const sanitiseError = (err: unknown): Error => {
  if (!(err instanceof Error)) {
    return new Error(`gdrive: unknown error: ${String(err)}`)
  }

  const bearerPattern = /Bearer\s+[A-Za-z0-9\-._~+/]+=*/g
  const sanitised = err.message.replace(bearerPattern, 'Bearer [REDACTED]')

  if (sanitised !== err.message) {
    return new Error(sanitised)
  }
  return err
}

export const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))
