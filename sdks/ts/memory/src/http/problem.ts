// SPDX-License-Identifier: Apache-2.0

/**
 * Problem+JSON helpers for the memory daemon. Mirrors the Go
 * `go/internal/httpd.Problem` shape so cross-SDK conformance assertions
 * see the same body on every error.
 *
 * The wire contract is documented in `spec/PROTOCOL.md` under "Error
 * shape" and "Error codes". A `Problem` body always carries `status`,
 * `title`, and usually a `code` plus `detail`.
 */

import {
  ErrConflict,
  ErrInvalidPath,
  ErrNotFound,
  ErrReadOnly,
  StoreError,
} from '../store/errors.js'

export type Problem = {
  readonly status: number
  readonly title: string
  readonly code?: string
  readonly detail?: string
}

export const CONTENT_TYPE_PROBLEM = 'application/problem+json'

export const problemResponse = (problem: Problem): Response => {
  const body: Record<string, unknown> = {
    status: problem.status,
    title: problem.title,
  }
  if (problem.code !== undefined) body.code = problem.code
  if (problem.detail !== undefined) body.detail = problem.detail
  return new Response(JSON.stringify(body), {
    status: problem.status,
    headers: { 'content-type': CONTENT_TYPE_PROBLEM },
  })
}

export const notFound = (detail: string): Response =>
  problemResponse({ status: 404, title: 'Not Found', code: 'not_found', detail })

export const validationError = (detail: string): Response =>
  problemResponse({
    status: 400,
    title: 'Bad Request',
    code: 'validation_error',
    detail,
  })

export const payloadTooLarge = (detail: string): Response =>
  problemResponse({
    status: 413,
    title: 'Payload Too Large',
    code: 'payload_too_large',
    detail,
  })

export const unauthorized = (detail: string): Response =>
  problemResponse({
    status: 401,
    title: 'Unauthorized',
    code: 'unauthorized',
    detail,
  })

export const forbidden = (detail: string): Response =>
  problemResponse({
    status: 403,
    title: 'Forbidden',
    code: 'forbidden',
    detail,
  })

export const conflict = (detail: string): Response =>
  problemResponse({
    status: 409,
    title: 'Conflict',
    code: 'conflict',
    detail,
  })

export const confirmationRequired = (detail: string): Response =>
  problemResponse({
    status: 428,
    title: 'Confirmation Required',
    code: 'confirmation_required',
    detail,
  })

export const internalError = (detail: string): Response =>
  problemResponse({
    status: 500,
    title: 'Internal Server Error',
    code: 'internal_error',
    detail,
  })

/**
 * Map a store-level error onto the Problem+JSON shape documented in
 * `spec/PROTOCOL.md`. Returns `undefined` when the error is unknown so
 * callers can fall back to `internalError`.
 */
export const storeProblem = (err: unknown): Response | undefined => {
  if (err instanceof ErrNotFound) {
    return notFound(err.message)
  }
  if (err instanceof ErrInvalidPath) {
    return validationError(err.message)
  }
  if (err instanceof ErrConflict) {
    return conflict(err.message)
  }
  if (err instanceof ErrReadOnly) {
    return conflict(err.message)
  }
  if (err instanceof StoreError) {
    return internalError(err.message)
  }
  return undefined
}

/**
 * Convenience JSON helper matching the Go daemon's writeJSON.
 */
export const jsonResponse = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
