// SPDX-License-Identifier: Apache-2.0

/**
 * Public surface of the memory HTTP daemon. Designed so both the
 * `memory serve` CLI and in-process tests can wire one up with a
 * single factory call.
 */

export {
  Daemon,
  BrainManager,
  BrainNotFoundError,
  BrainConflictError,
  defaultRoot,
} from './daemon.js'
export type { BrainResources, DaemonConfig } from './daemon.js'
export { createRouter, type Handler } from './router.js'
export {
  problemResponse,
  notFound,
  validationError,
  payloadTooLarge,
  unauthorized,
  forbidden,
  conflict,
  confirmationRequired,
  internalError,
  storeProblem,
  jsonResponse,
  type Problem,
} from './problem.js'
export { startSse } from './sse.js'
export type { SseSession, SseWriter } from './sse.js'
