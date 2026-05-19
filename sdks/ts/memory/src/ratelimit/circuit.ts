// SPDX-License-Identifier: Apache-2.0

/**
 * Circuit breaker implementation following the Netflix Hystrix pattern.
 * States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing recovery).
 *
 * Thread-safe in the single-threaded JS sense: no concurrent mutation
 * issues, but all state transitions are atomic within a single tick.
 */

import type { Logger } from '../llm/types.js'
import { noopLogger } from '../llm/types.js'

/** Circuit breaker states. */
export type CircuitState = 'closed' | 'open' | 'half_open'

/** Configuration for a circuit breaker instance. */
export type CircuitBreakerOptions = {
  /** Number of consecutive failures that trips the circuit. Must be >= 1. */
  readonly failureThreshold: number
  /** Duration in ms the circuit remains open before transitioning to half-open. */
  readonly resetTimeoutMs: number
  /** Max probe requests allowed in half-open state. Must be >= 1. */
  readonly halfOpenMaxAttempts: number
  /** Identifier for logging and metrics. */
  readonly name: string
  /** Optional logger. Defaults to noop. */
  readonly logger?: Logger
  /** Optional callback on state transitions. */
  readonly onStateChange?: (name: string, from: CircuitState, to: CircuitState) => void
}

/** Point-in-time snapshot of circuit breaker state. */
export type CircuitBreakerMetrics = {
  readonly state: CircuitState
  readonly consecutiveFailures: number
  readonly totalSuccesses: number
  readonly totalFailures: number
  readonly totalRejected: number
  readonly lastFailureAt: Date | undefined
  readonly lastSuccessAt: Date | undefined
}

/** Sentinel errors. */
export class CircuitOpenError extends Error {
  constructor(name: string) {
    super(`circuit breaker "${name}" is open`)
    this.name = 'CircuitOpenError'
  }
}

export class CircuitHalfOpenLimitError extends Error {
  constructor(name: string) {
    super(`circuit breaker "${name}" half-open probe limit reached`)
    this.name = 'CircuitHalfOpenLimitError'
  }
}

/** Circuit breaker interface. */
export type CircuitBreaker = {
  /** Check if a request is allowed. Throws CircuitOpenError or CircuitHalfOpenLimitError if rejected. */
  allow(): void
  /** Record a successful operation. */
  recordSuccess(): void
  /** Record a failed operation. */
  recordFailure(): void
  /** Execute a function through the circuit breaker. */
  execute<T>(fn: () => Promise<T>): Promise<T>
  /** Get current state. */
  state(): CircuitState
  /** Get metrics snapshot. */
  metrics(): CircuitBreakerMetrics
  /** Force reset to closed state. */
  reset(): void
}

/**
 * Create a circuit breaker with the given options.
 *
 * @throws {Error} if failureThreshold < 1, resetTimeoutMs <= 0, or halfOpenMaxAttempts < 1
 */
export const createCircuitBreaker = (opts: CircuitBreakerOptions): CircuitBreaker => {
  if (opts.failureThreshold < 1) {
    throw new Error(`failureThreshold must be >= 1, got ${opts.failureThreshold}`)
  }
  if (opts.resetTimeoutMs <= 0) {
    throw new Error(`resetTimeoutMs must be > 0, got ${opts.resetTimeoutMs}`)
  }
  if (opts.halfOpenMaxAttempts < 1) {
    throw new Error(`halfOpenMaxAttempts must be >= 1, got ${opts.halfOpenMaxAttempts}`)
  }

  const logger = opts.logger ?? noopLogger

  let currentState: CircuitState = 'closed'
  let consecutiveFailures = 0
  let halfOpenSuccesses = 0
  let halfOpenAttempts = 0
  let openedAt = 0

  let totalSuccesses = 0
  let totalFailures = 0
  let totalRejected = 0
  let lastFailureAt: Date | undefined
  let lastSuccessAt: Date | undefined

  const transitionTo = (to: CircuitState): void => {
    const from = currentState
    if (from === to) return
    currentState = to
    logger.info('circuit breaker state change', {
      name: opts.name,
      from,
      to,
    })
    opts.onStateChange?.(opts.name, from, to)
  }

  const allow = (): void => {
    switch (currentState) {
      case 'closed':
        return

      case 'open': {
        if (Date.now() - openedAt >= opts.resetTimeoutMs) {
          transitionTo('half_open')
          halfOpenSuccesses = 0
          halfOpenAttempts = 0
          halfOpenAttempts++
          return
        }
        totalRejected++
        throw new CircuitOpenError(opts.name)
      }

      case 'half_open': {
        if (halfOpenAttempts >= opts.halfOpenMaxAttempts) {
          totalRejected++
          throw new CircuitHalfOpenLimitError(opts.name)
        }
        halfOpenAttempts++
        return
      }
    }
  }

  const recordSuccess = (): void => {
    totalSuccesses++
    lastSuccessAt = new Date()
    consecutiveFailures = 0

    if (currentState === 'half_open') {
      halfOpenSuccesses++
      if (halfOpenSuccesses >= opts.halfOpenMaxAttempts) {
        transitionTo('closed')
      }
    }
  }

  const recordFailure = (): void => {
    totalFailures++
    consecutiveFailures++
    lastFailureAt = new Date()

    switch (currentState) {
      case 'closed':
        if (consecutiveFailures >= opts.failureThreshold) {
          openedAt = Date.now()
          transitionTo('open')
        }
        break

      case 'half_open':
        openedAt = Date.now()
        transitionTo('open')
        break
    }
  }

  const state = (): CircuitState => {
    if (currentState === 'open' && Date.now() - openedAt >= opts.resetTimeoutMs) {
      return 'half_open'
    }
    return currentState
  }

  const metricsSnapshot = (): CircuitBreakerMetrics => ({
    state: currentState,
    consecutiveFailures,
    totalSuccesses,
    totalFailures,
    totalRejected,
    lastFailureAt,
    lastSuccessAt,
  })

  const reset = (): void => {
    consecutiveFailures = 0
    halfOpenSuccesses = 0
    halfOpenAttempts = 0
    transitionTo('closed')
  }

  const execute = async <T>(fn: () => Promise<T>): Promise<T> => {
    allow()
    try {
      const result = await fn()
      recordSuccess()
      return result
    } catch (err) {
      recordFailure()
      throw err
    }
  }

  return {
    allow,
    recordSuccess,
    recordFailure,
    execute,
    state,
    metrics: metricsSnapshot,
    reset,
  }
}
