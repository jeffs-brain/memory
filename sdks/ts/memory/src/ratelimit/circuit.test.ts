// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { CircuitHalfOpenLimitError, CircuitOpenError, createCircuitBreaker } from './circuit.js'

describe('CircuitBreaker', () => {
  it('starts in closed state', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 3,
      resetTimeoutMs: 1000,
      halfOpenMaxAttempts: 2,
      name: 'test',
    })

    expect(cb.state()).toBe('closed')
  })

  it('allows requests when closed', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 3,
      resetTimeoutMs: 1000,
      halfOpenMaxAttempts: 2,
      name: 'test',
    })

    expect(() => cb.allow()).not.toThrow()
    cb.recordSuccess()
  })

  it('opens after failure threshold is reached', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 3,
      resetTimeoutMs: 60_000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    for (let i = 0; i < 3; i++) {
      cb.allow()
      cb.recordFailure()
    }

    expect(() => cb.allow()).toThrow(CircuitOpenError)
  })

  it('rejects when open and tracks rejected count', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 60_000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()

    expect(() => cb.allow()).toThrow(CircuitOpenError)

    const m = cb.metrics()
    expect(m.totalRejected).toBe(1)
  })

  it('transitions to half-open after reset timeout', async () => {
    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 10,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()

    await new Promise((resolve) => setTimeout(resolve, 20))

    expect(cb.state()).toBe('half_open')
    expect(() => cb.allow()).not.toThrow()
  })

  it('closes after all half-open probes succeed', async () => {
    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 10,
      halfOpenMaxAttempts: 2,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()

    await new Promise((resolve) => setTimeout(resolve, 20))

    cb.allow()
    cb.recordSuccess()
    cb.allow()
    cb.recordSuccess()

    expect(cb.state()).toBe('closed')
  })

  it('reopens on half-open failure', async () => {
    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 10,
      halfOpenMaxAttempts: 3,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()

    await new Promise((resolve) => setTimeout(resolve, 20))

    cb.allow()
    cb.recordSuccess()

    cb.allow()
    cb.recordFailure()

    expect(() => cb.allow()).toThrow(CircuitOpenError)
  })

  it('limits probes in half-open state', async () => {
    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 10,
      halfOpenMaxAttempts: 2,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()

    await new Promise((resolve) => setTimeout(resolve, 20))

    cb.allow()
    cb.allow()

    expect(() => cb.allow()).toThrow(CircuitHalfOpenLimitError)
  })

  it('success resets consecutive failure count', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 3,
      resetTimeoutMs: 60_000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()
    cb.allow()
    cb.recordFailure()
    cb.allow()
    cb.recordSuccess()

    // Two more failures should not trip (counter was reset).
    cb.allow()
    cb.recordFailure()
    cb.allow()
    cb.recordFailure()

    expect(cb.state()).toBe('closed')
  })

  it('execute wraps function with circuit breaker', async () => {
    const cb = createCircuitBreaker({
      failureThreshold: 2,
      resetTimeoutMs: 60_000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    const result = await cb.execute(async () => 42)
    expect(result).toBe(42)

    const err = new Error('downstream error')
    await expect(cb.execute(async () => { throw err })).rejects.toThrow('downstream error')
    await expect(cb.execute(async () => { throw err })).rejects.toThrow('downstream error')

    await expect(cb.execute(async () => 0)).rejects.toThrow(CircuitOpenError)
  })

  it('reset forces circuit to closed', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 60_000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    cb.allow()
    cb.recordFailure()

    cb.reset()

    expect(cb.state()).toBe('closed')
    expect(() => cb.allow()).not.toThrow()
    cb.recordSuccess()
  })

  it('metrics returns accurate snapshot', () => {
    const cb = createCircuitBreaker({
      failureThreshold: 2,
      resetTimeoutMs: 60_000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })

    cb.allow()
    cb.recordSuccess()
    cb.allow()
    cb.recordFailure()
    cb.allow()
    cb.recordFailure()

    try { cb.allow() } catch { /* expected */ }

    const m = cb.metrics()
    expect(m.totalSuccesses).toBe(1)
    expect(m.totalFailures).toBe(2)
    expect(m.totalRejected).toBe(1)
    expect(m.consecutiveFailures).toBe(2)
    expect(m.state).toBe('open')
    expect(m.lastFailureAt).toBeInstanceOf(Date)
    expect(m.lastSuccessAt).toBeInstanceOf(Date)
  })

  it('onStateChange fires on transitions', async () => {
    const transitions: string[] = []

    const cb = createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 10,
      halfOpenMaxAttempts: 1,
      name: 'test-cb',
      onStateChange: (_name, from, to) => {
        transitions.push(`${from}->${to}`)
      },
    })

    cb.allow()
    cb.recordFailure()

    await new Promise((resolve) => setTimeout(resolve, 20))

    cb.allow()
    cb.recordSuccess()

    expect(transitions).toEqual([
      'closed->open',
      'open->half_open',
      'half_open->closed',
    ])
  })

  it('throws on invalid options', () => {
    expect(() => createCircuitBreaker({
      failureThreshold: 0,
      resetTimeoutMs: 1000,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })).toThrow('failureThreshold')

    expect(() => createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 0,
      halfOpenMaxAttempts: 1,
      name: 'test',
    })).toThrow('resetTimeoutMs')

    expect(() => createCircuitBreaker({
      failureThreshold: 1,
      resetTimeoutMs: 1000,
      halfOpenMaxAttempts: 0,
      name: 'test',
    })).toThrow('halfOpenMaxAttempts')
  })
})
