// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import { CliUsageError } from '../config.js'
import { assertBindAllowed, parseAddr } from './serve.js'

describe('parseAddr', () => {
  it.each([
    [':8080', '127.0.0.1', 8080],
    ['127.0.0.1:18841', '127.0.0.1', 18841],
    ['0.0.0.0:9000', '0.0.0.0', 9000],
    ['[::1]:8080', '::1', 8080],
  ])('%s -> %s:%d', (addr, hostname, port) => {
    expect(parseAddr(addr)).toEqual({ hostname, port })
  })
})

describe('assertBindAllowed', () => {
  it.each(['127.0.0.1', 'localhost', '::1'])('allows loopback %s without a token', (host) => {
    expect(() => assertBindAllowed(host, undefined)).not.toThrow()
  })

  it.each(['0.0.0.0', '::', '192.0.2.10', 'memory.internal'])(
    'refuses %s without a token',
    (host) => {
      expect(() => assertBindAllowed(host, undefined)).toThrow(CliUsageError)
      expect(() => assertBindAllowed(host, '')).toThrow(CliUsageError)
    },
  )

  it('allows any host once a token is set', () => {
    expect(() => assertBindAllowed('0.0.0.0', 'secret')).not.toThrow()
  })
})
