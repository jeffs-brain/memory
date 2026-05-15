// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { isValid, nextOccurrence, parseCron } from './cron.js'

describe('parseCron', () => {
  it('parses "0 * * * *" -> every hour at minute 0', () => {
    const sched = parseCron('0 * * * *')
    expect(sched.minute).toEqual([0])
    expect(sched.hour).toHaveLength(24)
  })

  it('parses "30 2 * * 1" -> 2:30 AM every Monday', () => {
    const sched = parseCron('30 2 * * 1')
    expect(sched.minute).toEqual([30])
    expect(sched.hour).toEqual([2])
    expect(sched.dayOfWeek).toEqual([1])
  })

  it('parses "*/5 * * * *" -> every 5 minutes', () => {
    const sched = parseCron('*/5 * * * *')
    expect(sched.minute).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
  })

  it('parses ranges "0 9-17 * * 1-5"', () => {
    const sched = parseCron('0 9-17 * * 1-5')
    expect(sched.hour).toEqual([9, 10, 11, 12, 13, 14, 15, 16, 17])
    expect(sched.dayOfWeek).toEqual([1, 2, 3, 4, 5])
  })

  it('throws for invalid expressions', () => {
    expect(() => parseCron('')).toThrow()
    expect(() => parseCron('* * *')).toThrow()
    expect(() => parseCron('60 * * * *')).toThrow()
    expect(() => parseCron('* 25 * * *')).toThrow()
    expect(() => parseCron('* * 32 * *')).toThrow()
    expect(() => parseCron('* * * 13 *')).toThrow()
    expect(() => parseCron('* * * * 7')).toThrow()
    expect(() => parseCron('abc * * * *')).toThrow()
  })
})

describe('isValid', () => {
  it('returns true for valid expressions', () => {
    expect(isValid('0 * * * *')).toBe(true)
    expect(isValid('*/5 * * * *')).toBe(true)
  })

  it('returns false for invalid expressions', () => {
    expect(isValid('invalid')).toBe(false)
    expect(isValid('')).toBe(false)
  })
})

describe('parseField deduplication', () => {
  it('deduplicates overlapping list+range "1,1-3"', () => {
    const sched = parseCron('0 1,1-3 * * *')
    expect(sched.hour).toEqual([1, 2, 3])
  })
})

describe('nextOccurrence', () => {
  it('computes next hour for "0 * * * *"', () => {
    const sched = parseCron('0 * * * *')
    const ref = new Date('2026-05-15T10:30:00Z')
    const next = nextOccurrence(sched, ref)
    expect(next.getUTCHours()).toBe(11)
    expect(next.getUTCMinutes()).toBe(0)
  })

  it('returns next hour when at exact minute 0', () => {
    const sched = parseCron('0 * * * *')
    const ref = new Date('2026-05-15T10:00:00Z')
    const next = nextOccurrence(sched, ref)
    expect(next.getUTCHours()).toBe(11)
    expect(next.getUTCMinutes()).toBe(0)
  })

  it('computes next 5-minute mark', () => {
    const sched = parseCron('*/5 * * * *')
    const ref = new Date('2026-05-15T10:12:00Z')
    const next = nextOccurrence(sched, ref)
    expect(next.getUTCHours()).toBe(10)
    expect(next.getUTCMinutes()).toBe(15)
  })

  it('computes next Monday for "30 2 * * 1"', () => {
    const sched = parseCron('30 2 * * 1')
    // Thursday May 15, 2025.
    const ref = new Date('2025-05-15T10:00:00Z')
    const next = nextOccurrence(sched, ref)
    expect(next.getUTCDay()).toBe(1) // Monday
    expect(next.getUTCHours()).toBe(2)
    expect(next.getUTCMinutes()).toBe(30)
  })

  it('uses DOM+DOW union semantics when both are non-wildcard', () => {
    // "0 9 15 * 1" = at 9:00 on the 15th OR on Mondays
    const sched = parseCron('0 9 15 * 1')
    // Wednesday May 14, 2025 at 10:00 -> next should be May 15 (DOM match)
    const ref = new Date('2025-05-14T10:00:00Z')
    const next = nextOccurrence(sched, ref)
    expect(next.getUTCDate()).toBe(15) // DOM match (Thursday, not Monday)
    expect(next.getUTCHours()).toBe(9)
    expect(next.getUTCMinutes()).toBe(0)
  })
})
