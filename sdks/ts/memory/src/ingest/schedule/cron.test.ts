// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
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

  it('does not skip a month when starting late in a long month', () => {
    const feb = nextOccurrence(parseCron('0 0 * 2 *'), new Date('2026-01-31T12:00:00Z'))
    expect(feb.toISOString()).toBe('2026-02-01T00:00:00.000Z')
    const monthly = nextOccurrence(parseCron('0 0 1 * *'), new Date('2026-05-31T12:00:00Z'))
    expect(monthly.toISOString()).toBe('2026-06-01T00:00:00.000Z')
    const leap = nextOccurrence(parseCron('0 0 29 2 *'), new Date('2026-03-01T00:00:00Z'))
    expect(leap.toISOString()).toBe('2028-02-29T00:00:00.000Z')
  })

  it('evaluates in UTC whatever the host time zone', () => {
    const sched = parseCron('30 2 * * 1')
    const ref = new Date('2025-05-15T10:00:00Z')
    try {
      const results = ['UTC', 'Pacific/Kiritimati', 'America/Los_Angeles'].map((tz) => {
        vi.stubEnv('TZ', tz)
        return nextOccurrence(sched, ref).toISOString()
      })
      expect(results).toEqual(Array(3).fill('2025-05-19T02:30:00.000Z'))
    } finally {
      vi.unstubAllEnvs()
    }
  })

  it('returns the fallback four years out when nothing can match', () => {
    const never = nextOccurrence(parseCron('0 0 30 2 *'), new Date('2026-01-01T00:00:00Z'))
    expect(never.toISOString()).toBe('2030-01-01T00:00:00.000Z')
  })
})

describe('nextOccurrence with a time zone', () => {
  const at = (expr: string, after: string, timeZone: string): string =>
    nextOccurrence(parseCron(expr), new Date(after), { timeZone }).toISOString()

  it('matches fields against the zone wall clock', () => {
    expect(at('0 9 * * *', '2026-07-01T00:00:00Z', 'Europe/London')).toBe(
      '2026-07-01T08:00:00.000Z',
    )
    expect(at('0 9 * * *', '2026-01-01T00:00:00Z', 'Europe/London')).toBe(
      '2026-01-01T09:00:00.000Z',
    )
    expect(at('0 9 * * *', '2026-01-01T00:00:00Z', 'Asia/Kolkata')).toBe('2026-01-01T03:30:00.000Z')
  })

  it('uses the zone date, not the UTC date, for day fields', () => {
    // 23:30 UTC on Sunday is already Monday morning in Tokyo.
    expect(at('0 * * * 1', '2026-06-07T23:30:00Z', 'Asia/Tokyo')).toBe('2026-06-08T00:00:00.000Z')
  })

  it('skips a time that falls in the spring-forward gap', () => {
    // 02:30 does not exist in New York on 8 March 2026.
    expect(at('30 2 * * *', '2026-03-08T05:00:00Z', 'America/New_York')).toBe(
      '2026-03-09T06:30:00.000Z',
    )
  })

  it('fires once for a time repeated by the fall-back overlap', () => {
    // 01:30 happens twice in New York on 1 November 2026.
    const first = at('30 1 * * *', '2026-11-01T04:00:00Z', 'America/New_York')
    expect(first).toBe('2026-11-01T05:30:00.000Z')
    expect(at('30 1 * * *', first, 'America/New_York')).toBe('2026-11-02T06:30:00.000Z')
  })

  it('stays strictly after the reference inside the overlap', () => {
    // 06:10 UTC is the second 01:10; the next 01:30 is the second one.
    expect(at('30 1 * * *', '2026-11-01T06:10:00Z', 'America/New_York')).toBe(
      '2026-11-01T06:30:00.000Z',
    )
  })

  it('rejects an unknown zone', () => {
    expect(() =>
      nextOccurrence(parseCron('* * * * *'), new Date(), { timeZone: 'Mars/Olympus' }),
    ).toThrow('cron: unknown time zone "Mars/Olympus"')
  })
})
