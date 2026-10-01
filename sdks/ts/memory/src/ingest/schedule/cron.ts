// SPDX-License-Identifier: Apache-2.0

/**
 * Minimal cron expression parser supporting standard 5-field syntax:
 * minute, hour, day-of-month, month, day-of-week.
 *
 * Supports: numbers, ranges (1-5), steps (* /5), lists (1,3,5), and *.
 */

export type CronSchedule = {
  readonly minute: readonly number[]
  readonly hour: readonly number[]
  readonly dayOfMonth: readonly number[]
  readonly month: readonly number[]
  readonly dayOfWeek: readonly number[]
  /** True when the original day-of-month token was '*' (wildcard). */
  readonly dayOfMonthIsWild: boolean
  /** True when the original day-of-week token was '*' (wildcard). */
  readonly dayOfWeekIsWild: boolean
}

export const parseCron = (expression: string): CronSchedule => {
  const fields = expression.trim().split(/\s+/)
  if (fields.length !== 5) {
    throw new Error(`cron: expected 5 fields, got ${fields.length} in "${expression}"`)
  }

  const [minuteField, hourField, domField, monthField, dowField] = fields as [
    string,
    string,
    string,
    string,
    string,
  ]

  return {
    minute: parseField(minuteField, 0, 59, 'minute'),
    hour: parseField(hourField, 0, 23, 'hour'),
    dayOfMonth: parseField(domField, 1, 31, 'day-of-month'),
    month: parseField(monthField, 1, 12, 'month'),
    dayOfWeek: parseField(dowField, 0, 6, 'day-of-week'),
    dayOfMonthIsWild: isWildcard(domField),
    dayOfWeekIsWild: isWildcard(dowField),
  }
}

// Only a bare '*' is a true wildcard for DOM/DOW union semantics.
// Step forms like '*/N' restrict to every-Nth value, so not wildcards.
const isWildcard = (token: string): boolean => {
  return token.trim() === '*'
}

export const isValid = (expression: string): boolean => {
  try {
    parseCron(expression)
    return true
  } catch {
    return false
  }
}

export type NextOccurrenceOptions = {
  /**
   * IANA time zone the schedule's fields are written in, for example
   * `Europe/London`. Defaults to UTC.
   */
  readonly timeZone?: string
}

const MINUTE_MS = 60_000
const DAY_MS = 86_400_000

const formatters = new Map<string, Intl.DateTimeFormat>()

/** A formatter for `timeZone`, or an error naming the bad zone. */
const zoneFormatter = (timeZone: string): Intl.DateTimeFormat => {
  const cached = formatters.get(timeZone)
  if (cached !== undefined) return cached
  let formatter: Intl.DateTimeFormat
  try {
    formatter = new Intl.DateTimeFormat('en-GB', {
      timeZone,
      hourCycle: 'h23',
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
    })
  } catch {
    throw new Error(`cron: unknown time zone "${timeZone}"`)
  }
  formatters.set(timeZone, formatter)
  return formatter
}

/**
 * The zone's wall clock at `instant`, as a Date whose UTC fields hold
 * the local year, month, day, hour and minute.
 */
const toWallClock = (instant: number, formatter: Intl.DateTimeFormat): number => {
  const field = (parts: Intl.DateTimeFormatPart[], type: Intl.DateTimeFormatPartTypes): number =>
    Number(parts.find((part) => part.type === type)?.value)
  const parts = formatter.formatToParts(instant)
  return Date.UTC(
    field(parts, 'year'),
    field(parts, 'month') - 1,
    field(parts, 'day'),
    field(parts, 'hour'),
    field(parts, 'minute'),
  )
}

/**
 * Every instant whose wall clock is `wall`, earliest first: none inside
 * a spring-forward gap, two inside a fall-back overlap. Transitions are
 * months apart, so the offsets a day either side are the only candidates.
 */
const fromWallClock = (wall: number, formatter: Intl.DateTimeFormat): number[] => {
  const offsets = new Set(
    [wall - DAY_MS, wall + DAY_MS].map((probe) => toWallClock(probe, formatter) - probe),
  )
  return [...offsets]
    .map((offset) => wall - offset)
    .filter((instant) => toWallClock(instant, formatter) === wall)
    .sort((a, b) => a - b)
}

/**
 * Compute the next occurrence of the schedule strictly after `after`.
 * Searches up to 4 years ahead and returns `after` plus 4 years when
 * nothing matches.
 *
 * Fields are matched against the wall clock of `options.timeZone`, UTC
 * by default. In a zone with daylight saving, a time that does not exist
 * (inside the spring-forward gap) never matches, and a time that occurs
 * twice (inside the fall-back overlap) fires once, at its first instant
 * after `after`.
 *
 * DOM+DOW union semantics: per POSIX cron, when both day-of-month and
 * day-of-week are non-wildcard (not full-range), a date matches if
 * EITHER condition is true. When one or both are wildcard, standard
 * intersection (AND) applies.
 */
export const nextOccurrence = (
  sched: CronSchedule,
  after: Date,
  options: NextOccurrenceOptions = {},
): Date => {
  const formatter = options.timeZone !== undefined ? zoneFormatter(options.timeZone) : undefined
  const afterMs = after.getTime()

  const minuteSet = toSet(sched.minute)
  const hourSet = toSet(sched.hour)
  const domSet = toSet(sched.dayOfMonth)
  const monthSet = toSet(sched.month)
  const dowSet = toSet(sched.dayOfWeek)

  // Walk the wall clock with UTC arithmetic, so no step depends on the
  // host's time zone, starting at the minute after `after`.
  const wallAfter = formatter !== undefined ? toWallClock(afterMs, formatter) : afterMs
  const t = new Date(Math.floor(wallAfter / MINUTE_MS) * MINUTE_MS + MINUTE_MS)
  const limit = new Date(t)
  limit.setUTCFullYear(limit.getUTCFullYear() + 4)

  while (t < limit) {
    if (!monthSet.has(t.getUTCMonth() + 1)) {
      // Month and day together, so the 29th to 31st cannot overflow
      // into the month after next.
      t.setUTCMonth(t.getUTCMonth() + 1, 1)
      t.setUTCHours(0, 0, 0, 0)
      continue
    }
    if (
      !matchDay(
        domSet,
        dowSet,
        sched.dayOfMonthIsWild,
        sched.dayOfWeekIsWild,
        t.getUTCDate(),
        t.getUTCDay(),
      )
    ) {
      t.setUTCDate(t.getUTCDate() + 1)
      t.setUTCHours(0, 0, 0, 0)
      continue
    }
    if (!hourSet.has(t.getUTCHours())) {
      t.setUTCHours(t.getUTCHours() + 1, 0, 0, 0)
      continue
    }
    if (!minuteSet.has(t.getUTCMinutes())) {
      t.setUTCMinutes(t.getUTCMinutes() + 1, 0, 0)
      continue
    }
    if (formatter === undefined) return new Date(t)
    const instant = fromWallClock(t.getTime(), formatter).find((candidate) => candidate > afterMs)
    if (instant !== undefined) return new Date(instant)
    t.setUTCMinutes(t.getUTCMinutes() + 1, 0, 0)
  }

  const fallback = new Date(after)
  fallback.setUTCFullYear(fallback.getUTCFullYear() + 4)
  return fallback
}

/**
 * POSIX cron union semantics for day matching. When both DOM and DOW
 * are restricted (non-wildcard), match if EITHER is true. Otherwise
 * use standard AND logic.
 */
const matchDay = (
  domSet: Set<number>,
  dowSet: Set<number>,
  domIsWild: boolean,
  dowIsWild: boolean,
  day: number,
  weekday: number,
): boolean => {
  if (!domIsWild && !dowIsWild) {
    return domSet.has(day) || dowSet.has(weekday)
  }
  return domSet.has(day) && dowSet.has(weekday)
}

type CronRange = { readonly start: number; readonly end: number }

/**
 * Parses a single range token from a cron field. Handles three forms:
 * wildcard ('*'), explicit range ('1-5'), or single value ('3').
 */
const parseCronRange = (rangePart: string, min: number, max: number, name: string): CronRange => {
  if (rangePart === '*') {
    return { start: min, end: max }
  }

  if (rangePart.includes('-')) {
    const rangeSplit = rangePart.split('-')
    const s = rangeSplit[0] ?? ''
    const e = rangeSplit[1] ?? ''
    const start = Number.parseInt(s, 10)
    const end = Number.parseInt(e, 10)
    if (Number.isNaN(start) || Number.isNaN(end)) {
      throw new Error(`cron: ${name} field: invalid range "${rangePart}"`)
    }
    return { start, end }
  }

  const val = Number.parseInt(rangePart, 10)
  if (Number.isNaN(val)) {
    throw new Error(`cron: ${name} field: invalid value "${rangePart}"`)
  }
  return { start: val, end: val }
}

const parseField = (field: string, min: number, max: number, name: string): number[] => {
  const result: number[] = []

  for (const part of field.split(',')) {
    const trimmed = part.trim()
    const stepParts = trimmed.split('/')
    const rangePart = stepParts[0] ?? ''
    let step = 1

    if (stepParts.length === 2) {
      const stepStr = stepParts[1] ?? ''
      step = Number.parseInt(stepStr, 10)
      if (Number.isNaN(step) || step < 1) {
        throw new Error(`cron: ${name} field: invalid step "${stepStr}"`)
      }
    }

    const { start: rangeStart, end: rangeEnd } = parseCronRange(rangePart, min, max, name)

    if (rangeStart < min || rangeEnd > max || rangeStart > rangeEnd) {
      throw new Error(
        `cron: ${name} field: value out of range [${min}-${max}]: ${rangeStart}-${rangeEnd}`,
      )
    }

    for (let i = rangeStart; i <= rangeEnd; i += step) {
      result.push(i)
    }
  }

  if (result.length === 0) {
    throw new Error(`cron: ${name} field: empty`)
  }

  return [...new Set(result)]
}

const toSet = (values: readonly number[]): Set<number> => new Set(values)
