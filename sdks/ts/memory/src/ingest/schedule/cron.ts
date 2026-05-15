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

  const [minuteField, hourField, domField, monthField, dowField] = fields as [string, string, string, string, string]

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

/**
 * Returns true when the raw cron field token represents a wildcard.
 * Matches bare '*' and step-only forms like '* /N' (without space).
 */
const isWildcard = (token: string): boolean => {
  const trimmed = token.trim()
  return trimmed === '*' || trimmed.startsWith('*/')
}

export const isValid = (expression: string): boolean => {
  try {
    parseCron(expression)
    return true
  } catch {
    return false
  }
}

/**
 * Compute the next occurrence of the schedule after the given reference
 * time. Searches up to 4 years ahead.
 *
 * DOM+DOW union semantics: per POSIX cron, when both day-of-month and
 * day-of-week are non-wildcard (not full-range), a date matches if
 * EITHER condition is true. When one or both are wildcard, standard
 * intersection (AND) applies.
 */
export const nextOccurrence = (sched: CronSchedule, after: Date): Date => {
  // Start one minute after `after`, zeroing seconds.
  const start = new Date(after)
  start.setSeconds(0, 0)
  start.setMinutes(start.getMinutes() + 1)

  const minuteSet = toSet(sched.minute)
  const hourSet = toSet(sched.hour)
  const domSet = toSet(sched.dayOfMonth)
  const monthSet = toSet(sched.month)
  const dowSet = toSet(sched.dayOfWeek)

  const domIsWild = sched.dayOfMonthIsWild
  const dowIsWild = sched.dayOfWeekIsWild

  const limit = new Date(after)
  limit.setFullYear(limit.getFullYear() + 4)

  const t = new Date(start)
  while (t < limit) {
    const mon = t.getMonth() + 1 // JS months are 0-indexed
    if (!monthSet.has(mon)) {
      t.setMonth(t.getMonth() + 1)
      t.setDate(1)
      t.setHours(0, 0, 0, 0)
      continue
    }
    const dayMatch = matchDay(domSet, dowSet, domIsWild, dowIsWild, t.getDate(), t.getDay())
    if (!dayMatch) {
      t.setDate(t.getDate() + 1)
      t.setHours(0, 0, 0, 0)
      continue
    }
    if (!hourSet.has(t.getHours())) {
      t.setHours(t.getHours() + 1)
      t.setMinutes(0, 0, 0)
      continue
    }
    if (!minuteSet.has(t.getMinutes())) {
      t.setMinutes(t.getMinutes() + 1)
      t.setSeconds(0, 0)
      continue
    }
    return new Date(t)
  }

  return limit
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
      throw new Error(`cron: ${name} field: value out of range [${min}-${max}]: ${rangeStart}-${rangeEnd}`)
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
