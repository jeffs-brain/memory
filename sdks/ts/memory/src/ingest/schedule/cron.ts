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
}

export const parseCron = (expression: string): CronSchedule => {
  const fields = expression.trim().split(/\s+/)
  if (fields.length !== 5) {
    throw new Error(`cron: expected 5 fields, got ${fields.length} in "${expression}"`)
  }

  return {
    minute: parseField(fields[0], 0, 59, 'minute'),
    hour: parseField(fields[1], 0, 23, 'hour'),
    dayOfMonth: parseField(fields[2], 1, 31, 'day-of-month'),
    month: parseField(fields[3], 1, 12, 'month'),
    dayOfWeek: parseField(fields[4], 0, 6, 'day-of-week'),
  }
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

  const domIsWild = sched.dayOfMonth.length === 31
  const dowIsWild = sched.dayOfWeek.length === 7

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

const parseField = (field: string, min: number, max: number, name: string): number[] => {
  const result: number[] = []

  for (const part of field.split(',')) {
    const trimmed = part.trim()
    const stepParts = trimmed.split('/')
    const rangePart = stepParts[0]
    let step = 1

    if (stepParts.length === 2) {
      step = Number.parseInt(stepParts[1], 10)
      if (Number.isNaN(step) || step < 1) {
        throw new Error(`cron: ${name} field: invalid step "${stepParts[1]}"`)
      }
    }

    let rangeStart: number
    let rangeEnd: number

    if (rangePart === '*') {
      rangeStart = min
      rangeEnd = max
    } else if (rangePart.includes('-')) {
      const [s, e] = rangePart.split('-')
      rangeStart = Number.parseInt(s, 10)
      rangeEnd = Number.parseInt(e, 10)
      if (Number.isNaN(rangeStart) || Number.isNaN(rangeEnd)) {
        throw new Error(`cron: ${name} field: invalid range "${rangePart}"`)
      }
    } else {
      const val = Number.parseInt(rangePart, 10)
      if (Number.isNaN(val)) {
        throw new Error(`cron: ${name} field: invalid value "${rangePart}"`)
      }
      rangeStart = val
      rangeEnd = val
    }

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
