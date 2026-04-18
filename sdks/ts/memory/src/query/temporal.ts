// SPDX-License-Identifier: Apache-2.0

const QUESTION_DATE_RE =
  /^(\d{4})[/-](\d{2})[/-](\d{2})(?:\s+\([A-Za-z]{3}\))?(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?$/

const RELATIVE_TIME_RE =
  /(\d+)\s+(day|days|week|weeks|month|months)\s+ago/gi

const LAST_WEEKDAY_RE =
  /last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)/gi

const WEEKDAY_NAMES = [
  'Sunday',
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
] as const

const MONTH_NAMES = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
] as const

const WEEKDAY_BY_NAME = new Map(
  WEEKDAY_NAMES.map((name, index) => [name.toLowerCase(), index]),
)

export type TemporalExpansion = {
  readonly originalQuery: string
  readonly expandedQuery: string
  readonly dateHints: readonly string[]
  readonly resolved: boolean
}

export const parseQuestionDate = (value: string): Date | undefined => {
  const trimmed = value.trim()
  if (trimmed === '') return undefined

  const matched = QUESTION_DATE_RE.exec(trimmed)
  if (matched !== null) {
    const yearRaw = matched[1] ?? '0'
    const monthRaw = matched[2] ?? '1'
    const dayRaw = matched[3] ?? '1'
    const hourRaw = matched[4] ?? '0'
    const minuteRaw = matched[5] ?? '0'
    const secondRaw = matched[6] ?? '0'
    return new Date(
      Date.UTC(
        Number.parseInt(yearRaw, 10),
        Number.parseInt(monthRaw, 10) - 1,
        Number.parseInt(dayRaw, 10),
        Number.parseInt(hourRaw ?? '0', 10),
        Number.parseInt(minuteRaw ?? '0', 10),
        Number.parseInt(secondRaw ?? '0', 10),
      ),
    )
  }

  const parsed = new Date(trimmed)
  return Number.isNaN(parsed.getTime()) ? undefined : parsed
}

export const readerTodayAnchor = (questionDate: string | undefined): string => {
  const trimmed = questionDate?.trim() ?? ''
  if (trimmed === '') return 'unknown'
  const parsed = parseQuestionDate(trimmed)
  if (parsed === undefined) return trimmed
  return `${formatIsoDate(parsed)} (${WEEKDAY_NAMES[parsed.getUTCDay()]})`
}

export const expandTemporal = (
  question: string,
  questionDate: string | undefined,
): TemporalExpansion => {
  const anchor = questionDate !== undefined ? parseQuestionDate(questionDate) : undefined
  if (anchor === undefined) {
    return {
      originalQuery: question,
      expandedQuery: question,
      dateHints: [],
      resolved: false,
    }
  }

  const hints: string[] = []
  let expanded = question
  expanded = resolveRelativeTime(expanded, anchor, hints)
  expanded = resolveLastWeekday(expanded, anchor, hints)
  expanded = annotateOrdering(expanded)

  return {
    originalQuery: question,
    expandedQuery: expanded,
    dateHints: dedupeStrings(hints),
    resolved: hints.length > 0,
  }
}

export const augmentQueryWithTemporal = (
  question: string,
  questionDate: string | undefined,
): string => {
  const expansion = expandTemporal(question, questionDate)
  if (!expansion.resolved || expansion.dateHints.length === 0) return question
  const tokens: string[] = []
  for (const hint of expansion.dateHints) {
    const trimmed = hint.trim()
    if (trimmed === '') continue
    tokens.push(trimmed)
    tokens.push(trimmed.replaceAll('/', '-'))
  }
  const unique = dedupeStrings(tokens)
  return unique.length === 0 ? question : `${question} ${unique.join(' ')}`
}

export const resolvedTemporalHintLine = (
  question: string,
  questionDate: string | undefined,
): string | undefined => {
  const expansion = expandTemporal(question, questionDate)
  if (!expansion.resolved || expansion.dateHints.length === 0) return undefined
  return `[Resolved temporal references: ${expansion.dateHints.join(', ')}]`
}

export const dateSearchTokens = (value: string | undefined): readonly string[] => {
  const trimmed = value?.trim() ?? ''
  if (trimmed === '') return []
  const parsed = parseQuestionDate(trimmed)
  if (parsed === undefined) {
    const match = /\b(\d{4})[/-](\d{2})[/-](\d{2})\b/.exec(trimmed)
    if (match === null) return []
    const year = match[1] ?? ''
    const month = match[2] ?? ''
    const day = match[3] ?? ''
    return dedupeStrings([
      `${year}-${month}-${day}`,
      `${year}/${month}/${day}`,
      year,
    ])
  }

  return dedupeStrings([
    formatIsoDate(parsed),
    formatSlashDate(parsed),
    String(parsed.getUTCFullYear()),
    WEEKDAY_NAMES[parsed.getUTCDay()] ?? '',
    MONTH_NAMES[parsed.getUTCMonth()] ?? '',
  ])
}

const resolveRelativeTime = (
  question: string,
  anchor: Date,
  hints: string[],
): string =>
  question.replaceAll(RELATIVE_TIME_RE, (match, countRaw, unitRaw) => {
    const count = Number.parseInt(String(countRaw), 10)
    if (!Number.isFinite(count)) return match
    const unit = String(unitRaw).toLowerCase()
    const resolved = new Date(anchor.getTime())
    if (unit.startsWith('day')) {
      resolved.setUTCDate(resolved.getUTCDate() - count)
    } else if (unit.startsWith('week')) {
      resolved.setUTCDate(resolved.getUTCDate() - count * 7)
    } else if (unit.startsWith('month')) {
      resolved.setUTCMonth(resolved.getUTCMonth() - count)
    } else {
      return match
    }
    const slash = formatSlashDate(resolved)
    hints.push(slash)
    return `${match} (around ${slash})`
  })

const resolveLastWeekday = (
  question: string,
  anchor: Date,
  hints: string[],
): string =>
  question.replaceAll(LAST_WEEKDAY_RE, (match, weekdayRaw) => {
    const target = WEEKDAY_BY_NAME.get(String(weekdayRaw).toLowerCase())
    if (target === undefined) return match
    const resolved = new Date(anchor.getTime())
    for (let i = 0; i < 7; i++) {
      resolved.setUTCDate(resolved.getUTCDate() - 1)
      if (resolved.getUTCDay() === target) {
        const slash = formatSlashDate(resolved)
        hints.push(slash)
        return `${match} (${slash})`
      }
    }
    return match
  })

const annotateOrdering = (question: string): string => {
  const lower = question.toLowerCase()
  if (
    lower.includes('first') ||
    lower.includes('earlier') ||
    lower.includes('before')
  ) {
    return `${question} [Note: look for the earliest dated event]`
  }
  if (
    lower.includes('most recent') ||
    lower.includes('latest') ||
    lower.includes('last time')
  ) {
    return `${question} [Note: look for the most recently dated event]`
  }
  return question
}

const formatIsoDate = (date: Date): string =>
  `${String(date.getUTCFullYear()).padStart(4, '0')}-${String(
    date.getUTCMonth() + 1,
  ).padStart(2, '0')}-${String(date.getUTCDate()).padStart(2, '0')}`

const formatSlashDate = (date: Date): string => formatIsoDate(date).replaceAll('-', '/')

const dedupeStrings = (values: readonly string[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const trimmed = value.trim()
    if (trimmed === '' || seen.has(trimmed)) continue
    seen.add(trimmed)
    out.push(trimmed)
  }
  return out
}
