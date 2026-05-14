// SPDX-License-Identifier: Apache-2.0

export type Correction = {
  readonly snippet: string
  readonly phrase: string
}

export type CorrectionReminderOptions = {
  readonly searchTool?: string
  readonly createTool?: string
  readonly updateTool?: string
  readonly removeTool?: string
  readonly mentionChange?: boolean
}

const CORRECTION_PHRASES = [
  "that's wrong",
  'thats wrong',
  'you got that wrong',
  'you have that wrong',
  "you're wrong",
  'youre wrong',
  'you are wrong',
  "actually it's",
  'actually its',
  'actually it is',
  "it's actually",
  'its actually',
  'it is actually',
  'stop saying',
  "don't say",
  'do not say',
  'not right',
  "wrong, it's",
  'wrong, its',
  "wrong it's",
  'i never said',
  "i didn't say",
  'i did not say',
  "that's not",
  'thats not',
  'that is not',
  'correct that',
  'correction:',
  'please correct',
  'please update',
  'please remove',
  'please forget',
  'forget that',
  'forget what',
  'remember instead',
  'the correct',
] as const

const SOLO_CORRECTION_PATTERNS = [
  /^\s*no[\s,\-]+(it'?s|the|that|it is|its|that is|its actually)\b/i,
  /^\s*(wrong|incorrect)[\s,!.]/i,
  /^\s*actually[,\s\-]/i,
] as const

const FALSE_POSITIVE_SUBSTRINGS = [
  'no problem',
  'no idea',
  'no worries',
  'no rush',
  'no big deal',
  'wrong end of the stick',
  'wrong number',
  'wrong place',
  'nothing wrong',
  'not wrong',
  'no, before that',
  'no, but',
] as const

export const detectCorrection = (latestUserText: string): Correction | undefined => {
  if (latestUserText === '') return undefined

  const clean = latestUserText.toLowerCase().trim()
  if (clean === '') return undefined

  for (const falsePositive of FALSE_POSITIVE_SUBSTRINGS) {
    if (clean.includes(falsePositive)) return undefined
  }

  for (const phrase of CORRECTION_PHRASES) {
    const index = clean.indexOf(phrase)
    if (index >= 0) {
      return {
        snippet: extractCorrectionSnippet(latestUserText, index),
        phrase,
      }
    }
  }

  for (const pattern of SOLO_CORRECTION_PATTERNS) {
    const match = pattern.exec(clean)
    if (match?.index !== undefined) {
      return {
        snippet: extractCorrectionSnippet(latestUserText, match.index),
        phrase: pattern.source,
      }
    }
  }

  return undefined
}

export const buildCorrectionReminder = (userInput: string): string =>
  buildCorrectionReminderWithOptions(userInput, {
    searchTool: 'memory_search',
    updateTool: 'memory_update',
    removeTool: 'memory_remove',
    createTool: 'memory_create',
    mentionChange: true,
  })

export const buildCorrectionReminderWithOptions = (
  userInput: string,
  opts: CorrectionReminderOptions = {},
): string => {
  const correction = detectCorrection(userInput)
  if (correction === undefined) return ''

  const snippet = correction.snippet || userInput
  const searchTool = defaultString(opts.searchTool, 'memory_search')
  const updateTool = defaultString(opts.updateTool, 'memory_update')
  const removeTool = defaultString(opts.removeTool, 'memory_remove')
  const createTool = defaultString(opts.createTool, 'memory_create')

  let reminder = `User correction detected (${JSON.stringify(snippet)}). Before answering, call ${searchTool} for the disputed topic. If a stale entry exists, call ${updateTool} or ${removeTool} with a reason. If no entry exists yet but the correction is durable, call ${createTool}.`
  if (opts.mentionChange === true) {
    reminder += ' Mention in your reply what you changed.'
  }
  return reminder
}

const extractCorrectionSnippet = (original: string, matchStart: number): string => {
  const maxLen = 160
  const runes = Array.from(original)
  if (runes.length <= maxLen) return original.trim()

  const runeStart = Array.from(original.slice(0, matchStart)).length
  let start = runeStart - Math.floor(maxLen / 3)
  if (start < 0) start = 0

  let end = start + maxLen
  if (end > runes.length) {
    end = runes.length
    start = Math.max(0, end - maxLen)
  }

  const prefix = start > 0 ? '...' : ''
  const suffix = end < runes.length ? '...' : ''
  return `${prefix}${runes.slice(start, end).join('')}${suffix}`.trim()
}

const defaultString = (value: string | undefined, fallback: string): string => {
  const trimmed = value?.trim()
  return trimmed && trimmed !== '' ? trimmed : fallback
}
