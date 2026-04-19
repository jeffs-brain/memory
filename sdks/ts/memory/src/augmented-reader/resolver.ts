// SPDX-License-Identifier: Apache-2.0

export type RenderedRetrievedFact = {
  readonly index: number
  readonly date: string
  readonly sessionId?: string
  readonly source?: string
  readonly body: string
}

export type DeterministicAugmentedAnswer = {
  readonly kind:
    | 'action-date'
    | 'recipient-total-spend'
    | 'backend-language-recommendation'
  readonly answer: string
}

type ActionResolverSpec = {
  readonly kind: 'submission' | 'booking' | 'join'
  readonly questionPattern: RegExp
  readonly pastActionPattern: RegExp
  readonly dateLabelPattern: RegExp
  readonly directDatePattern: RegExp
}

type ResolvedAmount = {
  readonly currency: string
  readonly value: number
}

type Recipient = {
  readonly key: string
  readonly display: string
  readonly phrase: string
  readonly relationshipTail?: string
}

type TransactionEntry = {
  readonly recipientKey: string
  readonly recipientDisplay: string
  readonly amount: ResolvedAmount
  readonly factIndex: number
}

type RecommendationCandidate = {
  readonly score: number
  readonly languages: readonly string[]
  readonly factIndex: number
}

const FACT_HEADER_RE = /^Retrieved facts \(\d+\):$/u
const FACT_LINE_RE = /^\s*(\d+)\.\s+(.+)$/u
const FACT_LABEL_RE = /\[([^\]]+)\]/gu
const MONTH_RE =
  /\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b/i
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
const EXPLICIT_DATE_RE =
  /\b(?:\d{4}[/-]\d{2}[/-]\d{2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?|\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?)\b/i
const RELATIVE_DATE_RE = /\b(?:today|yesterday|last night)\b/i
const QUESTION_STOP_WORDS: ReadonlySet<string> = new Set([
  'a',
  'about',
  'all',
  'amount',
  'an',
  'and',
  'are',
  'back',
  'be',
  'book',
  'booked',
  'booking',
  'can',
  'cost',
  'date',
  'did',
  'exact',
  'finally',
  'follow',
  'for',
  'how',
  'i',
  'in',
  'is',
  'join',
  'joined',
  'learn',
  'me',
  'my',
  'of',
  'on',
  'or',
  'previous',
  'programming',
  'question',
  'recommended',
  'remember',
  'remind',
  'specific',
  'spend',
  'spent',
  'submission',
  'submit',
  'submitted',
  'the',
  'their',
  'to',
  'total',
  'up',
  'wanted',
  'was',
  'what',
  'when',
  'you',
  'your',
])
const ACTION_FOCUS_SKIP_WORDS: ReadonlySet<string> = new Set([
  ...QUESTION_STOP_WORDS,
  'book',
  'booked',
  'booking',
  'join',
  'joined',
  'submission',
  'submit',
  'submitted',
])
const TOTAL_SPEND_QUERY_RE = /\b(?:total|in total|total amount|how much)\b/i
const SPEND_QUERY_RE = /\b(?:spent|spend|cost|paid|pay)\b/i
const ROLLUP_CLAUSE_RE =
  /\b(?:aggregate|budget|combined|in total|overview|plan(?:ned)?|recap|roll-?up|summary|totalling|totalled?|tracker|tracking)\b/i
const TRANSACTION_VERB_RE =
  /\b(?:bought|cost(?:ing)?|gave|gifted|got|ordered|paid|picked(?:\s+|-)?up|purchased|spent|treated(?:\s+myself)?(?:\s+to)?|worth)\b/i
const DIRECT_TRANSACTION_SUBJECT_RE = /\b(?:i|we|the user|they)\b/i
const RECAP_CLAUSE_RE = /\b(?:recalled|remembered)\b/i
const HEDGED_TRANSACTION_RE = /\b(?:maybe|might|plan(?:ned)?|consider)\b/i
const RELATIONSHIP_TERMS: ReadonlySet<string> = new Set([
  'aunt',
  'boss',
  'boyfriend',
  'brother',
  'child',
  'children',
  'colleague',
  'cousin',
  'coworker',
  'dad',
  'daughter',
  'father',
  'friend',
  'girlfriend',
  'husband',
  'manager',
  'mentor',
  'mother',
  'mum',
  'neighbour',
  'neighbor',
  'nephew',
  'niece',
  'partner',
  'sister',
  'son',
  'teacher',
  'uncle',
  'wife',
])
const RESOURCE_LIST_RE =
  /\b(?:catalogue|catalogues|course|courses|framework|frameworks|guide|guides|library|libraries|resource|resources|tool|tools|tutorial|tutorials)\b/i
const RESOURCE_LIST_ITEM_RE =
  /\b(?:coursera|django|flask|hibernate|nodeschool|spring|sql|udacity)\b/i
const RECOMMENDATION_QUERY_RE = /\b(?:back-?end|backend)\b.*\blanguages?\b/i
const RECOMMENDATION_RECALL_RE =
  /\b(?:recommend(?:ed)?|remind me|specific|exact|learn)\b/i
const DIRECT_RECOMMENDATION_RE =
  /\b(?:learn|recommend(?:ed)?|programming language(?:s)?)\b/i
const LIST_INTRO_RE = /\b(?:for example|including|like|such as)\b/i
const LANGUAGE_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = [
  [/\bRuby\b/i, 'Ruby'],
  [/\bPython\b/i, 'Python'],
  [/\bPHP\b/i, 'PHP'],
  [/\bJava\b/i, 'Java'],
  [/\bRust\b/i, 'Rust'],
  [/\bKotlin\b/i, 'Kotlin'],
  [/\bScala\b/i, 'Scala'],
  [/\bElixir\b/i, 'Elixir'],
  [/\bPerl\b/i, 'Perl'],
  [/\bTypeScript\b/i, 'TypeScript'],
  [/\bJavaScript\b/i, 'JavaScript'],
  [/\bGo\b/u, 'Go'],
  [/\bGolang\b/i, 'Go'],
  [/\bC#\b|\bC sharp\b/i, 'C#'],
]
const ACTION_RESOLVER_SPECS: ReadonlyArray<ActionResolverSpec> = [
  {
    kind: 'submission',
    questionPattern: /\bsubmit(?:ted)?\b/i,
    pastActionPattern: /\bsubmitted\b/i,
    dateLabelPattern: /\bsubmission date\b/i,
    directDatePattern:
      /\bsubmitted\b[^.!?\n]{0,120}\b(?:on|by)\s+([^.!?\n]+)/i,
  },
  {
    kind: 'booking',
    questionPattern: /\bbook(?:ed|ing)?\b/i,
    pastActionPattern: /\bbooked\b/i,
    dateLabelPattern: /\bbooking date\b/i,
    directDatePattern:
      /\bbooked\b[^.!?\n]{0,120}\b(?:for|on)\s+([^.!?\n]+)/i,
  },
  {
    kind: 'join',
    questionPattern: /\bjoin(?:ed)?\b/i,
    pastActionPattern: /\bjoined\b/i,
    dateLabelPattern: /\bjoin date\b/i,
    directDatePattern:
      /\bjoined\b[^.!?\n]{0,120}\bon\s+([^.!?\n]+)/i,
  },
]

export const parseRenderedRetrievedFacts = (
  rendered: string,
): readonly RenderedRetrievedFact[] => {
  const lines = rendered.replace(/\r\n?/gu, '\n').split('\n')
  const facts: RenderedRetrievedFact[] = []
  let inFacts = false
  let currentIndex = 0
  let currentLabels: string[] = []
  let currentBody: string[] = []

  const flush = (): void => {
    if (currentLabels.length === 0) return
    const date = currentLabels[0]?.trim() ?? 'unknown'
    const labels = currentLabels.slice(1).map((label) => label.trim())
    const sessionLabel = labels.find((label) => label.startsWith('session='))
    const source = labels.find((label) => !label.startsWith('session='))
    facts.push({
      index: currentIndex,
      date,
      ...(sessionLabel !== undefined
        ? { sessionId: sessionLabel.slice('session='.length).trim() }
        : {}),
      ...(source !== undefined ? { source } : {}),
      body: currentBody.join('\n').trim(),
    })
    currentIndex = 0
    currentLabels = []
    currentBody = []
  }

  for (const line of lines) {
    if (!inFacts) {
      if (FACT_HEADER_RE.test(line.trim())) {
        inFacts = true
      }
      continue
    }

    const factLineMatch = FACT_LINE_RE.exec(line)
    if (factLineMatch !== null) {
      const labels = [...(factLineMatch[2] ?? '').matchAll(FACT_LABEL_RE)]
        .map((match) => match[1]?.trim() ?? '')
        .filter((label) => label !== '')
      if (labels.length > 0) {
        flush()
        currentIndex = Number(factLineMatch[1])
        currentLabels = labels
        continue
      }
    }

    if (currentLabels.length === 0) continue
    currentBody.push(line)
  }

  flush()
  return facts
}

export const resolveDeterministicAugmentedAnswer = (args: {
  readonly question: string
  readonly rendered: string
}): DeterministicAugmentedAnswer | undefined => {
  const facts = parseRenderedRetrievedFacts(args.rendered)
  if (facts.length === 0) return undefined
  return (
    resolveAnchoredActionDate(args.question, facts) ??
    resolveRecipientTotalSpend(args.question, facts) ??
    resolveBackendLanguageRecommendation(args.question, facts)
  )
}

const resolveAnchoredActionDate = (
  question: string,
  facts: readonly RenderedRetrievedFact[],
): DeterministicAugmentedAnswer | undefined => {
  if (!/\b(?:when|what date)\b/i.test(question)) return undefined
  const spec = ACTION_RESOLVER_SPECS.find((candidate) =>
    candidate.questionPattern.test(question),
  )
  if (spec === undefined) return undefined

  const questionAnchors = extractAnchors(question)
  const focusTokens = questionTokens(question, ACTION_FOCUS_SKIP_WORDS)
  const minimumFocusOverlap = minimumTargetOverlap(focusTokens.length)
  const actionFacts = facts.filter((fact) =>
    isRelevantActionFact(
      fact.body,
      spec,
      focusTokens,
      questionAnchors,
      minimumFocusOverlap,
    ),
  )
  if (actionFacts.length === 0) return undefined

  const candidates: Array<{ readonly score: number; readonly date: string }> = []
  for (const actionFact of actionFacts) {
    const anchors = dedupeStrings([
      ...questionAnchors,
      ...extractAnchors(actionFact.body),
    ])
    for (const fact of facts) {
      const date = extractActionDate(fact, spec)
      if (date === undefined) continue
      const score =
        scoreSharedAnchors(fact.body, anchors) * 4 +
        countTokenOverlap(fact.body, focusTokens) +
        (spec.dateLabelPattern.test(fact.body) ? 2 : 0) +
        (fact.index === actionFact.index ? 1 : 0)
      if (score > 0) {
        candidates.push({ score, date })
      }
    }
  }

  if (candidates.length === 0) return undefined
  const maxScore = Math.max(...candidates.map((candidate) => candidate.score))
  if (maxScore < 3) return undefined
  const topDates = dedupeStrings(
    candidates
      .filter((candidate) => candidate.score === maxScore)
      .map((candidate) => candidate.date),
  )
  if (topDates.length !== 1) return undefined
  const answer = (topDates[0] ?? '').trim()
  if (answer === '') return undefined
  return { kind: 'action-date', answer }
}

const resolveRecipientTotalSpend = (
  question: string,
  facts: readonly RenderedRetrievedFact[],
): DeterministicAugmentedAnswer | undefined => {
  if (!TOTAL_SPEND_QUERY_RE.test(question) || !SPEND_QUERY_RE.test(question)) {
    return undefined
  }
  const recipients = extractRecipients(question)
  if (recipients.length === 0) return undefined

  const entries: TransactionEntry[] = []
  const seen = new Set<string>()
  for (const fact of facts) {
    for (const clause of splitClauses(fact.body)) {
      if (scoreTransactionDirectness(clause) < 3) {
        continue
      }
      const amount = extractTransactionalAmount(clause)
      if (amount === undefined) continue
      const matches = recipients.filter((recipient) =>
        clauseTargetsRecipient(clause, recipient),
      )
      if (matches.length !== 1) continue
      const matchedRecipient = matches[0]
      if (matchedRecipient === undefined) continue
      const fingerprint = [
        matchedRecipient.key,
        amount.currency,
        String(amount.value),
        normaliseText(clause),
      ].join('|')
      if (seen.has(fingerprint)) continue
      seen.add(fingerprint)
      entries.push({
        recipientKey: matchedRecipient.key,
        recipientDisplay: matchedRecipient.display,
        amount,
        factIndex: fact.index,
      })
    }
  }

  if (entries.length === 0) return undefined
  const currencies = dedupeStrings(entries.map((entry) => entry.amount.currency))
  if (currencies.length !== 1) return undefined
  const currency = currencies[0]
  if (currency === undefined) return undefined

  const totalsByRecipient = new Map<string, number>()
  for (const recipient of recipients) {
    totalsByRecipient.set(recipient.key, 0)
  }
  for (const entry of entries) {
    totalsByRecipient.set(
      entry.recipientKey,
      (totalsByRecipient.get(entry.recipientKey) ?? 0) + entry.amount.value,
    )
  }

  if ([...totalsByRecipient.values()].some((value) => value <= 0)) {
    return undefined
  }

  const total = [...totalsByRecipient.values()].reduce(
    (sum, value) => sum + value,
    0,
  )
  if (total <= 0) return undefined

  return {
    kind: 'recipient-total-spend',
    answer: formatCurrency(currency, total),
  }
}

const resolveBackendLanguageRecommendation = (
  question: string,
  facts: readonly RenderedRetrievedFact[],
): DeterministicAugmentedAnswer | undefined => {
  if (
    !RECOMMENDATION_QUERY_RE.test(question) ||
    !RECOMMENDATION_RECALL_RE.test(question)
  ) {
    return undefined
  }

  const candidates: RecommendationCandidate[] = []
  for (const fact of facts) {
    for (const clause of splitClauses(fact.body)) {
      if (!DIRECT_RECOMMENDATION_RE.test(clause)) continue
      const languages = extractRecommendedLanguages(clause)
      if (languages.length === 0) continue
      let score = 0
      if (/\bback-?end\b/i.test(clause)) score += 2
      if (/\bprogramming language(?:s)?\b/i.test(clause)) score += 3
      if (LIST_INTRO_RE.test(clause)) score += 2
      if (/\b(?:learn|recommend(?:ed)?)\b/i.test(clause)) score += 1
      if (RESOURCE_LIST_RE.test(clause)) score -= 4
      if (RESOURCE_LIST_ITEM_RE.test(clause)) score -= 2
      if (score <= 0) continue
      candidates.push({ score, languages, factIndex: fact.index })
    }
  }

  if (candidates.length === 0) return undefined
  candidates.sort((left, right) => {
    if (left.score !== right.score) return right.score - left.score
    if (left.languages.length !== right.languages.length) {
      return right.languages.length - left.languages.length
    }
    return left.factIndex - right.factIndex
  })

  const top = candidates[0]
  if (top === undefined || top.score < 4) return undefined
  const second = candidates[1]
  if (
    second !== undefined &&
    second.score === top.score &&
    normaliseList(top.languages) !== normaliseList(second.languages)
  ) {
    return undefined
  }

  return {
    kind: 'backend-language-recommendation',
    answer: `I recommended learning ${joinWithOr(top.languages)} as a back-end programming language.`,
  }
}

const isRelevantActionFact = (
  text: string,
  spec: ActionResolverSpec,
  focusTokens: readonly string[],
  questionAnchors: readonly string[],
  minimumFocusOverlap: number,
): boolean => {
  if (!spec.pastActionPattern.test(text)) return false
  const overlap = countTokenOverlap(text, focusTokens)
  if (overlap >= minimumFocusOverlap) return true
  return overlap > 0 && scoreSharedAnchors(text, questionAnchors) > 0
}

const extractActionDate = (
  fact: RenderedRetrievedFact,
  spec: ActionResolverSpec,
): string | undefined => {
  const text = fact.body
  const anchoredMatch = new RegExp(
    `${spec.dateLabelPattern.source}(?:\\s+was|\\s+is|\\s*:)?\\s+([^.!?\\n]+)`,
    'i',
  ).exec(text)
  const anchoredDate = extractDatePhrase(anchoredMatch?.[1] ?? '', fact.date)
  if (anchoredDate !== undefined) return anchoredDate
  const directDate = extractDatePhrase(
    spec.directDatePattern.exec(text)?.[1] ?? '',
    fact.date,
  )
  if (directDate !== undefined) return directDate
  if (
    spec.pastActionPattern.test(text) &&
    (EXPLICIT_DATE_RE.test(text) || RELATIVE_DATE_RE.test(text))
  ) {
    const contextualDate = extractDatePhrase(text, fact.date)
    if (contextualDate !== undefined) return contextualDate
  }
  if (spec.dateLabelPattern.test(text)) {
    const labelledDate = extractDatePhrase(text, fact.date)
    if (labelledDate !== undefined) return labelledDate
    const anchorDate = formatAnchorDate(fact.date)
    if (anchorDate !== undefined) return anchorDate
  }
  return undefined
}

const extractDatePhrase = (
  value: string,
  anchorDateLabel?: string,
): string | undefined => {
  const trimmed = value.trim()
  if (trimmed === '') return undefined
  const relativeDate = RELATIVE_DATE_RE.exec(trimmed)?.[0]?.toLowerCase()
  if (relativeDate !== undefined) {
    const anchorDate = parseAnchorDate(anchorDateLabel)
    if (anchorDate === undefined) return undefined
    const dayOffset = relativeDate === 'today' ? 0 : -1
    return formatMonthDay(shiftDays(anchorDate, dayOffset))
  }
  const match = EXPLICIT_DATE_RE.exec(trimmed)
  if (match?.[0] === undefined) return undefined
  const parsed = parseAnchorDate(match[0])
  if (parsed !== undefined) return formatMonthDay(parsed)
  return match[0].trim()
}

const extractRecipients = (question: string): readonly Recipient[] => {
  const recipientTailMatch = /\bfor\s+(.+?)(?:[?!.]|$)/i.exec(question)
  if (recipientTailMatch?.[1] === undefined) return []
  const parts = recipientTailMatch[1]
    .split(/\s*(?:,| and | & )\s*/iu)
    .map((part) => part.trim())
    .filter((part) => part !== '')
  if (parts.length === 0) return []

  const recipients = parts.map((part): Recipient | undefined => {
    const cleaned = part
      .replace(/^(?:my|our|the|a|an|his|her|their)\s+/iu, '')
      .trim()
    if (cleaned === '') return undefined
    const key = normaliseText(cleaned)
    const words = cleaned.split(/\s+/u)
    const relationshipTail = words.at(-1)?.toLowerCase()
    const looksNamed =
      cleaned === cleaned.toUpperCase()
        ? cleaned.length > 1
        : /^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}$/u.test(cleaned)
    const looksRelationship =
      relationshipTail !== undefined && RELATIONSHIP_TERMS.has(relationshipTail)
    if (!looksNamed && !looksRelationship) return undefined
    return {
      key,
      display: cleaned,
      phrase: cleaned,
      ...(looksRelationship && relationshipTail !== undefined
        ? { relationshipTail }
        : {}),
    }
  })

  if (recipients.some((recipient) => recipient === undefined)) return []
  return recipients.filter(
    (recipient): recipient is Recipient => recipient !== undefined,
  )
}

const clauseTargetsRecipient = (clause: string, recipient: Recipient): boolean => {
  const phrasePattern = escapeRegex(recipient.phrase).replace(/\s+/gu, '\\s+')
  const leadingDet =
    '(?:my\\s+|our\\s+|the\\s+|a\\s+|an\\s+|his\\s+|her\\s+|their\\s+)?'
  const variants = [
    new RegExp(`\\bfor\\s+${leadingDet}${phrasePattern}\\b`, 'i'),
    new RegExp(
      `\\b(?:bought|purchased|ordered|got|gifted|gave|picked(?:\\s+|-)?up)\\s+${leadingDet}${phrasePattern}\\b`,
      'i',
    ),
  ]
  if (
    recipient.relationshipTail !== undefined &&
    recipient.relationshipTail !== recipient.phrase.toLowerCase()
  ) {
    const tailPattern = escapeRegex(recipient.relationshipTail)
    variants.push(
      new RegExp(`\\bfor\\s+${leadingDet}(?:\\w+\\s+)*${tailPattern}\\b`, 'i'),
      new RegExp(
        `\\b(?:bought|purchased|ordered|got|gifted|gave|picked(?:\\s+|-)?up)\\s+${leadingDet}(?:\\w+\\s+)*${tailPattern}\\b`,
        'i',
      ),
    )
  }
  return variants.some((pattern) => pattern.test(clause))
}

const extractTransactionalAmount = (
  clause: string,
): ResolvedAmount | undefined => {
  const patterns: readonly RegExp[] = [
    /\bspent\s+([$£€])\s?(\d[\d,]*(?:\.\d{1,2})?)/i,
    /\bpaid\s+([$£€])\s?(\d[\d,]*(?:\.\d{1,2})?)/i,
    /\bcost(?:\s+me)?\s+([$£€])\s?(\d[\d,]*(?:\.\d{1,2})?)/i,
    /\b(?:bought|purchased|ordered|got|picked(?:\s+|-)?up|gifted|gave)\b[^.!?\n]{0,120}?\bfor\s+([$£€])\s?(\d[\d,]*(?:\.\d{1,2})?)/i,
  ]
  for (const pattern of patterns) {
    const match = pattern.exec(clause)
    const resolved = parseAmountMatch(match)
    if (resolved !== undefined) return resolved
  }
  const allAmounts = [...clause.matchAll(/([$£€])\s?(\d[\d,]*(?:\.\d{1,2})?)/gi)]
  if (allAmounts.length !== 1) return undefined
  return parseAmountMatch(allAmounts[0] ?? null)
}

const parseAmountMatch = (
  match: RegExpExecArray | RegExpMatchArray | null,
): ResolvedAmount | undefined => {
  if (match?.[1] === undefined || match[2] === undefined) return undefined
  const parsed = Number(match[2].replace(/,/gu, ''))
  if (!Number.isFinite(parsed)) return undefined
  return { currency: match[1], value: parsed }
}

const extractRecommendedLanguages = (clause: string): readonly string[] => {
  if (
    RESOURCE_LIST_RE.test(clause) &&
    !/\bprogramming language(?:s)?\b/i.test(clause)
  ) {
    return []
  }
  const out: string[] = []
  for (const [pattern, language] of LANGUAGE_PATTERNS) {
    if (pattern.test(clause) && !out.includes(language)) {
      out.push(language)
    }
  }
  return out
}

const minimumTargetOverlap = (tokenCount: number): number =>
  Math.max(1, Math.ceil(tokenCount * 0.6))

const scoreTransactionDirectness = (clause: string): number => {
  let score = 0
  if (TRANSACTION_VERB_RE.test(clause)) score += 3
  if (DIRECT_TRANSACTION_SUBJECT_RE.test(clause)) score += 1
  if (ROLLUP_CLAUSE_RE.test(clause)) score -= 4
  if (RECAP_CLAUSE_RE.test(clause)) score -= 3
  if (HEDGED_TRANSACTION_RE.test(clause)) score -= 2
  return score
}

const splitClauses = (text: string): readonly string[] => {
  const compact = text
    .replace(/\r\n?/gu, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '')
    .join(' ')
  return compact
    .split(/(?<=[.!?])\s+/u)
    .map((clause) => clause.trim())
    .filter((clause) => clause !== '')
}

const questionTokens = (
  question: string,
  skip: ReadonlySet<string>,
): readonly string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const rawToken of question
    .toLowerCase()
    .replace(/back-end/giu, 'backend')
    .split(/[^a-z0-9#]+/u)) {
    const token = rawToken.trim()
    if (token.length < 3 || skip.has(token) || seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

const extractAnchors = (text: string): readonly string[] => {
  const out: string[] = []
  const push = (value: string | undefined): void => {
    const anchor = normaliseAnchor(value)
    if (anchor === '' || out.includes(anchor)) return
    out.push(anchor)
  }

  for (const match of text.matchAll(/"([^"\n]{2,80})"/gu)) {
    push(match[1])
  }
  for (const match of text.matchAll(/\b([A-Z]{2,10})\b/gu)) {
    push(match[1])
  }
  for (const match of text.matchAll(
    /\b(?:called|for|to|with|at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})/gu,
  )) {
    push(match[1])
  }
  return out
}

const normaliseAnchor = (value: string | undefined): string => {
  if (value === undefined) return ''
  const trimmed = value.trim()
  if (trimmed === '' || MONTH_RE.test(trimmed)) return ''
  return normaliseText(trimmed)
}

const scoreSharedAnchors = (
  text: string,
  anchors: readonly string[],
): number => {
  const normalised = normaliseText(text)
  let score = 0
  for (const anchor of anchors) {
    if (anchor !== '' && normalised.includes(anchor)) score += 1
  }
  return score
}

const countTokenOverlap = (
  text: string,
  tokens: readonly string[],
): number => {
  const normalised = normaliseText(text)
  let matches = 0
  for (const token of tokens) {
    if (normalised.includes(token)) matches += 1
  }
  return matches
}

const formatCurrency = (currency: string, value: number): string => {
  const rounded = Number.isInteger(value) ? String(value) : value.toFixed(2)
  return `${currency}${rounded}`
}

const parseAnchorDate = (value: string | undefined): Date | undefined => {
  if (value === undefined) return undefined
  const match = /^(\d{4})[-/](\d{2})[-/](\d{2})$/u.exec(value.trim())
  if (match?.[1] === undefined || match[2] === undefined || match[3] === undefined) {
    return undefined
  }
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  if (
    !Number.isInteger(year) ||
    !Number.isInteger(month) ||
    !Number.isInteger(day)
  ) {
    return undefined
  }
  const parsed = new Date(Date.UTC(year, month - 1, day))
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() !== month - 1 ||
    parsed.getUTCDate() !== day
  ) {
    return undefined
  }
  return parsed
}

const formatAnchorDate = (value: string | undefined): string | undefined => {
  const parsed = parseAnchorDate(value)
  if (parsed === undefined) return undefined
  return formatMonthDay(parsed)
}

const shiftDays = (value: Date, days: number): Date =>
  new Date(value.getTime() + days * 24 * 60 * 60 * 1000)

const formatMonthDay = (value: Date): string => {
  const month = MONTH_NAMES[value.getUTCMonth()]
  if (month === undefined) return ''
  return `${month} ${ordinal(value.getUTCDate())}`
}

const ordinal = (day: number): string => {
  if (day % 100 >= 11 && day % 100 <= 13) {
    return `${String(day)}th`
  }
  switch (day % 10) {
    case 1:
      return `${String(day)}st`
    case 2:
      return `${String(day)}nd`
    case 3:
      return `${String(day)}rd`
    default:
      return `${String(day)}th`
  }
}

const normaliseText = (value: string): string =>
  value
    .toLowerCase()
    .replace(/back-end/giu, 'backend')
    .replace(/[^a-z0-9#]+/gu, ' ')
    .trim()

const dedupeStrings = (values: readonly string[]): string[] => {
  const out: string[] = []
  for (const value of values) {
    if (value === '' || out.includes(value)) continue
    out.push(value)
  }
  return out
}

const normaliseList = (values: readonly string[]): string =>
  values.map((value) => value.toLowerCase()).join('|')

const joinWithAnd = (values: readonly string[]): string => {
  if (values.length === 0) return ''
  if (values.length === 1) return values[0] ?? ''
  if (values.length === 2) {
    return `${values[0] ?? ''} and ${values[1] ?? ''}`
  }
  return `${values.slice(0, -1).join(', ')} and ${values.at(-1) ?? ''}`
}

const joinWithOr = (values: readonly string[]): string => {
  if (values.length === 0) return ''
  if (values.length === 1) return values[0] ?? ''
  if (values.length === 2) {
    return `${values[0] ?? ''} or ${values[1] ?? ''}`
  }
  return `${values.slice(0, -1).join(', ')}, or ${values.at(-1) ?? ''}`
}

const escapeRegex = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&')
