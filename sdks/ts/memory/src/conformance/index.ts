// SPDX-License-Identifier: Apache-2.0

/**
 * Reusable HTTP conformance runner for `memory serve`.
 *
 * Loads the shared `spec/conformance/http-contract.json` fixture,
 * provisions an isolated brain per case, replays every documented
 * request against a remote daemon, and returns a pass/fail report.
 */

import { randomUUID } from 'node:crypto'
import { access, readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { isDeepStrictEqual } from 'node:util'

import { type SSEEvent, SSEParser, iterateSSE } from '../llm/sse.js'

export type ConformanceBodyAssertion =
  | { readonly kind: 'items-include-path'; readonly path: string }
  | { readonly kind: 'items-exclude-path'; readonly path: string }
  | { readonly kind: 'items-files-equal'; readonly paths: readonly string[] }
  | { readonly kind: 'items-dirs-equal'; readonly paths: readonly string[] }
  | { readonly kind: 'json-field-equals'; readonly field: string; readonly value: unknown }
  | ({ readonly kind: string } & Readonly<Record<string, unknown>>)

export type ConformanceStreamAssertion =
  | { readonly kind: 'expect-event'; readonly event: string }
  | ({ readonly kind: string } & Readonly<Record<string, unknown>>)

export type ConformanceStep = {
  readonly kind?: string
  readonly name?: string
  readonly method?: string
  readonly path?: string
  readonly headers?: Readonly<Record<string, string>>
  readonly bodyBase64?: string
  readonly bodyJson?: unknown
  readonly event?: string
  readonly expectedStatus?: number
  readonly expectedBodyBase64?: string
}

export type ConformanceExpectedResponse = {
  readonly status?: number
  readonly contentType?: string
  readonly bodyBase64?: string
  readonly body?: unknown
  readonly bodyAssertions?: readonly ConformanceBodyAssertion[]
  readonly streamAssertions?: readonly ConformanceStreamAssertion[]
}

export type ConformanceCase = {
  readonly name: string
  readonly notes?: string
  readonly setup?: readonly ConformanceStep[]
  readonly request: ConformanceStep
  readonly expectedResponse: ConformanceExpectedResponse
  readonly followUp?: readonly ConformanceStep[]
  readonly teardown?: readonly ConformanceStep[]
}

export type ConformanceDocument = {
  readonly description?: string
  readonly placeholders: Readonly<Record<string, string>>
  readonly cases: readonly ConformanceCase[]
}

export type RunConformanceSuiteOptions = {
  readonly baseUrl: string
  readonly authToken?: string
  readonly fixturePath?: string
  readonly brainIdPrefix?: string
  readonly requestTimeoutMs?: number
  readonly sseTimeoutMs?: number
}

export type ConformanceCaseResult = {
  readonly name: string
  readonly brainId: string
  readonly ok: boolean
  readonly error?: string
}

export type ConformanceSuiteResult = {
  readonly total: number
  readonly passed: number
  readonly failed: number
  readonly cases: readonly ConformanceCaseResult[]
}

type SseWaiter = {
  readonly resolve: (value: string) => void
  readonly reject: (error: Error) => void
  readonly timer: ReturnType<typeof setTimeout>
}

type SseSubscriber = {
  waitFor(event: string, timeoutMs?: number): Promise<string>
  close(): Promise<void>
}

type FetchResponse = Awaited<ReturnType<typeof fetch>>
type ResponseLike = {
  readonly status: number
  readonly headers: { get(name: string): string | null }
  readonly body: ReadableStream<Uint8Array> | null
  clone(): ResponseLike
  text(): Promise<string>
  arrayBuffer(): Promise<ArrayBuffer>
}

type RunCaseContext = {
  readonly baseUrl: URL
  readonly authToken: string | undefined
  readonly requestTimeoutMs: number
  readonly sseTimeoutMs: number
  readonly substitute: (value: string) => string
  readonly ssePool: Map<string, SseSubscriber>
}

const DEFAULT_REQUEST_TIMEOUT_MS = 5_000
const DEFAULT_SSE_TIMEOUT_MS = 3_000
const DEFAULT_BRAIN_ID_PREFIX = 'jb-conformance'
const DIST_FIXTURE_FILENAME = 'http-contract.json'
const REPO_FIXTURE_RELATIVE_PATH = ['spec', 'conformance', 'http-contract.json'] as const

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined

const formatValue = (value: unknown): string => {
  if (value === undefined) return 'undefined'
  try {
    return JSON.stringify(value) ?? String(value)
  } catch {
    return String(value)
  }
}

const normaliseBaseUrl = (raw: string): URL => {
  const url = new URL(raw)
  if (url.pathname.length > 1 && url.pathname.endsWith('/')) {
    url.pathname = url.pathname.slice(0, -1)
  }
  return url
}

const resolveRequestUrl = (baseUrl: URL, requestPath: string): string => {
  const requestUrl = new URL(requestPath, baseUrl.origin)
  const basePath = baseUrl.pathname === '/' ? '' : baseUrl.pathname
  const alreadyPrefixed =
    basePath !== '' &&
    (requestUrl.pathname === basePath || requestUrl.pathname.startsWith(`${basePath}/`))

  const url = new URL(baseUrl.toString())
  url.pathname = alreadyPrefixed ? requestUrl.pathname : `${basePath}${requestUrl.pathname}`
  url.search = requestUrl.search
  url.hash = requestUrl.hash
  return url.toString()
}

const substituteFactory = (
  placeholders: Readonly<Record<string, string>>,
  brainId: string,
): ((value: string) => string) => {
  const pairs = Object.entries({ ...placeholders, BRAIN_ID: brainId })
  pairs.sort(([left], [right]) => right.length - left.length)
  return (value: string) => {
    let resolved = value
    for (const [key, replacement] of pairs) {
      resolved = resolved.split(key).join(replacement)
    }
    return resolved
  }
}

const substituteJson = (value: unknown, substitute: (input: string) => string): unknown => {
  if (typeof value === 'string') return substitute(value)
  if (Array.isArray(value)) return value.map((item) => substituteJson(item, substitute))
  if (isRecord(value)) {
    const next: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(value)) {
      next[key] = substituteJson(child, substitute)
    }
    return next
  }
  return value
}

const buildHeaders = (
  headers: Readonly<Record<string, string>> | undefined,
  substitute: (value: string) => string,
  authToken: string | undefined,
): Headers => {
  const out = new Headers()
  if (headers !== undefined) {
    for (const [key, value] of Object.entries(headers)) {
      out.set(key, substitute(value))
    }
  }
  if (authToken !== undefined) {
    out.set('authorization', `Bearer ${authToken}`)
  }
  return out
}

const readResponseText = async (response: ResponseLike): Promise<string> => {
  try {
    return await response.text()
  } catch (error) {
    return `<failed to read response body: ${formatError(error)}>`
  }
}

const fetchWithTimeout = async (
  input: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<FetchResponse> => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(input, { ...init, signal: controller.signal })
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`request timed out after ${timeoutMs} ms`)
    }
    throw error
  } finally {
    clearTimeout(timer)
  }
}

const makeBrainId = (prefix: string, index: number): string =>
  `${prefix}-${index + 1}-${randomUUID().slice(0, 8)}`

const sameStringMultiset = (left: readonly string[], right: readonly string[]): boolean => {
  if (left.length !== right.length) return false
  const counts = new Map<string, number>()
  for (const value of left) {
    counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  for (const value of right) {
    const count = counts.get(value)
    if (count === undefined) return false
    if (count === 1) counts.delete(value)
    else counts.set(value, count - 1)
  }
  return counts.size === 0
}

const extractItems = (parsed: unknown): readonly Record<string, unknown>[] => {
  if (!isRecord(parsed)) {
    throw new Error(`expected JSON object with items array, got ${typeof parsed}`)
  }
  const items = parsed.items
  if (!Array.isArray(items)) {
    throw new Error('expected JSON object with items array')
  }
  return items.filter(isRecord)
}

const compareJson = (expected: unknown, actual: unknown, path = 'root'): void => {
  if (expected === null || expected === undefined) {
    if (actual !== expected) {
      throw new Error(`${path}: want ${String(expected)} got ${String(actual)}`)
    }
    return
  }

  if (typeof expected === 'string') {
    if (expected === '<ISO-8601>') {
      if (typeof actual !== 'string') {
        throw new Error(`${path}: want ISO-8601 string got ${typeof actual}`)
      }
      const parsed = new Date(actual)
      if (!Number.isFinite(parsed.getTime())) {
        throw new Error(`${path}: ${actual} is not ISO-8601`)
      }
      return
    }
    if (actual !== expected) {
      throw new Error(`${path}: want ${JSON.stringify(expected)} got ${JSON.stringify(actual)}`)
    }
    return
  }

  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) {
      throw new Error(`${path}: want array got ${typeof actual}`)
    }
    if (actual.length < expected.length) {
      throw new Error(`${path}: array length ${actual.length} < expected ${expected.length}`)
    }
    for (let index = 0; index < expected.length; index += 1) {
      compareJson(expected[index], actual[index], `${path}[${index}]`)
    }
    return
  }

  if (typeof expected === 'object') {
    if (!isRecord(actual)) {
      throw new Error(`${path}: want object got ${typeof actual}`)
    }
    for (const [key, value] of Object.entries(expected)) {
      if (!(key in actual)) {
        throw new Error(`${path}: missing key ${JSON.stringify(key)}`)
      }
      compareJson(value, actual[key], `${path}.${key}`)
    }
    return
  }

  if (actual !== expected) {
    throw new Error(`${path}: want ${String(expected)} got ${String(actual)}`)
  }
}

const runBodyAssertion = (assertion: ConformanceBodyAssertion, parsed: unknown): void => {
  switch (assertion.kind) {
    case 'items-include-path': {
      const items = extractItems(parsed)
      if (!items.some((item) => item.path === assertion.path)) {
        throw new Error(`items do not include ${JSON.stringify(assertion.path)}`)
      }
      return
    }
    case 'items-exclude-path': {
      const items = extractItems(parsed)
      if (items.some((item) => item.path === assertion.path)) {
        throw new Error(`items unexpectedly include ${JSON.stringify(assertion.path)}`)
      }
      return
    }
    case 'items-files-equal': {
      const expectedPaths = Array.isArray(assertion.paths)
        ? assertion.paths.filter((path): path is string => typeof path === 'string')
        : []
      const items = extractItems(parsed)
      const actual = items
        .filter((item) => item.is_dir !== true)
        .flatMap((item) => (typeof item.path === 'string' ? [item.path] : []))
      if (!sameStringMultiset(actual, expectedPaths)) {
        throw new Error(
          `items-files-equal: want ${JSON.stringify(expectedPaths)} got ${JSON.stringify(actual)}`,
        )
      }
      return
    }
    case 'items-dirs-equal': {
      const expectedPaths = Array.isArray(assertion.paths)
        ? assertion.paths.filter((path): path is string => typeof path === 'string')
        : []
      const items = extractItems(parsed)
      const actual = items
        .filter((item) => item.is_dir === true)
        .flatMap((item) => (typeof item.path === 'string' ? [item.path] : []))
      if (!sameStringMultiset(actual, expectedPaths)) {
        throw new Error(
          `items-dirs-equal: want ${JSON.stringify(expectedPaths)} got ${JSON.stringify(actual)}`,
        )
      }
      return
    }
    case 'json-field-equals': {
      if (!isRecord(parsed)) {
        throw new Error(`json-field-equals: expected JSON object got ${typeof parsed}`)
      }
      const field = typeof assertion.field === 'string' ? assertion.field : ''
      if (field === '') {
        throw new Error('json-field-equals: missing field name')
      }
      const actual = parsed[field]
      if (!isDeepStrictEqual(actual, assertion.value)) {
        throw new Error(
          `field ${JSON.stringify(field)}: want ${formatValue(assertion.value)} got ${formatValue(actual)}`,
        )
      }
      return
    }
    default:
      throw new Error(`unknown body assertion kind ${JSON.stringify(assertion.kind)}`)
  }
}

const assertExpectedBody = async (
  expected: ConformanceExpectedResponse,
  response: ResponseLike,
): Promise<void> => {
  if (expected.bodyBase64 !== undefined) {
    const wanted = Buffer.from(expected.bodyBase64, 'base64')
    const actual = Buffer.from(new Uint8Array(await response.clone().arrayBuffer()))
    if (!actual.equals(wanted)) {
      throw new Error(
        `body mismatch: want ${wanted.toString('base64')} got ${actual.toString('base64')}`,
      )
    }
  }

  if (expected.body !== undefined) {
    const text = await response.clone().text()
    let parsed: unknown
    try {
      parsed = text === '' ? null : JSON.parse(text)
    } catch (error) {
      throw new Error(`decode response JSON: ${formatError(error)} body=${text}`)
    }
    compareJson(expected.body, parsed)
  }

  if (expected.bodyAssertions !== undefined && expected.bodyAssertions.length > 0) {
    const text = await response.clone().text()
    let parsed: unknown
    try {
      parsed = text === '' ? {} : JSON.parse(text)
    } catch (error) {
      throw new Error(`decode response JSON: ${formatError(error)} body=${text}`)
    }
    for (const assertion of expected.bodyAssertions) {
      runBodyAssertion(assertion, parsed)
    }
  }
}

const assertStreamAssertions = async (
  expected: ConformanceExpectedResponse,
  response: ResponseLike,
  timeoutMs: number,
): Promise<void> => {
  const assertions = expected.streamAssertions
  if (assertions === undefined || assertions.length === 0) return
  if (response.body === null) {
    throw new Error('expected SSE response body, got none')
  }

  const wanted = new Set<string>()
  for (const assertion of assertions) {
    if (assertion.kind !== 'expect-event') {
      throw new Error(`unknown stream assertion kind ${JSON.stringify(assertion.kind)}`)
    }
    const eventName = typeof assertion.event === 'string' ? assertion.event : ''
    if (eventName === '') {
      throw new Error('expect-event assertion missing event name')
    }
    wanted.add(eventName)
  }

  const seen = new Set<string>()
  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  const parser = new SSEParser()
  const deadline = Date.now() + timeoutMs
  try {
    while (seen.size < wanted.size && Date.now() < deadline) {
      const remaining = Math.max(1, deadline - Date.now())
      const outcome = await Promise.race([
        reader.read(),
        new Promise<{ readonly timedOut: true }>((resolve) => {
          setTimeout(() => resolve({ timedOut: true }), Math.min(remaining, 200))
        }),
      ])
      if ('timedOut' in outcome) continue
      if (outcome.done) {
        for (const event of parser.flush()) {
          if (wanted.has(event.event)) {
            seen.add(event.event)
          }
        }
        break
      }
      const chunk = decoder.decode(outcome.value, { stream: true })
      for (const event of parser.feed(chunk)) {
        if (wanted.has(event.event)) {
          seen.add(event.event)
          if (seen.size === wanted.size) break
        }
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined)
    reader.releaseLock()
  }

  const missing = [...wanted].filter((event) => !seen.has(event))
  if (missing.length > 0) {
    throw new Error(`expected SSE events ${JSON.stringify(missing)} never arrived`)
  }
}

const assertExpectedResponse = async (
  rawExpected: ConformanceExpectedResponse,
  response: ResponseLike,
  timeoutMs: number,
): Promise<void> => {
  if (rawExpected.status !== undefined && response.status !== rawExpected.status) {
    const body = await readResponseText(response.clone())
    throw new Error(`want status ${rawExpected.status} got ${response.status} body=${body}`)
  }

  if (rawExpected.contentType !== undefined) {
    const actual = response.headers.get('content-type') ?? ''
    if (!actual.includes(rawExpected.contentType)) {
      throw new Error(
        `want content-type containing ${JSON.stringify(rawExpected.contentType)} got ${JSON.stringify(actual)}`,
      )
    }
  }

  await assertExpectedBody(rawExpected, response)
  await assertStreamAssertions(rawExpected, response, timeoutMs)
}

const assertEventPayload = (rawExpected: ConformanceExpectedResponse, payload: string): void => {
  if (rawExpected.bodyBase64 !== undefined) {
    const wanted = Buffer.from(rawExpected.bodyBase64, 'base64')
    const actual = Buffer.from(payload, 'utf8')
    if (!actual.equals(wanted)) {
      throw new Error(
        `event body mismatch: want ${wanted.toString('base64')} got ${actual.toString('base64')}`,
      )
    }
  }

  if (rawExpected.body !== undefined || rawExpected.bodyAssertions !== undefined) {
    let parsed: unknown
    try {
      parsed = JSON.parse(payload)
    } catch (error) {
      throw new Error(`decode event JSON: ${formatError(error)} payload=${payload}`)
    }
    if (rawExpected.body !== undefined) compareJson(rawExpected.body, parsed)
    if (rawExpected.bodyAssertions !== undefined) {
      for (const assertion of rawExpected.bodyAssertions) {
        runBodyAssertion(assertion, parsed)
      }
    }
  }
}

const createSseSubscriber = (url: string, headers: Headers): SseSubscriber => {
  const controller = new AbortController()
  const events: SSEEvent[] = []
  const waiters = new Map<string, SseWaiter[]>()
  let terminalError: Error | undefined

  const rejectWaiters = (error: Error): void => {
    for (const [event, pending] of waiters.entries()) {
      waiters.delete(event)
      for (const waiter of pending) {
        clearTimeout(waiter.timer)
        waiter.reject(error)
      }
    }
  }

  const resolveWaiters = (event: SSEEvent): void => {
    const pending = waiters.get(event.event)
    if (pending === undefined) return
    waiters.delete(event.event)
    for (const waiter of pending) {
      clearTimeout(waiter.timer)
      waiter.resolve(event.data)
    }
  }

  const run = (async (): Promise<void> => {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers,
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(`SSE subscribe failed: ${response.status} ${response.statusText}`)
      }
      if (response.body === null) {
        throw new Error('SSE subscribe failed: response body missing')
      }
      for await (const event of iterateSSE(response.body)) {
        events.push(event)
        resolveWaiters(event)
      }
      if (!controller.signal.aborted) {
        terminalError = new Error('SSE stream closed before the case finished')
        rejectWaiters(terminalError)
      }
    } catch (error) {
      if (controller.signal.aborted) return
      terminalError = asError(error)
      rejectWaiters(terminalError)
    }
  })()

  return {
    waitFor: async (eventName: string, timeoutMs = DEFAULT_SSE_TIMEOUT_MS): Promise<string> => {
      const existing = events.find((event) => event.event === eventName)
      if (existing !== undefined) return existing.data
      if (terminalError !== undefined) throw terminalError

      return new Promise<string>((resolve, reject) => {
        const timer = setTimeout(() => {
          const pending = waiters.get(eventName)
          if (pending === undefined) {
            reject(new Error(`timeout waiting for SSE event ${JSON.stringify(eventName)}`))
            return
          }
          waiters.set(
            eventName,
            pending.filter((candidate) => candidate.timer !== timer),
          )
          reject(new Error(`timeout waiting for SSE event ${JSON.stringify(eventName)}`))
        }, timeoutMs)

        const pending = waiters.get(eventName) ?? []
        pending.push({ resolve, reject, timer })
        waiters.set(eventName, pending)
      })
    },
    close: async (): Promise<void> => {
      controller.abort()
      rejectWaiters(new Error('SSE subscriber closed'))
      await run.catch(() => undefined)
    },
  }
}

const executeStepRequest = async (
  step: ConformanceStep,
  context: RunCaseContext,
): Promise<FetchResponse> => {
  const method = (step.method ?? 'GET').toUpperCase()
  const path = context.substitute(step.path ?? '')
  const url = resolveRequestUrl(context.baseUrl, path)
  const headers = buildHeaders(step.headers, context.substitute, context.authToken)

  let body: RequestInit['body'] | undefined
  if (step.bodyBase64 !== undefined && step.bodyBase64 !== '') {
    body = new Uint8Array(Buffer.from(context.substitute(step.bodyBase64), 'base64'))
    if (!headers.has('content-type')) {
      headers.set('content-type', 'application/octet-stream')
    }
  } else if (step.bodyJson !== undefined) {
    body = JSON.stringify(substituteJson(step.bodyJson, context.substitute))
    if (!headers.has('content-type')) {
      headers.set('content-type', 'application/json')
    }
  }

  return await fetchWithTimeout(
    url,
    {
      method,
      headers,
      ...(body !== undefined ? { body } : {}),
    },
    context.requestTimeoutMs,
  )
}

const runStep = async (rawStep: ConformanceStep, context: RunCaseContext): Promise<void> => {
  const resolvedStep = substituteJson(rawStep, context.substitute) as ConformanceStep

  switch (resolvedStep.kind) {
    case 'open-sse': {
      const name = resolvedStep.name ?? ''
      if (name === '') throw new Error('open-sse step missing name')
      const path = resolvedStep.path ?? ''
      if (path === '') throw new Error('open-sse step missing path')

      const existing = context.ssePool.get(name)
      if (existing !== undefined) {
        await existing.close()
      }

      const headers = buildHeaders(
        { accept: 'text/event-stream', ...(resolvedStep.headers ?? {}) },
        context.substitute,
        context.authToken,
      )
      context.ssePool.set(
        name,
        createSseSubscriber(resolveRequestUrl(context.baseUrl, path), headers),
      )
      return
    }
    case 'await-sse-event': {
      const name = resolvedStep.name ?? ''
      const event = resolvedStep.event ?? ''
      if (name === '' || event === '') {
        throw new Error('await-sse-event step missing name or event')
      }
      const subscriber = context.ssePool.get(name)
      if (subscriber === undefined) {
        throw new Error(`SSE subscriber ${JSON.stringify(name)} not opened`)
      }
      await subscriber.waitFor(event, context.sseTimeoutMs)
      return
    }
    case 'close-sse': {
      const name = resolvedStep.name ?? ''
      if (name === '') throw new Error('close-sse step missing name')
      const subscriber = context.ssePool.get(name)
      if (subscriber !== undefined) {
        await subscriber.close()
        context.ssePool.delete(name)
      }
      return
    }
    case undefined:
      break
    default:
      throw new Error(`unknown step kind ${JSON.stringify(resolvedStep.kind)}`)
  }

  const response = await executeStepRequest(resolvedStep, context)
  if (
    resolvedStep.expectedStatus !== undefined &&
    response.status !== resolvedStep.expectedStatus
  ) {
    const body = await readResponseText(response.clone())
    throw new Error(
      `setup step ${resolvedStep.method ?? 'GET'} ${resolvedStep.path ?? ''}: want status ${resolvedStep.expectedStatus} got ${response.status} body=${body}`,
    )
  }
  if (resolvedStep.expectedBodyBase64 !== undefined && resolvedStep.expectedBodyBase64 !== '') {
    const wanted = Buffer.from(resolvedStep.expectedBodyBase64, 'base64')
    const actual = Buffer.from(new Uint8Array(await response.arrayBuffer()))
    if (!actual.equals(wanted)) {
      throw new Error(
        `setup step body mismatch: want ${wanted.toString('base64')} got ${actual.toString('base64')}`,
      )
    }
  }
}

const createBrain = async (
  baseUrl: URL,
  authToken: string | undefined,
  requestTimeoutMs: number,
  brainId: string,
): Promise<void> => {
  const response = await fetchWithTimeout(
    resolveRequestUrl(baseUrl, '/v1/brains'),
    {
      method: 'POST',
      headers: buildHeaders({ 'content-type': 'application/json' }, (value) => value, authToken),
      body: JSON.stringify({ brainId }),
    },
    requestTimeoutMs,
  )
  if (response.status !== 201) {
    const body = await readResponseText(response)
    throw new Error(`create brain ${brainId}: want 201 got ${response.status} body=${body}`)
  }
}

const deleteBrain = async (
  baseUrl: URL,
  authToken: string | undefined,
  requestTimeoutMs: number,
  brainId: string,
): Promise<void> => {
  const response = await fetchWithTimeout(
    resolveRequestUrl(baseUrl, `/v1/brains/${encodeURIComponent(brainId)}`),
    {
      method: 'DELETE',
      headers: buildHeaders({ 'x-confirm-delete': 'yes' }, (value) => value, authToken),
    },
    requestTimeoutMs,
  )
  if (response.status !== 204) {
    const body = await readResponseText(response)
    throw new Error(`delete brain ${brainId}: want 204 got ${response.status} body=${body}`)
  }
}

const closeSubscribers = async (pool: Map<string, SseSubscriber>): Promise<void> => {
  for (const [name, subscriber] of pool.entries()) {
    await subscriber.close()
    pool.delete(name)
  }
}

const formatError = (error: unknown): string =>
  error instanceof Error ? error.message : String(error)

const asError = (error: unknown): Error =>
  error instanceof Error ? error : new Error(String(error))

const combineErrors = (
  main: string | undefined,
  cleanup: string | undefined,
): string | undefined => {
  if (main === undefined) return cleanup
  if (cleanup === undefined) return main
  return `${main}; cleanup failed: ${cleanup}`
}

const runCase = async (
  testCase: ConformanceCase,
  index: number,
  options: {
    readonly baseUrl: URL
    readonly authToken: string | undefined
    readonly requestTimeoutMs: number
    readonly sseTimeoutMs: number
    readonly brainIdPrefix: string
    readonly placeholders: Readonly<Record<string, string>>
  },
): Promise<ConformanceCaseResult> => {
  const brainId = makeBrainId(options.brainIdPrefix, index)
  const substitute = substituteFactory(options.placeholders, brainId)
  const context: RunCaseContext = {
    baseUrl: options.baseUrl,
    authToken: options.authToken,
    requestTimeoutMs: options.requestTimeoutMs,
    sseTimeoutMs: options.sseTimeoutMs,
    substitute,
    ssePool: new Map(),
  }

  let createdBrain = false
  let mainError: string | undefined
  try {
    await createBrain(options.baseUrl, options.authToken, options.requestTimeoutMs, brainId)
    createdBrain = true

    for (const step of testCase.setup ?? []) {
      await runStep(step, context)
    }

    const resolvedExpected = substituteJson(
      testCase.expectedResponse,
      substitute,
    ) as ConformanceExpectedResponse

    if (testCase.request.kind === 'await-sse-event') {
      const name = testCase.request.name ?? ''
      const event = testCase.request.event ?? ''
      if (name === '' || event === '') {
        throw new Error('await-sse-event request missing name or event')
      }
      const subscriber = context.ssePool.get(name)
      if (subscriber === undefined) {
        throw new Error(`SSE subscriber ${JSON.stringify(name)} not opened`)
      }
      const payload = await subscriber.waitFor(event, options.sseTimeoutMs)
      assertEventPayload(resolvedExpected, payload)
    } else {
      const response = await executeStepRequest(testCase.request, context)
      await assertExpectedResponse(resolvedExpected, response, options.sseTimeoutMs)
    }

    for (const step of testCase.followUp ?? []) {
      await runStep(step, context)
    }
    for (const step of testCase.teardown ?? []) {
      await runStep(step, context)
    }
  } catch (error) {
    mainError = formatError(error)
  }

  let cleanupError: string | undefined
  try {
    await closeSubscribers(context.ssePool)
    if (createdBrain) {
      await deleteBrain(options.baseUrl, options.authToken, options.requestTimeoutMs, brainId)
    }
  } catch (error) {
    cleanupError = formatError(error)
  }

  const error = combineErrors(mainError, cleanupError)
  if (error === undefined) {
    return { name: testCase.name, brainId, ok: true }
  }
  return { name: testCase.name, brainId, ok: false, error }
}

const assertDocument = (value: unknown, source: string): ConformanceDocument => {
  if (!isRecord(value)) {
    throw new Error(`conformance fixture ${source} must be an object`)
  }

  const cases = value.cases
  if (!Array.isArray(cases)) {
    throw new Error(`conformance fixture ${source} missing cases array`)
  }

  const placeholders = value.placeholders
  if (!isRecord(placeholders)) {
    throw new Error(`conformance fixture ${source} missing placeholders object`)
  }

  for (let index = 0; index < cases.length; index += 1) {
    const testCase = cases[index]
    if (!isRecord(testCase)) {
      throw new Error(`conformance fixture ${source} case ${index} must be an object`)
    }
    if (asString(testCase.name) === undefined) {
      throw new Error(`conformance fixture ${source} case ${index} missing name`)
    }
    if (!isRecord(testCase.request)) {
      throw new Error(`conformance fixture ${source} case ${index} missing request object`)
    }
    if (!isRecord(testCase.expectedResponse)) {
      throw new Error(`conformance fixture ${source} case ${index} missing expectedResponse object`)
    }
  }

  return value as ConformanceDocument
}

export const resolveConformanceFixturePath = async (): Promise<string> => {
  let dir = dirname(fileURLToPath(import.meta.url))
  for (;;) {
    for (const candidate of [
      join(dir, DIST_FIXTURE_FILENAME),
      join(dir, ...REPO_FIXTURE_RELATIVE_PATH),
    ]) {
      try {
        await access(candidate)
        return candidate
      } catch {
        /* keep walking */
      }
    }
    const parent = dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  throw new Error(
    `could not find ${DIST_FIXTURE_FILENAME} or ${REPO_FIXTURE_RELATIVE_PATH.join('/')} from ${fileURLToPath(import.meta.url)}`,
  )
}

export const loadConformanceDocument = async (
  fixturePath?: string,
): Promise<ConformanceDocument> => {
  const resolvedPath = fixturePath ?? (await resolveConformanceFixturePath())
  const raw = await readFile(resolvedPath, 'utf8')
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (error) {
    throw new Error(`parse conformance fixture ${resolvedPath}: ${formatError(error)}`)
  }
  return assertDocument(parsed, resolvedPath)
}

export const runConformanceSuite = async (
  options: RunConformanceSuiteOptions,
): Promise<ConformanceSuiteResult> => {
  const document = await loadConformanceDocument(options.fixturePath)
  const baseUrl = normaliseBaseUrl(options.baseUrl)
  const requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
  const sseTimeoutMs = options.sseTimeoutMs ?? DEFAULT_SSE_TIMEOUT_MS
  const brainIdPrefix = options.brainIdPrefix ?? DEFAULT_BRAIN_ID_PREFIX

  const cases: ConformanceCaseResult[] = []
  for (let index = 0; index < document.cases.length; index += 1) {
    const testCase = document.cases[index]
    if (testCase === undefined) continue
    cases.push(
      await runCase(testCase, index, {
        baseUrl,
        authToken: options.authToken,
        requestTimeoutMs,
        sseTimeoutMs,
        brainIdPrefix,
        placeholders: document.placeholders,
      }),
    )
  }

  const passed = cases.filter((testCase) => testCase.ok).length
  const failed = cases.length - passed
  return {
    total: cases.length,
    passed,
    failed,
    cases,
  }
}
