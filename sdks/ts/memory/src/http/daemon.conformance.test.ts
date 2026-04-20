// SPDX-License-Identifier: Apache-2.0

/**
 * Conformance harness for the memory HTTP daemon. Replays every case
 * from `spec/conformance/http-contract.json` against an in-process
 * daemon and asserts the documented HTTP response shape.
 *
 * Parity target: 28/29 — the one skipped case is the harness-side
 * "auth header forwarded when apiKey is set" test (same skip as Go).
 */

import { readFile } from 'node:fs/promises'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { Daemon, createRouter } from './index.js'

const HERE = dirname(fileURLToPath(import.meta.url))

const resolveConformanceFile = async (): Promise<string> => {
  let dir = HERE
  for (let i = 0; i < 8; i++) {
    const candidate = join(dir, 'spec', 'conformance', 'http-contract.json')
    try {
      await readFile(candidate, 'utf8')
      return candidate
    } catch {
      /* keep walking */
    }
    const parent = resolve(dir, '..')
    if (parent === dir) break
    dir = parent
  }
  throw new Error('conformance fixture not found')
}

type ConformanceStep = {
  kind?: string
  name?: string
  method?: string
  path?: string
  headers?: Record<string, string>
  bodyBase64?: string
  bodyJson?: unknown
  event?: string
  expectedStatus?: number
  expectedBodyBase64?: string
}

type ConformanceCase = {
  name: string
  setup?: ConformanceStep[]
  request: Record<string, unknown>
  expectedResponse: Record<string, unknown>
  followUp?: ConformanceStep[]
  teardown?: ConformanceStep[]
}

type ConformanceFile = {
  placeholders: Record<string, string>
  cases: ConformanceCase[]
}

const SKIP: Record<string, string> = {
  'auth header forwarded when apiKey is set':
    'harness asserts header forwarding rather than server behaviour',
}

// Provisioned per case; mirror Go so fresh state is guaranteed.
const BRAIN_ID = 'conformance-brain'

const BASE = 'http://conformance.test'

const substituteFactory = (placeholders: Record<string, string>): ((s: string) => string) => {
  const pairs = Object.entries({ ...placeholders, BRAIN_ID })
  // Longest keys first so nested substitutions don't clobber each other.
  pairs.sort(([a], [b]) => b.length - a.length)
  return (s: string) => {
    let out = s
    for (const [k, v] of pairs) out = out.split(k).join(v)
    return out
  }
}

type SseSubscriber = {
  events: Array<{ event: string; data: string }>
  waitFor: (event: string, timeoutMs?: number) => Promise<string>
  close: () => void
}

const subscribeSseFromHandler = (
  handler: (req: Request) => Promise<Response>,
  path: string,
): SseSubscriber => {
  const controller = new AbortController()
  const events: Array<{ event: string; data: string }> = []
  const waiters = new Map<string, Array<(data: string) => void>>()

  const run = async (): Promise<void> => {
    const resp = await handler(
      new Request(`${BASE}${path}`, {
        method: 'GET',
        headers: { accept: 'text/event-stream' },
        signal: controller.signal,
      }),
    )
    if (resp.body === null) return
    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    let currentEvent = ''
    let dataLines: string[] = []
    const flush = (): void => {
      if (currentEvent !== '' || dataLines.length > 0) {
        const payload = { event: currentEvent, data: dataLines.join('\n') }
        events.push(payload)
        const ws = waiters.get(currentEvent) ?? []
        while (ws.length > 0) {
          const w = ws.shift()
          w?.(payload.data)
        }
      }
      currentEvent = ''
      dataLines = []
    }
    try {
      while (!controller.signal.aborted) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        while (true) {
          const nl = buffer.indexOf('\n')
          if (nl === -1) break
          let line = buffer.slice(0, nl)
          buffer = buffer.slice(nl + 1)
          if (line.endsWith('\r')) line = line.slice(0, -1)
          if (line === '') {
            flush()
            continue
          }
          if (line.startsWith(':')) continue
          const colon = line.indexOf(':')
          const field = colon === -1 ? line : line.slice(0, colon)
          let value = colon === -1 ? '' : line.slice(colon + 1)
          if (value.startsWith(' ')) value = value.slice(1)
          if (field === 'event') currentEvent = value
          else if (field === 'data') dataLines.push(value)
        }
      }
    } catch {
      /* stream closed */
    } finally {
      try {
        reader.releaseLock()
      } catch {
        /* ignore */
      }
    }
  }

  void run().catch(() => undefined)

  return {
    events,
    waitFor: (event, timeoutMs = 2000) => {
      return new Promise<string>((resolve, reject) => {
        const existing = events.find((e) => e.event === event)
        if (existing !== undefined) {
          resolve(existing.data)
          return
        }
        const timer = setTimeout(() => {
          reject(new Error(`timeout waiting for SSE ${event}`))
        }, timeoutMs)
        const list = waiters.get(event) ?? []
        list.push((data) => {
          clearTimeout(timer)
          resolve(data)
        })
        waiters.set(event, list)
      })
    },
    close: () => controller.abort(),
  }
}

type RunCtx = {
  handler: (req: Request) => Promise<Response>
  substitute: (s: string) => string
  ssePool: Map<string, SseSubscriber>
}

const buildRequest = (step: ConformanceStep, ctx: RunCtx): Request => {
  const path = ctx.substitute(step.path ?? '')
  const url = `${BASE}${path}`
  const method = (step.method ?? 'GET').toUpperCase()
  const headers = new Headers()
  if (step.headers !== undefined) {
    for (const [k, v] of Object.entries(step.headers)) {
      headers.set(k, ctx.substitute(String(v)))
    }
  }
  let body: BodyInit | undefined
  if (step.bodyBase64 !== undefined && step.bodyBase64 !== '') {
    const decoded = Buffer.from(ctx.substitute(step.bodyBase64), 'base64')
    body = new Uint8Array(decoded)
    if (!headers.has('content-type')) headers.set('content-type', 'application/octet-stream')
  } else if (step.bodyJson !== undefined && step.bodyJson !== null) {
    body = JSON.stringify(step.bodyJson)
    if (!headers.has('content-type')) headers.set('content-type', 'application/json')
  }
  return new Request(url, { method, headers, ...(body !== undefined ? { body } : {}) })
}

const runStep = async (step: ConformanceStep, ctx: RunCtx): Promise<void> => {
  if (step.kind === 'open-sse') {
    const name = step.name ?? ''
    const path = ctx.substitute(step.path ?? '')
    const existing = ctx.ssePool.get(name)
    if (existing !== undefined) existing.close()
    ctx.ssePool.set(name, subscribeSseFromHandler(ctx.handler, path))
    return
  }
  if (step.kind === 'await-sse-event') {
    const sub = ctx.ssePool.get(step.name ?? '')
    if (sub === undefined) throw new Error(`SSE subscriber ${step.name} not opened`)
    await sub.waitFor(step.event ?? '')
    return
  }
  if (step.kind === 'close-sse') {
    const sub = ctx.ssePool.get(step.name ?? '')
    if (sub !== undefined) {
      sub.close()
      ctx.ssePool.delete(step.name ?? '')
    }
    return
  }
  const resp = await ctx.handler(buildRequest(step, ctx))
  if (step.expectedStatus !== undefined && resp.status !== step.expectedStatus) {
    const text = await resp.text()
    throw new Error(
      `step ${step.method} ${step.path}: want ${step.expectedStatus} got ${resp.status} body=${text}`,
    )
  }
  if (step.expectedBodyBase64 !== undefined && step.expectedBodyBase64 !== '') {
    const body = new Uint8Array(await resp.arrayBuffer())
    const want = Buffer.from(ctx.substitute(step.expectedBodyBase64), 'base64')
    expect(Buffer.from(body).equals(want)).toBe(true)
  }
}

const stepFromRequest = (req: Record<string, unknown>): ConformanceStep => {
  const out: ConformanceStep = {}
  if (typeof req.kind === 'string') out.kind = req.kind
  if (typeof req.name === 'string') out.name = req.name
  if (typeof req.method === 'string') out.method = req.method
  if (typeof req.path === 'string') out.path = req.path
  if (typeof req.event === 'string') out.event = req.event
  if (typeof req.bodyBase64 === 'string') out.bodyBase64 = req.bodyBase64
  if (req.bodyJson !== undefined) out.bodyJson = req.bodyJson
  const headers = req.headers
  if (typeof headers === 'object' && headers !== null) {
    const h: Record<string, string> = {}
    for (const [k, v] of Object.entries(headers as Record<string, unknown>)) h[k] = String(v)
    out.headers = h
  }
  return out
}

type BodyAssertion = Record<string, unknown>

const assertListAssertion = (assertion: BodyAssertion, parsed: unknown): void => {
  const items = Array.isArray((parsed as { items?: unknown }).items)
    ? ((parsed as { items: unknown[] }).items as Record<string, unknown>[])
    : []
  const kind = assertion.kind as string
  switch (kind) {
    case 'items-include-path': {
      const want = String(assertion.path)
      expect(items.some((it) => it.path === want)).toBe(true)
      break
    }
    case 'items-exclude-path': {
      const want = String(assertion.path)
      expect(items.every((it) => it.path !== want)).toBe(true)
      break
    }
    case 'items-files-equal': {
      const want = (assertion.paths as string[]) ?? []
      const got = items.filter((it) => it.is_dir !== true).map((it) => String(it.path))
      expect(new Set(got)).toEqual(new Set(want))
      break
    }
    case 'items-dirs-equal': {
      const want = (assertion.paths as string[]) ?? []
      const got = items.filter((it) => it.is_dir === true).map((it) => String(it.path))
      expect(new Set(got)).toEqual(new Set(want))
      break
    }
    case 'json-field-equals': {
      const field = String(assertion.field)
      const value = assertion.value
      expect((parsed as Record<string, unknown>)[field]).toEqual(value)
      break
    }
    default:
      throw new Error(`unknown bodyAssertion kind ${kind}`)
  }
}

const compareJson = (expected: unknown, actual: unknown, path = 'root'): void => {
  if (expected === null || expected === undefined) {
    expect(actual).toBe(expected)
    return
  }
  if (typeof expected === 'string') {
    if (expected === '<ISO-8601>') {
      expect(typeof actual).toBe('string')
      const when = new Date(String(actual))
      expect(Number.isFinite(when.getTime())).toBe(true)
      return
    }
    expect(actual).toBe(expected)
    return
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) throw new Error(`${path}: expected array`)
    expect(actual.length).toBeGreaterThanOrEqual(expected.length)
    for (let i = 0; i < expected.length; i++) {
      compareJson(expected[i], actual[i], `${path}[${i}]`)
    }
    return
  }
  if (typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object' || Array.isArray(actual)) {
      throw new Error(`${path}: expected object got ${typeof actual}`)
    }
    const a = actual as Record<string, unknown>
    for (const [k, v] of Object.entries(expected as Record<string, unknown>)) {
      expect(a).toHaveProperty(k)
      compareJson(v, a[k], `${path}.${k}`)
    }
    return
  }
  expect(actual).toBe(expected)
}

let fixtures: Array<{ daemon: Daemon; tempDir: string }> = []

beforeEach(() => {
  fixtures = []
})

afterEach(async () => {
  for (const f of fixtures) {
    await f.daemon.close()
    await rm(f.tempDir, { recursive: true, force: true })
  }
  fixtures = []
})

const makeFixture = async (): Promise<{
  handler: (req: Request) => Promise<Response>
  daemon: Daemon
}> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-conf-'))
  const daemon = new Daemon({ root: tempDir })
  await daemon.start()
  await daemon.brains.create(BRAIN_ID)
  const router = createRouter(daemon)
  const handler = async (req: Request): Promise<Response> => router(req)
  fixtures.push({ daemon, tempDir })
  return { handler, daemon }
}

describe('memory daemon conformance', async () => {
  const fixturePath = await resolveConformanceFile()
  const raw = await readFile(fixturePath, 'utf8')
  const doc = JSON.parse(raw) as ConformanceFile

  for (const c of doc.cases) {
    if (SKIP[c.name] !== undefined) {
      it.skip(c.name, () => undefined)
      continue
    }
    it(c.name, async () => {
      const { handler } = await makeFixture()
      const substitute = substituteFactory(doc.placeholders)
      const ctx: RunCtx = { handler, substitute, ssePool: new Map() }

      try {
        for (const step of c.setup ?? []) await runStep(step, ctx)

        const req = c.request
        if ((req as { kind?: string }).kind === 'await-sse-event') {
          const sub = ctx.ssePool.get(String((req as { name?: string }).name ?? ''))
          expect(sub).toBeDefined()
          await sub?.waitFor(String((req as { event?: string }).event ?? ''))
        } else {
          const step = stepFromRequest(c.request)
          const resp = await handler(buildRequest(step, ctx))
          await assertExpectedResponse(c.expectedResponse, resp, substitute)
        }

        for (const step of c.followUp ?? []) await runStep(step, ctx)
        for (const step of c.teardown ?? []) await runStep(step, ctx)
      } finally {
        for (const sub of ctx.ssePool.values()) sub.close()
      }
    })
  }
})

const assertExpectedResponse = async (
  expected: Record<string, unknown>,
  resp: Response,
  substitute: (s: string) => string,
): Promise<void> => {
  if (typeof expected.status === 'number' && resp.status !== expected.status) {
    const bodyText = await resp.clone().text()
    throw new Error(`status ${resp.status} != ${expected.status}; body=${bodyText}`)
  }
  if (typeof expected.contentType === 'string') {
    const actual = resp.headers.get('content-type') ?? ''
    expect(actual).toContain(expected.contentType)
  }
  if (typeof expected.bodyBase64 === 'string') {
    const want = Buffer.from(substitute(expected.bodyBase64), 'base64')
    const got = Buffer.from(new Uint8Array(await resp.clone().arrayBuffer()))
    expect(got.equals(want)).toBe(true)
  }
  if (expected.body !== undefined) {
    const got = await resp.clone().json()
    compareJson(expected.body, got)
  }
  if (Array.isArray(expected.bodyAssertions)) {
    const text = await resp.clone().text()
    const parsed = text === '' ? {} : JSON.parse(text)
    for (const raw of expected.bodyAssertions) {
      assertListAssertion(raw as BodyAssertion, parsed)
    }
  }
  if (Array.isArray(expected.streamAssertions)) {
    const want = new Set<string>()
    for (const a of expected.streamAssertions) {
      const kind = (a as { kind?: string }).kind
      if (kind === 'expect-event') want.add((a as { event?: string }).event ?? '')
    }
    const seen = new Set<string>()
    const reader = resp.body?.getReader()
    const decoder = new TextDecoder('utf-8')
    const deadline = Date.now() + 2500
    let buffer = ''
    let currentEvent = ''
    try {
      while (seen.size < want.size && Date.now() < deadline) {
        const race = Promise.race([
          reader.read(),
          new Promise<{ done: true; value?: undefined }>((resolve) =>
            setTimeout(() => resolve({ done: true }), 200),
          ),
        ])
        const { value, done } = (await race) as { value?: Uint8Array; done?: boolean }
        if (done === true) break
        if (value !== undefined) buffer += decoder.decode(value, { stream: true })
        while (true) {
          const nl = buffer.indexOf('\n')
          if (nl === -1) break
          let line = buffer.slice(0, nl)
          buffer = buffer.slice(nl + 1)
          if (line.endsWith('\r')) line = line.slice(0, -1)
          if (line === '') {
            if (want.has(currentEvent)) seen.add(currentEvent)
            currentEvent = ''
            continue
          }
          if (line.startsWith('event:')) currentEvent = line.slice(6).trim()
        }
      }
    } finally {
      await reader.cancel().catch(() => undefined)
    }
    expect(Array.from(seen).sort()).toEqual(Array.from(want).sort())
  }
}
