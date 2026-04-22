// SPDX-License-Identifier: Apache-2.0

/**
 * HTTP handlers for the memory daemon. Each handler matches the
 * behaviour of its Go counterpart in `sdks/go/cmd/memory/handler_*.go`
 * so the wire contract stays byte-equivalent.
 */

import {
  READER_AUGMENTED_MAX_TOKENS,
  READER_AUGMENTED_TEMPERATURE,
  buildAugmentedReaderPrompt,
} from '../augmented-reader/prompt.js'
import { resolveDeterministicAugmentedAnswer } from '../augmented-reader/resolver.js'
import { ingestDocument } from '../ingest/index.js'
import type { Message } from '../llm/index.js'
import { parseFrontmatter } from '../memory/frontmatter.js'
import type { ConsolidationReport, ExtractedMemory, Scope } from '../memory/index.js'
import { scopeTopic } from '../memory/index.js'
import { augmentQueryWithTemporal, resolvedTemporalHintLine } from '../query/index.js'
import type { RetrievalFilters } from '../retrieval/index.js'
import { createSseHeartbeat } from '../sse.js'
import {
  ErrConflict,
  ErrInvalidPath,
  ErrNotFound,
  type FileInfo,
  type Path,
  StoreError,
  batchJsonBodyLimit,
  joinPath,
  toPath,
  validatePath,
} from '../store/index.js'

import {
  BrainConflictError,
  BrainNotFoundError,
  type BrainResources,
  type Daemon,
} from './daemon.js'
import {
  confirmationRequired,
  conflict,
  internalError,
  jsonResponse,
  notFound,
  payloadTooLarge,
  storeProblem,
  validationError,
} from './problem.js'
import { startSse } from './sse.js'

const JSON_BODY_LIMIT = 1 * 1024 * 1024
const JSON_SMALL_LIMIT = 64 * 1024

type RawFileInfo = {
  readonly path: string
  readonly size: number
  readonly mtime: string
  readonly is_dir: boolean
}

const toRawFileInfo = (fi: FileInfo): RawFileInfo => ({
  path: fi.path,
  size: fi.size,
  mtime: fi.modTime.toISOString(),
  is_dir: fi.isDir,
})

const parseBool = (raw: string | null): boolean => raw === 'true' || raw === '1'

const readLimitedBody = async (req: Request, limit: number): Promise<Uint8Array | Response> => {
  const raw = await req.arrayBuffer()
  if (raw.byteLength > limit) {
    return payloadTooLarge(`body exceeds ${limit} bytes`)
  }
  return new Uint8Array(raw)
}

const readJsonBody = async <T>(req: Request, limit: number): Promise<T | Response> => {
  const body = await readLimitedBody(req, limit)
  if (body instanceof Response) return body
  if (body.byteLength === 0) {
    return validationError('empty body')
  }
  const text = new TextDecoder('utf-8').decode(body)
  try {
    return JSON.parse(text) as T
  } catch (err) {
    return validationError(`invalid JSON: ${err instanceof Error ? err.message : String(err)}`)
  }
}

const resolveBrain = async (
  daemon: Daemon,
  brainId: string,
): Promise<BrainResources | Response> => {
  if (brainId === '') return validationError('missing brainId')
  try {
    return await daemon.brains.get(brainId)
  } catch (err) {
    if (err instanceof BrainNotFoundError) {
      return notFound(`brain not found: ${brainId}`)
    }
    return internalError(err instanceof Error ? err.message : String(err))
  }
}

const respondError = (err: unknown): Response => {
  const mapped = storeProblem(err)
  if (mapped !== undefined) return mapped
  return internalError(err instanceof Error ? err.message : String(err))
}

type BrainSummary = {
  brainId: string
  description?: string
  created?: string
}

export const handleListBrains = async (daemon: Daemon): Promise<Response> => {
  const ids = await daemon.brains.list()
  const items: BrainSummary[] = ids.map((id) => ({ brainId: id }))
  return jsonResponse(200, { items })
}

export const handleGetBrain = async (daemon: Daemon, brainId: string): Promise<Response> => {
  if (brainId === '') return validationError('missing brainId')
  if (!(await daemon.brainExists(brainId))) {
    return notFound(`brain not found: ${brainId}`)
  }
  return jsonResponse(200, { brainId })
}

export const handleCreateBrain = async (daemon: Daemon, req: Request): Promise<Response> => {
  const body = await readJsonBody<{ brainId?: unknown; description?: unknown }>(
    req,
    JSON_SMALL_LIMIT,
  )
  if (body instanceof Response) return body
  const brainId = typeof body.brainId === 'string' ? body.brainId : ''
  if (brainId === '') return validationError('brainId required')
  const description = typeof body.description === 'string' ? body.description : undefined
  try {
    await daemon.brains.create(brainId)
  } catch (err) {
    if (err instanceof BrainConflictError) return conflict(err.message)
    return respondError(err)
  }
  const summary: BrainSummary = { brainId }
  if (description !== undefined) summary.description = description
  return jsonResponse(201, summary)
}

export const handleDeleteBrain = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  if (brainId === '') return validationError('missing brainId')
  if (req.headers.get('x-confirm-delete') !== 'yes') {
    return confirmationRequired('delete brain requires X-Confirm-Delete: yes header')
  }
  try {
    await daemon.brains.delete(brainId)
  } catch (err) {
    if (err instanceof BrainNotFoundError) return notFound(`brain not found: ${brainId}`)
    return respondError(err)
  }
  return new Response(null, { status: 204 })
}

const extractPath = (url: URL): Path | Response => {
  const raw = url.searchParams.get('path') ?? ''
  try {
    return toPath(raw)
  } catch (err) {
    if (err instanceof ErrInvalidPath) return validationError(err.message)
    return validationError(`invalid path: ${err instanceof Error ? err.message : String(err)}`)
  }
}

export const handleDocRead = async (
  daemon: Daemon,
  _req: Request,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const path = extractPath(url)
  if (path instanceof Response) return path
  try {
    const bytes = await br.store.read(path)
    return new Response(new Uint8Array(bytes), {
      status: 200,
      headers: {
        'content-type': 'application/octet-stream',
        'cache-control': 'no-store',
      },
    })
  } catch (err) {
    return respondError(err)
  }
}

export const handleDocHead = async (
  daemon: Daemon,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const path = extractPath(url)
  if (path instanceof Response) {
    // HEAD requests never carry a body; bubble the status only.
    if (path.status === 400) return new Response(null, { status: 400 })
    return path
  }
  try {
    const exists = await br.store.exists(path)
    if (!exists) return new Response(null, { status: 404 })
    return new Response(null, {
      status: 200,
      headers: { 'cache-control': 'no-store' },
    })
  } catch (err) {
    if (err instanceof ErrNotFound) return new Response(null, { status: 404 })
    if (err instanceof ErrInvalidPath) return new Response(null, { status: 400 })
    return respondError(err)
  }
}

export const handleDocStat = async (
  daemon: Daemon,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const path = extractPath(url)
  if (path instanceof Response) return path
  try {
    const info = await br.store.stat(path)
    return jsonResponse(200, toRawFileInfo(info))
  } catch (err) {
    return respondError(err)
  }
}

export const handleDocList = async (
  daemon: Daemon,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const rawDir = url.searchParams.get('dir') ?? ''
  const dir: Path | '' = rawDir === '' ? '' : (rawDir as Path)
  try {
    if (dir !== '') validatePath(dir)
  } catch (err) {
    if (err instanceof ErrInvalidPath) return validationError(err.message)
    return validationError(`invalid dir: ${err instanceof Error ? err.message : String(err)}`)
  }
  const globParam = url.searchParams.get('glob')
  const opts = {
    recursive: parseBool(url.searchParams.get('recursive')),
    includeGenerated: parseBool(url.searchParams.get('include_generated')),
    ...(globParam !== null && globParam !== '' ? { glob: globParam } : {}),
  }
  try {
    const items = await br.store.list(dir, opts)
    return jsonResponse(200, { items: items.map(toRawFileInfo) })
  } catch (err) {
    return respondError(err)
  }
}

export const handleDocWrite = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const path = extractPath(url)
  if (path instanceof Response) return path
  const body = await readLimitedBody(req, daemon.bodyLimits.singleDocumentBytes)
  if (body instanceof Response) return body
  try {
    await br.store.write(path, Buffer.from(body))
    return new Response(null, { status: 204 })
  } catch (err) {
    return respondError(err)
  }
}

export const handleDocAppend = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const path = extractPath(url)
  if (path instanceof Response) return path
  const body = await readLimitedBody(req, daemon.bodyLimits.singleDocumentBytes)
  if (body instanceof Response) return body
  try {
    await br.store.append(path, Buffer.from(body))
    return new Response(null, { status: 204 })
  } catch (err) {
    return respondError(err)
  }
}

export const handleDocDelete = async (
  daemon: Daemon,
  brainId: string,
  url: URL,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const path = extractPath(url)
  if (path instanceof Response) return path
  try {
    await br.store.delete(path)
    return new Response(null, { status: 204 })
  } catch (err) {
    return respondError(err)
  }
}

export const handleDocRename = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<{ from?: unknown; to?: unknown }>(req, JSON_SMALL_LIMIT)
  if (body instanceof Response) return body
  const fromRaw = typeof body.from === 'string' ? body.from : ''
  const toRaw = typeof body.to === 'string' ? body.to : ''
  let src: Path
  let dst: Path
  try {
    src = toPath(fromRaw)
    dst = toPath(toRaw)
  } catch (err) {
    if (err instanceof ErrInvalidPath) return validationError(err.message)
    return validationError(`invalid path: ${err instanceof Error ? err.message : String(err)}`)
  }
  try {
    await br.store.rename(src, dst)
    return new Response(null, { status: 204 })
  } catch (err) {
    return respondError(err)
  }
}

type BatchOpWire = {
  readonly type?: unknown
  readonly path?: unknown
  readonly to?: unknown
  readonly content_base64?: unknown
}

export const handleBatchOps = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<{
    reason?: unknown
    message?: unknown
    author?: unknown
    email?: unknown
    ops?: unknown
  }>(req, batchJsonBodyLimit(daemon.bodyLimits))
  if (body instanceof Response) return body
  const opsRaw = Array.isArray(body.ops) ? (body.ops as BatchOpWire[]) : []
  if (opsRaw.length > daemon.bodyLimits.batchOpCount) {
    return payloadTooLarge(`ops length exceeds ${daemon.bodyLimits.batchOpCount}`)
  }

  const decoded = new Array<Buffer | undefined>(opsRaw.length)
  let decodedSize = 0
  for (let i = 0; i < opsRaw.length; i++) {
    const op = opsRaw[i]
    if (op === undefined) continue
    const raw = typeof op.content_base64 === 'string' ? op.content_base64 : ''
    if (raw === '') continue
    try {
      const buf = Buffer.from(raw, 'base64')
      decodedSize += buf.length
      if (decodedSize > daemon.bodyLimits.batchDecodedBytes) {
        return payloadTooLarge(
          `batch payload exceeds ${daemon.bodyLimits.batchDecodedBytes} bytes after decode`,
        )
      }
      decoded[i] = buf
    } catch {
      return validationError(`invalid base64 at op ${i}`)
    }
  }

  const reason = typeof body.reason === 'string' ? body.reason : 'batch'
  const message = typeof body.message === 'string' ? body.message : undefined
  const author = typeof body.author === 'string' ? body.author : undefined
  const email = typeof body.email === 'string' ? body.email : undefined
  const opts = {
    reason,
    ...(message !== undefined ? { message } : {}),
    ...(author !== undefined ? { author } : {}),
    ...(email !== undefined ? { email } : {}),
  }

  let committed = 0
  try {
    await br.store.batch(opts, async (batch) => {
      for (let i = 0; i < opsRaw.length; i++) {
        const op = opsRaw[i]
        if (op === undefined) continue
        const type = typeof op.type === 'string' ? op.type : ''
        const pRaw = typeof op.path === 'string' ? op.path : ''
        const targetRaw = typeof op.to === 'string' ? op.to : ''
        const src = toPath(pRaw)
        switch (type) {
          case 'write':
            await batch.write(src, decoded[i] ?? Buffer.alloc(0))
            break
          case 'append':
            await batch.append(src, decoded[i] ?? Buffer.alloc(0))
            break
          case 'delete':
            await batch.delete(src)
            break
          case 'rename':
            await batch.rename(src, toPath(targetRaw))
            break
          default:
            throw new Error(`unknown op type: ${type}`)
        }
        committed += 1
      }
    })
  } catch (err) {
    return respondError(err)
  }
  return jsonResponse(200, { committed })
}

type SearchRequest = {
  readonly query?: unknown
  readonly topK?: unknown
  readonly candidateK?: unknown
  readonly rerankTopN?: unknown
  readonly mode?: unknown
  readonly questionDate?: unknown
  readonly filters?: unknown
}

type SearchResultPayload = {
  readonly chunks: readonly RetrievedChunk[]
  readonly trace?: unknown
  readonly attempts?: readonly unknown[]
}

export const handleSearch = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<SearchRequest>(req, 256 * 1024)
  if (body instanceof Response) return body
  const query = typeof body.query === 'string' ? body.query : ''
  if (query === '') return validationError('query required')
  const topK = typeof body.topK === 'number' && body.topK > 0 ? Math.trunc(body.topK) : 10
  const candidateK =
    typeof body.candidateK === 'number' && body.candidateK > 0
      ? Math.trunc(body.candidateK)
      : undefined
  const rerankTopN =
    typeof body.rerankTopN === 'number' && body.rerankTopN > 0
      ? Math.trunc(body.rerankTopN)
      : undefined
  const mode = typeof body.mode === 'string' ? body.mode : 'auto'
  const questionDate =
    typeof body.questionDate === 'string' && body.questionDate !== ''
      ? body.questionDate
      : undefined
  const filters = parseSearchFilters(body.filters)

  // Cheap after the first call — awaits the one-shot brain-open scan,
  // subsequent requests resolve instantly. Subscribe handles later
  // writes incrementally so no per-request re-scan is needed.
  await br.refresh()

  const started = Date.now()
  const search = await runSearch(br, query, topK, mode, {
    ...(candidateK !== undefined ? { candidateK } : {}),
    ...(rerankTopN !== undefined ? { rerankTopN } : {}),
    ...(questionDate !== undefined ? { questionDate } : {}),
    ...(filters !== undefined ? { filters } : {}),
  })
  const tookMs = Date.now() - started
  return jsonResponse(200, {
    chunks: search.chunks,
    tookMs,
    ...(search.trace !== undefined ? { trace: search.trace } : {}),
    ...(search.attempts !== undefined && search.attempts.length > 0
      ? { attempts: search.attempts }
      : {}),
  })
}

type RetrievedChunk = {
  chunkId: string
  documentId: string
  path: string
  score: number
  text: string
  title: string
  summary: string
  metadata?: Record<string, unknown>
  bm25Rank?: number
  vectorSimilarity?: number
  rerankScore?: number
}

const runSearch = async (
  br: BrainResources,
  query: string,
  topK: number,
  mode: string,
  opts: {
    readonly candidateK?: number
    readonly rerankTopN?: number
    readonly questionDate?: string
    readonly filters?: RetrievalFilters
  } = {},
): Promise<SearchResultPayload> => {
  let trace: unknown
  let attempts: readonly unknown[] | undefined
  if (br.retrieval !== undefined) {
    try {
      const response = await br.retrieval.searchRaw({
        query,
        ...(opts.candidateK !== undefined ? { candidateK: opts.candidateK } : {}),
        ...(opts.rerankTopN !== undefined ? { rerankTopN: opts.rerankTopN } : {}),
        ...(opts.questionDate !== undefined ? { questionDate: opts.questionDate } : {}),
        ...(opts.filters !== undefined ? { filters: opts.filters } : {}),
        topK,
        mode:
          mode === 'bm25' ||
          mode === 'semantic' ||
          mode === 'hybrid' ||
          mode === 'hybrid-rerank' ||
          mode === 'auto'
            ? mode
            : 'auto',
      })
      trace = response.trace
      attempts = response.trace.attempts
      if (response.results.length > 0) {
        return {
          chunks: response.results.map((r) => ({
            chunkId: r.id,
            documentId: r.path,
            path: r.path,
            score: r.score,
            text: r.content,
            title: r.title,
            summary: r.summary,
            ...(r.metadata !== undefined ? { metadata: r.metadata } : {}),
            ...(r.bm25Rank !== undefined ? { bm25Rank: r.bm25Rank } : {}),
            ...(r.vectorSimilarity !== undefined ? { vectorSimilarity: r.vectorSimilarity } : {}),
            ...(r.rerankScore !== undefined ? { rerankScore: r.rerankScore } : {}),
          })),
          ...(trace !== undefined ? { trace } : {}),
          ...(attempts !== undefined ? { attempts } : {}),
        }
      }
    } catch {
      /* fall through to naive scan */
    }
  }
  return {
    chunks: await naiveStoreSearch(br, query, opts.questionDate, topK, opts.filters),
    ...(trace !== undefined ? { trace } : {}),
    ...(attempts !== undefined ? { attempts } : {}),
  }
}

/** Fallback search used when the sqlite-vec index is unavailable.
 *  Walks the store, keeps every line with a case-insensitive match.
 */
const naiveStoreSearch = async (
  br: BrainResources,
  query: string,
  questionDate: string | undefined,
  topK: number,
  filters: RetrievalFilters | undefined,
): Promise<RetrievedChunk[]> => {
  const probes = [query, augmentQueryWithTemporal(query, questionDate ?? '')]
    .map((value) => value.trim().toLowerCase())
    .filter((value, index, all) => value !== '' && all.indexOf(value) === index)
  const entries = await br.store.list('', { recursive: true, includeGenerated: true })
  const hits: RetrievedChunk[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!/\.md$/i.test(entry.path)) continue
    if (!matchesSearchPathFilters(entry.path, filters)) continue
    try {
      const buf = await br.store.read(entry.path)
      const raw = buf.toString('utf8')
      const { frontmatter, body } = parseFrontmatter(raw)
      const text = body === '' ? raw : body
      const lowered = text.toLowerCase()
      if (!probes.some((probe) => lowered.includes(probe))) continue
      const metadata: Record<string, unknown> = {
        ...(frontmatter.scope !== undefined ? { scope: frontmatter.scope } : {}),
        ...(frontmatter.type !== undefined ? { type: frontmatter.type } : {}),
        ...(frontmatter.session_id !== undefined ? { sessionId: frontmatter.session_id } : {}),
        ...(frontmatter.session_date !== undefined
          ? { sessionDate: frontmatter.session_date }
          : {}),
        ...(frontmatter.observed_on !== undefined ? { observedOn: frontmatter.observed_on } : {}),
        ...(frontmatter.modified !== undefined ? { modified: frontmatter.modified } : {}),
      }
      hits.push({
        chunkId: entry.path,
        documentId: entry.path,
        path: entry.path,
        score: 1,
        text: text.slice(0, 800),
        title: frontmatter.name?.trim() ?? entry.path,
        summary: frontmatter.description ?? '',
        ...(Object.keys(metadata).length > 0 ? { metadata } : {}),
        bm25Rank: 0,
        vectorSimilarity: 0,
        rerankScore: 0,
      })
      if (hits.length >= topK) break
    } catch {
      /* ignore */
    }
  }
  return hits
}

export type ReaderMode = 'basic' | 'augmented'

type AskRequest = {
  readonly question?: unknown
  readonly topK?: unknown
  readonly candidateK?: unknown
  readonly rerankTopN?: unknown
  readonly mode?: unknown
  readonly model?: unknown
  readonly readerMode?: unknown
  readonly reader_mode?: unknown
  readonly questionDate?: unknown
  readonly filters?: unknown
}

export const handleAsk = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<AskRequest>(req, 256 * 1024)
  if (body instanceof Response) return body
  const question = typeof body.question === 'string' ? body.question : ''
  if (question === '') return validationError('question required')
  const topK = typeof body.topK === 'number' && body.topK > 0 ? Math.trunc(body.topK) : 5
  const candidateK =
    typeof body.candidateK === 'number' && body.candidateK > 0
      ? Math.trunc(body.candidateK)
      : undefined
  const rerankTopN =
    typeof body.rerankTopN === 'number' && body.rerankTopN > 0
      ? Math.trunc(body.rerankTopN)
      : undefined
  const mode = typeof body.mode === 'string' ? body.mode : 'auto'
  const model = typeof body.model === 'string' ? body.model : undefined
  const readerModeRaw = body.readerMode ?? body.reader_mode
  let readerMode: ReaderMode = 'basic'
  if (readerModeRaw !== undefined) {
    if (typeof readerModeRaw === 'string') {
      const normalisedReaderMode = readerModeRaw.trim().toLowerCase()
      if (normalisedReaderMode === 'basic' || normalisedReaderMode === 'augmented') {
        readerMode = normalisedReaderMode
      } else {
        return validationError("readerMode must be 'basic' or 'augmented'")
      }
    } else {
      return validationError("readerMode must be 'basic' or 'augmented'")
    }
  }
  const questionDate =
    typeof body.questionDate === 'string' && body.questionDate !== ''
      ? body.questionDate
      : undefined
  const filters = parseSearchFilters(body.filters)

  await br.refresh()
  const search = await runSearch(br, question, topK, mode, {
    ...(candidateK !== undefined ? { candidateK } : {}),
    ...(rerankTopN !== undefined ? { rerankTopN } : {}),
    ...(questionDate !== undefined ? { questionDate } : {}),
    ...(filters !== undefined ? { filters } : {}),
  })
  const chunks = search.chunks

  const session = startSse(req.signal)
  const { writer } = session

  // Kick off the streamer async so the Response returns immediately.
  void (async () => {
    try {
      writer.sendJson('retrieve', { chunks, topK, mode })

      const augmentedEvidence =
        readerMode === 'augmented'
          ? formatAugmentedChunks(question, chunks, questionDate)
          : undefined
      const deterministicAnswer =
        augmentedEvidence !== undefined
          ? resolveDeterministicAugmentedAnswer({
              question,
              rendered: augmentedEvidence,
            })
          : undefined
      if (deterministicAnswer !== undefined) {
        writer.sendJson('answer_delta', { text: deterministicAnswer.answer })
        for (const c of chunks) {
          if (writer.closed) return
          writer.sendJson('citation', {
            chunkId: c.chunkId,
            path: c.path,
            title: c.title,
            score: c.score,
          })
        }
        writer.sendJson('done', { ok: true })
        return
      }

      const provider = daemon.provider
      if (provider === undefined) {
        writer.sendJson('error', { message: 'no LLM provider configured' })
        writer.sendJson('done', { ok: false })
        writer.close()
        return
      }

      const messages: Message[] =
        readerMode === 'augmented'
          ? [
              {
                role: 'user',
                content: buildAugmentedAskPrompt(question, augmentedEvidence ?? '', questionDate),
              },
            ]
          : [
              { role: 'system', content: ASK_SYSTEM_PROMPT },
              { role: 'user', content: buildAskPrompt(question, chunks) },
            ]
      const maxTokens = readerMode === 'augmented' ? READER_AUGMENTED_MAX_TOKENS : 1024
      const temperature = readerMode === 'augmented' ? READER_AUGMENTED_TEMPERATURE : 0.2
      try {
        for await (const evt of provider.stream(
          {
            messages,
            maxTokens,
            temperature,
            ...(model !== undefined ? { model } : {}),
          },
          req.signal,
        )) {
          // Honour client disconnect: stop streaming as soon as the
          // writer closes (tab closed, reverse proxy dropped, etc).
          if (writer.closed) return
          if (evt.type === 'text_delta' && evt.text !== '') {
            writer.sendJson('answer_delta', { text: evt.text })
          } else if (evt.type === 'done') {
            break
          } else if (evt.type === 'error') {
            writer.sendJson('error', { message: evt.error.message })
            break
          }
        }
      } catch (err) {
        writer.sendJson('error', {
          message: err instanceof Error ? err.message : String(err),
        })
      }

      for (const c of chunks) {
        if (writer.closed) return
        writer.sendJson('citation', {
          chunkId: c.chunkId,
          path: c.path,
          title: c.title,
          score: c.score,
        })
      }
      writer.sendJson('done', { ok: true })
    } finally {
      writer.close()
    }
  })()

  return session.response
}

const ASK_SYSTEM_PROMPT =
  'You are Jeffs Brain, a helpful assistant. When evidence is supplied, ground your answer in it and cite the path of any source you rely on. When no evidence is supplied, answer concisely from general knowledge.'

const buildAskPrompt = (question: string, chunks: readonly RetrievedChunk[]): string => {
  const parts: string[] = []
  if (chunks.length > 0) {
    parts.push('## Evidence', '')
    for (const c of chunks) {
      parts.push(`### ${c.title !== '' ? c.title : c.path} (${c.path})`)
      parts.push(c.text !== '' ? c.text : c.summary)
      parts.push('')
    }
  }
  parts.push('## Question', '', question)
  return parts.join('\n')
}

/**
 * buildAugmentedAskPrompt mirrors readerUserTemplate in
 * sdks/go/eval/lme/reader.go so the TS /ask endpoint can opt in to the
 * paper-faithful LongMemEval CoT reader. Wording is byte-identical with
 * the Go template and the rendered evidence mirrors the Go and Python
 * retrieve-only format: temporal hint, numbered facts, date labels, and
 * session clustering with frontmatter stripped from the body.
 */
const buildAugmentedAskPrompt = (
  question: string,
  evidenceBlock: string,
  questionDate: string | undefined,
): string => buildAugmentedReaderPrompt(question, evidenceBlock, questionDate)

const formatAugmentedChunks = (
  question: string,
  chunks: readonly RetrievedChunk[],
  questionDate: string | undefined,
): string => {
  const ordered = clusterChunksBySession(chunks)
  if (ordered.length === 0) return ''

  const parts: string[] = []
  const temporalHint = resolvedTemporalHintLine(question, questionDate)
  if (temporalHint !== undefined) {
    parts.push(temporalHint, '')
  }
  parts.push(`Retrieved facts (${ordered.length}):`, '')
  for (const [index, chunk] of ordered.entries()) {
    const labels = [`[${chunkDate(chunk) || 'unknown'}]`]
    const sessionId = chunkSessionId(chunk)
    if (sessionId !== '') labels.push(`[session=${sessionId}]`)
    const source = sourceTagFromPath(chunk.path)
    if (source !== '') labels.push(`[${source}]`)
    parts.push(`${String(index + 1).padStart(2, ' ')}. ${labels.join(' ')}`)
    parts.push(displayBody(chunk), '')
  }
  return parts.join('\n').trim()
}

const metadataValue = (chunk: RetrievedChunk, ...keys: readonly string[]): string => {
  for (const key of keys) {
    const value = chunk.metadata?.[key]
    if (typeof value === 'string' && value.trim() !== '') return value.trim()
  }
  return ''
}

const chunkSessionId = (chunk: RetrievedChunk): string => {
  const fromMetadata = metadataValue(chunk, 'session_id', 'sessionId')
  if (fromMetadata !== '') return fromMetadata
  return firstFrontmatterValue(chunk.text, 'session_id')
}

const chunkDate = (chunk: RetrievedChunk): string => {
  const fromMetadata = metadataValue(
    chunk,
    'session_date',
    'sessionDate',
    'observed_on',
    'observedOn',
    'modified',
  )
  if (fromMetadata !== '') return fromMetadata
  for (const key of ['session_date', 'observed_on', 'modified']) {
    const value = firstFrontmatterValue(chunk.text, key)
    if (value !== '') return value
  }
  return ''
}

const displayBody = (chunk: RetrievedChunk): string => {
  const body = (chunk.text !== '' ? chunk.text : chunk.summary).trim()
  if (!body.startsWith('---')) return body
  const parsed = parseFrontmatter(body)
  return parsed.body.trim() !== '' ? parsed.body.trim() : body
}

const clusterChunksBySession = (chunks: readonly RetrievedChunk[]): RetrievedChunk[] => {
  if (chunks.length <= 1) return [...chunks]
  const order: string[] = []
  const groups = new Map<string, RetrievedChunk[]>()
  for (const [index, chunk] of chunks.entries()) {
    const key = chunkSessionId(chunk) || `__solo_${String(index)}__`
    if (!groups.has(key)) order.push(key)
    const group = groups.get(key) ?? []
    group.push(chunk)
    groups.set(key, group)
  }
  return order.flatMap((key) => groups.get(key) ?? [])
}

const firstFrontmatterValue = (content: string, key: string): string => {
  const lines = content.split('\n')
  let inFrontmatter = false
  const prefix = `${key}:`
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === '---') {
      if (inFrontmatter) break
      inFrontmatter = true
      continue
    }
    if (!inFrontmatter || !trimmed.startsWith(prefix)) continue
    return trimmed.slice(prefix.length).trim()
  }
  return ''
}

const sourceTagFromPath = (value: string): string => {
  const base = value.split('/').filter(Boolean).pop() ?? value
  return base.replace(/\.md$/i, '')
}

const parseSearchFilters = (raw: unknown): RetrievalFilters | undefined => {
  if (typeof raw !== 'object' || raw === null) return undefined
  const value = raw as Record<string, unknown>
  const pathsRaw = Array.isArray(value.paths)
    ? value.paths
    : Array.isArray(value.documentPaths)
      ? value.documentPaths
      : Array.isArray(value.document_paths)
        ? value.document_paths
        : []
  const tagsRaw = Array.isArray(value.tags) ? value.tags : []
  const filters: RetrievalFilters = {
    ...(pathsRaw.length > 0
      ? { paths: pathsRaw.filter((path): path is string => typeof path === 'string') }
      : {}),
    ...(typeof value.pathPrefix === 'string'
      ? { pathPrefix: value.pathPrefix }
      : typeof value.path_prefix === 'string'
        ? { pathPrefix: value.path_prefix }
        : {}),
    ...(typeof value.scope === 'string' ? { scope: value.scope } : {}),
    ...(typeof value.project === 'string' ? { project: value.project } : {}),
    ...(tagsRaw.length > 0
      ? { tags: tagsRaw.filter((tag): tag is string => typeof tag === 'string') }
      : {}),
  }
  return hasSearchFilters(filters) ? filters : undefined
}

const hasSearchFilters = (filters: RetrievalFilters): boolean =>
  (normaliseSearchPaths(filters.paths).length ?? 0) > 0 ||
  (filters.pathPrefix?.trim() ?? '') !== '' ||
  (filters.scope?.trim() ?? '') !== '' ||
  (filters.project?.trim() ?? '') !== '' ||
  (filters.tags?.length ?? 0) > 0

const matchesSearchPathFilters = (path: string, filters: RetrievalFilters | undefined): boolean => {
  if (filters === undefined) return true
  const paths = normaliseSearchPaths(filters.paths)
  if (paths.length > 0 && !paths.includes(path)) return false
  const pathPrefix = filters.pathPrefix?.trim() ?? ''
  if (pathPrefix !== '' && !path.startsWith(pathPrefix)) return false

  const scope = filters.scope?.trim().toLowerCase() ?? ''
  if (scope !== '') {
    switch (scope) {
      case 'global':
      case 'global_memory':
        if (!path.startsWith('memory/global/')) return false
        break
      case 'project':
      case 'project_memory':
        if (!path.startsWith('memory/project/')) return false
        break
      case 'agent':
      case 'agent_memory':
        if (!path.startsWith('memory/agent/')) return false
        break
      case 'wiki':
        if (!path.startsWith('wiki/')) return false
        break
      case 'raw':
      case 'raw_document':
        if (!path.startsWith('raw/')) return false
        break
      case 'raw_lme':
        if (!path.startsWith('raw/lme/')) return false
        break
      default:
        return false
    }
  }

  const project = filters.project?.trim().toLowerCase() ?? ''
  if (project !== '') {
    const segments = path.split('/')
    if (
      segments[0] !== 'memory' ||
      segments[1] !== 'project' ||
      segments[2]?.toLowerCase() !== project
    ) {
      return false
    }
  }

  return true
}

const normaliseSearchPaths = (paths: readonly string[] | undefined): string[] =>
  paths?.map((path) => path.trim()).filter((path) => path !== '') ?? []

type IngestFileRequest = {
  readonly path?: unknown
  readonly contentType?: unknown
  readonly title?: unknown
  readonly tags?: unknown
  readonly contentBase64?: unknown
}

export const handleIngestFile = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<IngestFileRequest>(req, batchJsonBodyLimit(daemon.bodyLimits))
  if (body instanceof Response) return body
  const name = typeof body.path === 'string' ? body.path : ''
  const contentB64 = typeof body.contentBase64 === 'string' ? body.contentBase64 : ''
  let bytes: Buffer
  if (contentB64 !== '') {
    try {
      bytes = Buffer.from(contentB64, 'base64')
    } catch {
      return validationError('invalid contentBase64')
    }
    if (bytes.length > daemon.bodyLimits.batchDecodedBytes) {
      return payloadTooLarge(
        `ingest payload exceeds ${daemon.bodyLimits.batchDecodedBytes} bytes after decode`,
      )
    }
  } else if (name !== '') {
    // Fallback: read from the daemon's local filesystem. Keeps parity
    // with the Go implementation which accepts either inline bytes or
    // a server-resolvable path.
    try {
      const { readFile } = await import('node:fs/promises')
      bytes = await readFile(name)
    } catch (err) {
      return validationError(
        `unable to read path: ${err instanceof Error ? err.message : String(err)}`,
      )
    }
  } else {
    return validationError('contentBase64 or path required')
  }

  const title = typeof body.title === 'string' ? body.title : ''
  const contentType = typeof body.contentType === 'string' ? body.contentType : ''
  return runIngest(br, brainId, bytes, { name, title, contentType })
}

export const handleIngestUrl = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<{ url?: unknown }>(req, JSON_SMALL_LIMIT)
  if (body instanceof Response) return body
  const url = typeof body.url === 'string' ? body.url : ''
  if (url === '') return validationError('url required')
  let bytes: Buffer
  let contentType: string
  try {
    const resp = await fetch(url)
    if (!resp.ok) return internalError(`fetch ${url}: HTTP ${resp.status}`)
    const buf = await resp.arrayBuffer()
    bytes = Buffer.from(buf)
    contentType = resp.headers.get('content-type') ?? ''
  } catch (err) {
    return internalError(`fetch ${url}: ${err instanceof Error ? err.message : String(err)}`)
  }
  const slug = slugify(url).slice(0, 64)
  return runIngest(br, brainId, bytes, { name: `${slug}.md`, title: url, contentType, source: url })
}

type IngestMeta = {
  readonly name: string
  readonly title: string
  readonly contentType: string
  readonly source?: string
}

/**
 * Run the full ingest pipeline (chunk + embed + index) when an embedder
 * is configured, falling back to a plain store write + the index
 * subscriber when the daemon has no embedder. Response shape mirrors
 * Go's knowledge.IngestResponse: {documentId, path, chunkCount, bytes,
 * tookMs}.
 */
const runIngest = async (
  br: BrainResources,
  brainId: string,
  bytes: Buffer,
  meta: IngestMeta,
): Promise<Response> => {
  const started = Date.now()
  const { createHash } = await import('node:crypto')
  const hash = createHash('sha256').update(bytes).digest('hex')
  const slug = meta.name !== '' ? slugify(meta.name) : hash.slice(0, 12)
  const storedPath = joinPath('raw/documents', `${slug}.md`)

  // Prefer the real ingest pipeline so chunking + embedding + index
  // upsert happens in a single call. Requires both a sqlite index and
  // an embedder; falls through to a raw store write otherwise so the
  // route stays usable against a daemon with no embedder configured.
  if (br.index !== undefined && br.embedder !== undefined) {
    try {
      const mime = pickIngestMime(meta.contentType, meta.name)
      const result = await ingestDocument({
        store: br.store,
        searchIndex: br.index,
        embedder: br.embedder,
        doc: {
          brainId,
          path: storedPath,
          content: bytes,
          mime,
          ...(meta.title !== '' ? { title: meta.title } : {}),
          ...(meta.source !== undefined ? { source: meta.source } : {}),
        },
      })
      return jsonResponse(200, {
        documentId: result.documentId,
        path: result.path,
        chunkCount: result.chunkCount,
        bytes: bytes.length,
        tookMs: Date.now() - started,
      })
    } catch {
      // Fall back to the naive write path rather than failing the
      // request; the store subscriber will still index the document as
      // a single chunk. Silent on purpose: the fallback below still
      // surfaces failures via respondError().
    }
  }

  try {
    await br.store.write(storedPath, bytes)
  } catch (err) {
    return respondError(err)
  }
  // Wait for the subscriber-driven index to catch up before returning.
  await br.refresh()
  return jsonResponse(200, {
    documentId: `${brainId}:${hash.slice(0, 16)}`,
    path: storedPath,
    chunkCount: 1,
    bytes: bytes.length,
    tookMs: Date.now() - started,
  })
}

const pickIngestMime = (contentType: string, name: string): string => {
  if (contentType !== '') {
    const simplified = contentType.split(';')[0]?.trim().toLowerCase() ?? ''
    if (simplified !== '') return simplified
  }
  if (/\.(md|markdown)$/i.test(name)) return 'text/markdown'
  if (/\.txt$/i.test(name)) return 'text/plain'
  if (/\.pdf$/i.test(name)) return 'application/pdf'
  if (/\.html?$/i.test(name)) return 'text/html'
  return 'text/markdown'
}

const slugify = (s: string): string => {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

type RememberRequest = {
  readonly note?: unknown
  readonly tags?: unknown
  readonly source?: unknown
  readonly slug?: unknown
  readonly scope?: unknown
}

export const handleRemember = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<RememberRequest>(req, 256 * 1024)
  if (body instanceof Response) return body
  const note = typeof body.note === 'string' ? body.note.trim() : ''
  if (note === '') return validationError('note required')
  const slug =
    typeof body.slug === 'string' && body.slug !== ''
      ? body.slug
      : `note-${new Date()
          .toISOString()
          .replace(/[^0-9a-z]/gi, '')
          .toLowerCase()}`

  const scope = typeof body.scope === 'string' ? body.scope.trim() : ''
  let path: Path
  if (scope === '' || scope === 'global') {
    path = joinPath('memory', 'global', `${slug}.md`)
  } else if (scope.startsWith('project:')) {
    const project = scope.slice('project:'.length)
    if (project === '') return validationError('project slug required for project scope')
    path = scopeTopic('project', project, `${slug}.md`)
  } else {
    return validationError(`unknown scope: ${scope}`)
  }

  const renderedBody = renderRememberBody(body, note)
  try {
    await br.store.write(path, Buffer.from(renderedBody, 'utf8'))
  } catch (err) {
    return respondError(err)
  }
  return jsonResponse(201, { path, slug })
}

const renderRememberBody = (req: RememberRequest, note: string): string => {
  const parts: string[] = []
  parts.push('---')
  parts.push('name: "remembered"')
  if (typeof req.source === 'string' && req.source !== '') {
    parts.push(`source: "${req.source.replace(/"/g, '\\"')}"`)
  }
  const now = new Date().toISOString()
  parts.push(`created: ${now}`)
  parts.push(`modified: ${now}`)
  if (Array.isArray(req.tags)) {
    const tags = (req.tags as unknown[]).filter((t): t is string => typeof t === 'string')
    if (tags.length > 0) {
      parts.push('tags:')
      for (const t of tags) parts.push(`  - ${t}`)
    }
  }
  parts.push('---')
  parts.push('')
  parts.push(note)
  parts.push('')
  return parts.join('\n')
}

type RecallRequestWire = {
  readonly query?: unknown
  readonly topK?: unknown
  readonly project?: unknown
  readonly scope?: unknown
  readonly actorId?: unknown
}

export const handleRecall = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<RecallRequestWire>(req, JSON_SMALL_LIMIT)
  if (body instanceof Response) return body
  const query = typeof body.query === 'string' ? body.query : ''
  if (query === '') return validationError('query required')
  const topK = typeof body.topK === 'number' && body.topK > 0 ? Math.trunc(body.topK) : 10
  const scope = asScope(body.scope)
  const actorId = typeof body.actorId === 'string' && body.actorId !== '' ? body.actorId : 'daemon'
  try {
    const hits = await br.memory.recall({ query, k: topK, scope, actorId })
    const memories = hits.map((h) => ({
      path: h.path,
      content: h.content,
      score: h.score,
      name: h.note.name,
      tags: h.note.tags,
    }))
    return jsonResponse(200, { memories })
  } catch (err) {
    return respondError(err)
  }
}

type ExtractRequestWire = {
  readonly project?: unknown
  readonly model?: unknown
  readonly scope?: unknown
  readonly actorId?: unknown
  readonly sessionId?: unknown
  readonly sessionDate?: unknown
  readonly messages?: unknown
}

const messagesFromWire = (raw: unknown): Message[] | Response => {
  if (!Array.isArray(raw) || raw.length === 0) return validationError('messages required')
  const out: Message[] = []
  for (const m of raw) {
    if (typeof m !== 'object' || m === null) return validationError('message must be an object')
    const role = (m as { role?: unknown }).role
    const content = (m as { content?: unknown }).content
    if (
      (role !== 'system' && role !== 'user' && role !== 'assistant' && role !== 'tool') ||
      typeof content !== 'string'
    ) {
      return validationError('message must include role and content')
    }
    out.push({ role, content })
  }
  return out
}

export const handleExtract = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<ExtractRequestWire>(req, JSON_BODY_LIMIT)
  if (body instanceof Response) return body
  const msgs = messagesFromWire(body.messages)
  if (msgs instanceof Response) return msgs
  const scope = asScope(body.scope)
  const actorId = typeof body.actorId === 'string' && body.actorId !== '' ? body.actorId : 'daemon'
  const sessionId =
    typeof body.sessionId === 'string' && body.sessionId !== '' ? body.sessionId : undefined
  const sessionDate =
    typeof body.sessionDate === 'string' && body.sessionDate !== '' ? body.sessionDate : undefined
  try {
    const extracted = await (br.memory.previewExtract ?? br.memory.extract)({
      messages: msgs,
      actorId,
      scope,
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(sessionDate !== undefined ? { sessionDate } : {}),
    })
    return jsonResponse(200, { memories: extracted as readonly ExtractedMemory[] })
  } catch (err) {
    return respondError(err)
  }
}

type ReflectRequestWire = {
  readonly project?: unknown
  readonly model?: unknown
  readonly scope?: unknown
  readonly actorId?: unknown
  readonly sessionId?: unknown
  readonly messages?: unknown
}

export const handleReflect = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  const body = await readJsonBody<ReflectRequestWire>(req, JSON_BODY_LIMIT)
  if (body instanceof Response) return body
  const msgs = messagesFromWire(body.messages)
  if (msgs instanceof Response) return msgs
  const sessionId =
    typeof body.sessionId === 'string' && body.sessionId !== ''
      ? body.sessionId
      : `session-${Date.now().toString(36)}`
  const scope = asScope(body.scope)
  const actorId = typeof body.actorId === 'string' && body.actorId !== '' ? body.actorId : 'daemon'
  try {
    const result = await br.memory.reflect({ messages: msgs, sessionId, scope, actorId })
    return jsonResponse(200, { result: result ?? null })
  } catch (err) {
    return respondError(err)
  }
}

type ConsolidateRequestWire = {
  readonly mode?: unknown
  readonly model?: unknown
  readonly scope?: unknown
  readonly actorId?: unknown
}

export const handleConsolidate = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br
  // Allow an empty body; Go daemon does the same for quick/full mode.
  let body: ConsolidateRequestWire = {}
  if ((req.headers.get('content-length') ?? '0') !== '0') {
    const parsed = await readJsonBody<ConsolidateRequestWire>(req, JSON_SMALL_LIMIT)
    if (parsed instanceof Response) {
      // Empty body tolerated (the Go handler does the same).
      if (parsed.status !== 400) return parsed
    } else {
      body = parsed
    }
  }
  const scope = asScope(body.scope)
  const actorId = typeof body.actorId === 'string' && body.actorId !== '' ? body.actorId : 'daemon'
  try {
    const report = await br.memory.consolidate({ scope, actorId })
    return jsonResponse(200, report as ConsolidationReport)
  } catch (err) {
    return respondError(err)
  }
}

const asScope = (raw: unknown): Scope => {
  if (raw === 'global' || raw === 'project' || raw === 'agent') return raw
  return 'global'
}

export const handleEvents = async (
  daemon: Daemon,
  req: Request,
  brainId: string,
): Promise<Response> => {
  const br = await resolveBrain(daemon, brainId)
  if (br instanceof Response) return br

  const session = startSse(req.signal)
  const { writer } = session
  writer.sendRaw('ready', 'ok')

  const unsubscribe = br.store.subscribe((evt) => {
    if (writer.closed) return
    const payload: Record<string, unknown> = {
      kind: evt.kind,
      path: evt.path,
      when: evt.when.toISOString(),
    }
    if (evt.oldPath !== undefined) payload.old_path = evt.oldPath
    if (evt.reason !== undefined) payload.reason = evt.reason
    writer.sendJson('change', payload)
  })

  let stopPing = (): void => undefined
  stopPing = createSseHeartbeat(25_000, () => {
    if (writer.closed || !writer.sendRaw('ping', 'keepalive')) {
      stopPing()
    }
  })

  void session.done.then(() => {
    stopPing()
    unsubscribe()
  })

  return session.response
}

// Re-export store sentinels so tests can inspect the mapping.
export { ErrConflict, ErrInvalidPath, ErrNotFound, StoreError }
