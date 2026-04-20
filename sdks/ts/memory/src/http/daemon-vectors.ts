// SPDX-License-Identifier: Apache-2.0

/**
 * Vector backfill routine for the memory HTTP daemon.
 *
 * Port of `sdks/go/cmd/memory/daemon_vectors.go` so TS and Go stay in
 * lockstep on vector coverage: after the initial FTS scan completes,
 * every indexed path that lacks a vector for the currently-configured
 * embedding model is embedded and persisted. Runs asynchronously so
 * brain open is not blocked by remote embed calls.
 *
 * Failure policy matches Go: a batch-level read or embed failure logs
 * and moves on rather than aborting the entire backfill. Signals and
 * cancellations are honoured between batches so shutdown still
 * completes promptly on a cold restart.
 */

import type { Embedder, Logger } from '../llm/types.js'
import type { SearchIndex as SqliteSearchIndex } from '../search/index.js'
import { type Store, toPath } from '../store/index.js'

const BACKFILL_BATCH_SIZE = 100
const BACKFILL_TEXT_CAP = 8192

/**
 * Resolve the effective embedding model name so the backfill and the
 * embedder stay aligned on a single identifier. Precedence:
 *   1. JB_EMBED_MODEL
 *   2. provider-specific default (JB_EMBED_PROVIDER or OPENAI_API_KEY
 *      presence)
 *   3. empty string when no embedder is configured (disables vectors)
 *
 * Mirrors the Go helper of the same name in daemon_vectors.go.
 */
export function resolveEmbedModel(
  env: Readonly<Record<string, string | undefined>>,
  embedder: Embedder | undefined,
): string {
  if (embedder === undefined) return ''
  const explicit = (env.JB_EMBED_MODEL ?? '').trim()
  if (explicit !== '') return explicit
  const provider = (env.JB_EMBED_PROVIDER ?? '').trim().toLowerCase()
  if (provider === 'openai') return 'text-embedding-3-small'
  if (provider === 'ollama') return 'bge-m3'
  if (provider === 'tei') return embedder.model() !== '' ? embedder.model() : 'tei'
  if (provider === 'hash') return embedder.model() !== '' ? embedder.model() : 'hash'
  // Auto-detect: OpenAI wins when an API key is present, matching the
  // Go fallback order.
  if ((env.OPENAI_API_KEY ?? '').trim() !== '') return 'text-embedding-3-small'
  // Last resort: lean on whatever the embedder reports. OllamaEmbedder
  // reports its bge-m3 default, HashEmbedder returns `hash-<dim>`.
  const fromEmbedder = embedder.model()
  return fromEmbedder !== '' ? fromEmbedder : 'bge-m3'
}

export type BackfillVectorsArgs = {
  readonly brainId: string
  readonly store: Store
  readonly index: SqliteSearchIndex
  readonly embedder: Embedder
  readonly model: string
  readonly logger: Logger
  readonly signal?: AbortSignal
}

/**
 * Embed every FTS-indexed path that lacks a vector for the configured
 * embedding model and store the result on the search index. Batched
 * embed calls amortise network round-trips; batch failures are logged
 * and skipped rather than aborting the run.
 */
export async function backfillVectors(args: BackfillVectorsArgs): Promise<void> {
  const { brainId, store, index, embedder, model, logger, signal } = args
  if (model === '') return

  const paths = index.indexedPaths()
  if (paths.length === 0) return

  const existingIds = new Set(index.chunkIdsWithVectorForModel(model))
  const toEmbed: string[] = []
  for (const p of paths) {
    if (!existingIds.has(p)) toEmbed.push(p)
  }

  if (toEmbed.length === 0) {
    logger.info('vectors: up to date', {
      brain: brainId,
      model,
      total: paths.length,
    })
    return
  }

  logger.info('vectors: backfill start', {
    brain: brainId,
    model,
    count: toEmbed.length,
    have: existingIds.size,
  })
  const started = Date.now()
  let embedded = 0

  for (let i = 0; i < toEmbed.length; i += BACKFILL_BATCH_SIZE) {
    if (signal?.aborted === true) {
      logger.info('vectors: backfill cancelled', {
        brain: brainId,
        embedded,
      })
      return
    }
    const batch = toEmbed.slice(i, i + BACKFILL_BATCH_SIZE)
    const texts: string[] = []
    const keptPaths: string[] = []
    for (const p of batch) {
      try {
        const buf = await store.read(toPath(p))
        let text = buf.toString('utf8')
        if (text.length > BACKFILL_TEXT_CAP) {
          text = text.slice(0, BACKFILL_TEXT_CAP)
        }
        texts.push(text)
        keptPaths.push(p)
      } catch (err) {
        logger.debug('vectors: skip unreadable path', {
          brain: brainId,
          path: p,
          err: err instanceof Error ? err.message : String(err),
        })
      }
    }
    if (texts.length === 0) continue

    let vectors: number[][]
    try {
      vectors = await embedder.embed(texts, signal)
    } catch (err) {
      logger.warn('vectors: embed batch failed', {
        brain: brainId,
        batch_start: i,
        err: err instanceof Error ? err.message : String(err),
      })
      continue
    }
    if (vectors.length !== keptPaths.length) {
      logger.warn('vectors: embedder returned mismatched count', {
        brain: brainId,
        got: vectors.length,
        want: keptPaths.length,
      })
      continue
    }

    let batchEmbedded = 0
    for (let j = 0; j < vectors.length; j += 1) {
      const vec = vectors[j]
      const path = keptPaths[j]
      if (vec === undefined || vec.length === 0 || path === undefined) continue
      if (vec.length !== index.vectorDim) {
        // sqlite-vec's virtual table bakes the dim in at creation; a
        // mismatch means the existing index was provisioned for a
        // different embedder. Warn once per batch and skip so stale
        // rows do not silently break the run.
        logger.warn('vectors: embedding dim mismatch', {
          brain: brainId,
          model,
          got: vec.length,
          want: index.vectorDim,
        })
        break
      }
      try {
        // Preserve the chunk row written by scanBrain (full content,
        // title, tags). The backfill only attaches the vector; missing
        // rows fall back to the path-as-id shape scanBrain would have
        // produced.
        const existing = index.getChunk(path)
        index.upsertChunk({
          id: existing?.id ?? path,
          path,
          ordinal: existing?.ordinal ?? 0,
          title: existing?.title ?? path,
          summary: existing?.summary ?? '',
          tags: existing?.tags ?? [],
          content: existing?.content ?? texts[j] ?? '',
          embedding: vec,
          embeddingModel: model,
        })
        batchEmbedded += 1
      } catch (err) {
        logger.warn('vectors: store upsert failed', {
          brain: brainId,
          path,
          err: err instanceof Error ? err.message : String(err),
        })
      }
    }
    embedded += batchEmbedded
    logger.debug('vectors: batch stored', {
      brain: brainId,
      done: embedded,
      total: toEmbed.length,
    })
  }

  logger.info('vectors: backfill done', {
    brain: brainId,
    model,
    embedded,
    durationMs: Date.now() - started,
  })
}
