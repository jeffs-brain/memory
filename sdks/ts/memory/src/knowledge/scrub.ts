// SPDX-License-Identifier: Apache-2.0

/**
 * PII scrub for raw/documents/ content. Regex-based (email + phone by
 * default), with pluggable custom patterns. Mutations go through a
 * single Batch with Reason: "scrub". Ported from apps/jeff/internal/
 * knowledge/scrub.go (rewritten: the Go version quarantines; we
 * redact in place per the TS port spec).
 */

import type { Logger } from '../llm/index.js'
import { type Path, type Store, pathUnder, toPath } from '../store/index.js'
import { RAW_DOCUMENTS_PREFIX } from './ingest.js'
import { appendLogInBatch } from './log.js'
import type { ScrubOptions, ScrubPattern, ScrubResult } from './types.js'

const EMAIL_RE = /[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/g
// Loose phone match: +?digits with separators, at least 7 digits overall.
const PHONE_RE = /(?:\+?\d[\s\-().]?){7,}\d/g

export const DEFAULT_PATTERNS: readonly ScrubPattern[] = [
  { name: 'email', pattern: EMAIL_RE, replacement: '[redacted:email]' },
  { name: 'phone', pattern: PHONE_RE, replacement: '[redacted:phone]' },
]

type ScrubDeps = {
  store: Store
  logger: Logger
}

export const createScrub = (deps: ScrubDeps) => {
  const { store, logger } = deps

  return async (opts: ScrubOptions = {}): Promise<readonly ScrubResult[]> => {
    const useDefaults = opts.useDefaults ?? true
    const patterns: readonly ScrubPattern[] = [
      ...(useDefaults ? DEFAULT_PATTERNS : []),
      ...(opts.patterns ?? []),
    ]
    if (patterns.length === 0) return []

    const prefix = toPath(RAW_DOCUMENTS_PREFIX)
    const exists = await store.exists(prefix).catch(() => false)
    if (!exists) return []

    const entries = await store.list(prefix, { recursive: true })
    const results: ScrubResult[] = []
    const mutations: { path: Path; content: string }[] = []

    for (const e of entries) {
      if (e.isDir) continue
      if (!pathUnder(e.path, RAW_DOCUMENTS_PREFIX, true)) continue
      if (!e.path.endsWith('.md')) continue
      const before = (await store.read(e.path)).toString('utf8')
      const { after, matches } = applyPatterns(before, patterns)
      if (after === before) continue
      results.push({ path: e.path, before, after, matches })
      mutations.push({ path: e.path, content: after })
    }

    if (opts.dryRun || mutations.length === 0) {
      return results
    }

    await store.batch({ reason: 'scrub' }, async (batch) => {
      for (const m of mutations) {
        await batch.write(m.path, Buffer.from(m.content, 'utf8'))
      }
      await appendLogInBatch(batch, {
        kind: 'scrub',
        title: `${mutations.length} files scrubbed`,
        detail: results
          .map((r) => `${r.path}: ${r.matches.map((m) => `${m.name}=${m.count}`).join(', ')}`)
          .join('\n'),
        when: new Date().toISOString(),
      })
    })

    logger.info('scrub complete', { files: mutations.length })
    return results
  }
}

export const applyPatterns = (
  content: string,
  patterns: readonly ScrubPattern[],
): { after: string; matches: readonly { name: string; count: number }[] } => {
  let out = content
  const counts: { name: string; count: number }[] = []
  for (const p of patterns) {
    // Clone regex so repeated use is safe regardless of g/sticky state.
    const re = new RegExp(p.pattern.source, ensureGlobal(p.pattern.flags))
    let count = 0
    out = out.replace(re, () => {
      count++
      return p.replacement
    })
    if (count > 0) counts.push({ name: p.name, count })
  }
  return { after: out, matches: counts }
}

const ensureGlobal = (flags: string): string => (flags.includes('g') ? flags : `${flags}g`)
