import type { Logger } from '../llm/index.js'
import { type Path, type Store, pathUnder, toPath } from '../store/index.js'
import { RAW_DOCUMENTS_PREFIX } from './ingest.js'
import { appendLogInBatch } from './log.js'
import type { ScrubOptions, ScrubPattern, ScrubResult } from './types.js'

const EMAIL_RE = /[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/g
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
    const mutations: Array<{ path: Path; content: string }> = []

    for (const entry of entries) {
      if (entry.isDir) continue
      if (!pathUnder(entry.path, RAW_DOCUMENTS_PREFIX, true)) continue
      if (!entry.path.endsWith('.md')) continue

      const before = await store.read(entry.path)
      const { after, matches } = applyPatterns(before, patterns)
      if (after === before) continue
      results.push({ path: entry.path, before, after, matches })
      mutations.push({ path: entry.path, content: after })
    }

    if (opts.dryRun === true || mutations.length === 0) {
      return results
    }

    await store.batch({ reason: 'scrub' }, async (batch) => {
      for (const mutation of mutations) {
        await batch.write(mutation.path, mutation.content)
      }
      await appendLogInBatch(batch, {
        kind: 'scrub',
        title: `${mutations.length} files scrubbed`,
        detail: results
          .map(
            (result) =>
              `${result.path}: ${result.matches.map((m) => `${m.name}=${m.count}`).join(', ')}`,
          )
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
  let output = content
  const counts: Array<{ name: string; count: number }> = []
  for (const pattern of patterns) {
    const regex = new RegExp(pattern.pattern.source, ensureGlobal(pattern.pattern.flags))
    let count = 0
    output = output.replace(regex, () => {
      count += 1
      return pattern.replacement
    })
    if (count > 0) counts.push({ name: pattern.name, count })
  }
  return { after: output, matches: counts }
}

const ensureGlobal = (flags: string): string => (flags.includes('g') ? flags : `${flags}g`)
