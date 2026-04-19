// SPDX-License-Identifier: Apache-2.0

import { defineCommand } from 'citty'
import { openBrain } from '../brain.js'
import {
  buildProvider,
  buildReranker,
  CliUsageError,
  buildEmbedder,
  embedderFromEnv,
  providerFromEnvOptional,
  rerankerFromEnv,
  resolveBrainDir,
} from '../config.js'
import { isSearchMode, runSearch } from '../search-runner.js'

const DEFAULT_LIMIT = 10

export const searchCommand = defineCommand({
  meta: {
    name: 'search',
    description: 'Search the brain (hybrid BM25 + vector by default)',
  },
  args: {
    query: {
      type: 'positional',
      description: 'Query string',
      required: true,
    },
    brain: {
      type: 'string',
      description: 'Brain directory (overrides JB_BRAIN)',
    },
    mode: {
      type: 'string',
      description: 'Search mode: hybrid|hybrid-rerank|bm25|semantic',
      default: 'hybrid',
    },
    rerank: {
      type: 'boolean',
      description: 'Request reranking for the selected mode when configured',
      default: false,
    },
    json: {
      type: 'boolean',
      description: 'Emit JSON results on stdout',
      default: false,
    },
    limit: {
      type: 'string',
      description: 'Maximum number of results',
      default: String(DEFAULT_LIMIT),
    },
  },
  run: async ({ args }) => {
    const query = args.query
    if (typeof query !== 'string' || query === '') {
      throw new CliUsageError('search: <query> is required')
    }
    const modeRaw = typeof args.mode === 'string' ? args.mode : 'hybrid'
    if (!isSearchMode(modeRaw)) {
      throw new CliUsageError(
        `search: invalid --mode '${modeRaw}'; expected hybrid|hybrid-rerank|bm25|semantic`,
      )
    }
    const limit = parseLimit(args.limit)
    const brainDir = resolveBrainDir(
      typeof args.brain === 'string' ? args.brain : undefined,
    )
    const rerankerSettings = rerankerFromEnv()
    const providerSettings =
      rerankerSettings !== undefined ? providerFromEnvOptional() : undefined
    const provider =
      providerSettings !== undefined ? buildProvider(providerSettings) : undefined
    const embedderSettings = embedderFromEnv()
    const embedder =
      embedderSettings !== undefined ? buildEmbedder(embedderSettings) : undefined
    const reranker =
      rerankerSettings !== undefined
        ? buildReranker(rerankerSettings, { ...(provider !== undefined ? { provider } : {}) })
        : undefined
    const rerankRequested = args.rerank === true || modeRaw === 'hybrid-rerank'

    const store = await openBrain(brainDir)
    try {
      const hits = await runSearch(query, {
        store,
        ...(embedder !== undefined ? { embedder } : {}),
        ...(reranker !== undefined ? { reranker } : {}),
        limit,
        mode: modeRaw,
        ...(rerankRequested ? { rerank: true } : {}),
      })
      if (args.json === true) {
        process.stdout.write(`${JSON.stringify({ query, mode: modeRaw, hits })}\n`)
      } else if (hits.length === 0) {
        process.stdout.write('no results\n')
      } else {
        for (const hit of hits) {
          process.stdout.write(
            `${hit.score.toFixed(4)}\t${hit.path}\n  ${hit.snippet.replace(/\s+/g, ' ').trim()}\n`,
          )
        }
      }
    } finally {
      await store.close()
    }
  },
})

const parseLimit = (raw: unknown): number => {
  if (typeof raw !== 'string' || raw === '') return DEFAULT_LIMIT
  const n = Number.parseInt(raw, 10)
  if (!Number.isFinite(n) || n <= 0) {
    throw new CliUsageError(`search: invalid --limit '${String(raw)}'`)
  }
  return n
}
