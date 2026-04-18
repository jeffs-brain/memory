// SPDX-License-Identifier: Apache-2.0

/**
 * Knowledge query. Collects wiki articles from the Store, ranks the
 * most relevant hits, and asks the provider to synthesise a cited
 * answer from the retrieved content.
 */

import type { Logger, Provider } from '../llm/index.js'
import { pathUnder, toPath, type Path, type Store } from '../store/index.js'
import { parseFrontmatter } from './frontmatter.js'
import { WIKI_PREFIX } from './promote.js'
import type {
  KnowledgeQueryRetriever,
  KnowledgeQueryRetrieverChunk,
  QueryOptions,
  QueryResult,
  QuerySortMode,
} from './types.js'

const QUERY_SYSTEM_PROMPT = `You are a knowledge base query engine. Given a question and wiki articles, synthesise a comprehensive answer.

Requirements:
- Answer the question directly using information from the provided articles
- Cite sources using [[wikilink]] notation (for example [[topic/article]])
- If articles contain conflicting information, note the discrepancy
- If the articles do not fully answer the question, say what is missing
- Be concise but thorough
- Use British English
- Do not fabricate information not present in the articles
- If the articles are not relevant, say "The knowledge base does not contain relevant information on this topic."`

const QUERY_MAX_BODY_CHARS = 3000
const DEFAULT_MAX_SOURCES = 5
const MAX_SOURCES_CAP = 10

type QueryDeps = {
  store: Store
  provider: Provider
  logger: Logger
  retriever?: KnowledgeQueryRetriever
}

type QueryArticle = {
  path: Path
  title: string
  summary: string
  body: string
  modified: Date
  score: number
}

export const createQuery = (deps: QueryDeps) => {
  const { store, provider, logger, retriever } = deps

  return async (question: string, opts: QueryOptions = {}): Promise<QueryResult> => {
    const maxSources = normaliseMaxSources(opts.maxSources)
    const root = toPath(WIKI_PREFIX)
    const exists = await store.exists(root).catch(() => false)
    if (!exists) {
      logger.debug('query: knowledge base is empty')
      return { answer: 'Knowledge base is empty.', sourcePaths: [], searchHits: 0 }
    }

    const ranked =
      (await retrieveArticles({ store, retriever, question, maxSources, sort: opts.sort, logger })) ??
      (await rankWikiArticles({ store, question, sort: opts.sort, logger, maxSources }))

    if (ranked.articles.length === 0) {
      logger.debug('query: no relevant articles found', {
        questionLength: question.trim().length,
      })
      return { answer: 'No relevant articles found.', sourcePaths: [], searchHits: 0 }
    }

    const promptArticles = orderForPrompt(ranked.articles, opts.sort)
    const sourcePaths = promptArticles.map((article) => article.path)
    const prompt = buildPrompt(question, promptArticles)

    try {
      const response = await provider.complete({
        system: QUERY_SYSTEM_PROMPT,
        messages: [{ role: 'user', content: prompt }],
        maxTokens: 2048,
        temperature: 0.3,
      })
      logger.info('query complete', {
        provider: provider.name(),
        model: provider.modelName(),
        sources: sourcePaths.length,
      })
      return {
        answer: response.content,
        sourcePaths,
        searchHits: ranked.searchHits,
      }
    } catch (error) {
      logger.warn('query synthesis failed, returning raw search results', {
        provider: provider.name(),
        model: provider.modelName(),
        error: error instanceof Error ? error.message : String(error),
      })

      const fallback = buildFallbackAnswer(provider, ranked.articles, error)
      return {
        answer: fallback,
        sourcePaths,
        searchHits: ranked.searchHits,
      }
    }
  }
}

type RankedQueryArticles = {
  readonly articles: readonly QueryArticle[]
  readonly searchHits: number
}

const retrieveArticles = async (input: {
  readonly store: Store
  readonly retriever: KnowledgeQueryRetriever | undefined
  readonly question: string
  readonly maxSources: number
  readonly sort: QuerySortMode | undefined
  readonly logger: Logger
}): Promise<RankedQueryArticles | undefined> => {
  if (input.retriever === undefined) return undefined

  try {
    const response = await input.retriever.retrieve({
      query: input.question,
      limit: Math.max(input.maxSources * 4, 20),
      mode: 'hybrid',
      candidateLimit: Math.max(input.maxSources * 8, 200),
    })
    const articles = await hydrateRetrievedArticles(input.store, response.chunks, input.sort)
    if (articles.length === 0) {
      input.logger.warn('query: retrieval returned no usable wiki articles, falling back to wiki scan')
      return undefined
    }
    return {
      articles: articles.slice(0, input.maxSources),
      searchHits: articles.length,
    }
  } catch (error) {
    input.logger.warn('query: retrieval failed, falling back to wiki scan', {
      error: error instanceof Error ? error.message : String(error),
    })
    return undefined
  }
}

const rankWikiArticles = async (input: {
  readonly store: Store
  readonly question: string
  readonly sort: QuerySortMode | undefined
  readonly logger: Logger
  readonly maxSources: number
}): Promise<RankedQueryArticles> => {
  const articles = await collectWikiArticles(input.store)
  if (articles.length === 0) {
    input.logger.debug('query: no wiki articles found')
    return { articles: [], searchHits: 0 }
  }

  const ranked = rankArticles(articles, input.question, input.sort)
  return {
    articles: ranked.slice(0, input.maxSources),
    searchHits: ranked.length,
  }
}

const normaliseMaxSources = (maxSources: number | undefined): number => {
  if (typeof maxSources !== 'number' || !Number.isFinite(maxSources)) {
    return DEFAULT_MAX_SOURCES
  }
  const rounded = Math.trunc(maxSources)
  if (rounded <= 0 || rounded > MAX_SOURCES_CAP) {
    return DEFAULT_MAX_SOURCES
  }
  return rounded
}

const collectWikiArticles = async (store: Store): Promise<readonly QueryArticle[]> => {
  const root = toPath(WIKI_PREFIX)
  const entries = await store.list(root, { recursive: true })
  const out: QueryArticle[] = []

  for (const entry of entries) {
    if (entry.isDir) continue
    if (!pathUnder(entry.path, WIKI_PREFIX, true)) continue
    if (!entry.path.endsWith('.md')) continue
    const filename = entry.path.slice(entry.path.lastIndexOf('/') + 1)
    if (filename.startsWith('_')) continue

    const raw = (await store.read(entry.path)).toString('utf8')
    const parsed = parseFrontmatter(raw)
    const title =
      parsed.frontmatter.title.trim() !== '' ? parsed.frontmatter.title.trim() : wikiLinkPath(entry.path)
    const summary = parsed.frontmatter.summary.trim()
    const modified = parsed.frontmatter.modified !== undefined ? parseIsoDate(parsed.frontmatter.modified) : entry.modTime
    const body = parsed.body.trim()
    out.push({
      path: entry.path,
      title,
      summary,
      body,
      modified,
      score: 0,
    })
  }

  return out
}

const hydrateRetrievedArticles = async (
  store: Store,
  chunks: readonly KnowledgeQueryRetrieverChunk[],
  sort: QuerySortMode | undefined,
): Promise<readonly QueryArticle[]> => {
  const bestByPath = new Map<
    string,
    {
      readonly score: number
      readonly modified: Date
      readonly title: string
    }
  >()

  for (const chunk of chunks) {
    if (!chunk.path.startsWith(`${WIKI_PREFIX}/`) || !chunk.path.endsWith('.md')) continue
    const current = bestByPath.get(chunk.path)
    const candidate = {
      score: chunk.score,
      modified: metadataDate(chunk.metadata),
      title: metadataTitle(chunk.metadata, chunk.path),
    }
    if (
      current === undefined ||
      candidate.score > current.score ||
      (candidate.score === current.score && candidate.modified.getTime() > current.modified.getTime())
    ) {
      bestByPath.set(chunk.path, candidate)
    }
  }

  const articles = await Promise.all(
    Array.from(bestByPath.entries()).map(async ([path, meta]) => {
      const raw = await store.read(path as Path).catch(() => undefined)
      if (raw === undefined) return undefined
      const parsed = parseFrontmatter(raw.toString('utf8'))
      return {
        path: path as Path,
        title: parsed.frontmatter.title.trim() !== '' ? parsed.frontmatter.title.trim() : meta.title,
        summary: parsed.frontmatter.summary.trim(),
        body: parsed.body.trim(),
        modified:
          parsed.frontmatter.modified !== undefined ? parseIsoDate(parsed.frontmatter.modified) : meta.modified,
        score: meta.score,
      } satisfies QueryArticle
    }),
  )

  const ranked = articles.filter((article): article is QueryArticle => article !== undefined)
  ranked.sort((left, right) => compareArticles(left, right, sort))
  return ranked
}

const rankArticles = (
  articles: readonly QueryArticle[],
  question: string,
  sort: QuerySortMode | undefined,
): readonly QueryArticle[] => {
  const terms = tokenise(question)
  if (terms.length === 0) return []

  const scored = articles
    .map((article) => ({
      ...article,
      score: scoreArticle(article, terms),
    }))
    .filter((article) => article.score > 0)

  scored.sort((left, right) => compareArticles(left, right, sort))

  return scored
}

const orderForPrompt = (articles: readonly QueryArticle[], sort: QuerySortMode | undefined): QueryArticle[] => {
  const ordered = [...articles]
  ordered.sort((a, b) => {
    const dateDiff = a.modified.getTime() - b.modified.getTime()
    if (sort === 'recency') {
      if (dateDiff !== 0) return -dateDiff
      return a.path.localeCompare(b.path)
    }
    if (dateDiff !== 0) return dateDiff
    return a.path.localeCompare(b.path)
  })
  return ordered
}

const scoreArticle = (article: QueryArticle, terms: readonly string[]): number => {
  const titleTerms = new Set(tokenise(article.title))
  const summaryTerms = new Set(tokenise(article.summary))
  const bodyTerms = new Set(tokenise(article.body))

  let score = 0
  for (const term of terms) {
    if (titleTerms.has(term)) score += 8
    if (summaryTerms.has(term)) score += 4
    if (bodyTerms.has(term)) score += 1
  }
  return score
}

const buildPrompt = (question: string, articles: readonly QueryArticle[]): string => {
  const lines: string[] = []
  lines.push('## Question')
  lines.push('')
  lines.push(question)
  lines.push('')
  lines.push('## Retrieved facts (ordered by date)')
  lines.push('')
  lines.push(
    'Each fact is numbered and tagged with its source date. Use the numbering when counting, summing, or listing. When facts conflict, prefer the most recent unless the question asks about history.',
  )
  lines.push('')

  for (const [index, article] of articles.entries()) {
    const dateTag = formatDate(article.modified)
    const linkPath = wikiLinkPath(article.path)
    lines.push(`### ${index + 1}. [${dateTag}] ${article.title} (\`${linkPath}\`)`)
    lines.push(trimBody(article.body))
    if (index < articles.length - 1) {
      lines.push('')
      lines.push('---')
      lines.push('')
    }
  }

  return lines.join('\n')
}

const buildFallbackAnswer = (
  provider: Provider,
  articles: readonly QueryArticle[],
  error: unknown,
): string => {
  const message = error instanceof Error ? error.message : String(error)
  const lines: string[] = []
  lines.push(`LLM synthesis unavailable (${provider.name()}/${provider.modelName()}): ${message}`)
  lines.push('')
  lines.push('Search results:')
  lines.push('')
  for (const [index, article] of articles.entries()) {
    const linkPath = wikiLinkPath(article.path)
    lines.push(`${index + 1}. **${article.title}** ([[${linkPath}]])`)
  }
  return lines.join('\n')
}

const compareArticles = (
  left: QueryArticle,
  right: QueryArticle,
  sort: QuerySortMode | undefined,
): number => {
  if (sort === 'recency') {
    const dateDiff = right.modified.getTime() - left.modified.getTime()
    if (dateDiff !== 0) return dateDiff
    const scoreDiff = right.score - left.score
    if (scoreDiff !== 0) return scoreDiff
    return left.path.localeCompare(right.path)
  }
  const scoreDiff = right.score - left.score
  if (scoreDiff !== 0) return scoreDiff
  const dateDiff = right.modified.getTime() - left.modified.getTime()
  if (dateDiff !== 0) return dateDiff
  return left.path.localeCompare(right.path)
}

const metadataDate = (
  metadata: Readonly<Record<string, string | number | boolean | null>> | undefined,
): Date => {
  const candidate =
    typeof metadata?.modified === 'string'
      ? metadata.modified
      : typeof metadata?.created === 'string'
        ? metadata.created
        : undefined
  return candidate === undefined ? new Date(0) : parseIsoDate(candidate)
}

const metadataTitle = (
  metadata: Readonly<Record<string, string | number | boolean | null>> | undefined,
  path: string,
): string =>
  typeof metadata?.title === 'string' && metadata.title.trim() !== ''
    ? metadata.title.trim()
    : wikiLinkPath(path)

const tokenise = (value: string): string[] => {
  const lower = value.toLowerCase()
  const matches = lower.match(/[a-z0-9]+/g) ?? []
  const seen = new Set<string>()
  const out: string[] = []
  for (const match of matches) {
    if (match.length < 2 && !/^\d+$/.test(match)) continue
    if (seen.has(match)) continue
    seen.add(match)
    out.push(match)
  }
  return out
}

const parseIsoDate = (value: string): Date => {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return new Date(0)
  }
  return parsed
}

const formatDate = (value: Date): string => {
  if (Number.isNaN(value.getTime()) || value.getTime() === 0) {
    return 'unknown'
  }
  return value.toISOString().slice(0, 10)
}

const trimBody = (body: string): string => {
  if (body.length <= QUERY_MAX_BODY_CHARS) {
    return body
  }
  return `${body.slice(0, QUERY_MAX_BODY_CHARS)}\n[...truncated]`
}

const stripMd = (value: string): string => (value.endsWith('.md') ? value.slice(0, -3) : value)

const wikiLinkPath = (value: string): string => {
  const stripped = stripMd(value)
  const prefix = `${WIKI_PREFIX}/`
  return stripped.startsWith(prefix) ? stripped.slice(prefix.length) : stripped
}
