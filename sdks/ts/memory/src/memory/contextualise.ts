// SPDX-License-Identifier: Apache-2.0

/**
 * Contextualise stage. Given a raw user message, run `recall` and wrap
 * the result into a structured `PromptContext` suitable for direct
 * injection into an LLM request. The system-reminder format mirrors
 * `FormatRecalledMemories` from Go so downstream templating stays
 * identical.
 *
 * This stage is intentionally side-effect free: no writes, no plugin
 * hooks, no state. It is a pure consumer of `recall`.
 */

import type { Path } from '../store/index.js'
import { lastSegment } from '../store/path.js'
import type {
  ContextualiseArgs,
  PromptContext,
  RecallHit,
  RecallSelectorMode,
  Scope,
} from './types.js'

const DEFAULT_TOP_N = 5
const DEFAULT_SELECTOR_MODE: RecallSelectorMode = 'auto'

export type ContextualiseDeps = {
  readonly recall: (opts: {
    query: string
    k?: number
    scope?: Scope
    actorId?: string
    fallbackScopes?: readonly Scope[]
    excludedPaths?: readonly Path[]
    surfacedPaths?: readonly Path[]
    selector?: RecallSelectorMode
  }) => Promise<readonly RecallHit[]>
  readonly defaultScope: Scope
  readonly defaultActorId: string
}

export const createContextualise = (deps: ContextualiseDeps) => {
  return async (args: ContextualiseArgs): Promise<PromptContext> => {
    const k = args.topK ?? DEFAULT_TOP_N
    const scope = args.scope ?? deps.defaultScope
    const fallbackScopes = args.fallbackScopes ?? (scope === 'project' ? ['global'] : undefined)
    const memories = await deps.recall({
      query: args.message,
      k,
      scope,
      actorId: args.actorId ?? deps.defaultActorId,
      ...(fallbackScopes !== undefined ? { fallbackScopes } : {}),
      ...(args.excludedPaths !== undefined ? { excludedPaths: args.excludedPaths } : {}),
      ...(args.surfacedPaths !== undefined ? { surfacedPaths: args.surfacedPaths } : {}),
      selector: args.selector ?? DEFAULT_SELECTOR_MODE,
    })
    return {
      userMessage: args.message,
      memories,
      systemReminder: formatRecall(memories),
    }
  }
}

const formatRecall = (memories: readonly RecallHit[]): string => {
  if (memories.length === 0) return ''
  const blocks: string[] = []
  for (const m of memories) {
    const scopeLabel = m.note.scope === 'global' ? 'Global memory' : 'Memory'
    const parts: string[] = []
    parts.push('<system-reminder>')
    parts.push(`${scopeLabel}: ${lastSegment(m.path)}`)
    if (m.note.modified) parts.push(`_modified: ${m.note.modified}_`)
    parts.push('')
    parts.push(m.content)
    parts.push('</system-reminder>')
    blocks.push(parts.join('\n'))
  }
  return blocks.join('\n').trim()
}
