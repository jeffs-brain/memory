// SPDX-License-Identifier: Apache-2.0

/**
 * Four pi lifecycle hooks that turn the runtime into a self-driving
 * memory layer:
 *
 *   - `before_agent_start` runs the recall pipeline against the user's
 *     prompt and injects a `<recalled-memory>...</recalled-memory>`
 *     block into the system prompt. Cache-friendly mode keeps the block
 *     stable across turns when the recalled set has not changed.
 *
 *   - `context` mirrors the recall block as a per-turn user message so
 *     callers that want the recall block to live below the prompt cache
 *     can flip `recall.onPrompt: false` and use the per-turn variant.
 *
 *   - `turn_end` enqueues a non-blocking extract job once the message
 *     threshold is met. The extract queue drains in the background.
 *
 *   - `session_shutdown` drains the extract queue, optionally runs
 *     reflect, optionally runs consolidate, then closes the underlying
 *     store and search index handles.
 */

import { createHash } from 'node:crypto'
import type {
  BeforeAgentStartEvent,
  BeforeAgentStartEventResult,
  ContextEvent,
  ExtensionAPI,
  SessionShutdownEvent,
  TurnEndEvent,
} from '@earendil-works/pi-coding-agent'
import type { Message, RecallHit, Scope } from '@jeffs-brain/memory'
import type { MemoryRuntime, RecallSnapshot } from './runtime.js'
import { renderRecallBlock } from './runtime.js'

/**
 * Local alias for the structurally-typed agent messages pi passes
 * through `context` and `turn_end`. The pi type lives in the agent
 * package and is not re-exported from the public extension barrel, so
 * we describe only the fields we touch and rely on duck-typing.
 */
type AgentMessageLike = {
  readonly role?: string
  readonly content?: unknown
} & Record<string, unknown>

/** Local mirror of pi's `ContextEventResult`. */
type ContextEventResult = {
  readonly messages?: ReadonlyArray<AgentMessageLike>
}

const RECALL_TAG_OPEN = '<recalled-memory>'
const RECALL_TAG_CLOSE = '</recalled-memory>'

const hashQuery = (query: string): string =>
  createHash('sha1').update(query, 'utf8').digest('hex').slice(0, 16)

const recallSetSignature = (hits: readonly RecallHit[]): string =>
  hits.map((hit) => `${hit.path}:${hit.score.toFixed(4)}`).join('|')

/**
 * Compute a fresh `RecallSnapshot` for the given query. Returns
 * `undefined` when recall returns no hits and the caller prefers to
 * skip injection (currently we always inject the labelled empty block
 * so the prompt cache sees a stable shape).
 */
export const buildRecallSnapshot = async (
  runtime: MemoryRuntime,
  query: string,
): Promise<RecallSnapshot> => {
  const cfg = runtime.config
  const hits =
    query.trim() === ''
      ? []
      : await runtime.memory
          .recall({
            query,
            k: cfg.recallTopK,
            scope: cfg.scope,
            actorId: cfg.actorId,
            ...(cfg.fallbackScopes.length > 0 ? { fallbackScopes: cfg.fallbackScopes } : {}),
          })
          .then((res) =>
            cfg.recallMinScore > 0 ? res.filter((h) => h.score >= cfg.recallMinScore) : res,
          )
  return {
    query,
    paths: hits.map((hit) => hit.path),
    block: renderRecallBlock(hits),
    hits,
  }
}

/**
 * Inject the recall block into the system prompt. When
 * `cacheFriendly` is on we check the in-memory `lastRecall` signature
 * before re-injecting so an unchanged set keeps the prompt-cache
 * boundary stable.
 */
const injectRecall = (systemPrompt: string, snapshot: RecallSnapshot): string => {
  const trimmed = stripPreviousRecall(systemPrompt)
  return `${trimmed}\n\n${snapshot.block}`
}

const stripPreviousRecall = (text: string): string => {
  const open = text.indexOf(RECALL_TAG_OPEN)
  if (open === -1) return text.trimEnd()
  const close = text.indexOf(RECALL_TAG_CLOSE, open)
  if (close === -1) return text.trimEnd()
  const before = text.slice(0, open).trimEnd()
  const after = text.slice(close + RECALL_TAG_CLOSE.length).trimStart()
  return `${before}${before !== '' && after !== '' ? '\n\n' : ''}${after}`.trimEnd()
}

const messagesFromTurnEnd = (event: TurnEndEvent): readonly Message[] => {
  const messages: Message[] = []
  pushAgentMessage(messages, event.message as unknown as AgentMessageLike)
  for (const tr of event.toolResults) {
    pushAgentMessage(messages, tr as unknown as AgentMessageLike)
  }
  return messages
}

const pushAgentMessage = (sink: Message[], msg: AgentMessageLike): void => {
  // Pi AgentMessage may be a custom type or the standard role-based
  // message. We only forward role-based messages.
  if (typeof msg !== 'object' || msg === null) return
  const role = msg.role
  if (role !== 'system' && role !== 'user' && role !== 'assistant' && role !== 'tool') {
    return
  }
  const content = msg.content
  if (typeof content === 'string') {
    sink.push({ role, content })
    return
  }
  if (Array.isArray(content)) {
    const text = content
      .map((part) => {
        if (typeof part === 'string') return part
        if (part && typeof part === 'object') {
          if ('text' in part && typeof (part as { text?: unknown }).text === 'string') {
            return (part as { text: string }).text
          }
        }
        return ''
      })
      .filter((s) => s.length > 0)
      .join('\n')
    if (text !== '') sink.push({ role, content: text })
  }
}

export type RegisterHooksOptions = {
  /** Override the message threshold from runtime config. */
  readonly extractMinMessages?: number
  /**
   * Inject the recall block at the per-turn boundary (`context`) instead
   * of the system prompt. Used when callers want the recall set below
   * the cache boundary.
   */
  readonly recallSurface?: 'system' | 'context'
}

/**
 * Wire the four lifecycle hooks into `pi`. Returns a small handle the
 * caller can drive directly in tests (and the host extension can use
 * to expose `flush()` / `close()`).
 */
export const registerMemoryHooks = (
  pi: ExtensionAPI,
  runtime: MemoryRuntime,
  options: RegisterHooksOptions = {},
): MemoryHookHandle => {
  const cfg = runtime.config
  const turnsSinceExtract = { count: 0 }
  const surface = options.recallSurface ?? 'system'
  const extractThreshold = options.extractMinMessages ?? cfg.extractMinMessages

  const beforeAgentStart = async (
    event: BeforeAgentStartEvent,
  ): Promise<BeforeAgentStartEventResult | undefined> => {
    if (!cfg.recallOnPrompt) return undefined
    if (surface !== 'system') return undefined
    const snapshot = await buildRecallSnapshot(runtime, event.prompt)
    if (cfg.recallCacheFriendly && runtime.lastRecall !== undefined) {
      const prev = runtime.lastRecall
      if (recallSetSignature(prev.hits) === recallSetSignature(snapshot.hits)) {
        // No change in recall set: keep the system prompt as-is so the
        // cache boundary stays stable.
        runtime.lastRecall = snapshot
        return undefined
      }
    }
    runtime.lastRecall = snapshot
    return { systemPrompt: injectRecall(event.systemPrompt, snapshot) }
  }

  const contextHook = async (event: ContextEvent): Promise<ContextEventResult | undefined> => {
    if (surface !== 'context') return undefined
    // Note: `cfg.recallOnPrompt` gates the system-prompt surface only.
    // When the host has explicitly opted into context-surface injection
    // we always run the recall, otherwise both surfaces would be
    // simultaneously gateable and recall could vanish entirely.
    const latestUser = findLatestUserText(event.messages as unknown as readonly AgentMessageLike[])
    if (latestUser === undefined) return undefined
    const snapshot = await buildRecallSnapshot(runtime, latestUser)
    if (cfg.recallCacheFriendly && runtime.lastRecall !== undefined) {
      if (recallSetSignature(runtime.lastRecall.hits) === recallSetSignature(snapshot.hits)) {
        runtime.lastRecall = snapshot
        return undefined
      }
    }
    runtime.lastRecall = snapshot
    const injected: AgentMessageLike = {
      role: 'user',
      content: snapshot.block,
    }
    return {
      messages: [...(event.messages as unknown as readonly AgentMessageLike[]), injected],
    }
  }

  const turnEnd = async (event: TurnEndEvent): Promise<void> => {
    if (!cfg.extractOnTurnEnd) return
    turnsSinceExtract.count += 1
    if (turnsSinceExtract.count < extractThreshold) return
    const messages = messagesFromTurnEnd(event)
    if (messages.length === 0) return
    runtime.extractQueue.enqueue({
      messages,
      actorId: cfg.actorId,
      scope: cfg.scope as Scope,
    })
    turnsSinceExtract.count = 0
  }

  const sessionShutdown = async (_event: SessionShutdownEvent): Promise<void> => {
    await runtime.extractQueue.flush()
    if (cfg.reflectOnSessionEnd && runtime.provider !== undefined) {
      try {
        await runtime.memory.reflect({
          sessionId: `pi-${Date.now()}`,
          messages: [],
        })
      } catch {
        // Reflection failures must not block shutdown.
      }
    }
    if (cfg.consolidateSchedule === 'session' && runtime.provider !== undefined) {
      try {
        await runtime.memory.consolidate({})
      } catch {
        // Same: best-effort consolidation only.
      }
    }
  }

  pi.on('before_agent_start', beforeAgentStart)
  // The pi `context` handler type lives in `@earendil-works/pi-agent-core`
  // (re-exported via the internal extensions barrel) and is not in the
  // public surface; cast through `unknown` so we can still register.
  ;(pi.on as (e: 'context', h: typeof contextHook) => void)('context', contextHook)
  pi.on('turn_end', turnEnd)
  pi.on('session_shutdown', sessionShutdown)

  return {
    runtime,
    handleBeforeAgentStart: beforeAgentStart,
    handleContext: contextHook,
    handleTurnEnd: turnEnd,
    handleSessionShutdown: sessionShutdown,
  }
}

const findLatestUserText = (messages: readonly AgentMessageLike[]): string | undefined => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg === undefined) continue
    if ((msg as { role?: string }).role !== 'user') continue
    const content = (msg as { content?: unknown }).content
    if (typeof content === 'string' && content.trim() !== '') return content
    if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === 'string' && part.trim() !== '') return part
        if (
          part &&
          typeof part === 'object' &&
          'text' in part &&
          typeof (part as { text?: unknown }).text === 'string' &&
          (part as { text: string }).text.trim() !== ''
        ) {
          return (part as { text: string }).text
        }
      }
    }
  }
  return undefined
}

export type MemoryHookHandle = {
  readonly runtime: MemoryRuntime
  handleBeforeAgentStart(
    event: BeforeAgentStartEvent,
  ): Promise<BeforeAgentStartEventResult | undefined>
  handleContext(event: ContextEvent): Promise<ContextEventResult | undefined>
  handleTurnEnd(event: TurnEndEvent): Promise<void>
  handleSessionShutdown(event: SessionShutdownEvent): Promise<void>
}

export { hashQuery, recallSetSignature, RECALL_TAG_OPEN, RECALL_TAG_CLOSE }
