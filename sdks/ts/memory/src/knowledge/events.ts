// SPDX-License-Identifier: Apache-2.0

import type { Logger } from '../llm/index.js'
import type { KnowledgeEvent, KnowledgeEventSink } from './types.js'

type KnowledgeEventEmitterDeps = {
  readonly logger: Logger
  readonly onEvent?: KnowledgeEventSink
}

export const createKnowledgeEventEmitter = (deps: KnowledgeEventEmitterDeps) => {
  const { logger, onEvent } = deps

  return async (event: KnowledgeEvent): Promise<void> => {
    if (onEvent === undefined) return
    try {
      await onEvent(event)
    } catch (err) {
      logger.warn('knowledge: event sink threw', {
        kind: event.kind,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
}
