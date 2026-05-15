// SPDX-License-Identifier: Apache-2.0

/**
 * In-process event bus backed by a simple subscriber list and async
 * dispatch queue. Zero external dependencies — this is the default
 * trigger delivery mechanism.
 */

import type { Logger } from '../../llm/types.js'
import type { IngestTriggerEvent, SubscribeOptions, TriggerBus, TriggerBusOptions, TriggerHandler, Unsubscribe } from './types.js'
import { validateTriggerEvent } from './types.js'

const DEFAULT_MAX_QUEUE_DEPTH = 1000

type Subscriber = {
  readonly id: number
  readonly handler: TriggerHandler
  readonly filter?: (event: IngestTriggerEvent) => boolean
}

export const createEventBus = (opts?: TriggerBusOptions): TriggerBus => {
  const maxQueueDepth = opts?.maxQueueDepth ?? DEFAULT_MAX_QUEUE_DEPTH
  const logger: Logger | undefined = opts?.logger

  let subscribers: Subscriber[] = []
  let nextId = 0
  let closed = false

  const queue: IngestTriggerEvent[] = []
  let flushing = false
  let drainResolve: (() => void) | undefined

  const deliverToAll = async (event: IngestTriggerEvent): Promise<void> => {
    const snapshot = [...subscribers]
    for (const sub of snapshot) {
      if (sub.filter && !sub.filter(event)) {
        continue
      }
      try {
        await sub.handler(event)
      } catch (err) {
        logger?.error('trigger: handler error', {
          error: String(err),
          eventId: event.id,
        })
      }
    }
  }

  const flush = async (): Promise<void> => {
    if (flushing) return
    flushing = true
    try {
      while (queue.length > 0) {
        const event = queue.shift()!
        await deliverToAll(event)
      }
    } finally {
      flushing = false
      if (closed && queue.length === 0 && drainResolve) {
        drainResolve()
      }
    }
  }

  const publish = (event: IngestTriggerEvent): void => {
    if (closed) throw new Error('trigger: bus is closed')
    validateTriggerEvent(event)

    if (queue.length >= maxQueueDepth) {
      queue.shift()
      logger?.warn('trigger: queue full, dropping oldest event', {
        maxQueueDepth: String(maxQueueDepth),
      })
    }

    queue.push(event)
    void flush()
  }

  const subscribe = (handler: TriggerHandler, opts?: SubscribeOptions): Unsubscribe => {
    const id = ++nextId
    subscribers.push({ id, handler, ...(opts?.filter ? { filter: opts.filter } : {}) })
    return () => {
      subscribers = subscribers.filter((s) => s.id !== id)
    }
  }

  const close = async (): Promise<void> => {
    if (closed) return
    closed = true
    if (queue.length === 0 && !flushing) return
    return new Promise<void>((resolve) => {
      drainResolve = resolve
      void flush()
    })
  }

  return { publish, subscribe, close }
}
