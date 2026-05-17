// SPDX-License-Identifier: Apache-2.0

/**
 * Sync state manager: persists and retrieves sync cursors per connector
 * per brain. Uses optimistic concurrency via a generation counter.
 *
 * Storage path: connector/<name>/<brainId>/sync-state.json
 */

import type { Store } from '../store/index.js'
import { isNotFound, toPath } from '../store/index.js'
import type { SyncCursor } from './types.js'

/** JSON wire format for sync state persistence. */
type SerialisedSyncState = {
  readonly cursor: {
    readonly value: string
    readonly updatedAt: string
    readonly metadata?: Readonly<Record<string, unknown>>
  }
  readonly generation: number
}

const syncStatePath = (connectorName: string, brainId: string) =>
  toPath(`connector/${connectorName}/${brainId}/sync-state.json`)

/**
 * SyncStateManager persists sync cursors in the brain Store with
 * optimistic concurrency.
 */
export class SyncStateManager {
  private readonly store: Store

  constructor(store: Store) {
    this.store = store
  }

  /**
   * Retrieve the last sync cursor. Returns undefined when no cursor
   * exists.
   */
  async getCursor(connectorName: string, brainId: string): Promise<SyncCursor | undefined> {
    try {
      const data = await this.store.read(syncStatePath(connectorName, brainId))
      const state = JSON.parse(data.toString('utf8')) as SerialisedSyncState
      return {
        value: state.cursor.value,
        updatedAt: new Date(state.cursor.updatedAt),
        ...(state.cursor.metadata !== undefined ? { metadata: state.cursor.metadata } : {}),
      }
    } catch (err: unknown) {
      if (isNotFound(err)) return undefined
      throw err
    }
  }

  /**
   * Persist a sync cursor. The generation counter is incremented on
   * each write for optimistic concurrency.
   */
  async setCursor(connectorName: string, brainId: string, cursor: SyncCursor): Promise<void> {
    const p = syncStatePath(connectorName, brainId)

    // Read current generation for optimistic concurrency.
    let generation = 0
    try {
      const data = await this.store.read(p)
      const existing = JSON.parse(data.toString('utf8')) as SerialisedSyncState
      generation = existing.generation
    } catch (err: unknown) {
      if (!isNotFound(err)) throw err
    }

    const state: SerialisedSyncState = {
      cursor: {
        value: cursor.value,
        updatedAt: new Date().toISOString(),
        ...(cursor.metadata !== undefined ? { metadata: cursor.metadata } : {}),
      },
      generation: generation + 1,
    }

    await this.store.write(p, Buffer.from(JSON.stringify(state, null, 2), 'utf8'))
  }

  /**
   * Remove the sync cursor, forcing a full sync on the next run.
   * No-op if no cursor exists.
   */
  async clearCursor(connectorName: string, brainId: string): Promise<void> {
    try {
      await this.store.delete(syncStatePath(connectorName, brainId))
    } catch (err: unknown) {
      if (isNotFound(err)) return
      throw err
    }
  }
}
