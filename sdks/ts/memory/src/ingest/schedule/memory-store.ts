// SPDX-License-Identifier: Apache-2.0

/**
 * In-memory ScheduleStore for testing and local development. Not suitable
 * for production use — jobs do not persist across restarts.
 */

import { isValid, nextOccurrence, parseCron } from './cron.js'
import type {
  CreateScheduleInput,
  ScheduleStore,
  ScheduleTarget,
  ScheduledJob,
  UpdateSchedulePatch,
} from './types.js'

/**
 * Create an in-memory ScheduleStore. Useful for testing. For production
 * use, swap in a SQLite or PostgreSQL adapter.
 */
export const createMemoryScheduleStore = (): ScheduleStore => {
  let idCounter = 0
  const nextId = (): string => {
    idCounter++
    return `sched-${idCounter}`
  }

  const jobs = new Map<string, ScheduledJob>()

  const create = async (input: CreateScheduleInput): Promise<ScheduledJob> => {
    if (!isValid(input.cronExpression)) {
      throw new Error(`schedule: invalid cron expression: "${input.cronExpression}"`)
    }

    const sched = parseCron(input.cronExpression)
    const now = new Date()
    const id = nextId()
    const job: ScheduledJob = {
      id,
      brainId: input.brainId,
      name: input.name,
      cronExpression: input.cronExpression,
      target: input.target,
      enabled: true,
      nextRunAt: nextOccurrence(sched, now),
      createdAt: now,
      updatedAt: now,
      metadata: input.metadata,
    }
    jobs.set(id, job)
    return job
  }

  const get = async (id: string): Promise<ScheduledJob | undefined> => jobs.get(id)

  const list = async (brainId: string): Promise<readonly ScheduledJob[]> =>
    [...jobs.values()].filter((j) => j.brainId === brainId)

  const update = async (id: string, patch: UpdateSchedulePatch): Promise<ScheduledJob> => {
    const existing = jobs.get(id)
    if (!existing) throw new Error(`schedule: job not found: ${id}`)

    const cronExpr = patch.cronExpression ?? existing.cronExpression
    if (patch.cronExpression !== undefined && !isValid(patch.cronExpression)) {
      throw new Error(`schedule: invalid cron expression: "${patch.cronExpression}"`)
    }

    const updated: ScheduledJob = {
      ...existing,
      name: patch.name ?? existing.name,
      cronExpression: cronExpr,
      target: (patch.target ?? existing.target) as ScheduleTarget,
      enabled: patch.enabled ?? existing.enabled,
      metadata: patch.metadata ?? existing.metadata,
      nextRunAt: patch.cronExpression
        ? nextOccurrence(parseCron(cronExpr), new Date())
        : existing.nextRunAt,
      updatedAt: new Date(),
    }
    jobs.set(id, updated)
    return updated
  }

  const del = async (id: string): Promise<void> => {
    jobs.delete(id)
  }

  const findDue = async (now: Date): Promise<readonly ScheduledJob[]> =>
    [...jobs.values()].filter((j) => j.enabled && j.nextRunAt <= now)

  const markRun = async (id: string, ranAt: Date, nextRunAt: Date): Promise<void> => {
    const existing = jobs.get(id)
    if (!existing) return
    jobs.set(id, {
      ...existing,
      lastRunAt: ranAt,
      nextRunAt,
      updatedAt: new Date(),
    })
  }

  return { create, get, list, update, delete: del, findDue, markRun }
}
