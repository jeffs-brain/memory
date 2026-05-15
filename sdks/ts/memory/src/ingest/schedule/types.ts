// SPDX-License-Identifier: Apache-2.0

/**
 * Schedule types for the cron-based ingestion scheduler. Jobs persist
 * across restarts via a ScheduleStore (SQLite or PostgreSQL adapter).
 */

import type { Logger } from '../../llm/types.js'

export type ScheduledJob = {
  readonly id: string
  readonly brainId: string
  readonly name: string
  readonly cronExpression: string
  readonly target: ScheduleTarget
  readonly enabled: boolean
  readonly lastRunAt?: Date
  readonly nextRunAt: Date
  readonly createdAt: Date
  readonly updatedAt: Date
  readonly metadata?: Readonly<Record<string, unknown>>
}

export type ScheduleTarget =
  | { readonly kind: 'url'; readonly url: string }
  | { readonly kind: 'file'; readonly path: string }
  | { readonly kind: 'directory'; readonly path: string; readonly glob?: string }

export type CreateScheduleInput = {
  readonly brainId: string
  readonly name: string
  readonly cronExpression: string
  readonly target: ScheduleTarget
  readonly metadata?: Readonly<Record<string, unknown>>
}

export type UpdateSchedulePatch = {
  readonly name?: string
  readonly cronExpression?: string
  readonly target?: ScheduleTarget
  readonly enabled?: boolean
  readonly metadata?: Readonly<Record<string, unknown>>
}

export type ScheduleStore = {
  create(input: CreateScheduleInput): Promise<ScheduledJob>
  get(id: string): Promise<ScheduledJob | undefined>
  list(brainId: string): Promise<readonly ScheduledJob[]>
  update(id: string, patch: UpdateSchedulePatch): Promise<ScheduledJob>
  delete(id: string): Promise<void>
  findDue(now: Date): Promise<readonly ScheduledJob[]>
  markRun(id: string, ranAt: Date, nextRunAt: Date): Promise<void>
}

/** Abstraction over cron expression parsing. Supply a custom engine to
 *  integrate third-party cron libraries. When omitted, the built-in
 *  parser is used. */
export type CronEngine = {
  readonly nextOccurrence: (expression: string, after: Date) => Date | undefined
  readonly isValid: (expression: string) => boolean
}

export type SchedulerOptions = {
  readonly scheduleStore: ScheduleStore
  readonly dispatch: (job: ScheduledJob) => Promise<void> | void
  readonly pollIntervalMs?: number
  readonly logger?: Logger
  /** Injectable clock for testing. Defaults to () => new Date(). */
  readonly now?: () => Date
  /** Custom cron engine. Defaults to the built-in parser. */
  readonly cronEngine?: CronEngine
}

export type Scheduler = {
  start(): void
  stop(): Promise<void>
  /** Exposed for testing. Returns count of jobs fired. */
  runDueJobs(): Promise<number>
}
