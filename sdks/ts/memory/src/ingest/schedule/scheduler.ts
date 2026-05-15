// SPDX-License-Identifier: Apache-2.0

/**
 * In-process scheduler that polls the ScheduleStore for due jobs and
 * fires them. No external cron daemon required.
 */

import { nextOccurrence, parseCron } from './cron.js'
import type { Scheduler, SchedulerOptions } from './types.js'

const DEFAULT_POLL_INTERVAL_MS = 30_000

export const createScheduler = (opts: SchedulerOptions): Scheduler => {
  const pollIntervalMs = opts.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS
  const logger = opts.logger
  const now = opts.now ?? (() => new Date())

  let timer: ReturnType<typeof setInterval> | undefined
  let stopped = false
  let runningPromise: Promise<void> | undefined
  let pollInFlight = false

  const runDueJobs = async (): Promise<number> => {
    const currentTime = now()
    const dueJobs = await opts.scheduleStore.findDue(currentTime)

    let fired = 0
    for (const job of dueJobs) {
      try {
        await opts.dispatch(job)
      } catch (err) {
        logger?.error('schedule: dispatch failed', {
          jobId: job.id,
          error: String(err),
        })
        continue
      }

      try {
        const sched = parseCron(job.cronExpression)
        const nextRun = nextOccurrence(sched, currentTime)
        await opts.scheduleStore.markRun(job.id, currentTime, nextRun)
      } catch (err) {
        logger?.error('schedule: markRun failed', {
          jobId: job.id,
          error: String(err),
        })
      }

      fired++
    }

    return fired
  }

  const poll = async (): Promise<void> => {
    if (stopped) return
    try {
      await runDueJobs()
    } catch (err) {
      logger?.error('schedule: poll error', { error: String(err) })
    }
  }

  const guardedPoll = async (): Promise<void> => {
    if (pollInFlight) return
    pollInFlight = true
    try {
      await poll()
    } finally {
      pollInFlight = false
    }
  }

  const start = (): void => {
    if (stopped) return
    // Run immediately on start.
    runningPromise = guardedPoll()
    timer = setInterval(() => {
      runningPromise = guardedPoll()
    }, pollIntervalMs)
  }

  const stop = async (): Promise<void> => {
    stopped = true
    if (timer !== undefined) {
      clearInterval(timer)
      timer = undefined
    }
    // Wait for any running poll to complete.
    if (runningPromise) {
      await runningPromise
    }
  }

  return { start, stop, runDueJobs }
}
