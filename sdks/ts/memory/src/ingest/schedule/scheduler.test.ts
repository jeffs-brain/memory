// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { createMemoryScheduleStore } from './memory-store.js'
import { createScheduler } from './scheduler.js'
import type { ScheduledJob } from './types.js'

describe('scheduler', () => {
  it('due job fires trigger event', async () => {
    const store = createMemoryScheduleStore()
    const dispatched: ScheduledJob[] = []

    const job = await store.create({
      brainId: 'brain-1',
      name: 'test job',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/data/test.md' },
    })

    // Force nextRunAt to the past.
    const past = new Date(Date.now() - 3600_000)
    await store.markRun(job.id, new Date(Date.now() - 7200_000), past)

    const scheduler = createScheduler({
      scheduleStore: store,
      dispatch: (j) => { dispatched.push(j) },
    })

    const fired = await scheduler.runDueJobs()
    expect(fired).toBe(1)
    expect(dispatched).toHaveLength(1)
    expect(dispatched[0].name).toBe('test job')
  })

  it('disabled job skipped even when due', async () => {
    const store = createMemoryScheduleStore()
    const dispatched: ScheduledJob[] = []

    const job = await store.create({
      brainId: 'brain-1',
      name: 'disabled job',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/data/test.md' },
    })

    await store.update(job.id, { enabled: false })

    const scheduler = createScheduler({
      scheduleStore: store,
      dispatch: (j) => { dispatched.push(j) },
    })

    const fired = await scheduler.runDueJobs()
    expect(fired).toBe(0)
    expect(dispatched).toHaveLength(0)
  })

  it('job rescheduled after firing (nextRunAt updated)', async () => {
    const store = createMemoryScheduleStore()

    const job = await store.create({
      brainId: 'brain-1',
      name: 'reschedule test',
      cronExpression: '0 * * * *',
      target: { kind: 'file', path: '/data/test.md' },
    })

    const past = new Date(Date.now() - 3600_000)
    await store.markRun(job.id, new Date(Date.now() - 7200_000), past)

    const scheduler = createScheduler({
      scheduleStore: store,
      dispatch: () => {},
    })

    await scheduler.runDueJobs()

    const updated = await store.get(job.id)
    expect(updated).toBeDefined()
    expect(updated!.nextRunAt.getTime()).toBeGreaterThan(Date.now())
  })

  it('no due jobs -> no events published', async () => {
    const store = createMemoryScheduleStore()
    const dispatched: ScheduledJob[] = []

    const scheduler = createScheduler({
      scheduleStore: store,
      dispatch: (j) => { dispatched.push(j) },
    })

    const fired = await scheduler.runDueJobs()
    expect(fired).toBe(0)
    expect(dispatched).toHaveLength(0)
  })

  it('job failure does not prevent other due jobs from firing', async () => {
    const store = createMemoryScheduleStore()

    const job1 = await store.create({
      brainId: 'brain-1',
      name: 'failing job',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/data/fail.md' },
    })
    const job2 = await store.create({
      brainId: 'brain-1',
      name: 'ok job',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/data/ok.md' },
    })

    const past = new Date(Date.now() - 3600_000)
    await store.markRun(job1.id, new Date(Date.now() - 7200_000), past)
    await store.markRun(job2.id, new Date(Date.now() - 7200_000), past)

    let succeeded = 0
    const scheduler = createScheduler({
      scheduleStore: store,
      logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
      dispatch: (j) => {
        if (j.name === 'failing job') throw new Error('dispatch error')
        succeeded++
      },
    })

    const fired = await scheduler.runDueJobs()
    expect(fired).toBe(1)
    expect(succeeded).toBe(1)
  })

  it('start() polls and stop() halts gracefully', async () => {
    const store = createMemoryScheduleStore()
    let dispatched = 0

    const job = await store.create({
      brainId: 'brain-1',
      name: 'lifecycle job',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/data/lifecycle.md' },
    })

    const past = new Date(Date.now() - 3600_000)
    await store.markRun(job.id, new Date(Date.now() - 7200_000), past)

    const scheduler = createScheduler({
      scheduleStore: store,
      pollIntervalMs: 50,
      dispatch: () => { dispatched++ },
    })

    scheduler.start()
    // Give the immediate poll time to fire.
    await new Promise((r) => setTimeout(r, 100))
    await scheduler.stop()

    expect(dispatched).toBeGreaterThanOrEqual(1)
  })

  it('per-brain scheduling isolation', async () => {
    const store = createMemoryScheduleStore()

    const job1 = await store.create({
      brainId: 'brain-a',
      name: 'job-a',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/a.md' },
    })
    const job2 = await store.create({
      brainId: 'brain-b',
      name: 'job-b',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/b.md' },
    })

    const past = new Date(Date.now() - 3600_000)
    await store.markRun(job1.id, new Date(Date.now() - 7200_000), past)
    await store.markRun(job2.id, new Date(Date.now() - 7200_000), past)

    const brains: Record<string, number> = {}
    const scheduler = createScheduler({
      scheduleStore: store,
      dispatch: (j) => {
        brains[j.brainId] = (brains[j.brainId] ?? 0) + 1
      },
    })

    const fired = await scheduler.runDueJobs()
    expect(fired).toBe(2)
    expect(brains['brain-a']).toBe(1)
    expect(brains['brain-b']).toBe(1)
  })
})

describe('memory-store CRUD', () => {
  it('create, get, list, update, delete', async () => {
    const store = createMemoryScheduleStore()

    const job = await store.create({
      brainId: 'brain-1',
      name: 'daily',
      cronExpression: '0 2 * * *',
      target: { kind: 'url', url: 'https://example.com' },
    })

    expect(job.id).toBeTruthy()
    expect(job.enabled).toBe(true)

    const got = await store.get(job.id)
    expect(got).toBeDefined()
    expect(got!.name).toBe('daily')

    const list = await store.list('brain-1')
    expect(list).toHaveLength(1)

    const updated = await store.update(job.id, { name: 'weekly' })
    expect(updated.name).toBe('weekly')

    await store.delete(job.id)
    const afterDelete = await store.list('brain-1')
    expect(afterDelete).toHaveLength(0)
  })

  it('findDue returns only enabled due jobs', async () => {
    const store = createMemoryScheduleStore()

    const job1 = await store.create({
      brainId: 'brain-1',
      name: 'due',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/a.md' },
    })

    const past = new Date(Date.now() - 3600_000)
    await store.markRun(job1.id, new Date(Date.now() - 7200_000), past)

    const job2 = await store.create({
      brainId: 'brain-1',
      name: 'disabled',
      cronExpression: '* * * * *',
      target: { kind: 'file', path: '/b.md' },
    })
    await store.update(job2.id, { enabled: false })

    const due = await store.findDue(new Date())
    expect(due).toHaveLength(1)
    expect(due[0].id).toBe(job1.id)
  })

  it('invalid cron expression rejected on create', async () => {
    const store = createMemoryScheduleStore()
    await expect(
      store.create({
        brainId: 'brain-1',
        name: 'bad cron',
        cronExpression: 'invalid',
        target: { kind: 'file', path: '/a.md' },
      }),
    ).rejects.toThrow('invalid cron expression')
  })
})
