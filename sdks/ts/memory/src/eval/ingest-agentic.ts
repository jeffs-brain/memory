/**
 * Phase 0 agentic ingest. Runs `Memory.extract` on a sliding window of
 * messages per session, simulating a live agent observing the haystack
 * one turn at a time. This keeps extraction call-frequency honest: real
 * production traffic calls extract after each user turn, not once per
 * entire session as `ingestReplay` does.
 */

import type { Logger, Message } from '../llm/index.js'
import type { Memory } from '../memory/types.js'
import type { IngestOutcome, LMEExample, LMESessionMessage } from './types.js'

export type AgenticOpts = {
  /**
   * Number of sessions to process in parallel. Extract calls within a
   * session remain serial so observed-order matches the simulated agent.
   * Defaults to 4.
   */
  readonly concurrency?: number
  /**
   * Minimum messages between extract calls, matching the Go extractor's
   * frequency cap (6 messages). Defaults to 6; values below 2 round up.
   */
  readonly extractEvery?: number
  readonly logger?: Logger
}

export const ingestAgentic = async (
  memory: Memory,
  examples: readonly LMEExample[],
  opts: AgenticOpts = {},
): Promise<IngestOutcome> => {
  const concurrency = clamp(opts.concurrency ?? 4, 1, 32)
  const extractEvery = Math.max(opts.extractEvery ?? 6, 2)
  const warnings: string[] = []
  let sessionsProcessed = 0

  const jobs = flattenSessions(examples)
  const queue = [...jobs]

  const workers: Promise<void>[] = []
  for (let w = 0; w < concurrency; w++) {
    workers.push(
      (async () => {
        while (true) {
          const job = queue.shift()
          if (!job) return
          try {
            await simulateSession(memory, job, extractEvery)
            sessionsProcessed++
          } catch (err) {
            warnings.push(`session ${job.id}: agentic extract failed: ${errText(err)}`)
            opts.logger?.warn('lme agentic: extract failed', {
              session: job.id,
              err: errText(err),
            })
          }
        }
      })(),
    )
  }
  await Promise.all(workers)

  return {
    mode: 'agentic',
    sessionsWritten: sessionsProcessed,
    examplesIngested: examples.length,
    warnings,
  }
}

type AgenticJob = {
  readonly id: string
  readonly date?: string
  readonly messages: readonly LMESessionMessage[]
}

const flattenSessions = (examples: readonly LMEExample[]): readonly AgenticJob[] => {
  const seen = new Set<string>()
  const out: AgenticJob[] = []
  for (const ex of examples) {
    const sessions = ex.haystackSessions ?? []
    const dates = ex.haystackDates ?? []
    for (let i = 0; i < ex.sessionIds.length; i++) {
      const id = ex.sessionIds[i]
      if (id === undefined || id === '' || seen.has(id)) continue
      seen.add(id)
      const msgs = sessions[i]
      if (!msgs || msgs.length === 0) continue
      const date = dates[i]
      const job: AgenticJob =
        date !== undefined && date !== ''
          ? { id, date, messages: msgs }
          : { id, messages: msgs }
      out.push(job)
    }
  }
  return out
}

const simulateSession = async (
  memory: Memory,
  job: AgenticJob,
  extractEvery: number,
): Promise<void> => {
  const base: Message[] = []
  if (job.date !== undefined && job.date !== '') {
    base.push({ role: 'system', content: `This conversation took place on ${job.date}.` })
  }

  const observed: Message[] = [...base]
  let sinceLast = 0

  for (const m of job.messages) {
    observed.push({ role: m.role, content: m.content })
    sinceLast++
    if (sinceLast >= extractEvery) {
      await memory.extract({
        messages: observed,
        sessionId: job.id,
        ...(job.date !== undefined && job.date !== '' ? { sessionDate: job.date } : {}),
      })
      sinceLast = 0
    }
  }

  // Final flush so sessions shorter than `extractEvery` still extract.
  if (sinceLast > 0 && observed.length > base.length) {
    await memory.extract({
      messages: observed,
      sessionId: job.id,
      ...(job.date !== undefined && job.date !== '' ? { sessionDate: job.date } : {}),
    })
  }
}

const clamp = (n: number, min: number, max: number): number =>
  n < min ? min : n > max ? max : n

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))
