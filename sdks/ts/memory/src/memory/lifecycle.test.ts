// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { createMemoryLifecycle } from './lifecycle.js'
import type { Memory } from './types.js'

describe('createMemoryLifecycle', () => {
  it('runs beforeTurn and afterTurn through the underlying memory instance', async () => {
    const contextualise = vi.fn<Memory['contextualise']>().mockResolvedValue({
      userMessage: 'How should I approach auth?',
      memories: [],
      systemReminder: 'Remember the previous auth decision.',
    })
    const extract = vi.fn<Memory['extract']>().mockResolvedValue([
      {
        action: 'create',
        filename: 'project-auth.md',
        name: 'Auth decision',
        description: 'Chosen auth provider',
        type: 'project',
        content: 'Use OIDC.',
        indexEntry: '- project-auth.md: auth decision',
        scope: 'project',
      },
    ])
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise,
        extract,
        reflect: vi.fn<Memory['reflect']>(),
        consolidate: vi.fn<Memory['consolidate']>(),
      },
    })

    const prompt = await lifecycle.beforeTurn({
      message: 'How should I approach auth?',
      actorId: 'tenant-a',
      scope: 'project',
    })
    const extracted = await lifecycle.afterTurn({
      messages: [{ role: 'user', content: 'We settled on OIDC.' }],
      actorId: 'tenant-a',
      sessionId: 'session-1',
      scope: 'project',
    })

    expect(contextualise).toHaveBeenCalledWith({
      message: 'How should I approach auth?',
      actorId: 'tenant-a',
      scope: 'project',
    })
    expect(extract).toHaveBeenCalledWith({
      messages: [{ role: 'user', content: 'We settled on OIDC.' }],
      actorId: 'tenant-a',
      sessionId: 'session-1',
      scope: 'project',
    })
    expect(prompt.systemReminder).toBe('Remember the previous auth decision.')
    expect(extracted).toHaveLength(1)
  })

  it('runs session-end reflection and optional consolidation in sequence', async () => {
    const calls: string[] = []
    const reflect = vi.fn<Memory['reflect']>().mockImplementation(async () => {
      calls.push('reflect')
      return {
        outcome: 'success',
        summary: 'Finished cleanly.',
        openQuestions: [],
        heuristics: [],
        path: 'reflections/session-1.md' as never,
      }
    })
    const consolidate = vi.fn<Memory['consolidate']>().mockImplementation(async () => {
      calls.push('consolidate')
      return { merged: 1, deleted: 0, promoted: 0, ops: [], errors: [] }
    })
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise: vi.fn<Memory['contextualise']>(),
        extract: vi.fn<Memory['extract']>(),
        reflect,
        consolidate,
      },
    })

    const result = await lifecycle.endSession({
      messages: [{ role: 'assistant', content: 'Done.' }],
      sessionId: 'session-1',
      actorId: 'tenant-a',
      scope: 'project',
      consolidate: true,
    })

    expect(reflect).toHaveBeenCalledWith({
      messages: [{ role: 'assistant', content: 'Done.' }],
      sessionId: 'session-1',
      actorId: 'tenant-a',
      scope: 'project',
      consolidate: true,
    })
    expect(consolidate).toHaveBeenCalledWith({
      actorId: 'tenant-a',
      scope: 'project',
    })
    expect(result.reflection?.outcome).toBe('success')
    expect(result.consolidation?.merged).toBe(1)
    expect(calls).toEqual(['reflect', 'consolidate'])
  })

  it('prepends buffered L0 observations ahead of recalled memory reminders', async () => {
    const contextualise = vi.fn<Memory['contextualise']>().mockResolvedValue({
      userMessage: 'How should I approach auth?',
      memories: [],
      systemReminder: 'Remember the previous auth decision.',
    })
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise,
        extract: vi.fn<Memory['extract']>().mockResolvedValue([]),
        reflect: vi.fn<Memory['reflect']>().mockResolvedValue(undefined),
        consolidate: vi.fn<Memory['consolidate']>().mockResolvedValue({
          merged: 0,
          deleted: 0,
          promoted: 0,
          ops: [],
          errors: [],
        }),
      },
    })

    await lifecycle.afterTurn({
      messages: [
        { role: 'user', content: 'We settled on OIDC for the auth provider.' },
        { role: 'assistant', content: 'I will remember that decision.' },
      ],
      sessionId: 'session-1',
      actorId: 'tenant-a',
      scope: 'project',
    })

    const prompt = await lifecycle.beforeTurn({
      message: 'How should I approach auth?',
      actorId: 'tenant-a',
      scope: 'project',
    })

    expect(prompt.systemReminder).toContain('Recent session observations:')
    expect(prompt.systemReminder).toContain('We settled on OIDC for the auth provider.')
    expect(prompt.systemReminder).toContain('Remember the previous auth decision.')
    expect(prompt.systemReminder.indexOf('Recent session observations:')).toBeLessThan(
      prompt.systemReminder.indexOf('Remember the previous auth decision.'),
    )
  })

  it('persists procedural records after extraction without changing the afterTurn result', async () => {
    const detectAndPersistProceduralRecords = vi
      .fn<NonNullable<Memory['detectAndPersistProceduralRecords']>>()
      .mockResolvedValue([])
    const extract = vi.fn<Memory['extract']>().mockResolvedValue([
      {
        action: 'create',
        filename: 'project-auth.md',
        name: 'Auth decision',
        description: 'Chosen auth provider',
        type: 'project',
        content: 'Use OIDC.',
        indexEntry: '- project-auth.md: auth decision',
        scope: 'project',
      },
    ])
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise: vi.fn<Memory['contextualise']>(),
        extract,
        reflect: vi.fn<Memory['reflect']>(),
        consolidate: vi.fn<Memory['consolidate']>(),
        detectAndPersistProceduralRecords,
      },
    })

    const extracted = await lifecycle.afterTurn({
      messages: [{ role: 'user', content: 'Deploy through the kubernetes skill.' }],
      actorId: 'tenant-a',
      sessionId: 'session-1',
      scope: 'project',
    })

    expect(extracted).toHaveLength(1)
    expect(detectAndPersistProceduralRecords).toHaveBeenCalledWith({
      messages: [{ role: 'user', content: 'Deploy through the kubernetes skill.' }],
      actorId: 'tenant-a',
      sessionId: 'session-1',
    })
  })

  it('respects explicit consolidation args and can skip consolidation', async () => {
    const reflect = vi.fn<Memory['reflect']>().mockResolvedValue(undefined)
    const consolidate = vi.fn<Memory['consolidate']>().mockResolvedValue({
      merged: 0,
      deleted: 0,
      promoted: 0,
      ops: [],
      errors: [],
    })
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise: vi.fn<Memory['contextualise']>(),
        extract: vi.fn<Memory['extract']>(),
        reflect,
        consolidate,
      },
    })

    const skipped = await lifecycle.endSession({
      messages: [],
      sessionId: 'session-1',
    })
    const explicit = await lifecycle.endSession({
      messages: [],
      sessionId: 'session-1',
      consolidate: { actorId: 'tenant-b', scope: 'global' },
    })

    expect(skipped).toEqual({ reflection: undefined })
    expect(consolidate).toHaveBeenCalledTimes(1)
    expect(consolidate).toHaveBeenCalledWith({ actorId: 'tenant-b', scope: 'global' })
    expect(explicit.consolidation?.merged).toBe(0)
  })

  it('records an episode at session end when reflection does not opt out', async () => {
    const calls: string[] = []
    const reflect = vi.fn<Memory['reflect']>().mockImplementation(async () => {
      calls.push('reflect')
      return {
        outcome: 'success',
        summary: 'Finished cleanly.',
        openQuestions: [],
        heuristics: [],
        shouldRecordEpisode: true,
        path: 'reflections/session-episode.md' as never,
      }
    })
    const recordEpisode = vi
      .fn<NonNullable<Memory['recordEpisode']>>()
      .mockImplementation(async () => {
        calls.push('episode')
        return {
          allowed: true,
          reason: 'passed' as const,
          signals: {
            messageCount: 8,
            substantiveMessageCount: 8,
            userMessageCount: 3,
            assistantMessageCount: 3,
            toolMessageCount: 2,
            toolCallCount: 1,
            writeSignal: true,
            editSignal: true,
            toolSignal: true,
          },
          recorded: true,
          disposition: 'created' as const,
          path: 'episodes/session-1.md' as never,
          episode: {
            path: 'episodes/session-1.md' as never,
            sessionId: 'session-1',
            actorId: 'tenant-a',
            scope: 'project',
            name: 'Episode session-1',
            summary: 'Refactored auth and fixed tests.',
            outcome: 'success',
            retryFeedback: '',
            shouldRecordEpisode: true,
            openQuestions: [],
            heuristics: [],
            tags: ['episode', 'auth', 'tests'],
            signals: {
              messageCount: 8,
              substantiveMessageCount: 8,
              userMessageCount: 3,
              assistantMessageCount: 3,
              toolMessageCount: 2,
              toolCallCount: 1,
              writeSignal: true,
              editSignal: true,
              toolSignal: true,
            },
          },
        }
      })
    const consolidate = vi.fn<Memory['consolidate']>().mockImplementation(async () => {
      calls.push('consolidate')
      return { merged: 1, deleted: 0, promoted: 0, ops: [], errors: [] }
    })
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise: vi.fn<Memory['contextualise']>(),
        extract: vi.fn<Memory['extract']>(),
        reflect,
        consolidate,
        recordEpisode,
      },
    })

    const result = await lifecycle.endSession({
      messages: [{ role: 'assistant', content: 'Done.' }],
      sessionId: 'session-1',
      actorId: 'tenant-a',
      scope: 'project',
      consolidate: true,
    })

    expect(recordEpisode).toHaveBeenCalledWith({
      messages: [{ role: 'assistant', content: 'Done.' }],
      sessionId: 'session-1',
      reflection: {
        outcome: 'success',
        summary: 'Finished cleanly.',
        retryFeedback: '',
        shouldRecordEpisode: true,
        openQuestions: [],
        heuristics: [],
      },
      actorId: 'tenant-a',
      scope: 'project',
    })
    expect(result.episode?.recorded).toBe(true)
    expect(result.episode?.episode?.outcome).toBe('success')
    expect(calls).toEqual(['reflect', 'episode', 'consolidate'])
  })

  it('skips episode capture when reflection explicitly opts out', async () => {
    const recordEpisode = vi.fn<NonNullable<Memory['recordEpisode']>>()
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise: vi.fn<Memory['contextualise']>(),
        extract: vi.fn<Memory['extract']>(),
        reflect: vi.fn<Memory['reflect']>().mockResolvedValue({
          outcome: 'partial',
          summary: 'Routine turn.',
          openQuestions: [],
          heuristics: [],
          shouldRecordEpisode: false,
          path: 'reflections/session-2.md' as never,
        }),
        consolidate: vi.fn<Memory['consolidate']>(),
        recordEpisode,
      },
    })

    await lifecycle.endSession({
      messages: [],
      sessionId: 'session-2',
    })

    expect(recordEpisode).not.toHaveBeenCalled()
  })

  it('compacts to the newest buffered observations and clears them on session end', async () => {
    const lifecycle = createMemoryLifecycle({
      memory: {
        contextualise: vi.fn<Memory['contextualise']>().mockResolvedValue({
          userMessage: 'What next?',
          memories: [],
          systemReminder: '',
        }),
        extract: vi.fn<Memory['extract']>().mockResolvedValue([]),
        reflect: vi.fn<Memory['reflect']>().mockResolvedValue(undefined),
        consolidate: vi.fn<Memory['consolidate']>().mockResolvedValue({
          merged: 0,
          deleted: 0,
          promoted: 0,
          ops: [],
          errors: [],
        }),
      },
      l0Buffer: {
        tokenBudget: 20,
        compactThresholdPercent: 100,
        keepRecentPercent: 50,
        maxObservationLength: 200,
      },
    })

    await lifecycle.afterTurn({
      messages: [{ role: 'user', content: 'First task note that should be compacted away.' }],
      sessionId: 'session-1',
    })
    await lifecycle.afterTurn({
      messages: [{ role: 'user', content: 'Second task note that should also be compacted away.' }],
      sessionId: 'session-1',
    })
    await lifecycle.afterTurn({
      messages: [{ role: 'user', content: 'Latest task note that must survive compaction.' }],
      sessionId: 'session-1',
    })

    const beforeReset = await lifecycle.beforeTurn({
      message: 'What next?',
    })
    expect(beforeReset.systemReminder).toContain('Latest task note that must survive compaction.')
    expect(beforeReset.systemReminder).not.toContain(
      'First task note that should be compacted away.',
    )

    await lifecycle.endSession({
      messages: [],
      sessionId: 'session-1',
    })

    const afterReset = await lifecycle.beforeTurn({
      message: 'Fresh session now.',
    })
    expect(afterReset.systemReminder).toBe('')
  })
})
