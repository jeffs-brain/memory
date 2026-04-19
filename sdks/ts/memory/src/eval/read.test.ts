// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/index.js'
import type { LMEExample } from './types.js'
import { createProviderReader, runRead, truncateSmartly } from './read.js'

const throwingProvider: Provider = {
  name: () => 'stub',
  modelName: () => 'stub-model',
  supportsStructuredDecoding: () => false,
  stream: async function* (_req: CompletionRequest): AsyncIterable<StreamEvent> {
    yield { type: 'done', stopReason: 'end_turn' }
  },
  complete: async (): Promise<CompletionResponse> => {
    throw new Error('boom')
  },
  structured: async (_req: StructuredRequest): Promise<string> => '{}',
}

describe('reader helpers', () => {
  it('includes the replay guidance for preferences and counting in the prompt', async () => {
    let captured = ''
    const provider: Provider = {
      ...throwingProvider,
      complete: async (req: CompletionRequest): Promise<CompletionResponse> => {
        captured = req.messages[0]?.content ?? ''
        return {
          content: 'ok',
          toolCalls: [],
          usage: { inputTokens: 0, outputTokens: 0 },
          stopReason: 'end_turn',
        }
      },
    }
    const reader = createProviderReader(provider, { budgetChars: 10_000 })
    await reader({
      question: 'Can you suggest a hotel for my upcoming trip to Miami?',
      questionDate: '2023/09/30 (Sat) 18:36',
      context: 'context',
    })
    expect(captured).toContain('Never use a fact dated after the current date.')
    expect(captured).toContain('For habit and routine questions')
    expect(captured).toContain('avoid double counting the roll-up')
    expect(captured).toContain('include all confirmed historical amounts for the same subject across sessions')
    expect(captured).toContain('Infer durable preferences from concrete desired features')
    expect(captured).toContain('Ignore unrelated hostel, budget, or solo-travel examples')
  })

  it('falls back to raw context when the provider fails', async () => {
    const reader = createProviderReader(throwingProvider, { budgetChars: 10_000 })
    await expect(
      reader({
        question: 'What colour?',
        questionDate: '2023/09/30 (Sat) 18:36',
        context: 'The answer is blue.',
      }),
    ).resolves.toBe('The answer is blue.')
  })

  it('truncates session-by-session instead of dropping later sections entirely', () => {
    const content = [
      '---\nsession_id: s1\nalpha '.repeat(200),
      '---\nsession_id: s2\nbravo '.repeat(200),
    ].join('\n\n---\n\n')
    const truncated = truncateSmartly(content, 1200)
    expect(truncated).toContain('session_id: s1')
    expect(truncated).toContain('session_id: s2')
  })

  it('keeps a relevant mid-session fact when the reader budget is tight', async () => {
    let captured = ''
    const provider: Provider = {
      ...throwingProvider,
      complete: async (req: CompletionRequest): Promise<CompletionResponse> => {
        captured = req.messages[0]?.content ?? ''
        return {
          content: 'ok',
          toolCalls: [],
          usage: { inputTokens: 0, outputTokens: 0 },
          stopReason: 'end_turn',
        }
      },
    }

    const context = [
      '---\nsession_id: answer_9282283d_2\nsession_date: 2023/11/03 (Fri) 19:56\n---',
      ...Array.from({ length: 80 }, (_, index) => `[user]: filler ${String(index)}`),
      '[user]: And by the way, speaking of boundaries, I see Dr. Smith every week, and she is helping me work on this stuff.',
      ...Array.from({ length: 80 }, (_, index) => `[user]: more filler ${String(index)}`),
    ].join('\n')

    const reader = createProviderReader(provider, { budgetChars: 500 })
    await reader({
      question: 'How often do I see my therapist, Dr. Smith?',
      questionDate: '2023/11/10 (Fri) 00:38',
      context,
    })

    expect(captured).toContain('I see Dr. Smith every week')
  })

  it('resolves anchored action dates before calling the reader', async () => {
    const example: LMEExample = {
      id: 'q1',
      category: 'time',
      question: 'When did I submit my research paper on sentiment analysis?',
      answer: 'February 1st',
      sessionIds: [],
    }

    const outcome = await runRead(
      {
        retrieval: async () => ({
          passages: [],
          rendered: [
            'Retrieved facts (2):',
            '',
            ' 1. [2024-01-20] [paper]',
            'I submitted my research paper on sentiment analysis to ACL.',
            '',
            ' 2. [2024-01-25] [acl-date]',
            "I'm reviewing for ACL, and their submission date was February 1st.",
          ].join('\n'),
        }),
        reader: async () => {
          throw new Error('reader should not be called')
        },
      },
      example,
    )

    expect(outcome.predicted).toBe('February 1st')
    expect(outcome.error).toBeUndefined()
  })
})
