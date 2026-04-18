import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/index.js'
import { createProviderReader, truncateSmartly } from './read.js'

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
})
