import { describe, expect, it } from 'vitest'

import type { InferenceBridge } from '../native/inference-bridge.js'
import type { GenerateParams, GenerateResult, ModelConfig, ModelInfo } from '../native/types.js'
import { LocalProvider } from './local-provider.js'

const createBridge = (
  implementation: (params: GenerateParams) => GenerateResult,
): {
  readonly bridge: InferenceBridge
  readonly calls: GenerateParams[]
} => {
  const calls: GenerateParams[] = []
  const bridge: InferenceBridge = {
    loadModel: async (_config: ModelConfig): Promise<void> => {},
    unloadModel: async (): Promise<void> => {},
    isModelLoaded: () => true,
    generate: async (params): Promise<GenerateResult> => {
      calls.push(params)
      return implementation(params)
    },
    getModelInfo: (): ModelInfo | null => null,
  }
  return { bridge, calls }
}

describe('LocalProvider', () => {
  it('prompts local structured calls with the schema without native JSON grammar', async () => {
    const schema = JSON.stringify({
      type: 'object',
      properties: {
        action: { enum: ['none', 'remember', 'recall'] },
        thinking: { enum: ['off', 'on'] },
      },
      required: ['action', 'thinking'],
    })
    const { bridge, calls } = createBridge(() => ({
      content: '{"action":"none","thinking":"off"}',
    }))
    const provider = new LocalProvider(bridge, 'gemma-test')

    await provider.structured({
      system: 'Choose the memory action.',
      messages: [{ role: 'user', content: 'hello' }],
      schema,
      schemaName: 'memory_tool_decision',
    })

    expect(calls).toHaveLength(1)
    expect(calls[0]?.responseFormat).toBeUndefined()
    expect(calls[0]?.enableThinking).toBe(false)
    expect(calls[0]?.messages[0]?.role).toBe('system')
    expect(calls[0]?.messages[0]?.content).toContain('Choose the memory action.')
    expect(calls[0]?.messages[0]?.content).toContain('memory_tool_decision')
    expect(calls[0]?.messages[0]?.content).toContain(schema)
  })

  it('passes tools through to the inference bridge and returns tool calls', async () => {
    const { bridge, calls } = createBridge(() => ({
      content: '',
      toolCalls: [
        {
          id: 'call-1',
          name: 'memory_recall',
          arguments: '{"query":"daughter"}',
        },
      ],
    }))
    const provider = new LocalProvider(bridge, 'gemma-test')

    const response = await provider.complete({
      taskType: 'chat',
      messages: [{ role: 'user', content: 'What is my daughter called?' }],
      tools: [
        {
          name: 'memory_recall',
          description: 'Recall memory',
          inputSchema: JSON.stringify({
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
          }),
        },
      ],
    })

    expect(calls[0]?.toolChoice).toBe('auto')
    expect(calls[0]?.tools?.[0]?.name).toBe('memory_recall')
    expect(response.toolCalls).toEqual([
      {
        id: 'call-1',
        name: 'memory_recall',
        arguments: '{"query":"daughter"}',
      },
    ])
  })

  it('caps local completions by default and preserves explicit request caps', async () => {
    const { bridge, calls } = createBridge(() => ({
      content: 'ok',
    }))
    const provider = new LocalProvider(bridge, 'gemma-test')

    await provider.complete({
      taskType: 'chat',
      messages: [{ role: 'user', content: 'hello' }],
    })
    await provider.complete({
      taskType: 'chat',
      messages: [{ role: 'user', content: 'hello again' }],
      maxTokens: 123,
    })

    expect(calls[0]?.maxTokens).toBe(512)
    expect(calls[1]?.maxTokens).toBe(123)
  })

  it('caps local streams by default and preserves explicit request caps', async () => {
    const calls: GenerateParams[] = []
    const bridge: InferenceBridge = {
      loadModel: async (_config: ModelConfig): Promise<void> => {},
      unloadModel: async (): Promise<void> => {},
      isModelLoaded: () => true,
      generate: async (_params): Promise<GenerateResult> => ({ content: '' }),
      generateStream: async function* (params) {
        calls.push(params)
        yield { type: 'done', stopReason: 'end_turn' }
      },
      getModelInfo: (): ModelInfo | null => null,
    }
    const provider = new LocalProvider(bridge, 'gemma-test')

    for await (const chunk of provider.stream({
      taskType: 'chat',
      messages: [{ role: 'user', content: 'hello' }],
    })) {
      void chunk
    }
    for await (const chunk of provider.stream({
      taskType: 'chat',
      messages: [{ role: 'user', content: 'hello again' }],
      maxTokens: 77,
    })) {
      void chunk
    }

    expect(calls[0]?.maxTokens).toBe(512)
    expect(calls[1]?.maxTokens).toBe(77)
  })
})
