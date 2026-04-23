import { startTransition, useEffectEvent, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'

import type { Provider, StreamEvent } from '../llm/types.js'
import type { Message } from '../llm/types.js'
import type { Scope } from '../memory/paths.js'
import type { MemoryClient } from '../memory/types.js'

type RouteAwareProvider = Provider & {
  lastRoute?: () => { readonly route: 'local' | 'cloud' } | null
}

export type UseChatConfig = {
  readonly memory: MemoryClient
  readonly provider: Provider
  readonly systemPrompt?: string
  readonly scope?: Scope
  readonly actorId?: string
  readonly rememberConversation?: boolean
}

const appendAssistantText = (messages: readonly Message[], text: string): readonly Message[] => {
  if (messages.length === 0) {
    return [{ role: 'assistant', content: text }]
  }
  const last = messages[messages.length - 1]
  if (last?.role !== 'assistant') {
    return [...messages, { role: 'assistant', content: text }]
  }
  const next = [...messages]
  next[next.length - 1] = {
    ...last,
    content: `${last.content ?? ''}${text}`,
  }
  return next
}

export const useChat = (
  config: UseChatConfig,
): {
  readonly messages: readonly Message[]
  readonly send: (text: string) => Promise<void>
  readonly isGenerating: boolean
  readonly isUsingCloud: boolean
  readonly error: Error | null
} => {
  const [messages, setMessages] = useState<readonly Message[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [isUsingCloud, setIsUsingCloud] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const sendEvent = useEffectEvent(async (text: string): Promise<void> => {
    const userMessage: Message = { role: 'user', content: text }
    let assistantContent = ''
    startTransition(() => {
      setMessages((current) => [...current, userMessage])
      setIsGenerating(true)
      setError(null)
    })

    try {
      const context = await config.memory.contextualise({
        userMessage: text,
        query: text,
        ...(config.scope === undefined ? {} : { scope: config.scope }),
        ...(config.actorId === undefined ? {} : { actorId: config.actorId }),
      })

      const systemContent =
        context.systemReminder === ''
          ? config.systemPrompt
          : config.systemPrompt === undefined
            ? context.systemReminder
            : `${config.systemPrompt}\n\n${context.systemReminder}`
      const systemMessages: Message[] =
        systemContent === undefined ? [] : [{ role: 'system', content: systemContent }]
      const promptMessages: Message[] = [...systemMessages, ...messages, userMessage]

      if (config.provider.stream !== undefined) {
        for await (const event of config.provider.stream({
          taskType: 'chat',
          messages: promptMessages,
        })) {
          switch (event.type) {
            case 'text_delta':
              assistantContent += event.text
              startTransition(() => {
                setMessages((current) => appendAssistantText(current, event.text))
              })
              break
            case 'done':
              startTransition(() => {
                setIsGenerating(false)
              })
              break
            case 'error':
              throw event.error
            case 'tool_call':
              break
          }
        }
      } else {
        const response = await config.provider.complete({
          taskType: 'chat',
          messages: promptMessages,
        })
        assistantContent = response.content
        startTransition(() => {
          setMessages((current) => [...current, { role: 'assistant', content: response.content }])
          setIsGenerating(false)
        })
      }

      const route = (config.provider as RouteAwareProvider).lastRoute?.()
      startTransition(() => {
        setIsUsingCloud(route?.route === 'cloud')
      })

      if (config.rememberConversation === true) {
        if (assistantContent !== '') {
          await config.memory.extract({
            messages: [userMessage, { role: 'assistant', content: assistantContent }],
            ...(config.scope === undefined ? {} : { scope: config.scope }),
            ...(config.actorId === undefined ? {} : { actorId: config.actorId }),
          })
        }
      }
    } catch (resolved) {
      const nextError = resolved instanceof Error ? resolved : new Error(String(resolved))
      startTransition(() => {
        setError(nextError)
        setIsGenerating(false)
      })
    }
  })

  return {
    messages,
    send: async (text: string) => {
      await sendEvent(text)
    },
    isGenerating,
    isUsingCloud,
    error,
  }
}
