import { useState } from 'react'
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native'

import {
  type MemoryClient,
  type Message,
  OpenAIEmbedder,
  OpenAIProvider,
  type Provider,
  useChat,
  useMemory,
  useRecall,
} from '@jeffs-brain/memory-react-native'

const API_KEY = process.env.EXPO_PUBLIC_OPENAI_API_KEY ?? ''
const BASE_URL = process.env.EXPO_PUBLIC_OPENAI_BASE_URL
const CHAT_MODEL = process.env.EXPO_PUBLIC_OPENAI_MODEL ?? 'gpt-4o-mini'
const EMBED_MODEL = process.env.EXPO_PUBLIC_OPENAI_EMBED_MODEL ?? 'text-embedding-3-small'
const BRAIN_ID = process.env.EXPO_PUBLIC_MEMORY_BRAIN_ID ?? 'hello-world'

const provider =
  API_KEY === ''
    ? undefined
    : new OpenAIProvider({
        apiKey: API_KEY,
        model: CHAT_MODEL,
        ...(BASE_URL === undefined ? {} : { baseURL: BASE_URL }),
      })

const embedder =
  API_KEY === ''
    ? undefined
    : new OpenAIEmbedder({
        apiKey: API_KEY,
        model: EMBED_MODEL,
        ...(BASE_URL === undefined ? {} : { baseURL: BASE_URL }),
      })

const statusTone = (memoryReady: boolean, providerReady: boolean): string => {
  if (!providerReady) return 'Provider not configured'
  if (!memoryReady) return 'Initialising brain'
  return 'Ready'
}

const seedMemory = async (memory: MemoryClient | null): Promise<void> => {
  if (memory === null) return
  await memory.remember({
    filename: 'favourite-breakfast.md',
    name: 'Favourite breakfast',
    description: 'Alex prefers a savoury breakfast with eggs and chilli oil.',
    content: 'Alex prefers a savoury breakfast with eggs, sourdough, and chilli oil.',
    tags: ['food', 'preference'],
    type: 'user',
  })
}

type ChatPaneProps = {
  readonly memory: MemoryClient
  readonly provider: Provider
}

const ChatPane = ({ memory, provider }: ChatPaneProps) => {
  const [draft, setDraft] = useState('')
  const { messages, send, isGenerating, isUsingCloud, error } = useChat({
    memory,
    provider,
    systemPrompt: 'You are a helpful assistant. Use the recalled memory when it helps.',
    rememberConversation: true,
  })

  const submit = async (): Promise<void> => {
    const trimmed = draft.trim()
    if (trimmed === '') return
    setDraft('')
    await send(trimmed)
  }

  return (
    <View style={styles.panel}>
      <View style={styles.row}>
        <Text style={styles.panelTitle}>Chat</Text>
        <Text style={styles.routePill}>{isUsingCloud ? 'Cloud route' : 'Route pending'}</Text>
      </View>
      <View style={styles.chatLog}>
        {messages.length === 0 ? (
          <Text style={styles.muted}>
            Start a chat after setting `EXPO_PUBLIC_OPENAI_API_KEY`. Memories extracted from the
            exchange will be written to the local brain.
          </Text>
        ) : (
          <ScrollView contentContainerStyle={styles.messageStack}>
            {messages.map((message, index) => (
              <View
                key={`${message.role}-${index}`}
                style={[
                  styles.messageBubble,
                  message.role === 'user' ? styles.userBubble : styles.assistantBubble,
                ]}
              >
                <Text style={styles.messageRole}>{message.role}</Text>
                <Text style={styles.messageText}>{message.content ?? ''}</Text>
              </View>
            ))}
          </ScrollView>
        )}
      </View>
      <TextInput
        value={draft}
        onChangeText={setDraft}
        placeholder="Ask something that can reuse memory"
        placeholderTextColor="#7c6e62"
        style={styles.input}
      />
      <Pressable onPress={() => void submit()} disabled={isGenerating} style={styles.primaryButton}>
        <Text style={styles.primaryButtonText}>{isGenerating ? 'Sending…' : 'Send'}</Text>
      </Pressable>
      {error !== null ? <Text style={styles.errorText}>{error.message}</Text> : null}
    </View>
  )
}

export default function App() {
  const [query, setQuery] = useState('breakfast')
  const { memory, isReady, error } = useMemory({
    brainId: BRAIN_ID,
    ...(provider === undefined ? {} : { provider }),
    ...(embedder === undefined ? {} : { embedder }),
  })
  const { results, isLoading } = useRecall(memory, query, { topK: 5 })

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>React Native Example</Text>
          <Text style={styles.title}>Cloud-backed memory on device</Text>
          <Text style={styles.subtitle}>
            This example uses the RN SDK with Expo file storage, `op-sqlite`, and the
            OpenAI-compatible provider path. Local on-device inference is a separate native
            integration step.
          </Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.panelTitle}>Status</Text>
          <View style={styles.row}>
            <Text style={styles.statusPill}>{statusTone(isReady, provider !== undefined)}</Text>
            {!isReady ? <ActivityIndicator color="#16322b" /> : null}
          </View>
          <Text style={styles.muted}>Brain id: {BRAIN_ID}</Text>
          {provider === undefined ? (
            <Text style={styles.warningText}>
              Set `EXPO_PUBLIC_OPENAI_API_KEY` to enable chat, extract, and reflect.
            </Text>
          ) : null}
          {error !== null ? <Text style={styles.errorText}>{error.message}</Text> : null}
          <Pressable
            onPress={() => void seedMemory(memory)}
            disabled={!isReady}
            style={styles.secondaryButton}
          >
            <Text style={styles.secondaryButtonText}>Seed a sample memory</Text>
          </Pressable>
        </View>

        <View style={styles.panel}>
          <Text style={styles.panelTitle}>Recall</Text>
          <TextInput
            value={query}
            onChangeText={setQuery}
            placeholder="Search the local brain"
            placeholderTextColor="#7c6e62"
            style={styles.input}
          />
          {isLoading ? <Text style={styles.muted}>Searching…</Text> : null}
          <View style={styles.resultStack}>
            {results.map((result) => (
              <View key={result.path} style={styles.resultCard}>
                <Text style={styles.resultTitle}>{result.note.name}</Text>
                <Text style={styles.resultMeta}>
                  {result.note.scope} · score {result.score.toFixed(3)}
                </Text>
                <Text style={styles.resultBody}>{result.note.content}</Text>
              </View>
            ))}
            {results.length === 0 && !isLoading ? (
              <Text style={styles.muted}>
                No matches yet. Seed the sample memory or chat once the provider is configured.
              </Text>
            ) : null}
          </View>
        </View>

        {memory !== null && provider !== undefined ? (
          <ChatPane memory={memory} provider={provider} />
        ) : (
          <View style={styles.panel}>
            <Text style={styles.panelTitle}>Chat</Text>
            <Text style={styles.muted}>
              The chat panel appears once the brain is ready and the cloud provider is configured.
            </Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#f6f0e8',
  },
  container: {
    padding: 20,
    gap: 16,
  },
  hero: {
    gap: 10,
    paddingTop: 12,
  },
  eyebrow: {
    color: '#7a4d2a',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.4,
    textTransform: 'uppercase',
  },
  title: {
    color: '#1c2d27',
    fontSize: 32,
    fontWeight: '800',
    lineHeight: 36,
  },
  subtitle: {
    color: '#4f5c57',
    fontSize: 15,
    lineHeight: 22,
  },
  panel: {
    backgroundColor: '#fffdf8',
    borderColor: '#d7c9b7',
    borderRadius: 22,
    borderWidth: 1,
    gap: 12,
    padding: 16,
  },
  panelTitle: {
    color: '#1c2d27',
    fontSize: 18,
    fontWeight: '700',
  },
  row: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 10,
    justifyContent: 'space-between',
  },
  statusPill: {
    backgroundColor: '#dbe9e2',
    borderRadius: 999,
    color: '#16322b',
    overflow: 'hidden',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  routePill: {
    backgroundColor: '#efe0c6',
    borderRadius: 999,
    color: '#6e4a23',
    overflow: 'hidden',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  muted: {
    color: '#5d605a',
    fontSize: 14,
    lineHeight: 20,
  },
  warningText: {
    color: '#81541d',
    fontSize: 14,
    lineHeight: 20,
  },
  errorText: {
    color: '#9b2335',
    fontSize: 14,
    lineHeight: 20,
  },
  input: {
    backgroundColor: '#f7f3eb',
    borderColor: '#d7c9b7',
    borderRadius: 16,
    borderWidth: 1,
    color: '#1c2d27',
    minHeight: 48,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#16322b',
    borderRadius: 16,
    minHeight: 48,
    justifyContent: 'center',
  },
  primaryButtonText: {
    color: '#f9f4ec',
    fontSize: 15,
    fontWeight: '700',
  },
  secondaryButton: {
    alignItems: 'center',
    borderColor: '#16322b',
    borderRadius: 16,
    borderWidth: 1,
    minHeight: 48,
    justifyContent: 'center',
  },
  secondaryButtonText: {
    color: '#16322b',
    fontSize: 15,
    fontWeight: '700',
  },
  resultStack: {
    gap: 10,
  },
  resultCard: {
    backgroundColor: '#f7f3eb',
    borderRadius: 16,
    gap: 6,
    padding: 12,
  },
  resultTitle: {
    color: '#1c2d27',
    fontSize: 15,
    fontWeight: '700',
  },
  resultMeta: {
    color: '#6b6b63',
    fontSize: 12,
  },
  resultBody: {
    color: '#38423f',
    fontSize: 14,
    lineHeight: 20,
  },
  chatLog: {
    maxHeight: 280,
  },
  messageStack: {
    gap: 10,
  },
  messageBubble: {
    borderRadius: 16,
    gap: 4,
    padding: 12,
  },
  userBubble: {
    backgroundColor: '#dfece7',
  },
  assistantBubble: {
    backgroundColor: '#f1e7d8',
  },
  messageRole: {
    color: '#5b544c',
    fontSize: 11,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  messageText: {
    color: '#1c2d27',
    fontSize: 14,
    lineHeight: 20,
  },
})
