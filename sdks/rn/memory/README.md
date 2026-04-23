# @jeffs-brain/memory-react-native

React Native SDK for `jeffs-brain`, built for offline-first assistants on iOS and Android.

## What It Includes

- Portable local store with the shared brain file layout
- SQLite search index with BM25 and vector retrieval
- Memory client with remember, recall, extract, reflect, consolidate, episodes, and procedural records
- Configurable routing between local and cloud providers
- OpenAI-compatible cloud provider and embedder helpers
- Document ingest helpers for markdown, plain text, and fetched URLs

## Install

```bash
bun add @jeffs-brain/memory-react-native
```

Peer dependencies:

- `react`
- `react-native`
- `expo-file-system`
- `@op-engineering/op-sqlite`
- `@react-native-community/netinfo`

Only the dependencies you use at runtime need to be installed.

## Quick Start

```ts
import { useMemory } from '@jeffs-brain/memory-react-native'

const { memory, isReady, error } = useMemory({
  brainId: 'assistant',
})
```

Once ready, the returned client exposes the same main memory pipeline surface as the other SDKs.

## Model Hosting

The open source package does not bake in deployment-specific model hosts. Upstream public artefacts remain the default. If you mirror models or route cloud calls through your own infrastructure, pass that configuration explicitly through the model manifest helpers or provider options.

## Repository

Project docs, the shared spec, and examples live in the main repository:

`https://github.com/jeffs-brain/memory`
