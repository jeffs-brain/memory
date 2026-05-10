# Getting Started

Welcome to the project documentation. This guide covers installation,
configuration, and basic usage patterns for new developers.

## Installation

Install the package using your preferred package manager:

```bash
npm install @jeffs-brain/memory
```

Verify the installation by running the version check:

```bash
npx memory --version
```

## Configuration

The system requires a configuration file at the project root. Create a
file named `memory.config.json` with the following structure:

```json
{
  "brain": "default",
  "store": "filesystem",
  "index": "sqlite"
}
```

Each field controls a different subsystem. The brain field selects which
knowledge base to target. The store field picks the persistence backend.
The index field determines the full-text search engine.

## Basic Usage

Once configured, ingest documents into the knowledge base:

```typescript
import { createBrain } from '@jeffs-brain/memory'

const brain = await createBrain({ config: './memory.config.json' })
await brain.ingest({ path: './docs/guide.md' })
const results = await brain.search({ query: 'installation steps' })
```

The search function returns ranked results using hybrid retrieval. Each
result includes a score, the matched chunk content, and metadata about
the source document.

## Advanced Topics

### Custom Embeddings

Override the default embedder by passing a provider configuration:

```typescript
const brain = await createBrain({
  config: './memory.config.json',
  embedder: { model: 'text-embedding-3-small', dimensions: 256 },
})
```

### Batch Ingestion

For large document sets, use the batch API to ingest multiple files in a
single transaction:

```typescript
await brain.batch({ reason: 'initial-load' }, async (b) => {
  for (const file of files) {
    await b.ingest({ path: file })
  }
})
```

This ensures atomicity: either all documents are indexed or none are.
