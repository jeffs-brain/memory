# @jeffs-brain/memory-postgres

Postgres adapter for [`@jeffs-brain/memory`](https://www.npmjs.com/package/@jeffs-brain/memory). Provides a Postgres-backed `Store`, a hybrid search index (tsvector BM25 plus halfvec cosine with RRF fusion), and a retriever factory that wires them together. Kept as a separate optional package so the core OSS surface has zero hard dependency on the `postgres` driver. Pull this in when running a high-scale or managed-cloud backend.

This adapter ships for TypeScript only at v0.1.0. The Go and Python SDKs do not yet have a Postgres adapter; if you need cross-SDK parity, stick to the core stores (`FsStore`, `MemStore`, `GitStore`, `HttpStore`) for now.

## Install

```bash
npm i @jeffs-brain/memory @jeffs-brain/memory-postgres
# or
bun add @jeffs-brain/memory @jeffs-brain/memory-postgres
```

The package requires Postgres 15 or later with the `vector` (pgvector) extension enabled.

## Usage

```ts
import postgres from 'postgres'
import {
  createPostgresStore,
  createPostgresSearchIndex,
  createPostgresRetriever,
} from '@jeffs-brain/memory-postgres'

const sql = postgres(process.env.DATABASE_URL!)

const store = await createPostgresStore({
  sql,
  tenantId: process.env.TENANT_ID!,
  brainId: process.env.BRAIN_ID!,
})

const index = createPostgresSearchIndex({
  sql,
  tenantId: process.env.TENANT_ID!,
  brainId: process.env.BRAIN_ID!,
})

const retriever = createPostgresRetriever({
  pg: sql,
  tenantId: process.env.TENANT_ID!,
  brainId: process.env.BRAIN_ID!,
  env: process.env,
})

const hits = await retriever.retrieve({
  query: 'which database did we pick?',
  limit: 5,
})
```

`tenantId` and `brainId` must be UUIDs; the adapter enforces this at construction time and pins `app.tenant_id` per transaction so Row Level Security policies stay honoured.

## Migrations

SQL migrations live in [`migrations/`](./migrations). Apply them in order with your preferred runner (`drizzle-kit`, `psql -f`, Atlas, Flyway).

## Feature support

- Postgres `Store` implementation matching the `spec/STORAGE.md` contract.
- Hybrid index: tsvector BM25 plus halfvec cosine fused with Reciprocal Rank Fusion at `k=60`.
- `createPostgresRetriever` returns a retriever compatible with the cross-SDK `spec/PROTOCOL.md` ask and search endpoints.

## Documentation

- Postgres store guide: https://docs.jeffsbrain.com/guides/stores/
- TypeScript getting started: https://docs.jeffsbrain.com/getting-started/typescript/
- Protocol and storage spec: [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec)

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
