# @jeffs-brain/memory-postgres

Postgres adapter for [`@jeffs-brain/memory`](https://www.npmjs.com/package/@jeffs-brain/memory). Provides a Postgres-backed `Store`, a hybrid search index (tsvector BM25 plus halfvec cosine with RRF fusion) and a retrieval factory that wires them together. Kept as a separate optional package so the core OSS surface has zero hard dependency on Drizzle or the `postgres` driver. Pull this in when running a high-scale or managed-cloud backend.

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
import { createPostgresStore, createPostgresRetrieval } from '@jeffs-brain/memory-postgres'

const sql = postgres(process.env.DATABASE_URL!)
const store = createPostgresStore({ sql })
const retrieval = createPostgresRetrieval({ sql, embedder })

const hits = await retrieval.search({ query: 'which database did we pick?', limit: 5 })
```

## Migrations

SQL migrations live in [`migrations/`](./migrations). Apply them in order with your preferred migration runner (`drizzle-kit`, `psql -f`, Atlas, Flyway).

## Docs

- Repo README: https://github.com/jeffs-brain/memory#readme
- Protocol and storage spec: [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec)

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
