# @jeffs-brain/memory-postgres

Postgres adapters for `@jeffs-brain/memory`. Provides a Postgres-backed `Store`, a hybrid search index (tsvector BM25 + halfvec cosine with RRF fusion), and a retrieval factory that wires them together. Kept as a separate optional package so the core `@jeffs-brain/memory` OSS surface has zero hard dependency on Drizzle or the `postgres` driver; pull this package in only when running the managed-cloud backend.
