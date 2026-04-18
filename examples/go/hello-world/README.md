# Go Hello World

Minimal example: ingest a markdown doc into a brain via `knowledge.Ingest` and run a hybrid search over it.

## Run

```
cd examples/go/hello-world
go run .
```

Expected output lists the top matches for "where do hedgehogs live?".

## Status

End-to-end:

1. Open an `fs.Store` + `brain.Brain`.
2. Construct a `knowledge.Base` over the brain store.
3. Call `kb.Ingest(...)` to persist `docs/hedgehogs.md` under `raw/documents/`.
4. Call `kb.Search(...)` for the top hits.

The search call uses the in-memory fallback because the FTS5 index covers `memory/`, `wiki/`, and `raw/.sources/`, while `knowledge.Ingest` writes to `raw/documents/`. Once the search index is taught to walk `raw/documents/`, the example can bind a `search.Index` for BM25 ranking.
