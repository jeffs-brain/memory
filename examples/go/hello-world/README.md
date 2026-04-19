# Go Hello World

Minimal example for `github.com/jeffs-brain/memory/go`: ingest a markdown doc into a brain via `knowledge.Ingest`, then run a hybrid search via `knowledge.Search`.

## Run

```bash
cd examples/go/hello-world
go run .
```

## Expected output

```
Ingested raw/documents/hedgehogs.md (1 chunks, 973 bytes)
Top 1 results for "where do hedgehogs live?":
1. [<score>] raw/documents/hedgehogs.md
   # <mark>Hedgehogs</mark>

   <mark>Hedgehogs</mark> are small, nocturnal mammals best known for the thousands of stiff spines...
```

The corpus is a single short markdown doc that fits in one chunk, so the BM25 path returns a single hit. FTS5 wraps query-term matches in `<mark>...</mark>` highlights. Drop more files into `docs/` to see real ranking.

## How it works

1. Opens an `fs.Store` under `./data/hello-world/`.
2. Opens the SQLite FTS5 index at `.search.db` and subscribes it to the store so writes flow into the index.
3. Constructs a `knowledge.Base` over the store + index.
4. Calls `kb.Ingest(...)` to persist `docs/hedgehogs.md` under `raw/documents/`.
5. Calls `kb.Search(...)` for the top hits.

The search index walks `raw/documents/`, `memory/`, and `wiki/`, so ingested content is immediately discoverable through BM25. No embedder is wired; swap in `search.NewIndex` with an `Embedder` to exercise the hybrid path.
