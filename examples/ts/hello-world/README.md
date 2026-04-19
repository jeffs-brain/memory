# TS Hello World

Minimal example for `@jeffs-brain/memory`: chunk and ingest a markdown file via the filesystem store, upsert the chunks into a SQLite search index, and run a BM25 query.

## Run

From the repo root:

```bash
bun install
bun run build:packages                 # builds the TS SDK dist/
cd examples/ts/hello-world
bun install
bun run start
```

## Expected output

```
Ingested raw/documents/hedgehogs.md (1 chunks, 973 bytes)
Top 1 results for "where do hedgehogs live?":
1. [<score>] raw/documents/hedgehogs.md
   # Hedgehogs Hedgehogs are small, nocturnal mammals best known for...
```

The corpus is a single short markdown doc that fits in one chunk, so the BM25 path returns a single hit. Scores are FTS5 ranks (lower magnitude is better in this configuration); exact values depend on corpus statistics. Drop more files into `docs/` to see real ranking.

## Notes

- Uses a `file:` dependency on the workspace SDK at `../../../sdks/ts/memory`. The SDK's `dist/` must be built (`bun run build:packages` from the repo root) before installing this example.
- Goes through the lower-level `chunkMarkdown` + `createSearchIndex` path rather than `createKnowledge`, because the high-level `Knowledge` factory requires an LLM `Provider` and this example is provider-free. The Go and Python siblings use the higher-level `knowledge.Base` path because their factories accept a store-only configuration.
- Documents are written to `raw/documents/<file>`, matching the layout used by the Go and Python `knowledge.Base` factories.
- No embedder is wired, so the example is BM25-only. To explore hybrid retrieval, supply an `Embedder` and call `createRetrieval` instead of `createSearchIndex`.
- Imports resolve through the SDK's public exports (`chunkMarkdown`, `createFsStore`, `createSearchIndex`, `toPath`). The SDK's `src/index.ts` is the canonical reference.
