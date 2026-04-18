# TS Hello World

Minimal example: ingest a markdown file and run a BM25 search against it.

## Run

```
bun install
bun run start
```

Expected output lists the top 3 matches for "where do hedgehogs live?", with the hedgerow sentence scoring highest.

## Notes

- Uses a `file:` dependency on the workspace SDK at `../../../sdks/ts/memory`. The SDK's `dist/` must be built (run `bun run build:packages` from the repo root) before installing this example.
- No embedder is wired, so the example is BM25 only. To explore hybrid retrieval, supply an `Embedder` and call `createRetrieval` instead.
