# Python Hello World

Minimal example for `jeffs_brain_memory`: ingest a markdown doc into a
filesystem-backed brain via `knowledge.Base.ingest`, then run a search
via `knowledge.Base.search`.

## Run

From this directory:

```bash
uv run --with ../../sdks/py python main.py
# or, with the SDK already on the active interpreter:
python main.py
```

If you have not installed the SDK yet:

```bash
cd ../../sdks/py
uv sync
cd -
uv run --with ../../sdks/py python main.py
```

## Expected output

```
Ingested raw/documents/hedgehogs.md (1 chunks, 973 bytes)
Top 1 results for "where do hedgehogs live?":
1. [8.000] raw/documents/hedgehogs.md
   read across Europe, Asia, Africa, and parts of New Zealand, where they were introduced by British settlers in the nineteenth century.

   Hedgehogs live in hedgero...
```

The corpus is a single short markdown doc that fits in one chunk, so retrieval returns a single hit. The score is the in-memory BM25-style scorer's term-frequency tally (no live FTS5 index in this minimal demo). Drop more files into `docs/` to see real ranking, or wire a `jeffs_brain_memory.search.Index` for the SQLite path.

## How it works

1. Opens an `FsStore` under `./data/hello-world/`.
2. Constructs a `knowledge.Base` over the store (no search index wired,
   so retrieval falls back to the in-memory BM25-style scorer that walks
   `raw/documents/`).
3. Calls `await kb.ingest(...)` to persist `docs/hedgehogs.md` under the
   brain's `raw/documents/` tree with regenerated frontmatter.
4. Calls `await kb.search(...)` for the top hits.

To exercise the SQLite FTS5 + sqlite-vec hybrid path, build a
`jeffs_brain_memory.search.Index`, pass it via `Options(index=...)`, and
call `index.upsert_chunks(...)` after each ingest (or use the daemon
which wires both ends together automatically).
