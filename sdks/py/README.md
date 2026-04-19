# jeffs-brain-memory

Python SDK for Jeffs Brain, the cross-language memory library for LLM agents.

Part of the polyglot [`jeffs-brain/memory`](https://github.com/jeffs-brain/memory) monorepo. This SDK tracks the same [`spec/`](../../spec/) and conformance fixtures as the TypeScript and Go SDKs.

Full documentation lives at [docs.jeffsbrain.com](https://docs.jeffsbrain.com).

## Install

```bash
pip install jeffs-brain-memory
# or
uv add jeffs-brain-memory
```

Optional LLM extras (each pulls in the matching provider client):

```bash
pip install "jeffs-brain-memory[openai]"
pip install "jeffs-brain-memory[anthropic]"
pip install "jeffs-brain-memory[ollama]"
```

The CLI is exposed as `memory`. Confirm with `memory --version` (currently `0.0.1`).

## Quickstart

Mirrors [`examples/py/hello-world/main.py`](../../examples/py/hello-world/main.py). `Store` operations are async (run them under `asyncio.run`). `search.Index` operations are sync.

```python
import asyncio
from pathlib import Path

from jeffs_brain_memory.knowledge import (
    CONTENT_TYPE_MARKDOWN,
    IngestRequest,
    Options,
    SearchRequest,
    new,
)
from jeffs_brain_memory.search import Index
from jeffs_brain_memory.store.fs import FsStore


async def amain() -> None:
    brain_root = Path("./data/hello-world")
    brain_root.mkdir(parents=True, exist_ok=True)

    # Async store, sync search index. Pass the index in via Options to
    # enable BM25 (and vector, when an embedder is configured).
    store = FsStore(brain_root)
    index = Index(brain_root / "index.sqlite")
    kb = new(Options(brain_id="hello-world", store=store, index=index))

    try:
        body = b"# Hedgehogs\n\nHedgehogs live in hedges and gardens.\n"
        ingest = await kb.ingest(
            IngestRequest(
                path="docs/hedgehogs.md",
                content=body,
                content_type=CONTENT_TYPE_MARKDOWN,
            )
        )
        print(f"Ingested {ingest.path} ({ingest.chunk_count} chunks)")

        resp = await kb.search(SearchRequest(query="where do hedgehogs live?", max_results=3))
        for hit in resp.hits:
            print(f"[{hit.score:.3f}] {hit.path} :: {(hit.snippet or hit.summary).strip()}")
    finally:
        await kb.close()
        await store.close()
        index.close()


if __name__ == "__main__":
    asyncio.run(amain())
```

## Module overview

- `jeffs_brain_memory.store` - async document persistence: `FsStore`, `MemStore`, `GitStore` (pygit2), `HttpStore` (wire client for `spec/PROTOCOL.md`).
- `jeffs_brain_memory.search` - sync SQLite FTS5 BM25, sqlite-vec vector search, trigram fuzzy fallback, query DSL parser.
- `jeffs_brain_memory.retrieval` - hybrid BM25 + vector retrieval with Reciprocal Rank Fusion, retry ladder, intent reweight, and reranker hooks.
- `jeffs_brain_memory.knowledge` - markdown, URL, file, and PDF ingest with frontmatter and the chunk compile pipeline; `new(Options(...))` returns a `Base`.
- `jeffs_brain_memory.memory` - extract, reflect, consolidate, recall, episodes, session buffer, contextualiser, distiller, procedural detection, wikilinks.
- `jeffs_brain_memory.acl` - access-control surface: `Provider` Protocol, in-process RBAC, `wrap_store(...)` Store wrapper.
- `jeffs_brain_memory.acl_openfga` - `httpx`-based OpenFGA HTTP adapter against the shared model in [`spec/openfga/`](../../spec/openfga).
- `jeffs_brain_memory.llm` - provider and embedder protocols plus OpenAI, Anthropic, Ollama, and Fake backends; `provider_from_env()` / `embedder_from_env()` resolve via `JB_LLM_PROVIDER` and `JB_LLM_MODEL`.
- `jeffs_brain_memory.errors` - canonical store exceptions (`ErrNotFound`, `ErrConflict`, `ErrReadOnly`, ...) shared across every backend.
- `jeffs_brain_memory.http` - the `memory serve` daemon: `Daemon`, `BrainManager`, and `create_app` matching `spec/PROTOCOL.md`.

## CLI

The `memory` CLI ships with the package. Available commands: `init`, `ingest`, `search`, `ask`, `serve`, `remember`, `recall`, `reflect`, `consolidate`, `create-brain`, `list-brains`. See the [CLI reference](https://docs.jeffsbrain.com/reference/cli/) for the full surface.

`memory serve` speaks the wire protocol in [`spec/PROTOCOL.md`](../../spec/PROTOCOL.md), so the cross-SDK eval runner and any TS or Go client can drive it identically across `ask-basic`, `ask-augmented`, and `search-retrieve-only`.

## Documentation

- Getting started: <https://docs.jeffsbrain.com/getting-started/python/>
- Examples: <https://docs.jeffsbrain.com/examples/python/>
- Guides: <https://docs.jeffsbrain.com/guides/>
- CLI reference: <https://docs.jeffsbrain.com/reference/cli/>
- Configuration reference: <https://docs.jeffsbrain.com/reference/configuration/>
- Spec: <https://docs.jeffsbrain.com/spec/>

## Cross-SDK parity

- The Python SDK does not currently ship a Postgres store adapter. Postgres-backed deployments are TypeScript-only today.
- Cross-SDK LongMemEval parity goes through the `memory serve` daemon. Python does not currently ship a native `memory eval lme` runner, so the shared scenarios (`ask-basic`, `ask-augmented`, `search-retrieve-only`) and the replay-backed tri-SDK flow in `eval/scripts/run_tri_lme.sh` are the parity surface for Python.

## Licence

Apache-2.0. See [LICENSE](./LICENSE).
