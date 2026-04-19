# SPDX-License-Identifier: Apache-2.0
"""Hello World example for the Jeffs Brain Python SDK.

Walks the canonical knowledge pipeline:

1. Open a filesystem-backed brain store.
2. Wire a `knowledge.Base` over it (no search index for this minimal
   path; the in-memory scorer carries the BM25-style fallback).
3. Ingest a markdown file via `kb.ingest`, which persists the document
   under `raw/documents/`.
4. Run a `kb.search` and print the top hits.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jeffs_brain_memory.knowledge import (
    CONTENT_TYPE_MARKDOWN,
    IngestRequest,
    Options,
    SearchRequest,
    new,
)
from jeffs_brain_memory.store.fs import FsStore

BRAIN_ID = "hello-world"


async def amain() -> None:
    here = Path(__file__).resolve().parent
    brain_root = here / "data" / BRAIN_ID
    brain_root.mkdir(parents=True, exist_ok=True)

    store = FsStore(brain_root)
    kb = new(Options(brain_id=BRAIN_ID, store=store))

    try:
        doc_path = here / "docs" / "hedgehogs.md"
        body = doc_path.read_bytes()
        ingest = await kb.ingest(
            IngestRequest(
                path=str(doc_path),
                content=body,
                content_type=CONTENT_TYPE_MARKDOWN,
            )
        )
        print(
            f"Ingested {ingest.path} ({ingest.chunk_count} chunks, "
            f"{ingest.bytes} bytes)"
        )

        query = "where do hedgehogs live?"
        resp = await kb.search(SearchRequest(query=query, max_results=3))
        print(f'Top {len(resp.hits)} results for "{query}":')
        for i, hit in enumerate(resp.hits, start=1):
            snippet = (hit.snippet or hit.summary).strip()
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"{i}. [{hit.score:.3f}] {hit.path}")
            print(f"   {snippet}")
    finally:
        await kb.close()
        await store.close()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
