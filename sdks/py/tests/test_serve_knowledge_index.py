# SPDX-License-Identifier: Apache-2.0
"""The daemon's knowledge base must be able to search through the brain's
index, and its store must honour the ``Store.list`` contract, so the
knowledge package never needs a fallback for either."""

from __future__ import annotations

from pathlib import Path

import pytest

from jeffs_brain_memory.http.daemon import Daemon
from jeffs_brain_memory.knowledge import IngestRequest, SearchMode, SearchRequest
from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.store import ListOpts


@pytest.mark.asyncio
async def test_knowledge_search_uses_the_brain_index(tmp_path: Path) -> None:
    daemon = await Daemon.create(root=tmp_path, llm=FakeProvider(["ok"]))
    try:
        br = await daemon.brains.create("kb")
        await br.knowledge_base.ingest(
            IngestRequest(
                brain_id="kb",
                path="badgers.md",
                content_type="text/markdown",
                content=b"# Badgers\n\nBadgers dig setts in woodland.\n",
            )
        )

        resp = await br.knowledge_base.search(
            SearchRequest(query="badgers setts", max_results=5, mode=SearchMode.BM25)
        )

        assert resp.hits, "expected a BM25 hit through the daemon's index"
        assert all(hit.source == "bm25" for hit in resp.hits)
        assert all(0.0 < hit.score <= 1.0 for hit in resp.hits)
        assert any(str(hit.path).startswith("raw/documents/") for hit in resp.hits)
    finally:
        await daemon.close()


@pytest.mark.asyncio
async def test_passthrough_store_list_takes_list_opts(tmp_path: Path) -> None:
    daemon = await Daemon.create(root=tmp_path, llm=FakeProvider(["ok"]))
    try:
        br = await daemon.brains.create("listing")
        await br.store.write("notes/deep/a.md", b"a")
        await br.store.write("notes/_generated.md", b"g")

        flat = await br.store.list("notes")
        recursive = await br.store.list("notes", ListOpts(recursive=True, include_generated=True))

        assert sorted(e.path for e in flat if not e.is_dir) == []
        assert sorted(e.path for e in recursive if not e.is_dir) == [
            "notes/_generated.md",
            "notes/deep/a.md",
        ]
    finally:
        await daemon.close()
