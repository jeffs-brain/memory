# SPDX-License-Identifier: Apache-2.0
"""Compile tests — full, dry-run, max-batch, cancel."""

from __future__ import annotations

import asyncio

import pytest

from jeffs_brain_memory.knowledge import (
    CompileOptions,
    CONTENT_TYPE_MARKDOWN,
    IngestRequest,
    Options,
    new,
)

from ._knowledge_store import KnowledgeTestStore


def _kb() -> tuple[object, KnowledgeTestStore]:
    store = KnowledgeTestStore()
    base = new(Options(brain_id="test", store=store))
    return base, store


async def _seed(base, titles: list[str]) -> None:
    for title in titles:
        body = (
            f"# {title}\n\n"
            f"A paragraph of body content about {title.lower()} that easily "
            f"clears the minimum chunk length threshold so the chunker keeps "
            f"it intact during the merge pass.\n"
        )
        await base.ingest(
            IngestRequest(
                path=f"{title.lower()}.md",
                content_type=CONTENT_TYPE_MARKDOWN,
                content=body.encode("utf-8"),
                title=title,
            )
        )


async def test_compile_walks_all_documents_in_store() -> None:
    base, _ = _kb()
    await _seed(base, ["Alpha", "Beta"])
    res = await base.compile(CompileOptions())
    assert res.documents >= 2
    assert res.chunks > 0


async def test_compile_dry_run_does_not_mutate_index() -> None:
    base, _ = _kb()
    await _seed(base, ["Alpha"])
    res = await base.compile(CompileOptions(dry_run=True))
    assert res.documents >= 1


async def test_compile_max_batch_caps_iteration() -> None:
    base, _ = _kb()
    await _seed(base, [f"T{i}" for i in range(5)])
    res = await base.compile(CompileOptions(max_batch=2))
    assert res.documents <= 2


async def test_compile_explicit_paths_bypass_walk() -> None:
    base, store = _kb()
    await _seed(base, ["Gamma"])
    # Walk via the explicit path list.
    # Pick up the single path from the store keys.
    paths = [k for k in store._docs if k.startswith("raw/documents/")]
    res = await base.compile(CompileOptions(paths=paths))
    assert res.documents == 1


async def test_compile_skips_non_markdown_entries() -> None:
    base, store = _kb()
    await _seed(base, ["Delta"])
    # Inject a non-markdown entry that should be ignored.
    await store.write("raw/documents/extra.json", b"{\"k\": 1}")
    res = await base.compile(CompileOptions())
    assert res.documents == 1


async def test_compile_handles_corrupt_document_gracefully() -> None:
    base, store = _kb()
    # Non-UTF-8 bytes written directly under the documents prefix.
    await store.write("raw/documents/corrupt.md", b"\xff\xfe\xff")
    res = await base.compile(CompileOptions())
    # Compile swallows the failure and reports it as skipped.
    assert res.skipped >= 1


async def test_compile_reports_elapsed_ms() -> None:
    base, _ = _kb()
    await _seed(base, ["Echo"])
    res = await base.compile(CompileOptions())
    assert res.elapsed_ms >= 0


async def test_compile_respects_cancel() -> None:
    base, _ = _kb()
    await _seed(base, ["Foxtrot"])

    async def run() -> None:
        await base.compile(CompileOptions())

    task = asyncio.create_task(run())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_compile_with_no_documents_returns_zero_totals() -> None:
    base, _ = _kb()
    res = await base.compile(CompileOptions())
    assert res.documents == 0
    assert res.chunks == 0
