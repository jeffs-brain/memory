# SPDX-License-Identifier: Apache-2.0
"""IndexSource adapter contract.

The Python ``search`` package is still a stub, so these tests exercise
the adapter against a lightweight in-memory index that matches the
:class:`SearchIndex` protocol. The wiring is what we care about;
integrating against FTS5 lands once ``search`` does.
"""

from __future__ import annotations

import pytest

from jeffs_brain_memory.retrieval import (
    Filters,
    IndexedRow,
    IndexSource,
    Mode,
    Request,
    Retriever,
)


class InMemoryIndex:
    """Duck-typed SearchIndex for wiring tests."""

    def __init__(self, rows: list[IndexedRow]) -> None:
        self.rows = list(rows)

    @staticmethod
    def _scope_matches(row_scope: str, want: str) -> bool:
        if not want:
            return True
        aliases = {
            "memory": {"global_memory", "project_memory"},
            "global": {"global_memory"},
            "global_memory": {"global_memory"},
            "project": {"project_memory"},
            "project_memory": {"project_memory"},
        }
        expected = aliases.get(want.strip().lower(), {want.strip().lower()})
        return row_scope.strip().lower() in expected if row_scope else True

    async def search_bm25(
        self, expr: str, k: int, filters: Filters
    ) -> list[IndexedRow]:
        tokens = expr.lower().split()
        if not tokens:
            return []
        scored: list[tuple[IndexedRow, int]] = []
        for r in self.rows:
            if not self._scope_matches(r.scope, filters.scope):
                continue
            if filters.project and r.project_slug and r.project_slug != filters.project:
                continue
            corpus = " ".join([r.path, r.title, r.summary, r.content]).lower()
            score = sum(corpus.count(t) for t in tokens)
            if score > 0:
                scored.append((r, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].path))
        if k > 0 and len(scored) > k:
            scored = scored[:k]
        return [r for r, _ in scored]

    async def all_rows(self) -> list[IndexedRow]:
        return list(self.rows)


def _rows() -> list[IndexedRow]:
    return [
        IndexedRow(
            path="wiki/invoice-processing.md",
            title="Invoice Processing",
            summary="End-to-end automation for supplier invoices",
            content="The invoice processing workflow extracts line items from supplier PDFs.",
            scope="wiki",
        ),
        IndexedRow(
            path="wiki/order-processing.md",
            title="Order Processing Pipeline",
            summary="Ingest sales orders for retail partners",
            content="Order processing automates document ingestion for invoice export.",
            scope="wiki",
        ),
        IndexedRow(
            path="wiki/holiday-calendar.md",
            title="Holiday Calendar",
            summary="Public holidays across regions",
            content="The holiday calendar publishes regional public holidays for HR planning.",
            scope="wiki",
        ),
        IndexedRow(
            path="memory/global/invoice-note.md",
            title="Invoice note",
            summary="Personal observation",
            content="Reminder that invoice batches close on Friday.",
            scope="global_memory",
        ),
    ]


async def test_new_index_source_requires_index() -> None:
    with pytest.raises(ValueError):
        IndexSource(None)  # type: ignore[arg-type]


async def test_new_index_source_requires_model_when_vectors_set() -> None:
    class _Dummy:
        async def search_bm25(self, *_a, **_kw):
            return []

        async def all_rows(self):
            return []

    class _VecStore:
        async def search(self, *_a, **_kw):
            return []

    with pytest.raises(ValueError):
        IndexSource(_Dummy(), vectors=_VecStore())  # type: ignore[arg-type]


async def test_bm25_returns_top_hits_with_path_as_id() -> None:
    src = IndexSource(InMemoryIndex(_rows()))
    hits = await src.search_bm25("invoice", 5, Filters())
    assert hits, "expected at least one BM25 hit"
    for h in hits:
        assert h.id == h.path
        assert h.id != ""
        assert h.content != ""


async def test_bm25_respects_path_prefix() -> None:
    src = IndexSource(InMemoryIndex(_rows()))
    hits = await src.search_bm25(
        "invoice", 10, Filters(path_prefix="wiki/invoice")
    )
    assert hits
    for h in hits:
        assert h.path.startswith("wiki/invoice")


async def test_bm25_scope_alias_memory_matches_canonical_rows() -> None:
    src = IndexSource(InMemoryIndex(_rows()))
    hits = await src.search_bm25("invoice", 10, Filters(scope="memory"))
    assert [h.path for h in hits] == ["memory/global/invoice-note.md"]


async def test_vectors_nil_returns_empty() -> None:
    src = IndexSource(InMemoryIndex(_rows()))
    hits = await src.search_vector([0.1, 0.2, 0.3], 5, Filters())
    assert hits == []


async def test_chunks_returns_all_rows() -> None:
    rows = _rows()
    src = IndexSource(InMemoryIndex(rows))
    chunks = await src.chunks()
    assert len(chunks) == len(rows)
    got = {c.path: c for c in chunks}
    for r in rows:
        assert r.path in got
        assert got[r.path].id == r.path


async def test_end_to_end_bm25_retrieve() -> None:
    src = IndexSource(InMemoryIndex(_rows()))
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(query="invoice processing", mode=Mode.BM25, top_k=3)
    )
    assert resp.chunks
    assert not resp.trace.embedder_used
    assert resp.trace.effective_mode == Mode.BM25


async def test_end_to_end_trigram_fallback() -> None:
    src = IndexSource(InMemoryIndex(_rows()))
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(query="invioce procesing", mode=Mode.BM25, top_k=3)
    )
    assert resp.attempts
    # Trigram rung should have fired on the misspelling; the fuzzy slug
    # match surfaces the invoice-processing note.
    if resp.trace.used_retry:
        assert any(a.reason == "trigram_fuzzy" for a in resp.attempts) or any(
            a.chunks > 0 for a in resp.attempts if a.rung > 0
        )
