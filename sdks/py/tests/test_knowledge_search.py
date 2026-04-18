# SPDX-License-Identifier: Apache-2.0
"""Search tests — in-memory fallback, index delegation, hybrid retriever, fallback on error."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jeffs_brain_memory.knowledge import (
    CONTENT_TYPE_MARKDOWN,
    IngestRequest,
    Options,
    SearchHit,
    SearchMode,
    SearchRequest,
    new,
)
from jeffs_brain_memory.knowledge.search import (
    score_document,
    snippet_for,
    tokenise_query,
)
from jeffs_brain_memory.path import BrainPath

from ._knowledge_store import KnowledgeTestStore


def _kb(*, index=None, retriever=None) -> tuple[object, KnowledgeTestStore]:
    store = KnowledgeTestStore()
    base = new(Options(brain_id="test", store=store, index=index, retriever=retriever))
    return base, store


async def _seed(base, docs: list[tuple[str, str]]) -> None:
    for title, body in docs:
        md = f"# {title}\n\n{body}\n"
        await base.ingest(
            IngestRequest(
                path=f"{title.lower()}.md",
                content_type=CONTENT_TYPE_MARKDOWN,
                content=md.encode("utf-8"),
                title=title,
            )
        )


async def test_in_memory_fallback_scorer_returns_top_hit() -> None:
    base, _ = _kb()
    await _seed(
        base,
        [
            ("Alpha", "talks about memory retrieval with enough content to populate a chunk."),
            ("Beta", "covers unrelated topics that have nothing to do with the query."),
        ],
    )
    resp = await base.search(SearchRequest(query="memory", max_results=5))
    assert len(resp.hits) >= 1
    assert resp.hits[0].title == "Alpha"
    assert resp.hits[0].source == "memory"


async def test_empty_query_yields_zero_hits() -> None:
    base, _ = _kb()
    resp = await base.search(SearchRequest(query="   "))
    assert resp.hits == []


async def test_explicit_bm25_mode_uses_in_memory_when_no_index() -> None:
    base, _ = _kb()
    await _seed(
        base,
        [("Alpha", "alpha body about memory retrieval that populates a chunk easily")],
    )
    resp = await base.search(SearchRequest(query="memory", mode=SearchMode.BM25))
    assert resp.mode == "bm25"
    assert any(hit.title == "Alpha" for hit in resp.hits)


@dataclass
class FakeIndex:
    """Minimal ``IndexLike`` that echoes a canned hit list."""

    hits: list[SearchHit] = field(default_factory=list)
    last_query: str = ""

    async def search_bm25(self, query: str, *, limit: int) -> list[SearchHit]:
        self.last_query = query
        return list(self.hits[:limit])


async def test_bound_index_supersedes_in_memory_fallback() -> None:
    canned = [
        SearchHit(path=BrainPath("wiki/a/alpha.md"), title="Alpha", score=1.0, source="bm25"),
    ]
    index = FakeIndex(hits=canned)
    base, _ = _kb(index=index)
    resp = await base.search(SearchRequest(query="memory", mode=SearchMode.BM25))
    assert resp.mode == "bm25"
    assert resp.hits[0].title == "Alpha"
    assert index.last_query == "memory"


@dataclass
class FakeRetriever:
    """Minimal hybrid retriever stub."""

    hits: list[SearchHit] = field(default_factory=list)
    error: Exception | None = None
    last_query: str = ""
    called: int = 0

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        brain_id: str = "",
        mode: SearchMode = SearchMode.AUTO,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        self.called += 1
        self.last_query = query
        if self.error is not None:
            raise self.error
        return list(self.hits[:top_k]), {"effective_mode": "hybrid"}


async def test_hybrid_retriever_supersedes_index() -> None:
    retriever = FakeRetriever(
        hits=[
            SearchHit(path=BrainPath("wiki/a/alpha.md"), title="Alpha", score=0.9, source="fused"),
            SearchHit(path=BrainPath("wiki/b/beta.md"), title="Beta", score=0.3, source="fused"),
        ]
    )
    base, _ = _kb(retriever=retriever)
    resp = await base.search(SearchRequest(query="alpha"))
    assert retriever.called == 1
    assert resp.mode == "hybrid"
    assert len(resp.hits) == 2
    assert resp.hits[0].title == "Alpha"
    assert resp.hits[0].source == "fused"


async def test_hybrid_retriever_falls_back_to_bm25_on_error() -> None:
    retriever = FakeRetriever(error=RuntimeError("retriever down"))
    base, _ = _kb(retriever=retriever)
    await _seed(
        base,
        [("Alpha", "alpha body text long enough for the chunker to index it")],
    )
    resp = await base.search(SearchRequest(query="alpha", mode=SearchMode.HYBRID))
    assert resp.fell_back is True
    assert resp.mode == "bm25"


async def test_hybrid_retriever_empty_result_triggers_fallback() -> None:
    retriever = FakeRetriever(hits=[])
    base, _ = _kb(retriever=retriever)
    await _seed(
        base,
        [("Alpha", "some alpha content long enough to survive the chunker pass")],
    )
    resp = await base.search(SearchRequest(query="alpha", mode=SearchMode.HYBRID))
    assert resp.fell_back is True
    assert resp.mode == "bm25"


async def test_auto_mode_chooses_hybrid_when_retriever_bound() -> None:
    retriever = FakeRetriever(
        hits=[SearchHit(path=BrainPath("wiki/a/a.md"), title="A", score=0.5, source="fused")]
    )
    base, _ = _kb(retriever=retriever)
    resp = await base.search(SearchRequest(query="a"))
    assert resp.mode == "hybrid"


async def test_auto_mode_uses_bm25_when_no_retriever() -> None:
    base, _ = _kb()
    await _seed(base, [("Alpha", "some body about memory that survives the chunker pass")])
    resp = await base.search(SearchRequest(query="memory"))
    assert resp.mode == "bm25"


def test_tokenise_query_strips_punctuation() -> None:
    assert tokenise_query(" Hello, World?  ") == ["hello", "world"]


def test_score_document_weighs_title_highest() -> None:
    from jeffs_brain_memory.knowledge.frontmatter import Frontmatter

    fm = Frontmatter(title="memory", summary="memory", tags=["memory"])
    score = score_document(["memory"], fm, "memory body")
    assert score == 3 + 2 + 2 + 1


def test_snippet_for_centres_on_first_hit() -> None:
    body = "alpha beta gamma delta epsilon zeta eta theta"
    snippet = snippet_for(body, ["gamma"])
    assert "gamma" in snippet
