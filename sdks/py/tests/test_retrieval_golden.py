# SPDX-License-Identifier: Apache-2.0
"""Golden fixtures: synthesise a minimal corpus keyed to the golden
``any_of`` expectations and assert the top-5 contains at least one of
the expected paths.

The fixtures were captured against a 5K-article corpus that cannot be
redistributed. This port mirrors the Go golden tests: synthesise chunks
with title/summary/content seeded by the slug so both BM25 and semantic
cosine plausibly surface them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jeffs_brain_memory.llm.fake import FakeEmbedder
from jeffs_brain_memory.retrieval import Mode, Request, Retriever

from ._retrieval_fakes import FakeChunk, FakeSource
from jeffs_brain_memory.retrieval import slug_text_for


SPEC_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures" / "retrieval"


def _load(name: str) -> list[dict]:
    raw = (SPEC_DIR / name).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return list(data.get("queries") or [])


def _pick(queries: list[dict], ids: list[str]) -> list[dict]:
    idx = {q["id"]: q for q in queries if "id" in q}
    return [idx[i] for i in ids if i in idx]


def _corpus_for(queries: list[dict]) -> list[FakeChunk]:
    chunks: list[FakeChunk] = []
    seen: set[str] = set()
    for q in queries:
        for p in q.get("any_of") or []:
            if p in seen:
                continue
            seen.add(p)
            chunks.append(_chunk_for_path(p, q.get("q", "")))
        for p in q.get("must_retrieve") or []:
            if p in seen:
                continue
            seen.add(p)
            chunks.append(_chunk_for_path(p, q.get("q", "")))
    distractors = [
        FakeChunk(
            id="d1",
            path="wiki/holiday-calendar.md",
            title="Holiday calendar",
            content="Public holidays across regions.",
        ),
        FakeChunk(
            id="d2",
            path="wiki/office-stationery.md",
            title="Stationery budget",
            content="Pen and paper stock ledger.",
        ),
        FakeChunk(
            id="d3",
            path="wiki/hr-handbook.md",
            title="HR handbook",
            content="Policies on annual leave and expenses.",
        ),
        FakeChunk(
            id="d4",
            path="wiki/company-wifi.md",
            title="Office wifi",
            content="Joining the guest wifi network.",
        ),
    ]
    chunks.extend(distractors)
    return chunks


def _chunk_for_path(path: str, query: str) -> FakeChunk:
    slug = slug_text_for(path)
    words = slug.split()
    title = " ".join(words)
    summary = "Reference note about " + " ".join(words)
    content = summary + ". Related query context: " + query + "."
    return FakeChunk(
        id=path, path=path, title=title, summary=summary, content=content
    )


def _top_paths(chunks) -> list[str]:
    return [c.path for c in chunks]


@pytest.mark.skipif(
    not SPEC_DIR.exists(), reason="spec/fixtures/retrieval not reachable"
)
async def test_golden_hybrid_bm25() -> None:
    queries = _pick(
        _load("golden-hybrid.yaml"),
        ["invoice-automation", "quote-generation-tools"],
    )
    assert queries, "expected at least one golden query"
    corpus = _corpus_for(queries)
    src = FakeSource(corpus)
    r = Retriever(source=src)
    for q in queries:
        resp = await r.retrieve(
            Request(query=q["q"], mode=Mode.BM25, top_k=5)
        )
        hit_paths = _top_paths(resp.chunks)
        wanted = set(q.get("any_of") or [])
        assert wanted & set(hit_paths), (
            f"{q['id']}: top-5 {hit_paths} missed any_of {sorted(wanted)}"
        )


@pytest.mark.skipif(
    not SPEC_DIR.exists(), reason="spec/fixtures/retrieval not reachable"
)
async def test_golden_hybrid_mode() -> None:
    queries = _pick(
        _load("golden-hybrid.yaml"),
        ["invoice-automation", "quote-generation-tools"],
    )
    corpus = _corpus_for(queries)
    src = FakeSource(corpus)
    r = Retriever(source=src, embedder=FakeEmbedder(src.embed_dim))
    for q in queries:
        resp = await r.retrieve(
            Request(query=q["q"], mode=Mode.HYBRID, top_k=5)
        )
        hit_paths = _top_paths(resp.chunks)
        wanted = set(q.get("any_of") or [])
        assert wanted & set(hit_paths), (
            f"{q['id']} (hybrid): top-5 {hit_paths} missed any_of {sorted(wanted)}"
        )


@pytest.mark.skipif(
    not SPEC_DIR.exists(), reason="spec/fixtures/retrieval not reachable"
)
async def test_golden_realworld_bm25() -> None:
    queries = _pick(_load("golden-realworld.yaml"), ["bosch-direct"])
    if not queries:
        pytest.skip("bosch-direct query missing from fixture")
    corpus = _corpus_for(queries)
    src = FakeSource(corpus)
    r = Retriever(source=src)
    for q in queries:
        resp = await r.retrieve(
            Request(query=q["q"], mode=Mode.BM25, top_k=5)
        )
        hit_paths = _top_paths(resp.chunks)
        wanted = set(q.get("any_of") or [])
        assert wanted & set(hit_paths), (
            f"{q['id']}: top-5 {hit_paths} missed any_of {sorted(wanted)}"
        )
