# SPDX-License-Identifier: Apache-2.0
"""Golden fixtures: synthesise a minimal corpus keyed to the public golden
set and assert the top-5 satisfies the fixture pass criterion.

``spec/fixtures/retrieval/golden-public.yaml`` is fully synthetic. This
port mirrors the Go golden tests: synthesise chunks with
title/summary/content seeded by the slug so both BM25 and semantic cosine
plausibly surface them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jeffs_brain_memory.llm.fake import FakeEmbedder
from jeffs_brain_memory.retrieval import Mode, Request, Retriever, slug_text_for

from ._retrieval_fakes import FakeChunk, FakeSource

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures" / "retrieval"


def _load(name: str) -> list[dict]:
    raw = (SPEC_DIR / name).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return list(data.get("queries") or [])


GOLDEN_SET = "golden-public.yaml"


def _passes(hit_paths: list[str], query: dict) -> bool:
    got = set(hit_paths)
    if got & set(query.get("any_of") or []):
        return True
    must = query.get("must_retrieve") or []
    return bool(must) and all(p in got for p in must)


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
@pytest.mark.parametrize("mode", [Mode.BM25, Mode.HYBRID])
async def test_golden_public(mode: Mode) -> None:
    queries = _load(GOLDEN_SET)
    assert queries, "expected at least one golden query"
    corpus = _corpus_for(queries)
    src = FakeSource(corpus)
    embedder = FakeEmbedder(src.embed_dim) if mode is Mode.HYBRID else None
    r = Retriever(source=src, embedder=embedder)
    for q in queries:
        resp = await r.retrieve(Request(query=q["q"], mode=mode, top_k=5))
        hit_paths = _top_paths(resp.chunks)
        assert _passes(hit_paths, q), (
            f"{q['id']} ({mode.value}): top-5 {hit_paths} did not satisfy "
            f"any_of {q.get('any_of')} / must_retrieve {q.get('must_retrieve')}"
        )
