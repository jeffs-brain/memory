# SPDX-License-Identifier: Apache-2.0
"""Retry ladder: each rung fires from a seeded fixture."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.retrieval import (
    BM25Hit,
    Mode,
    Request,
    Retriever,
    TRIGRAM_JACCARD_THRESHOLD,
    build_trigram_index,
    compute_trigrams,
    jaccard,
    query_tokens,
    sanitise_query,
    slug_text_for,
    strongest_term,
)

from ._retrieval_fakes import FakeChunk, FakeSource


def _retry_corpus() -> list[FakeChunk]:
    return [
        FakeChunk(
            id="r1",
            path="wiki/kubernetes-cluster-setup.md",
            title="Kubernetes Cluster Setup",
            summary="Bootstrapping a kind cluster",
            content="Provision a three-node kind cluster and apply the base manifests.",
        ),
        FakeChunk(
            id="r2",
            path="wiki/archipelago-tooling.md",
            title="Archipelago tooling",
            summary="Custom build tooling for the archipelago stack",
            content="The archipelago build set manages intra-service contracts.",
        ),
        FakeChunk(
            id="r3",
            path="wiki/miscellaneous-notes.md",
            title="Miscellaneous Notes",
            summary="Catch-all",
            content="Kubernetes operations runbooks are kept here for ad-hoc lookup.",
        ),
    ]


def test_sanitise_query_strips_punctuation() -> None:
    assert sanitise_query("!!!hello, world??? ") == "hello world"


def test_sanitise_query_empty_input_returns_empty() -> None:
    assert sanitise_query("") == ""


def test_strongest_term_picks_longest_non_stopword() -> None:
    assert strongest_term("the alphabet soup") == "alphabet"


def test_strongest_term_rejects_short_and_stopwords() -> None:
    assert strongest_term("the and or") == ""


def test_query_tokens_dedupes_and_lowercases() -> None:
    got = query_tokens("Kubernetes KUBERNETES cluster CLUSTER the")
    assert got == ["kubernetes", "cluster"]


def test_slug_text_for_keeps_stem_only() -> None:
    assert slug_text_for("wiki/parent/my-doc.md") == "my doc"


def test_compute_trigrams_pads_with_dollar() -> None:
    grams = compute_trigrams("ab")
    # "ab" -> padded "$ab$" -> 2 3-grams.
    assert grams == {"$ab", "ab$"}


def test_jaccard_similarity() -> None:
    a = {"$ab", "abc", "bc$"}
    b = {"$ab", "abd", "bd$"}
    # |{$ab}| = 1 intersection, |union| = 5.
    assert jaccard(a, b) == pytest.approx(1.0 / 5.0)


def test_trigram_index_finds_typo_match() -> None:
    from jeffs_brain_memory.retrieval import TrigramChunk

    idx = build_trigram_index(
        [TrigramChunk(id="k", path="wiki/kubernetes.md")]
    )
    hits = idx.search(["kubernets"], 5)
    assert len(hits) == 1
    assert hits[0].id == "k"
    assert hits[0].similarity >= TRIGRAM_JACCARD_THRESHOLD


async def test_rung0_initial_hit_no_retry() -> None:
    src = FakeSource(_retry_corpus())
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="kubernetes", mode=Mode.BM25))
    assert not resp.trace.used_retry
    assert len(resp.attempts) == 1
    assert resp.attempts[0].reason == "initial"


async def test_initial_fanout_includes_strongest_term_probe() -> None:
    src = FakeSource(_retry_corpus())

    def override(expr: str) -> tuple[list[BM25Hit], bool]:
        if "xyz" in expr:
            return [], True
        return [], False

    src.bm25_override = override
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="xyz kubernetes", mode=Mode.BM25))
    assert not resp.trace.used_retry
    assert len(resp.attempts) == 1
    assert resp.attempts[0].reason == "initial"
    assert "kubernetes" in resp.attempts[0].query


async def test_rung3_refreshed_sanitised_fires() -> None:
    src = FakeSource(
        [FakeChunk(id="a", path="wiki/foo.md", title="foo", content="alphabet")]
    )
    calls = {"n": 0}

    def override(expr: str) -> tuple[list[BM25Hit], bool]:
        calls["n"] += 1
        if calls["n"] <= 2:
            return [], True
        return [], False

    src.bm25_override = override
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="!!! alphabet ???", mode=Mode.BM25))
    assert resp.trace.used_retry
    sanitised = [a for a in resp.attempts if a.reason == "refreshed_sanitised"]
    assert sanitised and sanitised[0].rung == 3


async def test_rung4_refreshed_strongest_fires() -> None:
    src = FakeSource(
        [
            FakeChunk(
                id="a",
                path="wiki/kubernetes.md",
                title="Kubernetes",
                content="runbook",
            )
        ]
    )
    calls = {"n": 0}

    def override(expr: str) -> tuple[list[BM25Hit], bool]:
        calls["n"] += 1
        if calls["n"] <= 3:
            return [], True
        return [], False

    src.bm25_override = override
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="!? kubernetes ?!", mode=Mode.BM25))
    assert resp.trace.used_retry
    r4 = [a for a in resp.attempts if a.reason == "refreshed_strongest"]
    assert r4 and r4[0].rung == 4


async def test_rung5_trigram_fuzzy_fires() -> None:
    src = FakeSource(
        [
            FakeChunk(id="kube", path="wiki/kubernetes.md", title="Kubernetes"),
            FakeChunk(
                id="arch", path="wiki/archipelago.md", title="Archipelago"
            ),
        ]
    )
    src.bm25_override = lambda expr: ([], True)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(query="kubernets", mode=Mode.BM25, top_k=5)
    )
    assert resp.trace.used_retry
    trigram = [a for a in resp.attempts if a.reason == "trigram_fuzzy"]
    assert trigram and trigram[0].rung == 5
    assert trigram[0].chunks > 0
    assert resp.chunks, "trigram hits should bubble up as final results"


async def test_skip_retry_ladder_bypasses_rungs() -> None:
    src = FakeSource(_retry_corpus())
    src.bm25_override = lambda expr: ([], True)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(query="kubernetes", mode=Mode.BM25, skip_retry_ladder=True)
    )
    assert not resp.trace.used_retry
    assert len(resp.attempts) == 1
