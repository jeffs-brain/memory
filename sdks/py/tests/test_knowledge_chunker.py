# SPDX-License-Identifier: Apache-2.0
"""Chunker / segmenter tests."""

from __future__ import annotations

from jeffs_brain_memory.knowledge.compile import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    estimate_tokens,
    segment_document,
)
from jeffs_brain_memory.knowledge.types import Document
from jeffs_brain_memory.path import BrainPath, DocumentID


def _doc(body: str) -> Document:
    return Document(
        id=DocumentID("doc-id"),
        brain_id="",
        path=BrainPath("raw/documents/doc.md"),
        title="Doc",
        body=body,
    )


def test_empty_body_yields_no_chunks() -> None:
    assert segment_document(_doc("   \n   ")) == []


def test_chunker_respects_headings() -> None:
    body = (
        "# One\n\n"
        "This paragraph about the first section carries enough body content "
        "to clear the minimum chunk length threshold used by the segmenter "
        "and merge pass.\n\n"
        "## Two\n\n"
        "This paragraph about the second section is also long enough to "
        "survive the small chunk merge pass so the segmenter emits two "
        "distinct chunks for this document.\n"
    )
    chunks = segment_document(_doc(body))
    assert len(chunks) >= 2
    assert chunks[0].heading == "One"
    assert chunks[1].heading == "Two"


def test_chunker_merges_small_stubs_upward() -> None:
    body = (
        "# Big\n\n"
        "A reasonably sized paragraph of body content that comfortably "
        "exceeds the minimum chunk length threshold used by the segmenter "
        "so the merge pass keeps it intact.\n\n"
        "## Tiny\n\nshort\n"
    )
    chunks = segment_document(_doc(body))
    # Tiny section collapses into Big by the merge pass.
    assert len(chunks) == 1
    assert "short" in chunks[0].text


def test_chunker_splits_long_sections_at_sentence_boundaries() -> None:
    sentence = "This sentence describes one discrete piece of content clearly. "
    body = "# Long\n\n" + (sentence * 80)
    chunks = segment_document(_doc(body))
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= CHUNK_MAX_CHARS + 10  # small tolerance for boundary inclusion


def test_chunker_assigns_contiguous_ordinals() -> None:
    body = "# One\n\n" + ("alpha " * 300) + "\n\n## Two\n\n" + ("beta " * 300)
    chunks = segment_document(_doc(body))
    ordinals = [c.ordinal for c in chunks]
    assert ordinals == list(range(len(chunks)))


def test_chunker_constants_match_go_reference() -> None:
    # Preserves parity with go/knowledge/compile.go.
    assert CHUNK_MIN_CHARS == 120
    assert CHUNK_MAX_CHARS == 1800
    assert MIN_CHUNK_TOKENS == (CHUNK_MIN_CHARS + 3) // 4
    assert MAX_CHUNK_TOKENS == (CHUNK_MAX_CHARS + 3) // 4


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
