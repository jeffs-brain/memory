# SPDX-License-Identifier: Apache-2.0
"""Fusion maths and tie-break ordering."""

from __future__ import annotations

import math

from jeffs_brain_memory.retrieval import (
    RRF_DEFAULT_K,
    RRFCandidate,
    reciprocal_rank_fusion,
)


def _by_id(rs):
    return {r.chunk_id: r for r in rs}


def test_single_list_matches_formula() -> None:
    lst = [
        RRFCandidate(id="a", path="a.md"),
        RRFCandidate(id="b", path="b.md"),
        RRFCandidate(id="c", path="c.md"),
    ]
    out = reciprocal_rank_fusion([lst], RRF_DEFAULT_K)
    assert len(out) == 3
    want = [
        1.0 / (RRF_DEFAULT_K + 1),
        1.0 / (RRF_DEFAULT_K + 2),
        1.0 / (RRF_DEFAULT_K + 3),
    ]
    for i, r in enumerate(out):
        assert math.isclose(r.score, want[i], rel_tol=1e-12)


def test_two_lists_sum_contributions() -> None:
    list1 = [
        RRFCandidate(id="a", path="a.md", title="A", bm25_rank=0, have_bm25_rank=True),
        RRFCandidate(id="b", path="b.md", bm25_rank=1, have_bm25_rank=True),
    ]
    list2 = [
        RRFCandidate(
            id="c", path="c.md", vector_similarity=0.9, have_vector_sim=True
        ),
        RRFCandidate(
            id="a", path="a.md", vector_similarity=0.8, have_vector_sim=True
        ),
    ]
    out = reciprocal_rank_fusion([list1, list2], RRF_DEFAULT_K)
    assert len(out) == 3
    by_id = _by_id(out)
    want_a = 1.0 / (RRF_DEFAULT_K + 1) + 1.0 / (RRF_DEFAULT_K + 2)
    assert math.isclose(by_id["a"].score, want_a, rel_tol=1e-12)
    assert out[0].chunk_id == "a"
    assert by_id["a"].bm25_rank == 0
    assert by_id["a"].vector_similarity != 0


def test_tie_break_by_path_asc() -> None:
    list1 = [RRFCandidate(id="zebra", path="z.md")]
    list2 = [RRFCandidate(id="alpha", path="a.md")]
    out = reciprocal_rank_fusion([list1, list2], RRF_DEFAULT_K)
    assert len(out) == 2
    assert out[0].path == "a.md"


def test_metadata_fill_from_later_lists() -> None:
    list1 = [RRFCandidate(id="a", path="a.md")]
    list2 = [
        RRFCandidate(
            id="a", path="a.md", title="Hydrated", summary="Sum", content="body"
        )
    ]
    out = reciprocal_rank_fusion([list1, list2], RRF_DEFAULT_K)
    assert out[0].title == "Hydrated"
    assert out[0].summary == "Sum"
    assert out[0].text == "body"


def test_no_overwrite_of_early_metadata() -> None:
    list1 = [
        RRFCandidate(
            id="a", path="a.md", title="First", summary="FirstSum"
        )
    ]
    list2 = [
        RRFCandidate(
            id="a", path="a.md", title="Second", summary="SecondSum"
        )
    ]
    out = reciprocal_rank_fusion([list1, list2], RRF_DEFAULT_K)
    assert out[0].title == "First"
    assert out[0].summary == "FirstSum"


def test_zero_k_falls_back_to_default() -> None:
    out = reciprocal_rank_fusion([[RRFCandidate(id="a", path="a.md")]], 0)
    want = 1.0 / (RRF_DEFAULT_K + 1)
    assert math.isclose(out[0].score, want, rel_tol=1e-12)


def test_empty_inputs_return_empty() -> None:
    assert reciprocal_rank_fusion([], RRF_DEFAULT_K) == []
    assert reciprocal_rank_fusion([[]], RRF_DEFAULT_K) == []


def test_skips_candidates_with_empty_id() -> None:
    lst = [
        RRFCandidate(id="", path="a.md"),
        RRFCandidate(id="real", path="b.md"),
    ]
    out = reciprocal_rank_fusion([lst], RRF_DEFAULT_K)
    assert len(out) == 1
    assert out[0].chunk_id == "real"
