# SPDX-License-Identifier: Apache-2.0
"""Temporal BM25 fanout helpers."""

from jeffs_brain_memory.retrieval.temporal import derive_sub_queries, temporal_query_variants
from jeffs_brain_memory.retrieval.temporal import resolved_date_hints


def test_derive_sub_queries_picks_two_longest_non_stop_tokens() -> None:
    assert derive_sub_queries(
        "What happened to the Radiation Amplified zombie last Friday?"
    ) == ["radiation amplified zombie", "radiation"]


def test_derive_sub_queries_split_compound_focus_phrases() -> None:
    assert derive_sub_queries(
        "What is the total amount I spent on the designer handbag and "
        "high-end skincare products?"
    ) == ["handbag cost", "high-end products"]


def test_derive_sub_queries_prioritise_action_date_probe_for_when_did_i_submit() -> None:
    assert derive_sub_queries(
        "When did I submit my research paper on sentiment analysis?"
    ) == ["sentiment analysis submission date", "research paper submission date"]


def test_derive_sub_queries_add_inspiration_source_probe() -> None:
    assert derive_sub_queries(
        "How can I find new inspiration for my paintings?"
    ) == ["paintings social media tutorials"]


def test_derive_sub_queries_add_specific_back_end_language_probe() -> None:
    assert derive_sub_queries(
        "I wanted to follow up on our previous conversation about front-end "
        "and back-end development. Can you remind me of the specific "
        "back-end programming languages you recommended I learn?"
    ) == ["back-end programming language", "back-end development"]


def test_temporal_query_variants_include_raw_augmented_and_token_probes() -> None:
    variants = temporal_query_variants(
        "What happened last Friday to the Radiation Amplified zombie?",
        "2024/03/13 (Wed) 10:00",
    )

    assert variants[0] == "radiation amplified zombie"
    assert "What happened last Friday to the Radiation Amplified zombie?" in variants
    assert (
        "What happened last Friday to the Radiation Amplified zombie? "
        '"2024/03/08" "2024-03-08"'
    ) in variants
    assert variants[-1] == "radiation"


def test_resolved_date_hints_expand_last_week_day_by_day() -> None:
    assert resolved_date_hints(
        "Where did I volunteer last week?",
        "2024/03/13 (Wed) 10:00",
    ) == [
        "2024/03/06",
        "2024/03/07",
        "2024/03/08",
        "2024/03/09",
        "2024/03/10",
        "2024/03/11",
        "2024/03/12",
    ]
