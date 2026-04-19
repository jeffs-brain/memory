# SPDX-License-Identifier: Apache-2.0
"""English intent regex coverage and multiplier behaviour."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.retrieval import (
    RetrievedChunk,
    concrete_fact_intent_multiplier,
    detect_retrieval_intent,
    preference_intent_multiplier,
    retrieval_result_text,
    reweight_shared_memory_ranking,
)


@pytest.mark.parametrize(
    "query, want",
    [
        ("recommend a coffee shop", True),
        ("Can you suggest a book?", True),
        ("what should i read next", True),
        ("which should I buy, the blue or the red?", True),
        ("tip for next release", True),
        ("any ideas for dinner tonight", True),
        ("advice on cabling", True),
        ("how many invoices did we send", False),
        ("hola amigo", False),
    ],
)
def test_detect_preference(query: str, want: bool) -> None:
    assert detect_retrieval_intent(query).preference_query is want


@pytest.mark.parametrize(
    "query, want",
    [
        ("how many invoices did we process", True),
        ("count the line items", True),
        ("in total how much was spent", True),
        ("list the clients", True),
        ("what are all the projects", True),
        ("did I pick up milk", True),
        ("have I finished the report", True),
        ("was I booked for dinner", True),
        ("were I the one who ordered", True),
        ("did i travelled to bosch yesterday", True),
        ("How long is my daily commute to work?", True),
        ("What specific languages did you recommend for learning back-end programming?", True),
        ("recommend a flat white", False),
        ("non english text abc xyz", False),
    ],
)
def test_detect_concrete_fact(query: str, want: bool) -> None:
    assert detect_retrieval_intent(query).concrete_fact_query is want


def test_both_intents_compose_in_label() -> None:
    intent = detect_retrieval_intent("recommend how many to buy")
    assert intent.preference_query
    assert intent.concrete_fact_query
    assert intent.label() == "preference+concrete-fact"


def test_preference_multiplier_user_preference_path() -> None:
    r = RetrievedChunk(
        path="memory/global/user-preference-coffee.md",
        title="Coffee",
        text="I prefer oat milk",
    )
    text = retrieval_result_text(r)
    assert preference_intent_multiplier(r, text) == 2.35


def test_preference_multiplier_global_preference_note() -> None:
    r = RetrievedChunk(
        path="memory/global/notes.md",
        title="Notes",
        text="I really love flat whites",
    )
    text = retrieval_result_text(r)
    assert preference_intent_multiplier(r, text) == 2.1


def test_preference_multiplier_generic_non_global() -> None:
    r = RetrievedChunk(
        path="wiki/guides/tips.md",
        text="here are some tips for improving throughput",
    )
    text = retrieval_result_text(r)
    assert preference_intent_multiplier(r, text) == 0.82


def test_preference_multiplier_rollup() -> None:
    r = RetrievedChunk(
        path="memory/global/roll-up.md",
        text="overview summary totalling everything",
    )
    text = retrieval_result_text(r)
    assert preference_intent_multiplier(r, text) == 0.9


def test_concrete_fact_user_fact_path() -> None:
    r = RetrievedChunk(
        path="memory/global/user-fact-birthday.md",
        text="no date tag",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier(r, text) == 2.2


def test_concrete_fact_question_like_user_fact_penalty() -> None:
    r = RetrievedChunk(
        path="memory/global/user-fact-commute-question.md",
        text="What are some tips for staying awake during morning commutes?",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier(r, text) == pytest.approx(0.99)


def test_concrete_fact_rollup_penalty() -> None:
    r = RetrievedChunk(
        path="wiki/recap.md",
        text="summary recap overview totalling monthly figures",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier(r, text) == 0.45


def test_concrete_fact_generic_non_global_penalty() -> None:
    r = RetrievedChunk(
        path="wiki/tips/misc.md",
        text="general guide and tips",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier(r, text) == 0.75


def test_reweight_tie_break_by_original_rank() -> None:
    results = [
        RetrievedChunk(path="a.md", score=1.0, title="A"),
        RetrievedChunk(path="b.md", score=1.0, title="B"),
    ]
    out = reweight_shared_memory_ranking("recommend a thing", results)
    assert out[0].path == "a.md"
    assert out[1].path == "b.md"


def test_reweight_no_intent_returns_inputs() -> None:
    in_ = [RetrievedChunk(path="a.md", score=0.1)]
    out = reweight_shared_memory_ranking("regular fact lookup", in_)
    assert len(out) == 1
    assert out[0].score == 0.1


def test_reweight_boosts_global_preference_on_preference_query() -> None:
    results = [
        RetrievedChunk(path="wiki/options.md", score=1.0, text="options guide"),
        RetrievedChunk(
            path="memory/global/user-preference-coffee.md",
            score=0.5,
            text="I prefer flat whites",
        ),
    ]
    out = reweight_shared_memory_ranking("recommend a coffee shop", results)
    # Boosted 0.5 * 2.35 = 1.175 > 1.0 * 0.82 = 0.82.
    assert out[0].path == "memory/global/user-preference-coffee.md"
