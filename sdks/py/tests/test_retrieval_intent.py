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
        ("When did I submit my research paper on sentiment analysis?", True),
        ("What specific languages did you recommend for learning back-end programming?", True),
        (
            "Can you remind me of the specific back-end programming languages you recommended I learn?",
            True,
        ),
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
    assert concrete_fact_intent_multiplier("how many birthdays", r, text) == 2.2


def test_concrete_fact_question_like_user_fact_penalty() -> None:
    r = RetrievedChunk(
        path="memory/global/user-fact-commute-question.md",
        text="What are some tips for staying awake during morning commutes?",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier("how many commutes", r, text) == pytest.approx(0.99)


def test_concrete_fact_rollup_penalty() -> None:
    r = RetrievedChunk(
        path="wiki/recap.md",
        text="summary recap overview totalling monthly figures",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier("how many figures", r, text) == 0.45


def test_concrete_fact_generic_non_global_penalty() -> None:
    r = RetrievedChunk(
        path="wiki/tips/misc.md",
        text="general guide and tips",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier("how many guides", r, text) == 0.75


def test_concrete_fact_boosts_explicit_date_for_action_date_query() -> None:
    r = RetrievedChunk(
        path="memory/project/eval-lme/research-paper.md",
        text="I submitted the research paper on February 1st and felt relieved.",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier(
        "When did I submit my research paper on sentiment analysis?",
        r,
        text,
    ) == pytest.approx(3.19)


def test_concrete_fact_penalises_measurementless_duration_notes() -> None:
    r = RetrievedChunk(
        path="memory/global/user-commute-note.md",
        text="The user has been driving their car to work every day since mid-January.",
    )
    text = retrieval_result_text(r)
    assert concrete_fact_intent_multiplier(
        "How long is my daily commute to work?",
        r,
        text,
    ) == pytest.approx(0.72)


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


def test_reweight_exact_recall_prefers_focused_back_end_language_note() -> None:
    results = [
        RetrievedChunk(
            path="memory/project/eval-lme/back-end-learning-resources.md",
            score=1.0,
            text=(
                "Recommended back-end resources include NodeSchool, Udacity, "
                "Coursera, Flask, Django, Spring, Hibernate, SQL."
            ),
        ),
        RetrievedChunk(
            path="memory/project/eval-lme/study-tips-for-becoming-full-stack.md",
            score=0.8,
            text="Learn a back-end programming language, such as Ruby, Python, or PHP.",
        ),
    ]
    out = reweight_shared_memory_ranking(
        (
            "I wanted to follow up on our previous conversation about front-end "
            "and back-end development. Can you remind me of the specific "
            "back-end programming languages you recommended I learn?"
        ),
        results,
    )
    assert out[0].path == "memory/project/eval-lme/study-tips-for-becoming-full-stack.md"


def test_reweight_exact_duration_prefers_phrase_aligned_commute_fact() -> None:
    results = [
        RetrievedChunk(
            path="memory/global/user_commute_duration.md",
            score=1.0,
            text="Typically has a 30-minute train commute; some days the commute is shorter.",
        ),
        RetrievedChunk(
            path="memory/global/user-fact-2023-05-22-listening-audiobooks-during-daily-commute.md",
            score=0.4,
            text=(
                "I've been listening to audiobooks during my daily commute to work, "
                "which takes 45 minutes each way."
            ),
        ),
    ]
    out = reweight_shared_memory_ranking(
        "How long is my daily commute to work?",
        results,
    )
    assert (
        out[0].path
        == "memory/global/user-fact-2023-05-22-listening-audiobooks-during-daily-commute.md"
    )


def test_reweight_first_person_duration_prefers_routine_user_fact_over_project_tips() -> None:
    results = [
        RetrievedChunk(
            path="memory/project/eval-lme/morning-commute-tips.md",
            score=1.0,
            text="Tips for staying awake during a 30-minute morning commute.",
        ),
        RetrievedChunk(
            path="memory/global/user-morning-commute-duration.md",
            score=0.92,
            text=(
                "User is often on a train for a 30-minute morning commute. "
                "Some days the commute is shorter, around 15-20 minutes."
            ),
        ),
        RetrievedChunk(
            path="memory/global/user-commute-time.md",
            score=0.75,
            text="I listen to audiobooks during my daily commute, which takes 45 minutes each way.",
        ),
    ]
    out = reweight_shared_memory_ranking(
        "How long is my daily morning commute to work?",
        results,
    )
    assert out[0].path == "memory/global/user-commute-time.md"


def test_reweight_composite_totals_diversify_across_focuses() -> None:
    results = [
        RetrievedChunk(
            path="memory/global/coach-handbag-800.md",
            score=1.0,
            text=(
                "User recently treated themself to a Coach handbag which cost $800 "
                "and they are really loving the quality."
            ),
        ),
        RetrievedChunk(
            path="memory/global/user-fact-2023-05-28-recently-invested-some-high-end-products.md",
            score=0.78,
            text=(
                "I've recently invested $500 in some high-end products during the "
                "Nordstrom anniversary sale."
            ),
        ),
        RetrievedChunk(
            path="memory/global/user_ebay_handbag_deal.md",
            score=0.63,
            text="The user bought a designer handbag on eBay that originally retailed for $1,500 for $200.",
        ),
        RetrievedChunk(
            path="memory/global/user_high-end-moisturizer.md",
            score=0.5,
            text=(
                "The user recently splurged on a $150 moisturizer and is asking for "
                "affordable alternatives to high-end skincare products."
            ),
        ),
    ]
    out = reweight_shared_memory_ranking(
        "What is the total amount I spent on the designer handbag and high-end skincare products?",
        results,
    )
    assert sorted([out[0].path, out[1].path]) == [
        "memory/global/coach-handbag-800.md",
        "memory/global/user-fact-2023-05-28-recently-invested-some-high-end-products.md",
    ]
