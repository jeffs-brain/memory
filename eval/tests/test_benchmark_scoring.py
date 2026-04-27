# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from benchmarks.base import EvalQuestion
from benchmarks.scoring import (
    AdversarialAbstentionScorer,
    ExactContainmentScorer,
    JudgeBridgeScorer,
    TokenF1Scorer,
)


def test_token_f1_normalises_articles_case_and_punctuation() -> None:
    question = EvalQuestion(
        id="q1",
        question="Where did Jeff move?",
        gold_answers=["The Netherlands."],
        category="single-hop",
    )

    result = TokenF1Scorer().score(question=question, answer="netherlands", citations=[])

    assert result.score == 1.0
    assert result.passed is True


def test_token_f1_uses_best_gold_answer() -> None:
    question = EvalQuestion(
        id="q1",
        question="Who attended?",
        gold_answers=["Alice and Bob", "Charlie"],
        category="single-hop",
    )

    result = TokenF1Scorer().score(question=question, answer="Charlie", citations=[])

    assert result.score == 1.0
    assert result.passed is True


def test_token_f1_counts_duplicate_tokens() -> None:
    question = EvalQuestion(
        id="q1",
        question="What was repeated?",
        gold_answers=["red red blue"],
        category="single-hop",
    )

    result = TokenF1Scorer().score(question=question, answer="red blue", citations=[])

    assert round(result.score, 6) == 0.8
    assert result.passed is True


def test_adversarial_abstention_detects_known_phrase() -> None:
    question = EvalQuestion(
        id="q1",
        question="What is unavailable?",
        gold_answers=["Not answerable"],
        category="adversarial",
    )

    result = AdversarialAbstentionScorer().score(
        question=question,
        answer="There is no information about that in the conversation.",
        citations=[],
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.detail["matched_signal"] == "no information"


def test_adversarial_abstention_rejects_substantive_answer() -> None:
    question = EvalQuestion(
        id="q1",
        question="What is unavailable?",
        gold_answers=["Not answerable"],
        category="adversarial",
    )

    result = AdversarialAbstentionScorer().score(
        question=question,
        answer="The answer is Tuesday.",
        citations=[],
    )

    assert result.score == 0.0
    assert result.passed is False


def test_exact_containment_matches_any_gold_answer() -> None:
    question = EvalQuestion(
        id="q1",
        question="Where did Jeff move?",
        gold_answers=["Amersfoort", "the Netherlands"],
        category="single-hop",
    )

    result = ExactContainmentScorer().score(
        question=question,
        answer="Jeff moved to the Netherlands in 2021.",
        citations=[],
    )

    assert result.score == 1.0
    assert result.passed is True


class FakeJudge:
    model = "fake-judge"

    def __init__(self) -> None:
        self.item: dict[str, object] | None = None
        self.answer: str | None = None

    def score(self, *, item: dict[str, object], answer: str) -> float:
        self.item = item
        self.answer = answer
        return 0.75


def test_judge_bridge_maps_question_to_existing_judge_shape() -> None:
    fake_judge = FakeJudge()
    scorer = JudgeBridgeScorer(judge=fake_judge)
    question = EvalQuestion(
        id="q1",
        question="What happened?",
        gold_answers=["The first gold answer", "The second gold answer"],
        category="single-hop",
    )

    result = scorer.score(question=question, answer="Candidate", citations=[])

    assert result.score == 0.75
    assert result.passed is True
    assert fake_judge.item == {
        "question": "What happened?",
        "reference_answer": "The first gold answer",
    }
    assert fake_judge.answer == "Candidate"
    assert result.detail["judge_model"] == "fake-judge"
