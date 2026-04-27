# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from benchmarks.base import EvalQuestion
from benchmarks.scoring import (
    AdversarialAbstentionScorer,
    BPIContainmentScorer,
    ExactContainmentScorer,
    JudgeBridgeScorer,
    TokenF1Scorer,
    bpi_action_score,
    bpi_macro_average,
    bpi_rule_score,
    extract_bpi_answer,
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


def _bpi_question(
    *,
    expected_rules: list[str],
    expected_action: str = "Auto-approve the expense.",
    action_keywords: list[str] | None = None,
    valid_rules: list[str] | None = None,
) -> EvalQuestion:
    return EvalQuestion(
        id="bpi-1",
        question="What should happen?",
        gold_answers=[expected_action],
        category="single-rule-recall",
        metadata={
            "expected_rules": expected_rules,
            "expected_action": expected_action,
            "action_keywords": action_keywords or ["auto-approve", "approved"],
            "valid_rules": valid_rules or ["R1", "R2", "R3"],
        },
    )


def test_bpi_rule_score_exact_match() -> None:
    assert bpi_rule_score(predicted_rules={"R1", "R3"}, expected_rules={"R1", "R3"}) == 1.0


def test_bpi_rule_score_partial_match_uses_f1() -> None:
    score = bpi_rule_score(predicted_rules={"R1", "R2"}, expected_rules={"R1", "R3"})

    assert score == 0.5


def test_bpi_rule_score_handles_correct_abstention() -> None:
    assert bpi_rule_score(predicted_rules=set(), expected_rules=set()) == 1.0


def test_bpi_rule_score_rejects_spurious_abstention_rules() -> None:
    assert bpi_rule_score(predicted_rules={"R1"}, expected_rules=set()) == 0.0


def test_bpi_action_score_normalises_keyword_punctuation_case_and_spacing() -> None:
    score = bpi_action_score(
        predicted_action="The expense is AUTO APPROVED after review.",
        expected_action="",
        keywords=["auto-approve", "review"],
    )

    assert score == 1.0


def test_bpi_action_score_requires_all_keywords() -> None:
    score = bpi_action_score(
        predicted_action="Refer to HR for guidance.",
        expected_action="",
        keywords=["refer", "HR", "not covered"],
    )

    assert score == 0.0


def test_extract_bpi_answer_prefers_structured_json() -> None:
    rules, action = extract_bpi_answer(
        """
        {
          "applicable_rules": ["R1", "R3"],
          "action": "Escalate to manager."
        }
        """
    )

    assert rules == {"R1", "R3"}
    assert action == "Escalate to manager."


def test_bpi_containment_scorer_combines_rule_and_action_scores() -> None:
    result = BPIContainmentScorer().score(
        question=_bpi_question(expected_rules=["R1"]),
        answer='{"applicable_rules": ["R1"], "action": "Auto approved."}',
        citations=[],
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.detail["rule_score"] == 1.0
    assert result.detail["action_score"] == 1.0
    assert result.detail["predicted_rules"] == ["R1"]


def test_bpi_containment_scorer_flags_hallucinated_rule_ids() -> None:
    result = BPIContainmentScorer().score(
        question=_bpi_question(
            expected_rules=["R1"],
            action_keywords=["director"],
            valid_rules=["R1", "R2"],
        ),
        answer='{"applicable_rules": ["R1", "R99"], "action": "Ask a director."}',
        citations=[],
    )

    assert round(result.score, 6) == 0.8
    assert round(result.detail["rule_score"], 6) == round(2 / 3, 6)
    assert result.detail["spurious_rules"] == ["R99"]
    assert result.detail["phantom_rules"] == ["R99"]


def test_bpi_macro_average_uses_category_means() -> None:
    assert bpi_macro_average({"single-rule-recall": 1.0, "abstention": 0.5}) == 0.75
