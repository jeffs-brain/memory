# SPDX-License-Identifier: Apache-2.0
"""Scorer unit tests."""
from __future__ import annotations

import pytest

from scorer.exact import ExactScorer
from scorer.judge import BudgetExceededError, JudgeScorer, _parse_score


class TestExactScorer:
    def test_matches_substring_case_insensitive(self) -> None:
        scorer = ExactScorer()
        item = {"expected_substrings": ["Canberra"]}
        assert scorer.score(item=item, answer="The capital is canberra.") == 1.0

    def test_no_match_returns_zero(self) -> None:
        scorer = ExactScorer()
        item = {"expected_substrings": ["Sydney"]}
        assert scorer.score(item=item, answer="Canberra is the capital.") == 0.0

    def test_any_match_wins(self) -> None:
        scorer = ExactScorer()
        item = {"expected_substrings": ["alpha", "beta", "gamma"]}
        assert scorer.score(item=item, answer="it is clearly GAMMA") == 1.0

    def test_empty_expected_is_not_a_silent_pass(self) -> None:
        scorer = ExactScorer()
        assert scorer.score(item={"expected_substrings": []}, answer="anything") == 0.0
        assert scorer.score(item={}, answer="anything") == 0.0

    def test_case_sensitive_mode(self) -> None:
        scorer = ExactScorer(case_sensitive=True)
        item = {"expected_substrings": ["Foo"]}
        assert scorer.score(item=item, answer="contains Foo") == 1.0
        assert scorer.score(item=item, answer="contains foo") == 0.0

    def test_empty_answer(self) -> None:
        scorer = ExactScorer()
        assert scorer.score(item={"expected_substrings": ["x"]}, answer="") == 0.0

    def test_empty_needle_is_ignored(self) -> None:
        scorer = ExactScorer()
        item = {"expected_substrings": ["", "valid"]}
        assert scorer.score(item=item, answer="valid answer") == 1.0
        assert scorer.score(item={"expected_substrings": [""]}, answer="anything") == 0.0


class TestJudgeParse:
    def test_parses_clean_json(self) -> None:
        assert _parse_score('{"score": 0.83, "reason": "close"}') == 0.83

    def test_clamps_above_one(self) -> None:
        assert _parse_score('{"score": 4.2}') == 1.0

    def test_clamps_below_zero(self) -> None:
        assert _parse_score('{"score": -0.1}') == 0.0

    def test_falls_back_to_regex(self) -> None:
        # Model occasionally returns prose despite the system prompt.
        assert _parse_score("score is 0.5, looks okay") == 0.5

    def test_garbage_returns_zero(self) -> None:
        assert _parse_score("") == 0.0
        assert _parse_score("no numbers here") == 0.0


class TestJudgeBudget:
    def test_budget_exceeded_raises(self) -> None:
        scorer = JudgeScorer(budget_usd=0.00001)
        # Force a booking that blows the tiny budget.
        with pytest.raises(BudgetExceededError):
            scorer._book_spend({"input_tokens": 1_000_000, "output_tokens": 1_000_000})

    def test_budget_respected_under_threshold(self) -> None:
        scorer = JudgeScorer(budget_usd=10.0)
        scorer._book_spend({"input_tokens": 100, "output_tokens": 100})
        assert scorer.spend_usd > 0
        assert scorer.spend_usd < 10.0

    def test_no_budget_means_no_raise(self) -> None:
        scorer = JudgeScorer(budget_usd=None)
        scorer._book_spend({"input_tokens": 10_000_000, "output_tokens": 10_000_000})
        # Should simply accumulate, never throw.
        assert scorer.spend_usd > 0
