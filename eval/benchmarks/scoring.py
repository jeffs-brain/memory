# SPDX-License-Identifier: Apache-2.0
"""Benchmark scorers."""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from benchmarks.base import EvalQuestion, ScorerResult
from scorer.judge import JudgeScorer

_ARTICLES = {"a", "an", "the"}
_PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)
_BPI_PUNCTUATION_TRANSLATION = str.maketrans(
    {char: " " for char in string.punctuation}
)
_ABSTENTION_SIGNALS = (
    "i don't know",
    "i do not know",
    "no information",
    "cannot answer",
    "can't answer",
    "not mentioned",
    "unknown",
    "not enough information",
)
_BPI_RULE_ID_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
_BPI_NONE_VALUES = {"", "none", "no rule", "no rules", "n/a", "not applicable"}


def normalise_tokens(value: str) -> list[str]:
    cleaned = value.lower().translate(_PUNCTUATION_TRANSLATION)
    return [token for token in cleaned.split() if token and token not in _ARTICLES]


def token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = normalise_tokens(predicted)
    gold_tokens = normalise_tokens(gold)
    if not predicted_tokens or not gold_tokens:
        return 1.0 if predicted_tokens == gold_tokens else 0.0

    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bpi_rule_score(
    *,
    predicted_rules: set[str],
    expected_rules: set[str],
) -> float:
    if not expected_rules:
        return 1.0 if not predicted_rules else 0.0

    overlap = len(predicted_rules & expected_rules)
    precision = overlap / len(predicted_rules) if predicted_rules else 1.0
    recall = overlap / len(expected_rules)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalise_bpi_text(value: str) -> str:
    cleaned = value.lower().translate(_BPI_PUNCTUATION_TRANSLATION)
    return re.sub(r"\s+", " ", cleaned).strip()


def bpi_action_score(*, predicted_action: str, expected_action: str, keywords: list[str]) -> float:
    haystack = normalise_bpi_text(predicted_action)
    if keywords:
        valid_keywords = [keyword for keyword in keywords if keyword]
        if not valid_keywords:
            return 0.0
        matched = all(_bpi_phrase_matches(haystack, normalise_bpi_text(keyword)) for keyword in valid_keywords)
        return 1.0 if matched else 0.0

    expected = normalise_bpi_text(expected_action)
    if not expected:
        return 0.0
    return 1.0 if _bpi_phrase_matches(haystack, expected) else 0.0


def bpi_macro_average(category_scores: dict[str, float]) -> float:
    if not category_scores:
        return 0.0
    return sum(category_scores.values()) / len(category_scores)


def extract_bpi_answer(answer: str) -> tuple[set[str], str]:
    data = _json_object(answer)
    if data is not None:
        rules = _rule_set_from_value(
            data.get("applicable_rules")
            or data.get("predicted_rules")
            or data.get("rule_ids")
            or data.get("rules")
        )
        action = (
            data.get("action")
            or data.get("recommended_action")
            or data.get("predicted_action")
            or ""
        )
        return rules, action if isinstance(action, str) else str(action)

    rule_line = next(
        (
            line
            for line in answer.splitlines()
            if "rule" in line.lower() and ("appl" in line.lower() or ":" in line)
        ),
        answer,
    )
    if normalise_bpi_text(rule_line) in _BPI_NONE_VALUES:
        return set(), answer
    return {
        candidate
        for candidate in _BPI_RULE_ID_RE.findall(rule_line)
        if any(char.isdigit() for char in candidate)
    }, answer


def _json_object(value: str) -> dict[str, Any] | None:
    import json

    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _rule_set_from_value(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        normalised = normalise_bpi_text(value)
        if normalised in _BPI_NONE_VALUES or "no rule" in normalised:
            return set()
        return {
            candidate
            for candidate in _BPI_RULE_ID_RE.findall(value)
            if any(char.isdigit() for char in candidate)
        }
    if isinstance(value, list):
        rules: set[str] = set()
        for item in value:
            rules.update(_rule_set_from_value(item))
        return rules
    return set()


@dataclass
class ExactContainmentScorer:
    name: str = "exact-containment"
    case_sensitive: bool = False

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> ScorerResult:
        del citations
        haystack = answer if self.case_sensitive else answer.lower()
        for gold in question.gold_answers:
            candidate = gold if self.case_sensitive else gold.lower()
            if candidate and candidate in haystack:
                return ScorerResult(score=1.0, passed=True)
        return ScorerResult(score=0.0, passed=False)


@dataclass
class TokenF1Scorer:
    threshold: float = 0.5
    name: str = "token-f1"

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> ScorerResult:
        del citations
        score = max((token_f1(answer, gold) for gold in question.gold_answers), default=0.0)
        return ScorerResult(
            score=score,
            passed=score >= self.threshold,
            detail={"threshold": self.threshold},
        )


@dataclass
class AdversarialAbstentionScorer:
    name: str = "adversarial"
    signals: tuple[str, ...] = _ABSTENTION_SIGNALS

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> ScorerResult:
        del question, citations
        normalised = re.sub(r"\s+", " ", answer.lower()).strip()
        matched = next((signal for signal in self.signals if signal in normalised), None)
        score = 1.0 if matched else 0.0
        return ScorerResult(
            score=score,
            passed=score == 1.0,
            detail={"matched_signal": matched},
        )


@dataclass
class JudgeBridgeScorer:
    judge: JudgeScorer = field(default_factory=JudgeScorer)
    name: str = "judge"
    pass_threshold: float = 0.5

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> ScorerResult:
        del citations
        item = {
            "question": question.question,
            "reference_answer": question.gold_answers[0] if question.gold_answers else "",
        }
        score = self.judge.score(item=item, answer=answer)
        return ScorerResult(
            score=score,
            passed=score >= self.pass_threshold,
            detail={"threshold": self.pass_threshold, "judge_model": self.judge.model},
        )


@dataclass
class BPIContainmentScorer:
    name: str = "bpi-containment"
    rule_weight: float = 0.6
    action_weight: float = 0.4
    pass_threshold: float = 0.8

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> ScorerResult:
        del citations
        expected_rules = set(_strings(question.metadata.get("expected_rules")))
        valid_rules = set(_strings(question.metadata.get("valid_rules")))
        keywords = _strings(question.metadata.get("action_keywords"))
        expected_action = str(
            question.metadata.get("expected_action")
            or (question.gold_answers[0] if question.gold_answers else "")
        )
        predicted_rules, predicted_action = extract_bpi_answer(answer)
        rule_score = bpi_rule_score(
            predicted_rules=predicted_rules,
            expected_rules=expected_rules,
        )
        action_score = bpi_action_score(
            predicted_action=predicted_action,
            expected_action=expected_action,
            keywords=keywords,
        )
        score = self.rule_weight * rule_score + self.action_weight * action_score
        phantom_rules = sorted(predicted_rules - valid_rules) if valid_rules else []
        spurious_rules = sorted(predicted_rules - expected_rules)
        exact_rule_match = predicted_rules == expected_rules

        return ScorerResult(
            score=score,
            passed=score >= self.pass_threshold,
            detail={
                "rule_score": rule_score,
                "action_score": action_score,
                "predicted_rules": sorted(predicted_rules),
                "expected_rules": sorted(expected_rules),
                "predicted_action": predicted_action,
                "expected_action": expected_action,
                "exact_rule_match": exact_rule_match,
                "spurious_rules": spurious_rules,
                "phantom_rules": phantom_rules,
                "threshold": self.pass_threshold,
            },
        )


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _bpi_phrase_matches(haystack: str, needle: str) -> bool:
    if needle == "":
        return False
    if needle in haystack:
        return True
    haystack_tokens = haystack.split()
    needle_tokens = needle.split()
    pos = 0
    for token in needle_tokens:
        found_at = next(
            (
                index
                for index in range(pos, len(haystack_tokens))
                if haystack_tokens[index].startswith(token)
                or token.startswith(haystack_tokens[index])
            ),
            None,
        )
        if found_at is None:
            return False
        pos = found_at + 1
    return True


__all__ = [
    "AdversarialAbstentionScorer",
    "BPIContainmentScorer",
    "ExactContainmentScorer",
    "JudgeBridgeScorer",
    "TokenF1Scorer",
    "bpi_action_score",
    "bpi_macro_average",
    "bpi_rule_score",
    "extract_bpi_answer",
    "normalise_bpi_text",
    "normalise_tokens",
    "token_f1",
]
