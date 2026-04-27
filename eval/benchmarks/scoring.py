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


__all__ = [
    "AdversarialAbstentionScorer",
    "ExactContainmentScorer",
    "JudgeBridgeScorer",
    "TokenF1Scorer",
    "normalise_tokens",
    "token_f1",
]
