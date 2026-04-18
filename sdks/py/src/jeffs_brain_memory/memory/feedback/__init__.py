# SPDX-License-Identifier: Apache-2.0
"""Implicit user feedback classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Reaction(str, Enum):
    REINFORCED = "reinforced"
    CORRECTED = "corrected"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class FeedbackEvent:
    memory_path: str = ""
    reaction: Reaction = Reaction.NEUTRAL
    confidence: float = 0.0
    pattern: str = ""
    snippet: str = ""


@dataclass(slots=True)
class ClassifyResult:
    events: list[FeedbackEvent] = field(default_factory=list)
    turn_content: str = ""


POSITIVE_PATTERNS = [
    r"\b(perfect|exactly|great|thanks|correct|right|yes)\b",
    r"\bthat('s| is| was) (right|correct|helpful|useful|what i needed)\b",
    r"\b(good|nice) (memory|recall|find)\b",
    r"\byou remembered\b",
    r"\bthat helps\b",
    r"\bspot on\b",
]

NEGATIVE_PATTERNS = [
    r"\b(wrong|incorrect|no|nope|not right)\b",
    r"\bthat('s| is| was) (wrong|incorrect|outdated|old|stale)\b",
    r"\b(forget|remove|delete) (that|this|it)\b",
    r"\bnot what i (meant|asked|wanted)\b",
    r"\btry again\b",
    r"\bthat('s| is) (not|no longer) (true|accurate|relevant)\b",
    r"\bactually[,.]?\s",
]


class Classifier:
    def __init__(self) -> None:
        self._positive = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]
        self._negative = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]

    def classify(
        self, user_input: str, surfaced_this_turn: list[str]
    ) -> ClassifyResult:
        result = ClassifyResult(turn_content=_truncate_snippet(user_input, 500))
        if not surfaced_this_turn or not user_input.strip():
            return result

        reaction, confidence, pattern = self._detect_reaction(user_input)
        for path in surfaced_this_turn:
            result.events.append(
                FeedbackEvent(
                    memory_path=path,
                    reaction=reaction,
                    confidence=confidence,
                    pattern=pattern,
                    snippet=_truncate_snippet(user_input, 200),
                )
            )
        return result

    def _detect_reaction(
        self, user_input: str
    ) -> tuple[Reaction, float, str]:
        pos_matches = 0
        pos_pattern = ""
        for r in self._positive:
            m = r.search(user_input)
            if m:
                pos_matches += 1
                if not pos_pattern:
                    pos_pattern = m.group(0)
        neg_matches = 0
        neg_pattern = ""
        for r in self._negative:
            m = r.search(user_input)
            if m:
                neg_matches += 1
                if not neg_pattern:
                    neg_pattern = m.group(0)

        if pos_matches == 0 and neg_matches == 0:
            return Reaction.NEUTRAL, 0.0, ""
        if pos_matches > neg_matches:
            return Reaction.REINFORCED, _clamp(pos_matches * 0.3), pos_pattern
        if neg_matches > pos_matches:
            return Reaction.CORRECTED, _clamp(neg_matches * 0.3), neg_pattern
        return Reaction.NEUTRAL, 0.2, ""


def _clamp(v: float) -> float:
    return 1.0 if v > 1.0 else v


def _truncate_snippet(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."


__all__ = [
    "Classifier",
    "ClassifyResult",
    "FeedbackEvent",
    "Reaction",
]
