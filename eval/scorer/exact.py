# SPDX-License-Identifier: Apache-2.0
"""Deterministic scorer. No network. Substring match against `expected_substrings`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExactScorer:
    case_sensitive: bool = False

    def score(self, *, item: dict[str, Any], answer: str) -> float:
        """Return 1.0 if `answer` contains any `expected_substrings` entry, else 0.0.

        An empty `expected_substrings` list is treated as a scorer
        misconfiguration and returns 0.0 rather than a silent pass.
        """
        expected: list[str] = item.get("expected_substrings") or []
        if not expected:
            return 0.0
        haystack = answer if self.case_sensitive else answer.lower()
        for needle in expected:
            candidate = needle if self.case_sensitive else needle.lower()
            if candidate and candidate in haystack:
                return 1.0
        return 0.0
