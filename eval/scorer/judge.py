# SPDX-License-Identifier: Apache-2.0
"""LLM-as-judge scorer. Defaults to OpenAI `gpt-4o`.

Budget guard: set `JB_EVAL_BUDGET_USD` to fail-fast when the running
spend estimate exceeds the threshold. The estimate is coarse (it uses
OpenAI-published per-token prices for the default model) but enough to
catch runaway CI costs.

Override the model via `JB_EVAL_JUDGE_MODEL`. For LongMemEval, `gpt-4o`
is the recommended actor + judge. Lighter alternatives for cheap PR
smoke runs: `gpt-4o-mini`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "gpt-4o"

# Conservative USD per 1M tokens as of 2026-04 for gpt-4o. Override via env for drift.
_DEFAULT_INPUT_COST_PER_M = 2.50
_DEFAULT_OUTPUT_COST_PER_M = 10.00

JUDGE_SYSTEM = (
    "You are a strict eval judge for a retrieval-augmented memory system. "
    "Score each answer from 0.0 to 1.0 against the reference answer. "
    "Rubric: (1) faithfulness to the reference, (2) presence of supporting "
    "citations when the reference includes them, (3) semantic match. "
    "Reply with a compact JSON object: {\"score\": <float 0..1>, \"reason\": \"<string>\"}."
)


class BudgetExceededError(RuntimeError):
    """Raised when `JB_EVAL_BUDGET_USD` is breached mid-run."""


@dataclass
class JudgeScorer:
    model: str = field(default_factory=lambda: os.environ.get("JB_EVAL_JUDGE_MODEL", DEFAULT_MODEL))
    budget_usd: float | None = field(default=None)
    input_cost_per_m: float = field(default=_DEFAULT_INPUT_COST_PER_M)
    output_cost_per_m: float = field(default=_DEFAULT_OUTPUT_COST_PER_M)
    spend_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.budget_usd is None:
            raw = os.environ.get("JB_EVAL_BUDGET_USD")
            self.budget_usd = float(raw) if raw else None

    def score(self, *, item: dict[str, Any], answer: str) -> float:
        reference = item.get("reference_answer") or ""
        question = item.get("question", "")
        prompt = (
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference}\n\n"
            f"Candidate answer:\n{answer}\n"
        )
        # TODO(eval): wire an optional cache so re-runs against identical
        # (model, prompt) pairs skip the API call.
        content, usage = self._call_openai(prompt)
        self._book_spend(usage)
        return _parse_score(content)

    def _call_openai(self, prompt: str) -> tuple[str, dict[str, int]]:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        choice = resp.choices[0].message.content or "{}"
        usage = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
        }
        return choice, usage

    def _book_spend(self, usage: dict[str, int]) -> None:
        cost = (
            usage.get("input_tokens", 0) / 1_000_000 * self.input_cost_per_m
            + usage.get("output_tokens", 0) / 1_000_000 * self.output_cost_per_m
        )
        self.spend_usd += cost
        if self.budget_usd is not None and self.spend_usd > self.budget_usd:
            raise BudgetExceededError(
                f"JB_EVAL_BUDGET_USD exceeded: spent ${self.spend_usd:.4f} > ${self.budget_usd:.4f}"
            )


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(content: str) -> float:
    """Parse a 0..1 float from the judge's JSON response, clamping on noise."""
    try:
        data = json.loads(content)
        value = float(data.get("score", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        match = _NUMBER_RE.search(content or "")
        value = float(match.group(0)) if match else 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
