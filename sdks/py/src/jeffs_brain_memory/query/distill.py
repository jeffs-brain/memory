# SPDX-License-Identifier: Apache-2.0
"""LLM-backed query distiller with an LRU result cache.

The prompt is ported verbatim from ``go/query/prompt.go`` to keep
behaviour aligned across SDKs.
"""

from __future__ import annotations

from collections import OrderedDict

from ..llm.provider import Provider
from ..llm.types import Message, Role, CompleteRequest

__all__ = ["Distiller", "DISTILL_SYSTEM_PROMPT"]

# Ported verbatim from Go's ``query/prompt.go``. Do NOT reflow or edit:
# any SDK divergence shows up as a cache-key mismatch downstream.
DISTILL_SYSTEM_PROMPT = """You are a search query distiller. Given a raw user message (which may be a huge error paste, a vague question, or a multi-part request), produce structured search queries that will retrieve the most relevant information from a knowledge base.

Respond with ONLY a JSON array of query objects:
[{"text": "concise search query", "domain": "optional domain hint", "entities": ["extracted entities"], "recency_bias": "recent|historical|", "confidence": 0.0-1.0}]

Rules:
- Extract the actual question from noise (error logs, pasted code, etc.)
- Split multi-intent queries into separate query objects
- Expand abbreviations and jargon where possible
- Resolve anaphoric references ("it", "that") using context if available
- Maximum 3 queries per input
- Each query text should be 5-30 words, focused and searchable
- Set confidence to 0.0-1.0 based on how certain you are the rewrite captures the intent"""

_MAX_DISTILL_TOKENS = 512
_DISTILL_TEMPERATURE = 0.1


class Distiller:
    """LLM-backed query distiller with an LRU cache.

    The cache key is ``(model, query.lower().strip())``; identical
    queries against different models are cached independently. Each
    ``get`` bumps the entry to the most-recently-used position via
    ``OrderedDict.move_to_end`` in O(1).
    """

    def __init__(self, provider: Provider, *, cache_size: int = 512) -> None:
        if cache_size <= 0:
            cache_size = 1
        self._provider = provider
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._cache_size = cache_size

    async def distill(self, query: str, *, model: str = "") -> str:
        """Return the distilled form of ``query``.

        Empty or whitespace-only input is returned untouched (and never
        cached). A provider error propagates to the caller; the cache is
        not poisoned with an error result.
        """

        stripped = query.strip()
        if not stripped:
            return stripped

        key = (model, stripped.lower())
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        response = await self._provider.complete(
            CompleteRequest(
                model=model,
                messages=[
                    Message(role=Role.SYSTEM, content=DISTILL_SYSTEM_PROMPT),
                    Message(role=Role.USER, content=stripped),
                ],
                temperature=_DISTILL_TEMPERATURE,
                max_tokens=_MAX_DISTILL_TOKENS,
            )
        )
        distilled = (response.text or "").strip()
        if not distilled:
            distilled = stripped

        self._cache_put(key, distilled)
        return distilled

    def cache_size(self) -> int:
        """Return the current number of cached entries. Primarily for tests."""

        return len(self._cache)

    def _cache_get(self, key: tuple[str, str]) -> str | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: tuple[str, str], value: str) -> None:
        if key in self._cache:
            self._cache[key] = value
            self._cache.move_to_end(key)
            return
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
