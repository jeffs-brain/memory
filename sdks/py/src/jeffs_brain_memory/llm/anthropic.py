# SPDX-License-Identifier: Apache-2.0
"""Anthropic provider. Stub — requires `jeffs-brain-memory[anthropic]`."""

from __future__ import annotations

from typing import Sequence

from . import ChatMessage, LlmProvider


class AnthropicProvider(LlmProvider):
    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key

    async def complete(self, messages: Sequence[ChatMessage], *, model: str) -> str:
        raise NotImplementedError("AnthropicProvider.complete")
