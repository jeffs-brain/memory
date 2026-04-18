# SPDX-License-Identifier: Apache-2.0
"""OpenAI provider. Stub — requires `jeffs-brain-memory[openai]`."""

from __future__ import annotations

from typing import Sequence

from . import ChatMessage, EmbedProvider, LlmProvider


class OpenAIProvider(LlmProvider, EmbedProvider):
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def complete(self, messages: Sequence[ChatMessage], *, model: str) -> str:
        raise NotImplementedError("OpenAIProvider.complete")

    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        raise NotImplementedError("OpenAIProvider.embed")
