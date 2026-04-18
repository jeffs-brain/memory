# SPDX-License-Identifier: Apache-2.0
"""Ollama provider. Stub — requires `jeffs-brain-memory[ollama]`."""

from __future__ import annotations

from typing import Sequence

from . import ChatMessage, EmbedProvider, LlmProvider


class OllamaProvider(LlmProvider, EmbedProvider):
    def __init__(self, *, host: str = "http://localhost:11434") -> None:
        self.host = host

    async def complete(self, messages: Sequence[ChatMessage], *, model: str) -> str:
        raise NotImplementedError("OllamaProvider.complete")

    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        raise NotImplementedError("OllamaProvider.embed")
