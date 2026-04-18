# SPDX-License-Identifier: Apache-2.0
"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

__all__ = ["LlmProvider", "ChatMessage", "EmbedProvider"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LlmProvider(ABC):
    """Completion provider."""

    @abstractmethod
    async def complete(self, messages: Sequence[ChatMessage], *, model: str) -> str: ...


class EmbedProvider(ABC):
    """Embedding provider."""

    @abstractmethod
    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]: ...
