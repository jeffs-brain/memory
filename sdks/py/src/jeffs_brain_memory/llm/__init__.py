# SPDX-License-Identifier: Apache-2.0
"""LLM provider abstraction.

Exports the provider and embedder protocols along with the shared dataclass
shapes and concrete backends (OpenAI, Anthropic, Ollama, Fake).
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .config import embedder_from_env, provider_from_env
from .fake import FakeEmbedder, FakeProvider
from .ollama import OllamaEmbedder, OllamaProvider
from .openai import OpenAIEmbedder, OpenAIProvider
from .provider import Embedder, Provider
from .types import (
    CompleteRequest,
    CompleteResponse,
    EmptyMessagesError,
    LLMError,
    Message,
    NoProviderError,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolDef,
)

__all__ = [
    "AnthropicProvider",
    "CompleteRequest",
    "CompleteResponse",
    "Embedder",
    "EmptyMessagesError",
    "FakeEmbedder",
    "FakeProvider",
    "LLMError",
    "Message",
    "NoProviderError",
    "OllamaEmbedder",
    "OllamaProvider",
    "OpenAIEmbedder",
    "OpenAIProvider",
    "Provider",
    "Role",
    "StopReason",
    "StreamChunk",
    "ToolCall",
    "ToolDef",
    "embedder_from_env",
    "provider_from_env",
]
