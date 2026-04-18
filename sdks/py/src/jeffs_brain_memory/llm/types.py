# SPDX-License-Identifier: Apache-2.0
"""Shared dataclasses and enums for the llm package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """Speaker of a :class:`Message`."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    """Why generation halted."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    STOP = "stop_sequence"


@dataclass(slots=True)
class Message:
    """Single conversation turn."""

    role: Role
    content: str


@dataclass(slots=True)
class ToolDef:
    """Callable tool a model may invoke."""

    name: str
    description: str = ""
    schema: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """Model's request to invoke a tool.

    ``arguments`` is the raw JSON string as produced by the provider.
    """

    name: str
    arguments: str = ""
    id: str = ""


@dataclass(slots=True)
class CompleteRequest:
    """Common request every :class:`Provider` accepts."""

    model: str = ""
    messages: list[Message] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 0
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    tools: list[ToolDef] = field(default_factory=list)


@dataclass(slots=True)
class CompleteResponse:
    """Returned by :meth:`Provider.complete`."""

    text: str = ""
    stop: StopReason = StopReason.END_TURN
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class StreamChunk:
    """One unit emitted on a streaming iterator.

    ``delta_text`` is incremental; ``stop`` is non-empty on the final chunk.
    ``tool_call`` is set when a provider streams tool calls inline.
    """

    delta_text: str = ""
    tool_call: ToolCall | None = None
    stop: StopReason | None = None


class LLMError(Exception):
    """Base error type raised by providers."""


class EmptyMessagesError(LLMError):
    """Raised when a provider receives no messages to complete on."""

    def __init__(self) -> None:
        super().__init__("llm: empty messages")


class NoProviderError(LLMError):
    """Raised when no provider can be resolved from the environment."""

    def __init__(self) -> None:
        super().__init__("llm: no provider configured")
