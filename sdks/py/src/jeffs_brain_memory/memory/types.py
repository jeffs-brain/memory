# SPDX-License-Identifier: Apache-2.0
"""Memory-layer message shape with tool-call metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.types import Message as LLMMessage
from ..llm.types import Role


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str = ""
    name: str = ""
    arguments: str = ""  # raw JSON string


@dataclass(slots=True)
class ToolResultBlock:
    """A tool-result attachment on a message."""

    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass(slots=True)
class ContentBlock:
    """Structured content fragment attached to a :class:`Message`."""

    type: str = ""
    text: str = ""
    tool_use: ToolCall | None = None
    tool_result: ToolResultBlock | None = None


@dataclass(slots=True)
class Message:
    """Conversation turn carrying memory-specific metadata.

    Convert to the simpler :class:`llm.Message` via :meth:`as_llm`.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""
    blocks: list[ContentBlock] = field(default_factory=list)

    def as_llm(self) -> LLMMessage:
        """Return the :class:`llm.Message` equivalent."""
        return LLMMessage(role=self.role, content=self.content)


def messages_as_llm(msgs: list[Message]) -> list[LLMMessage]:
    """Convert a list of memory messages for provider calls."""
    return [m.as_llm() for m in msgs]


@dataclass(slots=True)
class TopicFile:
    """Metadata for a memory topic file."""

    name: str = ""
    description: str = ""
    type: str = ""
    path: str = ""
    created: str = ""
    modified: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: str = ""
    source: str = ""
    scope: str = ""


@dataclass(slots=True)
class Frontmatter:
    """Parsed YAML frontmatter from a memory markdown file."""

    name: str = ""
    description: str = ""
    type: str = ""
    created: str = ""
    modified: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: str = ""
    source: str = ""
    supersedes: str = ""
    superseded_by: str = ""
