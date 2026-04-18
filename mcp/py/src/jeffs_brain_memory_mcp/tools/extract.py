# SPDX-License-Identifier: Apache-2.0
"""``memory_extract`` tool definition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..client import ExtractArgs, ExtractMessage, MemoryClient, ProgressEmitter
from ._types import ToolDef


class ExtractMessageInput(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ExtractInput(BaseModel):
    messages: list[ExtractMessageInput] = Field(min_length=1, max_length=500)
    brain: str | None = None
    actor_id: str | None = None
    session_id: str | None = None


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, ExtractInput)
    messages = tuple(
        ExtractMessage(role=m.role, content=m.content) for m in args.messages
    )
    return await client.extract(
        ExtractArgs(
            messages=messages,
            brain=args.brain,
            actor_id=args.actor_id,
            session_id=args.session_id,
        ),
        progress,
    )


extract_tool = ToolDef(
    name="memory_extract",
    description=(
        "Submit a conversation transcript so the server can asynchronously "
        "extract memorable facts. If session_id is provided the messages are "
        "appended to that session; otherwise a transcript document is created."
    ),
    input_model=ExtractInput,
    handler=_handle,
)
