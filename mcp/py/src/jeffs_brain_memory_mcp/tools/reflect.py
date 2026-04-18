# SPDX-License-Identifier: Apache-2.0
"""``memory_reflect`` tool definition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..client import MemoryClient, ProgressEmitter, ReflectArgs
from ._types import ToolDef


class ReflectInput(BaseModel):
    session_id: str = Field(min_length=1)
    brain: str | None = None


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, ReflectInput)
    return await client.reflect(
        ReflectArgs(session_id=args.session_id, brain=args.brain),
        progress,
    )


reflect_tool = ToolDef(
    name="memory_reflect",
    description="Close a session and trigger server-side reflection over its messages.",
    input_model=ReflectInput,
    handler=_handle,
)
