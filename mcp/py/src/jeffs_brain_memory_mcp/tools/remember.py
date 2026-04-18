# SPDX-License-Identifier: Apache-2.0
"""``memory_remember`` tool definition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..client import MemoryClient, ProgressEmitter, RememberArgs
from ._types import ToolDef


class RememberInput(BaseModel):
    content: str = Field(
        min_length=1,
        max_length=5_000_000,
        description="Markdown body of the new memory.",
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Title. Derived from the first heading if omitted.",
    )
    brain: str | None = Field(
        default=None,
        description="Brain id or slug. Defaults to JB_BRAIN or the singleton brain.",
    )
    tags: list[str] | None = Field(
        default=None,
        max_length=64,
        description="Optional tags, each 1..64 chars.",
    )
    path: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description="Destination path within the brain.",
    )


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    _progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, RememberInput)
    return await client.remember(
        RememberArgs(
            content=args.content,
            title=args.title,
            brain=args.brain,
            tags=tuple(args.tags or ()),
            path=args.path,
        )
    )


remember_tool = ToolDef(
    name="memory_remember",
    description=(
        "Store a new memory (markdown document) in the brain. "
        "Returns the created document id and path."
    ),
    input_model=RememberInput,
    handler=_handle,
)
