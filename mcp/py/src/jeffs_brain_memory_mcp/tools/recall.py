# SPDX-License-Identifier: Apache-2.0
"""``memory_recall`` tool definition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..client import MemoryClient, ProgressEmitter, RecallArgs
from ._types import ToolDef


class RecallInput(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    brain: str | None = None
    scope: Literal["global", "project", "agent"] | None = None
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    _progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, RecallInput)
    return await client.recall(
        RecallArgs(
            query=args.query,
            brain=args.brain,
            scope=args.scope,
            session_id=args.session_id,
            top_k=args.top_k,
        )
    )


recall_tool = ToolDef(
    name="memory_recall",
    description=(
        "Recall memories for a query. Pass session_id to weight recent session "
        "context; otherwise uses the dedicated memory-search surface. `scope` "
        "selects the memory namespace rather than a generic metadata filter."
    ),
    input_model=RecallInput,
    handler=_handle,
)
