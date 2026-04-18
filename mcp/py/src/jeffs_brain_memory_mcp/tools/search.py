# SPDX-License-Identifier: Apache-2.0
"""``memory_search`` tool definition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..client import MemoryClient, ProgressEmitter, SearchArgs
from ._types import ToolDef


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    brain: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    scope: Literal["all", "global", "project", "agent"] | None = None
    sort: Literal["relevance", "recency", "relevance_then_recency"] | None = None


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    _progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, SearchInput)
    return await client.search(
        SearchArgs(
            query=args.query,
            brain=args.brain,
            top_k=args.top_k,
            scope=args.scope,
            sort=args.sort,
        )
    )


search_tool = ToolDef(
    name="memory_search",
    description=(
        "Search memory notes in a brain and return matching note content with "
        "citations. `scope` selects the memory namespace, and `sort` controls "
        "whether relevance or recency wins."
    ),
    input_model=SearchInput,
    handler=_handle,
)
