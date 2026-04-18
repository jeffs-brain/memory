# SPDX-License-Identifier: Apache-2.0
"""``memory_ask`` tool definition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..client import AskArgs, MemoryClient, ProgressEmitter
from ._types import ToolDef


class AskInput(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    brain: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    progress: ProgressEmitter | None,
) -> dict[str, Any]:
    # TODO(next-pass): switch to streaming once the Python SDK surfaces a
    # token-level `ask` iterator. Today the local implementation returns a
    # final payload and emits two coarse progress events (retrieved +
    # answered), mirroring the TypeScript wrapper.
    assert isinstance(args, AskInput)
    return await client.ask(
        AskArgs(query=args.query, brain=args.brain, top_k=args.top_k),
        progress,
    )


ask_tool = ToolDef(
    name="memory_ask",
    description=(
        "Ask a question grounded in the brain. Streams answer tokens as MCP "
        "progress notifications and returns the final answer with citations."
    ),
    input_model=AskInput,
    handler=_handle,
)
