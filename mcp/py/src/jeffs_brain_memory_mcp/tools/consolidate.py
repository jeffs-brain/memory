# SPDX-License-Identifier: Apache-2.0
"""``memory_consolidate`` tool definition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..client import ConsolidateArgs, MemoryClient, ProgressEmitter
from ._types import ToolDef


class ConsolidateInput(BaseModel):
    brain: str | None = None


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, ConsolidateInput)
    return await client.consolidate(
        ConsolidateArgs(brain=args.brain),
        progress,
    )


consolidate_tool = ToolDef(
    name="memory_consolidate",
    description=(
        "Trigger a consolidation pass on the brain (compile summaries, "
        "promote stable notes, prune stale episodic memory)."
    ),
    input_model=ConsolidateInput,
    handler=_handle,
)
