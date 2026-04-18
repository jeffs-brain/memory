# SPDX-License-Identifier: Apache-2.0
"""``memory_list_brains`` tool definition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..client import MemoryClient, ProgressEmitter
from ._types import ToolDef


class ListBrainsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def _handle(
    _args: BaseModel,
    client: MemoryClient,
    _progress: ProgressEmitter | None,
) -> dict[str, Any]:
    return await client.list_brains()


list_brains_tool = ToolDef(
    name="memory_list_brains",
    description="List all brains the caller has access to.",
    input_model=ListBrainsInput,
    handler=_handle,
)
