# SPDX-License-Identifier: Apache-2.0
"""``memory_ingest_file`` tool definition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..client import IngestFileArgs, MemoryClient, ProgressEmitter
from ._types import ToolDef


class IngestFileInput(BaseModel):
    path: str = Field(min_length=1, description="Absolute or relative local path.")
    brain: str | None = None
    as_: Literal["markdown", "text", "pdf", "json"] | None = Field(
        default=None,
        alias="as",
    )

    model_config = {"populate_by_name": True}


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, IngestFileInput)
    return await client.ingest_file(
        IngestFileArgs(path=args.path, brain=args.brain, as_=args.as_),
        progress,
    )


ingest_file_tool = ToolDef(
    name="memory_ingest_file",
    description=(
        "Ingest a local file (<= 25 MB) into the brain. Returns the ingest result."
    ),
    input_model=IngestFileInput,
    handler=_handle,
)
