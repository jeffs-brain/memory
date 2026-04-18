# SPDX-License-Identifier: Apache-2.0
"""``memory_ingest_url`` tool definition."""

from __future__ import annotations

from typing import Any

from pydantic import AnyUrl, BaseModel, Field

from ..client import IngestUrlArgs, MemoryClient, ProgressEmitter
from ._types import ToolDef


class IngestUrlInput(BaseModel):
    url: AnyUrl = Field(description="Absolute URL to fetch and ingest.")
    brain: str | None = None


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, IngestUrlInput)
    return await client.ingest_url(
        IngestUrlArgs(url=str(args.url), brain=args.brain),
        progress,
    )


ingest_url_tool = ToolDef(
    name="memory_ingest_url",
    description=(
        "Fetch a URL and ingest its contents into the brain. Uses the "
        "server-side /ingest/url endpoint when available; otherwise fetches "
        "locally and creates a document."
    ),
    input_model=IngestUrlInput,
    handler=_handle,
)
