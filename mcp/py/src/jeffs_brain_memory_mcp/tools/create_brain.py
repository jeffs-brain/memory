# SPDX-License-Identifier: Apache-2.0
"""``memory_create_brain`` tool definition."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..client import CreateBrainArgs, MemoryClient, ProgressEmitter
from ._types import ToolDef


_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class CreateBrainInput(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    slug: str | None = Field(default=None, min_length=1, max_length=64)
    visibility: Literal["private", "tenant", "public"] | None = None

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _SLUG_PATTERN.fullmatch(value):
            raise ValueError("slug must match /^[a-z0-9][a-z0-9-]*$/")
        return value


async def _handle(
    args: BaseModel,
    client: MemoryClient,
    _progress: ProgressEmitter | None,
) -> dict[str, Any]:
    assert isinstance(args, CreateBrainInput)
    return await client.create_brain(
        CreateBrainArgs(
            name=args.name,
            slug=args.slug,
            visibility=args.visibility,
        )
    )


create_brain_tool = ToolDef(
    name="memory_create_brain",
    description=(
        "Create a new brain. Generates a slug from the name if one is not provided."
    ),
    input_model=CreateBrainInput,
    handler=_handle,
)
