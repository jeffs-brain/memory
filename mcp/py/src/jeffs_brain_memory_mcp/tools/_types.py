# SPDX-License-Identifier: Apache-2.0
"""Shared tool metadata types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from ..client import MemoryClient, ProgressEmitter


ToolHandler = Callable[
    [BaseModel, MemoryClient, ProgressEmitter | None], Awaitable[dict[str, Any]]
]


@dataclass(frozen=True, slots=True)
class ToolDef:
    """Static tool metadata plus its async handler."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler


__all__ = ["ToolDef", "ToolHandler"]
