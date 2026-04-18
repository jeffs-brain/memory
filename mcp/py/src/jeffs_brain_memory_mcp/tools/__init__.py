# SPDX-License-Identifier: Apache-2.0
"""The eleven ``memory_*`` tool definitions."""

from __future__ import annotations

from ._types import ToolDef, ToolHandler
from .ask import ask_tool
from .consolidate import consolidate_tool
from .create_brain import create_brain_tool
from .extract import extract_tool
from .ingest_file import ingest_file_tool
from .ingest_url import ingest_url_tool
from .list_brains import list_brains_tool
from .recall import recall_tool
from .reflect import reflect_tool
from .remember import remember_tool
from .search import search_tool


TOOLS: tuple[ToolDef, ...] = (
    remember_tool,
    recall_tool,
    search_tool,
    ask_tool,
    ingest_file_tool,
    ingest_url_tool,
    extract_tool,
    reflect_tool,
    consolidate_tool,
    create_brain_tool,
    list_brains_tool,
)


__all__ = ["ToolDef", "ToolHandler", "TOOLS"]
