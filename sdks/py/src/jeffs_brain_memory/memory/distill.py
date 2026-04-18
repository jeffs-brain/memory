# SPDX-License-Identifier: Apache-2.0
"""Distilled recall: run query distillation before a standard recall."""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

from ..llm.provider import Provider
from ..llm.types import Message as LLMMessage
from .recall import RecallWeights, SurfacedMemory, recall

if TYPE_CHECKING:
    from .manager import MemoryManager


class Distiller(Protocol):
    async def distill(
        self, query: str, history: list[LLMMessage], options: dict[str, Any]
    ) -> Any: ...


async def distilled_recall(
    mem: "MemoryManager",
    provider: Provider,
    model: str,
    distiller: Distiller | None,
    project_path: str,
    user_query: str,
    history: list[LLMMessage],
    surfaced: set[str] | None,
    weights: RecallWeights,
) -> tuple[list[SurfacedMemory], Any]:
    """Return (memories, trace). ``trace`` is whatever the distiller emits."""
    search_query = user_query
    trace: Any = None
    if distiller is not None:
        try:
            result = await distiller.distill(
                user_query,
                history,
                {"scope": "recall", "cloud_provider": provider},
            )
        except Exception:
            result = None
        if result is not None:
            trace = getattr(result, "trace", None)
            queries = getattr(result, "queries", None) or []
            if queries:
                first = queries[0]
                txt = getattr(first, "text", None) or str(first)
                if txt:
                    search_query = txt

    memories = await recall(
        mem, provider, model, project_path, search_query, surfaced, weights
    )
    return memories, trace
