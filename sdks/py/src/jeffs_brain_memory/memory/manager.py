# SPDX-License-Identifier: Apache-2.0
"""High-level memory orchestration. Stub."""

from __future__ import annotations

from ..store import Store


class MemoryManager:
    """Coordinates extraction, reflection, and consolidation."""

    def __init__(self, store: Store) -> None:
        self.store = store

    async def remember(self, content: str, *, source: str | None = None) -> None:
        raise NotImplementedError("MemoryManager.remember")

    async def recall(self, query: str, *, limit: int = 20) -> list[str]:
        raise NotImplementedError("MemoryManager.recall")
