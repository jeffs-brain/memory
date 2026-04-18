# SPDX-License-Identifier: Apache-2.0
"""SQLite FTS5 BM25 index. Stub."""

from __future__ import annotations

from pathlib import Path

from . import SearchHit


class Bm25Index:
    """Wrapper over an FTS5 virtual table."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    def upsert(self, chunk_id: str, text: str) -> None:
        raise NotImplementedError("Bm25Index.upsert")

    def delete(self, chunk_id: str) -> None:
        raise NotImplementedError("Bm25Index.delete")

    def search(self, query: str, *, limit: int = 50) -> list[SearchHit]:
        raise NotImplementedError("Bm25Index.search")
