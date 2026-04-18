# SPDX-License-Identifier: Apache-2.0
"""sqlite-vec vector index. Stub."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import SearchHit


class VectorIndex:
    """Wrapper over a `vec0` virtual table."""

    def __init__(self, db_path: Path | str, *, dim: int) -> None:
        self.db_path = Path(db_path)
        self.dim = dim

    def upsert(self, chunk_id: str, embedding: Sequence[float]) -> None:
        raise NotImplementedError("VectorIndex.upsert")

    def delete(self, chunk_id: str) -> None:
        raise NotImplementedError("VectorIndex.delete")

    def search(self, embedding: Sequence[float], *, limit: int = 50) -> list[SearchHit]:
        raise NotImplementedError("VectorIndex.search")
