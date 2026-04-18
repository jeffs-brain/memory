# SPDX-License-Identifier: Apache-2.0
"""Search stage — SQLite BM25 (FTS5) and sqlite-vec.

See `spec/ALGORITHMS.md` for ranking details.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..path import ChunkID

__all__ = ["SearchHit"]


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: ChunkID
    score: float
    snippet: str | None = None
