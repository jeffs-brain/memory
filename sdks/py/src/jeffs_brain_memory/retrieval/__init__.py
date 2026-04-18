# SPDX-License-Identifier: Apache-2.0
"""Retrieval — hybrid fusion over BM25 and vector stages.

See `spec/ALGORITHMS.md` for RRF, the retry ladder, and unanimity rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..path import ChunkID

__all__ = ["RetrievedChunk"]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: ChunkID
    score: float
    text: str
    source_path: str | None = None
