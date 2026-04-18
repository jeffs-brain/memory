# SPDX-License-Identifier: Apache-2.0
"""Cross-encoder reranker. Stub."""

from __future__ import annotations

from ..retrieval import RetrievedChunk


async def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank retrieved chunks against the query. Stub."""
    raise NotImplementedError("rerank.cross_encoder.rerank")
