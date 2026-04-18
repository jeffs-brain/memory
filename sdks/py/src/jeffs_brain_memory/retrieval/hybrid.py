# SPDX-License-Identifier: Apache-2.0
"""Reciprocal Rank Fusion + retry ladder. Stub."""

from __future__ import annotations

from ..query import CompiledQuery
from . import RetrievedChunk


async def retrieve(query: CompiledQuery, *, limit: int = 20) -> list[RetrievedChunk]:
    """Run the hybrid retrieval pipeline. Stub."""
    raise NotImplementedError("retrieval.hybrid.retrieve")
