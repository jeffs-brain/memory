# SPDX-License-Identifier: Apache-2.0
"""Query DSL — parsing, AST, and compilation to search stages.

See `spec/QUERY-DSL.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Query", "CompiledQuery"]


@dataclass(frozen=True, slots=True)
class Query:
    """Parsed query AST. Concrete shape lands in Phase 4b."""

    raw: str


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    """Query compiled into BM25 and/or vector stages."""

    bm25: str | None = None
    vector_text: str | None = None
