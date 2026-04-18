# SPDX-License-Identifier: Apache-2.0
"""Query DSL: parsing, normalisation, temporal, distillation.

See ``spec/QUERY-DSL.md`` for the normative behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Options, Query, Result, TemporalAnnotation, Trace

__all__ = [
    "Query",
    "CompiledQuery",
    "Options",
    "Result",
    "TemporalAnnotation",
    "Trace",
]


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    """Query compiled into BM25 and/or vector stages.

    Retained for compatibility with :mod:`jeffs_brain_memory.retrieval`;
    the full compilation pipeline lands in a later phase.
    """

    bm25: str | None = None
    vector_text: str | None = None
