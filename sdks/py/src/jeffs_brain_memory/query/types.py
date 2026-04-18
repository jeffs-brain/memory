# SPDX-License-Identifier: Apache-2.0
"""Dataclasses that describe a parsed, annotated query.

See ``spec/QUERY-DSL.md`` for the normative behaviour of each field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..llm.provider import Provider

__all__ = [
    "Query",
    "TemporalAnnotation",
    "Options",
    "Trace",
    "Result",
]


@dataclass(slots=True)
class TemporalAnnotation:
    """A resolved temporal reference within a query.

    ``range_start`` and ``range_end`` form a half-open UTC interval
    ``[range_start, range_end)``. ``recogniser`` identifies which of the
    three spec-defined recognisers produced the annotation (one of
    ``"relative"``, ``"last_weekday"`` or ``"ordering"``).
    """

    range_start: datetime
    range_end: datetime
    recogniser: str


@dataclass(slots=True)
class Query:
    """A parsed query with downstream-friendly annotations.

    - ``raw`` is the caller's original input.
    - ``normalised`` is lower-cased, whitespace-collapsed text used by the
      cache key and significance checks.
    - ``tokens`` is a whitespace-split token list for rough length signals.
    - ``significant_terms`` lists non-stopword tokens (lowercased, stripped
      of non-alphanumeric runs).
    - ``temporal`` carries the first resolved temporal annotation, if any.
    - ``distilled`` is the LLM-rewritten query text, set only when the
      distiller runs and succeeds.
    """

    raw: str
    normalised: str
    tokens: list[str] = field(default_factory=list)
    significant_terms: list[str] = field(default_factory=list)
    temporal: TemporalAnnotation | None = None
    distilled: str | None = None


@dataclass(slots=True)
class Options:
    """Parser options. Mirrors the Go ``query.Options`` shape.

    ``anchor`` is used by temporal recognisers; when absent or ``None``
    temporal annotation is skipped. ``model`` flows through to the
    distiller's cache key so the same query against a different model
    does not collide.
    """

    distill: bool = False
    cache: bool = True
    provider: Provider | None = None
    anchor: datetime | None = None
    model: str = ""


@dataclass(slots=True)
class Trace:
    """Side-channel describing what the parser did on this call."""

    used_cache: bool = False
    distilled: bool = False


@dataclass(slots=True)
class Result:
    """Pair returned from :func:`query.parser.parse`."""

    query: Query
    trace: Trace
