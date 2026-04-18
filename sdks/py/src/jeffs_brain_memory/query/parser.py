# SPDX-License-Identifier: Apache-2.0
"""Query DSL parser. Stub — target behaviour described in `spec/QUERY-DSL.md`."""

from __future__ import annotations

from . import CompiledQuery, Query


def parse(raw: str) -> Query:
    """Parse a query string into an AST. Stub."""
    raise NotImplementedError("query.parser.parse")


def compile(query: Query) -> CompiledQuery:
    """Compile an AST to search-stage inputs. Stub."""
    raise NotImplementedError("query.parser.compile")
