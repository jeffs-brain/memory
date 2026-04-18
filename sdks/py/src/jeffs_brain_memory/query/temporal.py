# SPDX-License-Identifier: Apache-2.0
"""Temporal expansion for query filters (e.g. `last:week`). Stub."""

from __future__ import annotations

from datetime import datetime


def expand(expression: str, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Resolve a relative temporal expression to a `[start, end)` range. Stub."""
    raise NotImplementedError("query.temporal.expand")
