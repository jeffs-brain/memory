# SPDX-License-Identifier: Apache-2.0
"""Temporal recognisers for the Python SDK.

Ports the three recognisers defined in ``sdks/go/query/temporal.go`` and
the normative spec in ``spec/QUERY-DSL.md``. All date arithmetic is UTC
only: callers that care about local time must normalise their anchor
before passing it in.

The two explicit recognisers return :class:`TemporalAnnotation` instances
with half-open UTC intervals ``[range_start, range_end)``. The ordering
recogniser does not produce a date range: it returns the possibly
annotated query string, mirroring the Go helper.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .types import TemporalAnnotation

__all__ = [
    "RELATIVE_TIME_RE",
    "RELATIVE_DAY_RE",
    "LAST_WEEK_RE",
    "LAST_WEEKDAY_RE",
    "parse_question_date",
    "resolve_relative_day",
    "resolve_last_week",
    "resolve_relative_time",
    "resolve_last_weekday",
    "annotate_ordering",
    "annotate",
]

# Regexes ported verbatim from Go. The Go source uses ``(?i)`` as the
# case-insensitive flag; in Python we pass ``re.IGNORECASE`` explicitly.
RELATIVE_TIME_RE: re.Pattern[str] = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(day|days|week|weeks|month|months)\s+ago\b",
    re.IGNORECASE,
)

RELATIVE_DAY_RE: re.Pattern[str] = re.compile(
    r"\b(yesterday|today)\b",
    re.IGNORECASE,
)

LAST_WEEK_RE: re.Pattern[str] = re.compile(
    r"\blast\s+week\b",
    re.IGNORECASE,
)

LAST_WEEKDAY_RE: re.Pattern[str] = re.compile(
    r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    re.IGNORECASE,
)

_WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_RELATIVE_TIME_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d",
)


def parse_question_date(value: str) -> datetime:
    """Parse an anchor string into a UTC datetime.

    Accepts the same surface forms as Go's ``parseQuestionDate``. Raises
    :class:`ValueError` on unparseable input.
    """

    value = value.strip()
    if not value:
        raise ValueError("empty date")
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return dt.replace(tzinfo=timezone.utc)
    raise ValueError(f"unrecognised date format: {value!r}")


def _add_months(anchor: datetime, delta: int) -> datetime:
    """Subtract or add ``delta`` months, rolling forward on overflow.

    Mirrors the JavaScript / Go ``setUTCMonth`` semantics called out in
    the spec: if the target month has fewer days than the anchor day, the
    date rolls forward into the next month.
    """

    year = anchor.year
    month = anchor.month + delta
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1

    # Start with the anchor day and allow it to overflow into the next
    # month. ``timedelta`` handles the overflow carry after we clamp to
    # the month maximum and then add the remainder as extra days.
    day = anchor.day
    try:
        return anchor.replace(year=year, month=month, day=day)
    except ValueError:
        # Day exceeds this month's length: clamp to month end and carry.
        # Determine the last day of the target month by stepping to the
        # first of next month and subtracting one day.
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year += 1
        last_day = (
            datetime(next_year, next_month, 1, tzinfo=timezone.utc)
            - timedelta(days=1)
        ).day
        overflow = day - last_day
        clamped = anchor.replace(year=year, month=month, day=last_day)
        return clamped + timedelta(days=overflow)


def resolve_relative_time(text: str, anchor: datetime) -> TemporalAnnotation | None:
    """Resolve ``N days|weeks|months ago`` against ``anchor`` (UTC).

    Returns ``None`` if no match is found. If multiple matches appear,
    the first one wins — the parser layer can call again on the
    residual string if it needs to accumulate annotations.
    """

    match = RELATIVE_TIME_RE.search(text)
    if match is None:
        return None

    n = _parse_relative_time_count(match.group(1))
    if n is None:
        return None
    unit = match.group(2).lower()

    if unit.startswith("day"):
        resolved = anchor - timedelta(days=n)
    elif unit.startswith("week"):
        resolved = anchor - timedelta(days=n * 7)
    elif unit.startswith("month"):
        resolved = _add_months(anchor, -n)
    else:
        return None

    # Half-open interval from the resolved date through the anchor.
    return TemporalAnnotation(
        range_start=resolved,
        range_end=anchor,
        recogniser="relative",
    )


def _parse_relative_time_count(raw: str) -> int | None:
    trimmed = raw.strip().lower()
    try:
        return int(trimmed)
    except ValueError:
        return _RELATIVE_TIME_NUMBER_WORDS.get(trimmed)


def resolve_last_weekday(text: str, anchor: datetime) -> TemporalAnnotation | None:
    """Resolve ``last <weekday>`` against ``anchor`` (UTC).

    The anchor day itself is never returned: ``last monday`` on a Monday
    resolves to seven days earlier. Returns ``None`` when no match.
    """

    match = LAST_WEEKDAY_RE.search(text)
    if match is None:
        return None

    target = _WEEKDAY_MAP.get(match.group(1).lower())
    if target is None:
        return None

    day = anchor
    for _ in range(7):
        day = day - timedelta(days=1)
        if day.weekday() == target:
            return TemporalAnnotation(
                range_start=day,
                range_end=day + timedelta(days=1),
                recogniser="last_weekday",
            )
    return None


def resolve_relative_day(text: str, anchor: datetime) -> TemporalAnnotation | None:
    """Resolve ``today`` and ``yesterday`` against ``anchor`` (UTC)."""

    match = RELATIVE_DAY_RE.search(text)
    if match is None:
        return None

    day = match.group(1).lower()
    if day == "today":
        start = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    elif day == "yesterday":
        start = (anchor - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        return None

    return TemporalAnnotation(
        range_start=start,
        range_end=start + timedelta(days=1),
        recogniser="relative_day",
    )


def resolve_last_week(text: str, anchor: datetime) -> TemporalAnnotation | None:
    """Resolve bare ``last week`` to the prior seven-day window."""

    if LAST_WEEK_RE.search(text) is None:
        return None

    end = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=7)
    return TemporalAnnotation(
        range_start=start,
        range_end=end,
        recogniser="last_week",
    )


_FIRST_HINTS = ("first", "earlier", "before")
_LATEST_HINTS = ("most recent", "latest", "last time")


def annotate_ordering(text: str) -> str:
    """Append an ordering hint if ``text`` contains a known trigger word.

    Mirrors the Go ``annotateOrdering`` helper. Case-insensitive substring
    matching; the original text is returned verbatim when no trigger is
    present.
    """

    lower = text.lower()
    for trigger in _FIRST_HINTS:
        if trigger in lower:
            return text + " [Note: look for the earliest dated event]"
    for trigger in _LATEST_HINTS:
        if trigger in lower:
            return text + " [Note: look for the most recently dated event]"
    return text


def annotate(text: str, anchor: datetime | None) -> TemporalAnnotation | None:
    """Run the two explicit recognisers in order and return the first hit.

    ``anchor`` must be UTC. If ``None``, no annotation is produced.
    """

    if anchor is None:
        return None
    relative = resolve_relative_time(text, anchor)
    if relative is not None:
        return relative
    relative_day = resolve_relative_day(text, anchor)
    if relative_day is not None:
        return relative_day
    last_week = resolve_last_week(text, anchor)
    if last_week is not None:
        return last_week
    return resolve_last_weekday(text, anchor)
