# SPDX-License-Identifier: Apache-2.0
"""Temporal recogniser tests — ported from ``query/temporal_test.go``.

The Python SDK exposes resolved dates as :class:`TemporalAnnotation`
rather than the Go ``TemporalExpansion``; tests assert on the resolved
``range_start`` / ``range_end`` rather than on the formatted hint string,
but reuse the same anchor dates so parity with Go is mechanical to check.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jeffs_brain_memory.query.temporal import (
    annotate,
    annotate_ordering,
    parse_question_date,
    resolve_last_weekday,
    resolve_relative_time,
)

ANCHOR = datetime(2023, 4, 10, 23, 7, tzinfo=timezone.utc)  # Monday


def _d(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def test_two_weeks_ago() -> None:
    # 2023/04/10 - 2 weeks = 2023/03/27.
    annotation = resolve_relative_time("What did we discuss 2 weeks ago?", ANCHOR)
    assert annotation is not None
    assert annotation.recogniser == "relative"
    assert annotation.range_start == ANCHOR - timedelta(weeks=2)
    assert annotation.range_end == ANCHOR


def test_three_days_ago() -> None:
    annotation = resolve_relative_time("Who messaged me 3 days ago?", ANCHOR)
    assert annotation is not None
    assert annotation.range_start == ANCHOR - timedelta(days=3)
    assert annotation.range_end == ANCHOR


def test_one_month_ago() -> None:
    # 2023/04/10 minus one month = 2023/03/10 under JS/Go setUTCMonth semantics.
    annotation = resolve_relative_time("What happened 1 month ago?", ANCHOR)
    assert annotation is not None
    assert annotation.range_start == _d(2023, 3, 10).replace(
        hour=ANCHOR.hour, minute=ANCHOR.minute
    )


def test_last_saturday() -> None:
    # 2023/04/10 (Mon) → last Saturday = 2023/04/08.
    annotation = resolve_last_weekday("What was said last Saturday?", ANCHOR)
    assert annotation is not None
    assert annotation.recogniser == "last_weekday"
    assert annotation.range_start.date() == _d(2023, 4, 8).date()
    assert annotation.range_end - annotation.range_start == timedelta(days=1)


def test_last_monday_on_monday_returns_seven_days_earlier() -> None:
    # Spec: the anchor day itself is never returned.
    annotation = resolve_last_weekday("last Monday", ANCHOR)
    assert annotation is not None
    assert annotation.range_start.date() == _d(2023, 4, 3).date()


def test_no_temporal_reference_returns_none() -> None:
    assert resolve_relative_time("What is the capital of France?", ANCHOR) is None
    assert resolve_last_weekday("What is the capital of France?", ANCHOR) is None


def test_annotate_orders_first_then_weekday() -> None:
    # ``annotate`` runs the relative recogniser first; if it matches we do
    # not also consult ``last <weekday>``.
    got = annotate("Compare 2 weeks ago vs last Friday", ANCHOR)
    assert got is not None
    assert got.recogniser == "relative"


def test_annotate_falls_back_to_weekday() -> None:
    got = annotate("What did we say last Wednesday?", ANCHOR)
    assert got is not None
    assert got.recogniser == "last_weekday"


def test_annotate_returns_none_without_anchor() -> None:
    assert annotate("2 weeks ago", None) is None


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("2023/04/10 (Mon) 23:07", _d(2023, 4, 10).replace(hour=23, minute=7)),
        ("2023/04/10 23:07", _d(2023, 4, 10).replace(hour=23, minute=7)),
        ("2023/04/10", _d(2023, 4, 10)),
        ("2023-04-10", _d(2023, 4, 10)),
    ],
)
def test_parse_question_date_accepts_all_spec_formats(text: str, want: datetime) -> None:
    assert parse_question_date(text) == want


@pytest.mark.parametrize(
    "value",
    ["", "not a date", "10/04/2023", "April 10, 2023"],
)
def test_parse_question_date_rejects_unknown_formats(value: str) -> None:
    with pytest.raises(ValueError):
        parse_question_date(value)


def test_annotate_ordering_first() -> None:
    assert "earliest dated event" in annotate_ordering(
        "When did we first discuss the project?"
    )


def test_annotate_ordering_most_recent() -> None:
    assert "most recently dated event" in annotate_ordering(
        "What was the most recent update?"
    )


def test_annotate_ordering_no_trigger() -> None:
    text = "What colour is the sky?"
    assert annotate_ordering(text) == text


def test_month_end_overflow_rolls_forward() -> None:
    # Spec: anchor 2026-03-31, one month ago → 2026-03-03 (overflow of
    # three days from February into March). Reuses the JavaScript /
    # Go setUTCMonth behaviour.
    anchor = datetime(2026, 3, 31, tzinfo=timezone.utc)
    got = resolve_relative_time("1 month ago", anchor)
    assert got is not None
    assert got.range_start == datetime(2026, 3, 3, tzinfo=timezone.utc)
