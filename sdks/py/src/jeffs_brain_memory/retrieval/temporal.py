# SPDX-License-Identifier: Apache-2.0
"""Temporal helpers shared by retrieval and augmented ask rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re

from ..query.temporal import (
    parse_question_date,
    resolve_last_week,
    resolve_last_weekday,
    resolve_relative_day,
    resolve_relative_time,
)

__all__ = [
    "augment_query_with_temporal",
    "build_bm25_query_plan",
    "compile_bm25_fanout_query",
    "derive_sub_queries",
    "question_tokens",
    "resolved_temporal_hint_line",
    "temporal_query_variants",
]

QUESTION_TOKEN_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "what",
        "who", "when", "where", "why", "how",
        "did", "does", "was", "were", "are",
        "you", "your", "about", "this", "that",
        "have", "has", "had", "from", "into",
        "than", "then", "them", "they", "their",
    }
)
MAX_BM25_FANOUT_QUERIES = 4
MAX_DERIVED_SUB_QUERIES = 2
PHRASE_PROBE_MIN_TOKENS = 2
PHRASE_PROBE_MAX_TOKENS = 4
PHRASE_PROBE_CONNECTORS: frozenset[str] = frozenset({"and", "or", "plus"})
PHRASE_PROBE_BOUNDARY_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "plus",
        "for", "with", "what", "who", "when", "where", "why", "how",
        "did", "does", "do", "was", "were", "is", "are", "am",
        "you", "your", "about", "this", "that", "these", "those",
        "have", "has", "had", "from", "into", "than", "then", "them", "they", "their",
        "i", "me", "my", "we", "our", "us", "it", "if", "to", "of", "on", "in", "at", "by",
        "amount", "total", "all", "list",
        "finally", "decided", "decide", "wondering", "wonder",
        "remembered", "remember", "thinking", "back", "previous", "conversation",
        "can", "could", "would", "should", "remind", "follow", "specific", "exact",
        "spent", "spend", "bought", "buy", "ordered", "order",
        "purchased", "purchase", "paid", "pay", "submitted", "submit",
        "many", "much", "long",
        "last", "today", "yesterday", "tomorrow", "week", "month", "year",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }
)
PHRASE_PROBE_TRIM_WORDS: frozenset[str] = frozenset({"many", "much", "long"})
ACTION_DATE_PROBE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsubmit(?:ted)?\b", re.IGNORECASE), "submission date"),
    (re.compile(r"\bbook(?:ed|ing)?\b", re.IGNORECASE), "booking date"),
    (
        re.compile(r"\b(?:buy|bought|purchase(?:d)?|order(?:ed)?)\b", re.IGNORECASE),
        "purchase date",
    ),
    (re.compile(r"\bjoin(?:ed)?\b", re.IGNORECASE), "join date"),
    (re.compile(r"\b(?:start(?:ed)?|begin|began)\b", re.IGNORECASE), "start date"),
    (re.compile(r"\b(?:finish(?:ed)?|complete(?:d)?)\b", re.IGNORECASE), "completion date"),
    (re.compile(r"\baccept(?:ed|ance)?\b", re.IGNORECASE), "acceptance date"),
)
ACTION_DATE_FOCUS_SKIP_WORDS: frozenset[str] = frozenset(
    {
        "accept", "accepted", "acceptance",
        "begin", "began", "book", "booked", "booking",
        "buy", "bought", "complete", "completed", "completion",
        "date", "finish", "finished",
        "join", "joined", "order", "ordered",
        "purchase", "purchased", "start", "started",
        "submit", "submitted", "submission",
    }
)
INSPIRATION_QUERY_HINTS: tuple[str, ...] = (
    "inspiration",
    "inspired",
    "ideas",
    "stuck",
    "uninspired",
)
INSPIRATION_FOCUS_SKIP_WORDS: frozenset[str] = frozenset(
    {
        "find", "finding", "fresh", "idea", "ideas",
        "inspiration", "inspired", "new", "stuck", "uninspired",
    }
)
LOW_SIGNAL_PHRASE_PROBE_WORDS: frozenset[str] = frozenset(
    {
        "after", "before", "day", "days", "event", "events",
        "first", "happen", "happened", "month", "months",
        "second", "third", "time", "times", "week", "weeks",
        "year", "years",
    }
)
ENUMERATION_OR_TOTAL_QUERY_RE = re.compile(
    r"\b(?:how many|count|total|in total|sum|add up|list|what are all)\b",
    re.IGNORECASE,
)
SPECIFIC_RECOMMENDATION_QUERY_RE = re.compile(r"\b(?:specific|exact)\b", re.IGNORECASE)
MONEY_EVENT_QUERY_RE = re.compile(
    r"\b(?:spent|spend|cost|costed|paid|pay)\b", re.IGNORECASE
)
HEAD_BIGRAM_LAST_TOKENS: frozenset[str] = frozenset(
    {
        "development",
        "item",
        "items",
        "language",
        "languages",
        "product",
        "products",
    }
)


@dataclass(frozen=True, slots=True)
class BM25QueryPlan:
    queries: list[str]
    phrase_probes: list[str]


def _resolved_date_hints(question: str, question_date: str) -> list[str]:
    trimmed = question_date.strip()
    if not trimmed:
        return []
    try:
        anchor = parse_question_date(trimmed)
    except ValueError:
        return []

    hints: list[str] = []
    relative = resolve_relative_time(question, anchor)
    if relative is not None:
        hints.append(relative.range_start.strftime("%Y/%m/%d"))
    relative_day = resolve_relative_day(question, anchor)
    if relative_day is not None:
        hints.append(relative_day.range_start.strftime("%Y/%m/%d"))
    last_week = resolve_last_week(question, anchor)
    if last_week is not None:
        day = last_week.range_start
        while day < last_week.range_end:
            hints.append(day.strftime("%Y/%m/%d"))
            day += timedelta(days=1)
    weekday = resolve_last_weekday(question, anchor)
    if weekday is not None:
        hints.append(weekday.range_start.strftime("%Y/%m/%d"))

    seen: set[str] = set()
    out: list[str] = []
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        out.append(hint)
    return out


def resolved_date_hints(question: str, question_date: str) -> list[str]:
    return _resolved_date_hints(question, question_date)


def augment_query_with_temporal(question: str, question_date: str) -> str:
    """Append concrete date tokens for temporal questions when possible."""
    hints = _resolved_date_hints(question, question_date)
    if not hints:
        return question

    tokens: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        for variant in (f'"{hint}"', f'"{hint.replace("/", "-")}"'):
            if variant in seen:
                continue
            seen.add(variant)
            tokens.append(variant)
    if not tokens:
        return question
    return f"{question} {' '.join(tokens)}"


def resolved_temporal_hint_line(question: str, question_date: str) -> str | None:
    """Render the reader-facing resolved temporal hint line."""
    hints = _resolved_date_hints(question, question_date)
    if not hints:
        return None
    return f"[Resolved temporal references: {', '.join(hints)}]"


def question_tokens(question: str) -> list[str]:
    """Return lowercased, deduplicated, non-stop-word query tokens."""
    if not question.strip():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in question.lower().split():
        token = raw.strip(""".,;:!?"'()[]{}<>""")
        if len(token) < 3:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        if token in QUESTION_TOKEN_STOP_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _phrase_probe_tokens(question: str) -> list[str]:
    if not question.strip():
        return []
    out: list[str] = []
    for raw in question.lower().split():
        token = raw.strip(""".,;:!?"'()[]{}<>""")
        if not token:
            continue
        out.append(token)
    return out


def _join_phrase_tokens(tokens: list[str]) -> str | None:
    if len(tokens) < PHRASE_PROBE_MIN_TOKENS:
        return None
    return " ".join(tokens)


def _collect_left_phrase(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    collected: list[str] = []
    for token in reversed(tokens):
        if token in PHRASE_PROBE_BOUNDARY_WORDS:
            if collected:
                break
            continue
        if len(token) < 2 or any(ch.isdigit() for ch in token):
            if collected:
                break
            continue
        collected.append(token)
        if len(collected) >= PHRASE_PROBE_MAX_TOKENS:
            break
    collected.reverse()
    return _join_phrase_tokens(collected)


def _collect_right_phrase(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    collected: list[str] = []
    for token in tokens:
        if token in PHRASE_PROBE_BOUNDARY_WORDS:
            if collected:
                break
            continue
        if len(token) < 2 or any(ch.isdigit() for ch in token):
            if collected:
                break
            continue
        collected.append(token)
        if len(collected) >= PHRASE_PROBE_MAX_TOKENS:
            break
    return _join_phrase_tokens(collected)


def _derive_phrase_probes(question: str) -> list[str]:
    tokens = _phrase_probe_tokens(question)
    if len(tokens) < PHRASE_PROBE_MIN_TOKENS:
        return []

    out: list[str] = []
    seen: set[str] = set()
    def append_phrase(phrase: str | None) -> bool:
        if phrase is None:
            return False
        trimmed = phrase.strip()
        if not trimmed or trimmed in seen:
            return False
        seen.add(trimmed)
        out.append(trimmed)
        return len(out) >= MAX_DERIVED_SUB_QUERIES

    for index, token in enumerate(tokens):
        if token not in PHRASE_PROBE_CONNECTORS:
            continue
        left = _collect_left_phrase(tokens[:index])
        if append_phrase(left):
            return out
        right = _collect_right_phrase(tokens[index + 1 :])
        if append_phrase(right):
            return out
    for phrase in _derive_boundary_span_probes(tokens):
        if append_phrase(phrase):
            return out
    return out


def _filtered_phrase_probes(question: str) -> list[str]:
    return [
        phrase
        for phrase in _derive_phrase_probes(question)
        if _filter_question_tokens(phrase, LOW_SIGNAL_PHRASE_PROBE_WORDS)
    ]


def _derive_boundary_span_probes(tokens: list[str]) -> list[str]:
    if len(tokens) < PHRASE_PROBE_MIN_TOKENS:
        return []

    out: list[str] = []
    segment: list[str] = []

    def flush() -> None:
        nonlocal segment
        phrase = _best_segment_phrase(segment)
        if phrase is not None:
            out.append(phrase)
        segment = []

    for token in tokens:
        if (
            not token
            or token in PHRASE_PROBE_BOUNDARY_WORDS
            or len(token) < 2
            or any(ch.isdigit() for ch in token)
        ):
            flush()
            continue
        segment.append(token)
    flush()

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in out:
        if phrase in seen:
            continue
        seen.add(phrase)
        deduped.append(phrase)
    return deduped


def _best_segment_phrase(tokens: list[str]) -> str | None:
    trimmed = _trim_phrase_probe_tokens(tokens)
    if len(trimmed) < PHRASE_PROBE_MIN_TOKENS:
        return None
    if len(trimmed) <= PHRASE_PROBE_MAX_TOKENS:
        return _join_phrase_tokens(trimmed)

    for size in range(PHRASE_PROBE_MAX_TOKENS, PHRASE_PROBE_MIN_TOKENS - 1, -1):
        best: str | None = None
        best_score = -1
        for start in range(0, len(trimmed) - size + 1):
            candidate = _trim_phrase_probe_tokens(trimmed[start : start + size])
            if len(candidate) < PHRASE_PROBE_MIN_TOKENS:
                continue
            score = _phrase_probe_score(candidate)
            if score > best_score:
                best_score = score
                best = _join_phrase_tokens(candidate)
        if best is not None:
            return best
    return None


def _trim_phrase_probe_tokens(tokens: list[str]) -> list[str]:
    start = 0
    while start < len(tokens) and tokens[start] in PHRASE_PROBE_TRIM_WORDS:
        start += 1
    return tokens[start:]


def _phrase_probe_score(tokens: list[str]) -> int:
    return len(tokens) * 100 + sum(len(token) for token in tokens)


def derive_sub_queries(question: str) -> list[str]:
    """Return phrase probes first, then strongest-token fallbacks."""
    out: list[str] = []
    seen: set[str] = {question.strip().lower()}
    inspiration_query = any(
        hint in question.strip().lower() for hint in INSPIRATION_QUERY_HINTS
    )

    for probe in _derive_specific_recommendation_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_money_focus_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_action_date_context_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_inspiration_source_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_action_date_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    phrases = _filtered_phrase_probes(question)
    for phrase in phrases:
        if inspiration_query and not _filter_question_tokens(
            phrase, INSPIRATION_FOCUS_SKIP_WORDS
        ):
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    token_source = " ".join(phrases) if phrases else question
    tokens = sorted(question_tokens(token_source), key=len, reverse=True)
    if len(tokens) < 2:
        return out
    for token in tokens:
        if inspiration_query and token in INSPIRATION_FOCUS_SKIP_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            break
    return out


def _derive_priority_sub_queries(question: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = {question.strip().lower()}

    for probe in _derive_specific_recommendation_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_money_focus_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_action_date_context_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for probe in _derive_inspiration_source_probes(question):
        if probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out

    for phrase in _filtered_phrase_probes(question):
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out
    return out


def _derive_action_date_context_probes(question: str) -> list[str]:
    action_date_probes = _derive_action_date_probes(question)
    if not action_date_probes:
        return []
    focuses = _derive_action_date_focuses(
        _filter_question_tokens(question, ACTION_DATE_FOCUS_SKIP_WORDS)
    )
    if not focuses:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for probe in action_date_probes:
        for focus in focuses:
            candidate = f"{focus} {probe}".strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= MAX_DERIVED_SUB_QUERIES:
                return out
    return out


def _derive_action_date_focuses(tokens: list[str]) -> list[str]:
    if not tokens:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def append_window(start: int, end: int) -> None:
        if start < 0 or end > len(tokens) or start >= end:
            return
        candidate = " ".join(tokens[start:end]).strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        out.append(candidate)

    if len(tokens) == 1:
        append_window(0, 1)
        return out
    if len(tokens) == 2:
        append_window(0, 2)
        return out
    if len(tokens) == 3:
        append_window(1, 3)
        append_window(0, 2)
        return out
    append_window(len(tokens) - 2, len(tokens))
    append_window(0, 2)
    return out


def _derive_inspiration_source_probes(question: str) -> list[str]:
    lowered = question.strip().lower()
    if not lowered or not any(hint in lowered for hint in INSPIRATION_QUERY_HINTS):
        return []
    tokens = _filter_question_tokens(question, INSPIRATION_FOCUS_SKIP_WORDS)
    if not tokens:
        return []
    focus = tokens[-1]
    if not focus:
        return []
    return [f"{focus} social media tutorials"]


def _derive_specific_recommendation_probes(question: str) -> list[str]:
    lowered = question.strip().lower()
    if not lowered or SPECIFIC_RECOMMENDATION_QUERY_RE.search(lowered) is None:
        return []
    if (
        re.search(r"\brecommend(?:ed)?\b", lowered) is None
        and re.search(r"\bremind me\b", lowered) is None
    ):
        return []
    if "back-end" in lowered and re.search(r"\blanguages?\b", lowered) is not None:
        return ["back-end programming language", "back-end development"]
    return []


def _derive_money_focus_probes(question: str) -> list[str]:
    lowered = question.strip().lower()
    if (
        not lowered
        or ENUMERATION_OR_TOTAL_QUERY_RE.search(lowered) is None
        or MONEY_EVENT_QUERY_RE.search(lowered) is None
    ):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for phrase in _filtered_phrase_probes(question):
        candidate = _money_focus_probe_from_phrase(phrase)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            return out
    return out


def _money_focus_probe_from_phrase(phrase: str) -> str:
    edge = _derive_phrase_edge_focus(phrase)
    if edge:
        return edge
    head = _derive_phrase_head_focus(phrase)
    if not head:
        return ""
    if head != phrase and len(phrase.split()) <= 2:
        return f"{head} cost"
    return phrase


def _derive_phrase_edge_focus(phrase: str) -> str:
    tokens = [token for token in phrase.strip().lower().split() if token]
    if len(tokens) < 3:
        return ""
    first = tokens[0]
    last = tokens[-1]
    if "-" not in first or last not in HEAD_BIGRAM_LAST_TOKENS:
        return ""
    return f"{first} {last}"


def _derive_phrase_head_focus(phrase: str) -> str:
    tokens = [token for token in phrase.strip().lower().split() if token]
    if not tokens:
        return ""
    last = tokens[-1]
    if len(tokens) >= 2 and last in HEAD_BIGRAM_LAST_TOKENS:
        return " ".join(tokens[-2:])
    return last


def _derive_action_date_probes(question: str) -> list[str]:
    lowered = question.strip().lower()
    if not lowered or "when" not in lowered:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pattern, probe in ACTION_DATE_PROBE_RULES:
        if pattern.search(lowered) is None or probe in seen:
            continue
        seen.add(probe)
        out.append(probe)
        if len(out) >= MAX_DERIVED_SUB_QUERIES:
            break
    return out


def _filter_question_tokens(question: str, skip: frozenset[str]) -> list[str]:
    return [token for token in question_tokens(question) if token not in skip]


def build_bm25_query_plan(question: str, question_date: str) -> BM25QueryPlan:
    base = augment_query_with_temporal(question, question_date)
    queries: list[str] = []
    seen: set[str] = set()
    priority_queries = _derive_priority_sub_queries(question)
    candidates = (
        [*priority_queries]
        if _should_use_priority_only_bm25(question) and len(priority_queries) >= 2
        else [*priority_queries, question, base, *derive_sub_queries(question)]
    )
    for candidate in candidates:
        trimmed = " ".join(candidate.split())
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        queries.append(trimmed)
        if len(queries) >= MAX_BM25_FANOUT_QUERIES:
            break
    return BM25QueryPlan(queries=queries, phrase_probes=_derive_phrase_probes(question))


def _should_use_priority_only_bm25(question: str) -> bool:
    lowered = question.strip().lower()
    return bool(_derive_action_date_context_probes(question)) or (
        len(_filtered_phrase_probes(question)) >= 2 and " and " in lowered
    )


def compile_bm25_fanout_query(query: str, phrase_probes: list[str]) -> str:
    trimmed = " ".join(query.split())
    if (
        trimmed
        and trimmed in phrase_probes
        and " " in trimmed
        and '"' not in trimmed
    ):
        return f'"{trimmed}"'
    return trimmed


def temporal_query_variants(question: str, question_date: str) -> list[str]:
    """Return deduplicated BM25 fanout variants for ``question``."""
    return build_bm25_query_plan(question, question_date).queries
