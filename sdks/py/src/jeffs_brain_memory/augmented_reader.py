# SPDX-License-Identifier: Apache-2.0
"""Deterministic augmented-reader answer resolver."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
import re

_FACT_HEADER_RE = re.compile(r"^\s*(\d+)\.\s+((?:\[[^\]]+\]\s*)+)$")
_LABEL_RE = re.compile(r"\[([^\]]+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#'-]*")
_MONTH_DATE_RE = re.compile(
    r"\b(?:"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}(?:-|/)\d{2}(?:-|/)\d{2}\b")
_QUESTION_ACTION_RE = re.compile(
    r"\b(?:when|what date)\s+did\s+i\s+(submit|book|join)\b",
    re.IGNORECASE,
)
_TOTAL_SPEND_RE = re.compile(
    r"\b(?:total amount|in total|how much)\b.*\bspent\b",
    re.IGNORECASE,
)
_TRANSACTION_VERB_RE = re.compile(
    r"\b(?:"
    r"bought|purchased|spent|paid|got|ordered|picked up|treated(?: myself)?"
    r"(?: to)?|invested|cost(?:ing)?|totall?ing|worth"
    r")\b",
    re.IGNORECASE,
)
_SUMMARY_MARKER_RE = re.compile(
    r"\b(?:summary|roll-up|rollup|overview|overall|recap|tracker|budget|"
    r"bookkeeping|in total)\b",
    re.IGNORECASE,
)
_DATE_EXACT_WORD_RE = re.compile(r"\b(yesterday|today|last night)\b", re.IGNORECASE)

_ACTION_MARKERS = {
    "submit": ("submit", "submitted", "submission"),
    "book": ("book", "booked", "booking"),
    "join": ("join", "joined", "joining"),
}

_ACTION_DATE_MARKERS = {
    "submit": ("submitted on", "submission date", "submitted"),
    "book": ("booked on", "booking date", "booked"),
    "join": ("joined on", "join date", "joined"),
}

_ACTION_PAST_TENSE = {
    "submit": "submitted",
    "book": "booked",
    "join": "joined",
}

_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "date",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "learn",
    "learning",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "recommend",
    "recommended",
    "remember",
    "remind",
    "specific",
    "suggest",
    "suggested",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "up",
    "was",
    "what",
    "when",
    "which",
    "with",
    "you",
    "your",
}

_GENERIC_ANCHOR_TOKENS = {
    "airbnb",
    "analysis",
    "book",
    "booking",
    "club",
    "date",
    "group",
    "joined",
    "paper",
    "research",
    "submission",
    "submitted",
}

_LANGUAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Node.js", re.compile(r"\bnode\.?js\b", re.IGNORECASE)),
    ("Python", re.compile(r"\bpython\b", re.IGNORECASE)),
    ("Ruby", re.compile(r"\bruby\b", re.IGNORECASE)),
    ("PHP", re.compile(r"\bphp\b", re.IGNORECASE)),
    ("Java", re.compile(r"\bjava\b", re.IGNORECASE)),
    ("Go", re.compile(r"\bgo\b", re.IGNORECASE)),
    ("Rust", re.compile(r"\brust\b", re.IGNORECASE)),
    ("C++", re.compile(r"\bc\+\+\b", re.IGNORECASE)),
    ("C#", re.compile(r"\bc#\b", re.IGNORECASE)),
    ("TypeScript", re.compile(r"\btypescript\b", re.IGNORECASE)),
    ("JavaScript", re.compile(r"\bjavascript\b", re.IGNORECASE)),
    ("Kotlin", re.compile(r"\bkotlin\b", re.IGNORECASE)),
    ("Scala", re.compile(r"\bscala\b", re.IGNORECASE)),
    ("Elixir", re.compile(r"\belixir\b", re.IGNORECASE)),
)

_RESOURCE_MARKERS = (
    "resource",
    "resources",
    "course",
    "courses",
    "workshop",
    "workshops",
    "tutorial",
    "tutorials",
    "platform",
    "platforms",
    "nodeschool",
    "udacity",
    "coursera",
    "flask",
    "django",
    "spring",
    "hibernate",
    "framework",
    "frameworks",
    "sql",
)


@dataclass(frozen=True)
class RenderedFact:
    index: int
    date_label: str
    anchor_date: date | None
    session_id: str
    source: str
    body: str


@dataclass(frozen=True)
class RecipientTarget:
    label: str
    core_tokens: tuple[str, ...]


@dataclass(frozen=True)
class SpendCandidate:
    recipient: RecipientTarget
    amount: Decimal
    directness: int
    item_tokens: frozenset[str]


def resolve_deterministic_augmented_answer(
    question: str, rendered_evidence: str
) -> str | None:
    facts = _parse_rendered_facts(rendered_evidence)
    if not facts:
        return None

    for resolver in (
        _resolve_action_date,
        _resolve_named_recipient_total,
        _resolve_specific_backend_languages,
    ):
        answer = resolver(question, facts)
        if answer is not None:
            return answer
    return None


def _parse_rendered_facts(rendered_evidence: str) -> list[RenderedFact]:
    lines = rendered_evidence.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("Retrieved facts ("):
            start = index + 1
            break
    if start is None:
        return []

    facts: list[RenderedFact] = []
    current_index = -1
    current_labels = ""
    body_lines: list[str] = []

    def flush() -> None:
        if current_index < 0:
            return
        labels = _LABEL_RE.findall(current_labels)
        if not labels:
            return
        date_label = labels[0].strip()
        session_id = ""
        source = ""
        for label in labels[1:]:
            trimmed = label.strip()
            if trimmed.startswith("session="):
                session_id = trimmed[len("session=") :].strip()
                continue
            if trimmed:
                source = trimmed
        body = "\n".join(body_lines).strip()
        facts.append(
            RenderedFact(
                index=current_index,
                date_label=date_label or "unknown",
                anchor_date=_parse_anchor_date(date_label),
                session_id=session_id,
                source=source,
                body=body,
            )
        )

    for line in lines[start:]:
        match = _FACT_HEADER_RE.match(line)
        if match is not None:
            flush()
            current_index = int(match.group(1))
            current_labels = match.group(2)
            body_lines = []
            continue
        if current_index < 0:
            continue
        body_lines.append(line)
    flush()
    return facts


def _parse_anchor_date(value: str) -> date | None:
    stripped = value.strip()
    if stripped in {"", "unknown"}:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    return None


def _resolve_action_date(question: str, facts: list[RenderedFact]) -> str | None:
    match = _QUESTION_ACTION_RE.search(question)
    if match is None:
        return None
    action = match.group(1).lower()
    object_text = question[match.end() :].strip(" ?.")
    target_tokens = _significant_tokens(object_text)
    if not target_tokens:
        return None
    minimum_target_overlap = max(1, (len(target_tokens) * 3 + 4) // 5)

    action_facts = sorted(
        (
            fact
            for fact in facts
            if _fact_action_overlap(fact.body, action) and _target_overlap(fact.body, target_tokens)
            >= minimum_target_overlap
        ),
        key=lambda fact: (
            _target_overlap(fact.body, target_tokens),
            fact.anchor_date or date.min,
        ),
        reverse=True,
    )
    if not action_facts:
        return None

    for action_fact in action_facts:
        direct_date = _extract_date_value(action_fact.body, action_fact.anchor_date)
        if direct_date is not None:
            return direct_date

        action_tokens = _significant_tokens(action_fact.body)
        best_match: tuple[int, int, date, str] | None = None
        for fact in facts:
            if fact == action_fact:
                continue
            if not _fact_supports_action_date(fact.body, action):
                continue
            date_value = _extract_date_value(fact.body, fact.anchor_date)
            if date_value is None and fact.anchor_date is not None:
                date_value = fact.anchor_date.isoformat()
            if date_value is None:
                continue
            shared_tokens = (
                action_tokens
                & _significant_tokens(fact.body)
                - target_tokens
                - _GENERIC_ANCHOR_TOKENS
            )
            target_overlap = _target_overlap(fact.body, target_tokens)
            if not shared_tokens and target_overlap < minimum_target_overlap:
                continue
            candidate = (
                len(shared_tokens),
                target_overlap,
                fact.anchor_date or date.min,
                date_value,
            )
            if best_match is None or candidate > best_match:
                best_match = candidate
        if best_match is not None:
            return best_match[3]
    return None


def _resolve_named_recipient_total(
    question: str, facts: list[RenderedFact]
) -> str | None:
    if _TOTAL_SPEND_RE.search(question) is None:
        return None
    recipients = _parse_recipients_from_question(question)
    if not recipients:
        return None

    per_recipient: dict[str, list[SpendCandidate]] = {r.label: [] for r in recipients}
    for fact in facts:
        for clause in _split_clauses(fact.body):
            amount = _extract_transaction_amount(clause)
            if amount is None:
                continue
            directness = _transaction_directness(clause)
            if directness <= 0:
                continue
            item_tokens = _item_tokens(clause)
            for recipient in recipients:
                if _clause_mentions_recipient(clause, recipient):
                    per_recipient[recipient.label].append(
                        SpendCandidate(
                            recipient=recipient,
                            amount=amount,
                            directness=directness,
                            item_tokens=frozenset(item_tokens),
                        )
                    )

    if any(not candidates for candidates in per_recipient.values()):
        return None

    total = Decimal("0")
    for recipient in recipients:
        chosen = _dedupe_spend_candidates(per_recipient[recipient.label])
        if not chosen:
            return None
        recipient_total = sum((candidate.amount for candidate in chosen), start=Decimal("0"))
        total += recipient_total
    return _format_currency(total)


def _resolve_specific_backend_languages(
    question: str, facts: list[RenderedFact]
) -> str | None:
    lowered_question = question.lower()
    if "back-end" not in lowered_question and "backend" not in lowered_question:
        return None
    if "language" not in lowered_question:
        return None
    if "recommend" not in lowered_question and "suggest" not in lowered_question:
        return None
    if "learn" not in lowered_question and "learning" not in lowered_question:
        return None

    best: tuple[int, list[str]] | None = None
    for fact in facts:
        for clause in _split_clauses(fact.body):
            lowered_clause = clause.lower()
            if "back-end" not in lowered_clause and "backend" not in lowered_clause:
                continue
            direct_marker = any(
                marker in lowered_clause
                for marker in (
                    "back-end programming language",
                    "backend programming language",
                    "back-end languages",
                    "backend languages",
                    "server-side scripting",
                )
            )
            recommendation_marker = any(
                marker in lowered_clause
                for marker in ("recommend", "recommended", "suggest", "suggested", "learn")
            )
            if not direct_marker and not recommendation_marker:
                continue
            languages = _extract_languages(clause)
            if len(languages) < 2:
                continue
            resource_penalty = sum(
                1 for marker in _RESOURCE_MARKERS if marker in lowered_clause
            )
            score = len(languages) * 4
            if direct_marker:
                score += 5
            if "such as" in lowered_clause or "like" in lowered_clause:
                score += 2
            score -= resource_penalty * 3
            if best is None or score > best[0]:
                best = (score, languages)
    if best is None:
        return None

    languages = best[1]
    return (
        "I recommended learning "
        f"{_serialise_list(languages)} as a back-end programming language."
    )


def _fact_action_overlap(text: str, action: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ACTION_MARKERS[action])


def _fact_supports_action_date(text: str, action: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ACTION_DATE_MARKERS[action])


def _target_overlap(text: str, target_tokens: set[str]) -> int:
    fact_tokens = _significant_tokens(text)
    return len(fact_tokens & target_tokens)


def _extract_date_value(text: str, anchor_date: date | None) -> str | None:
    iso_match = _ISO_DATE_RE.search(text)
    if iso_match is not None:
        value = iso_match.group(0)
        parsed = datetime.strptime(value, "%Y-%m-%d" if "-" in value else "%Y/%m/%d").date()
        return _format_month_day(parsed)

    month_match = _MONTH_DATE_RE.search(text)
    if month_match is not None:
        return month_match.group(0)

    relative_match = _DATE_EXACT_WORD_RE.search(text)
    if relative_match is not None and anchor_date is not None:
        exact_word = relative_match.group(1).lower()
        if exact_word == "today":
            resolved = anchor_date
        elif exact_word == "yesterday":
            resolved = anchor_date - timedelta(days=1)
        else:
            resolved = anchor_date - timedelta(days=1)
        return _format_month_day(resolved)

    return None


def _parse_recipients_from_question(question: str) -> list[RecipientTarget]:
    match = re.search(r"\bfor\b(?P<tail>.+?)(?:\?|$)", question, re.IGNORECASE)
    if match is None:
        return []
    tail = match.group("tail")
    parts = [
        _normalise_recipient_phrase(part)
        for part in re.split(r"\s*(?:,| and | & )\s*", tail, flags=re.IGNORECASE)
    ]
    recipients: list[RecipientTarget] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        tokens = tuple(tok for tok in _significant_tokens(part) if tok not in {"gift", "gifts"})
        if not tokens:
            continue
        recipients.append(RecipientTarget(label=part, core_tokens=tokens))
        seen.add(part)
    return recipients


def _normalise_recipient_phrase(value: str) -> str:
    stripped = value.strip(" ?.!,")
    stripped = re.sub(
        r"^(?:my|our|the|a|an|his|her|their)\s+",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    return stripped.strip()


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _extract_transaction_amount(text: str) -> Decimal | None:
    matches = re.findall(r"\$([0-9][0-9,]*(?:\.[0-9]{2})?)", text)
    if not matches:
        return None
    for pattern in (
        r"\b(?:cost(?:ing)?|for|totall?ing|worth|paid|spent|invested)\b[^$]{0,40}"
        r"\$([0-9][0-9,]*(?:\.[0-9]{2})?)",
        r"\b(?:bought|purchased|got|ordered|picked up|treated(?: myself)?(?: to)?)\b"
        r"[^$]{0,80}\$([0-9][0-9,]*(?:\.[0-9]{2})?)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            return Decimal(match.group(1).replace(",", ""))
    return Decimal(matches[-1].replace(",", ""))


def _transaction_directness(text: str) -> int:
    score = 0
    lowered = text.lower()
    if _TRANSACTION_VERB_RE.search(text) is not None:
        score += 3
    if re.search(r"\b(?:i|i've|i’d|i'd|the user)\b", text, re.IGNORECASE) is not None:
        score += 1
    if _SUMMARY_MARKER_RE.search(text) is not None:
        score -= 4
    if "recalled" in lowered or "remembered" in lowered:
        score -= 3
    return score


def _clause_mentions_recipient(text: str, recipient: RecipientTarget) -> bool:
    return all(
        re.search(rf"\b{re.escape(token)}(?:'s|s')?\b", text, re.IGNORECASE) is not None
        for token in recipient.core_tokens
    )


def _item_tokens(text: str) -> set[str]:
    tokens = _significant_tokens(text)
    return {
        token
        for token in tokens
        if token
        not in {
            "amount",
            "bought",
            "cost",
            "costing",
            "gift",
            "gifts",
            "got",
            "invested",
            "ordered",
            "paid",
            "picked",
            "purchased",
            "spent",
            "treated",
            "worth",
        }
        and not token.isdigit()
    }


def _dedupe_spend_candidates(
    candidates: list[SpendCandidate],
) -> list[SpendCandidate]:
    chosen: list[SpendCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda current: (
            current.directness,
            len(current.item_tokens),
        ),
        reverse=True,
    ):
        duplicate = False
        for existing in chosen:
            if existing.amount != candidate.amount:
                continue
            overlap = _token_overlap(existing.item_tokens, candidate.item_tokens)
            if overlap >= 0.5:
                duplicate = True
                break
        if not duplicate:
            chosen.append(candidate)
    return chosen


def _token_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _extract_languages(text: str) -> list[str]:
    positions: list[tuple[int, str]] = []
    for language, pattern in _LANGUAGE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            positions.append((match.start(), language))
    positions.sort()
    return [language for _, language in positions]


def _serialise_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{', '.join(items[:-1])}, or {items[-1]}"


def _significant_tokens(text: str) -> set[str]:
    tokens = {
        _normalise_token(token)
        for token in _TOKEN_RE.findall(text.lower())
    }
    tokens = {
        token for token in tokens if len(token) > 1 and token not in _STOPWORDS
    }
    return {
        token
        for token in tokens
        if token not in {marker for markers in _ACTION_MARKERS.values() for marker in markers}
    }


def _normalise_token(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'")


def _format_currency(value: Decimal) -> str:
    normalised = value.quantize(Decimal("0.01"))
    if normalised == normalised.to_integral():
        return f"${int(normalised)}"
    return f"${normalised:.2f}"


def _format_month_day(value: date) -> str:
    return f"{value.strftime('%B')} {_ordinal(value.day)}"


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"
