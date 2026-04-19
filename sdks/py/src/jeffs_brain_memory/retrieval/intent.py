# SPDX-License-Identifier: Apache-2.0
"""English-only; non-English queries receive base RRF scores without
modification.

When Python lands a retrieval parity patch first, update the sibling
SDK ports to match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .temporal import _derive_priority_sub_queries, _filtered_phrase_probes
from .types import RetrievedChunk

# Case-insensitive only; no ``g``, ``s`` or ``u`` flags (spec rule).
PREFERENCE_QUERY_RE = re.compile(
    r"\b(?:recommend|suggest|recommendation|suggestion|tips?|advice|ideas?|what should i|which should i)\b",
    re.IGNORECASE,
)
ENUMERATION_OR_TOTAL_QUERY_RE = re.compile(
    r"\b(?:how many|count|total|in total|sum|add up|list|what are all)\b",
    re.IGNORECASE,
)
PROPERTY_LOOKUP_QUERY_RE = re.compile(
    r"\b(?:how long is my|what percentage(?: of)?|(?:what was the )?page count|what specific|which specific|what exact|which exact)\b",
    re.IGNORECASE,
)
FIRST_PERSON_PROPERTY_QUERY_RE = re.compile(
    r"\b(?:how often do i|what time do i|what time is my|where do i|where did i|where have i|where am i|where is my|what speed is my|how fast is my|which mode of transport did i|what mode of transport did i|which transport did i|what transport did i)\b",
    re.IGNORECASE,
)
SPECIFIC_RECOMMENDATION_QUERY_RE = re.compile(
    r"\b(?:specific|exact)\b", re.IGNORECASE
)
FIRST_PERSON_FACT_LOOKUP_RE = re.compile(
    r"\b(?:did i|have i|was i|were i)\b",
    re.IGNORECASE,
)
FIRST_PERSON_CONCRETE_QUERY_RE = re.compile(
    r"\b(?:my|me|i)\b",
    re.IGNORECASE,
)
FACT_LOOKUP_VERB_RE = re.compile(
    r"\b(?:pick(?:ed)? up|bought|ordered|spent|earned|sold|drove|travelled|traveled|watched|visited|completed|finished|submitted|booked|take|took|keep|kept|see|saw)\b",
    re.IGNORECASE,
)
PREFERENCE_NOTE_RE = re.compile(
    r"\b(?:prefer(?:s|red)?|like(?:s|d)?|love(?:s|d)?|want(?:s|ed)?|need(?:s|ed)?|avoid(?:s|ed)?|dislike(?:s|d)?|hate(?:s|d)?|enjoy(?:s|ed)?|interested in|looking for)\b",
    re.IGNORECASE,
)
GENERIC_NOTE_RE = re.compile(
    r"\b(?:tips?|advice|suggest(?:ion|ed)?s?|recommend(?:ation|ed)?s?|ideas?|options?|guide|tracking|tracker|checklist)\b",
    re.IGNORECASE,
)
ROLLUP_NOTE_RE = re.compile(
    r"\b(?:roll-?up|summary|recap|overview|aggregate|combined|overall|in total|totalled?|totalling)\b",
    re.IGNORECASE,
)
ATOMIC_EVENT_NOTE_RE = re.compile(
    r"\b(?:i|we)\s+(?:picked up|bought|ordered|spent|earned|sold|drove|travelled|traveled|went|watched|visited|completed|finished|started|booked|got|took|submitted)\b",
    re.IGNORECASE,
)
DATE_TAG_RE = re.compile(r"\[(?:date|observed on):", re.IGNORECASE)
MONEY_EVENT_QUERY_RE = re.compile(
    r"\b(?:spent|spend|cost|costed|paid|pay)\b", re.IGNORECASE
)
DURATION_QUERY_RE = re.compile(r"\bhow long\b", re.IGNORECASE)
BODY_ABSOLUTE_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
MEASUREMENT_VALUE_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?(?:\s+|-)(?:minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
ROUTINE_SCOPE_QUERY_RE = re.compile(
    r"\b(?:daily|every|weekday|each way)\b",
    re.IGNORECASE,
)
ROUTINE_SCOPE_NOTE_RE = re.compile(
    r"\b(?:daily commute|every day|every weekday|weekday|weekdays|each way)\b",
    re.IGNORECASE,
)
SEGMENT_QUALIFIER_NOTE_RE = re.compile(
    r"\b(?:morning commute|often|some days?|sometimes|around)\b",
    re.IGNORECASE,
)
DIRECT_USER_FACT_NOTE_RE = re.compile(
    r"\b(?:i|i'm|i’ve|i've|my|we|we're|we’ve|we've|our|the user)\b",
    re.IGNORECASE,
)
CADENCE_NOTE_RE = re.compile(
    r"\b(?:every (?:day|week|month|year|weekday|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|each (?:day|week|month|year|way|morning|evening|night)|daily|weekly|monthly|fortnightly|annually|usually|normally|once a (?:day|week|month|year)|twice a (?:day|week|month|year)|per (?:day|week|month|year))\b",
    re.IGNORECASE,
)
TIME_VALUE_NOTE_RE = re.compile(
    r"\b(?:at|around)\s+\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)
LOCATION_STORAGE_NOTE_RE = re.compile(
    r"\b(?:keep(?:ing)?|kept|store(?:d|ing)?|stash(?:ed|ing)?|leave|left|put|putting)\b[^.!?\n]{0,120}\b(?:under|beneath|inside|in|on|behind|beside|next to|near|at\s+(?:home|work|the office|the house))\b",
    re.IGNORECASE,
)
SPEED_VALUE_NOTE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mbps|gbps|mph|km/?h|kph)\b",
    re.IGNORECASE,
)
TRANSPORT_MODE_NOTE_RE = re.compile(
    r"\b(?:train|bus|car|bike|bicycle|cycling|walking|on foot|tram|metro|tube|subway|ferry|taxi|uber|rideshare)\b",
    re.IGNORECASE,
)
QUESTION_LIKE_NOTE_RE = re.compile(
    r"(?:^|\n)(?:what\s+(?:are|is|should|could)|which\s+(?:should|would)|how\s+(?:can|should|could|long)|can\s+you|could\s+you|should\s+i|would\s+you|when\s+did|where\s+(?:can|should)|why\s+(?:is|does|did))\b",
    re.IGNORECASE,
)
SUPERSEDED_MARKER_RE = re.compile(
    r"\b(?:superseded[_ ]by|replaced by|archived)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RetrievalIntent:
    """Outcome of the regex-driven intent detection step."""

    preference_query: bool = False
    concrete_fact_query: bool = False

    def label(self) -> str:
        parts: list[str] = []
        if self.preference_query:
            parts.append("preference")
        if self.concrete_fact_query:
            parts.append("concrete-fact")
        return "+".join(parts)


def detect_retrieval_intent(query: str) -> RetrievalIntent:
    """Apply the spec's patterns to the lowercased query.

    Non-English queries bypass every regex and therefore always report
    both intents as ``False``.
    """
    normalised = query.lower()
    return RetrievalIntent(
        preference_query=bool(PREFERENCE_QUERY_RE.search(normalised)),
        concrete_fact_query=(
            bool(PROPERTY_LOOKUP_QUERY_RE.search(normalised))
            or bool(FIRST_PERSON_PROPERTY_QUERY_RE.search(normalised))
            or
            bool(ENUMERATION_OR_TOTAL_QUERY_RE.search(normalised))
            or bool(_derive_action_date_probes(query))
            or _has_specific_recall_cue(normalised)
            or (
                bool(FIRST_PERSON_FACT_LOOKUP_RE.search(normalised))
                and bool(FACT_LOOKUP_VERB_RE.search(normalised))
            )
        ),
    )


def _has_specific_recall_cue(normalised_query: str) -> bool:
    if not normalised_query:
        return False
    if "remind me" not in normalised_query and "remember" not in normalised_query:
        return False
    return "the specific" in normalised_query or "the exact" in normalised_query


def retrieval_result_text(r: RetrievedChunk) -> str:
    """Compose the text blob the document-side regexes run against."""
    return "\n".join([r.path, r.title, r.summary, r.text]).lower()


def preference_intent_multiplier(r: RetrievedChunk, text: str) -> float:
    path = r.path.lower()
    is_global_preference_note = "memory/global/" in path and (
        "user-preference-" in path or bool(PREFERENCE_NOTE_RE.search(text))
    )
    if is_global_preference_note:
        if "user-preference-" in path:
            return 2.35
        return 2.1
    if "memory/global/" not in path and GENERIC_NOTE_RE.search(text):
        return 0.82
    if ROLLUP_NOTE_RE.search(text):
        return 0.9
    return 1.0


def _derive_action_date_probes(query: str) -> list[str]:
    lowered = query.strip().lower()
    if not lowered or "when" not in lowered:
        return []
    probes: list[str] = []
    for pattern, probe in (
        (re.compile(r"\bsubmit(?:ted)?\b", re.IGNORECASE), "submission date"),
        (re.compile(r"\bbook(?:ed|ing)?\b", re.IGNORECASE), "booking date"),
        (
            re.compile(
                r"\b(?:buy|bought|purchase(?:d)?|order(?:ed)?)\b",
                re.IGNORECASE,
            ),
            "purchase date",
        ),
        (re.compile(r"\bjoin(?:ed)?\b", re.IGNORECASE), "join date"),
        (
            re.compile(r"\b(?:start(?:ed)?|begin|began)\b", re.IGNORECASE),
            "start date",
        ),
        (
            re.compile(r"\b(?:finish(?:ed)?|complete(?:d)?)\b", re.IGNORECASE),
            "completion date",
        ),
        (
            re.compile(r"\baccept(?:ed|ance)?\b", re.IGNORECASE),
            "acceptance date",
        ),
    ):
        if pattern.search(lowered) is None or probe in probes:
            continue
        probes.append(probe)
        if len(probes) >= 2:
            break
    return probes


def concrete_fact_intent_multiplier(
    query: str, r: RetrievedChunk, text: str
) -> float:
    path = r.path.lower()
    is_rollup = bool(ROLLUP_NOTE_RE.search(text))
    is_question_like_note = bool(QUESTION_LIKE_NOTE_RE.search(text)) and bool(
        GENERIC_NOTE_RE.search(text)
    )
    is_concrete_fact = (
        "user-fact-" in path
        or "milestone-" in path
        or (
            not is_rollup
            and (bool(DATE_TAG_RE.search(text)) or bool(ATOMIC_EVENT_NOTE_RE.search(text)))
        )
    )

    multiplier = 1.0
    if is_concrete_fact:
        multiplier *= 2.2
    if _derive_action_date_probes(query):
        multiplier *= 1.45 if BODY_ABSOLUTE_DATE_RE.search(text) else 0.78
    if DURATION_QUERY_RE.search(query):
        multiplier *= 1.35 if MEASUREMENT_VALUE_RE.search(text) else 0.72
    if is_question_like_note:
        multiplier *= 0.45
    if is_rollup:
        multiplier *= 0.45
    if (
        not is_concrete_fact
        and "memory/global/" not in path
        and GENERIC_NOTE_RE.search(text)
    ):
        multiplier *= 0.75
    return multiplier


def retrieval_intent_multiplier(
    intent: RetrievalIntent, query: str, r: RetrievedChunk
) -> float:
    multiplier = 1.0
    text = retrieval_result_text(r)
    if intent.preference_query:
        multiplier *= preference_intent_multiplier(r, text)
    if intent.concrete_fact_query:
        multiplier *= concrete_fact_intent_multiplier(query, r, text)
        multiplier *= _focus_aligned_concrete_fact_multiplier(query, text)
        multiplier *= _first_person_concrete_fact_multiplier(query, r, text)
        multiplier *= _stale_superseded_multiplier(r, text)
    return multiplier


def _focus_aligned_concrete_fact_multiplier(query: str, text: str) -> float:
    phrases = _derive_priority_sub_queries(query)
    if not phrases:
        phrases = _filtered_phrase_probes(query)
    if not phrases:
        return 1.0

    lowered_text = _normalise_focus_alignment_text(text)
    best = 0.0
    for phrase in phrases:
        best = max(best, _focus_alignment_score(lowered_text, phrase))

    if best >= 0.99:
        return 1.6
    if best >= 0.66:
        return 1.25
    return 1.0


def _first_person_concrete_fact_multiplier(
    query: str, r: RetrievedChunk, text: str
) -> float:
    normalised_query = query.strip().lower()
    if not normalised_query:
        return 1.0
    if (
        FIRST_PERSON_CONCRETE_QUERY_RE.search(normalised_query) is None
        and FIRST_PERSON_FACT_LOOKUP_RE.search(normalised_query) is None
    ):
        return 1.0

    path = r.path.lower()
    is_global_memory = "memory/global/" in path
    is_direct_fact = _is_concrete_fact_like(path, text)

    if is_global_memory and is_direct_fact:
        multiplier = 1.35
    elif is_global_memory:
        multiplier = 1.22
    elif is_direct_fact:
        multiplier = 0.88
    else:
        multiplier = 0.58

    if not is_global_memory and GENERIC_NOTE_RE.search(text) is not None:
        multiplier *= 0.82

    if (
        DURATION_QUERY_RE.search(normalised_query) is not None
        and ROUTINE_SCOPE_QUERY_RE.search(normalised_query) is not None
    ):
        if ROUTINE_SCOPE_NOTE_RE.search(text) is not None:
            multiplier *= 1.25
        if (
            SEGMENT_QUALIFIER_NOTE_RE.search(text) is not None
            and ROUTINE_SCOPE_NOTE_RE.search(text) is None
        ):
            multiplier *= 0.15
    return multiplier


def _stale_superseded_multiplier(r: RetrievedChunk, text: str) -> float:
    for key in ("superseded_by", "supersededBy"):
        value = r.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return 0.18
    if SUPERSEDED_MARKER_RE.search(text) is not None:
        return 0.18
    return 1.0


def _is_concrete_fact_like(path: str, text: str) -> bool:
    if "user-fact-" in path or "milestone-" in path:
        return True
    if ROLLUP_NOTE_RE.search(text) is not None:
        return False
    return DATE_TAG_RE.search(text) is not None or ATOMIC_EVENT_NOTE_RE.search(text) is not None


def _query_targets_cadence(normalised_query: str) -> bool:
    return "how often do i" in normalised_query


def _query_targets_time(normalised_query: str) -> bool:
    return "what time do i" in normalised_query or "what time is my" in normalised_query


def _query_targets_location(normalised_query: str) -> bool:
    return (
        "where do i" in normalised_query
        or "where did i" in normalised_query
        or "where have i" in normalised_query
        or "where am i" in normalised_query
        or "where is my" in normalised_query
    )


def _query_targets_speed(normalised_query: str) -> bool:
    return "what speed is my" in normalised_query or "how fast is my" in normalised_query


def _query_targets_transport_mode(normalised_query: str) -> bool:
    return (
        "which mode of transport did i" in normalised_query
        or "what mode of transport did i" in normalised_query
        or "which transport did i" in normalised_query
        or "what transport did i" in normalised_query
    )


def _focus_alignment_score(lowered_text: str, phrase: str) -> float:
    lowered_phrase = _normalise_focus_alignment_text(phrase)
    if not lowered_phrase:
        return 0.0
    if lowered_phrase in lowered_text:
        return 1.0

    text_tokens = _token_set(lowered_text.split())
    phrase_tokens = [token for token in lowered_phrase.split() if token]
    if not phrase_tokens:
        return 0.0

    matched = 0
    for token in phrase_tokens:
        if token in text_tokens:
            matched += 1
    return matched / len(phrase_tokens)


def _normalise_focus_alignment_text(raw: str) -> str:
    if not raw:
        return ""
    return " ".join(
        re.sub(r"[-/()\[\],.:;?!]", " ", raw.lower()).split()
    )


def _token_set(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token}


def reweight_shared_memory_ranking(
    query: str, results: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Apply multiplicative score adjustments when the query pattern
    matches. Original fused rank is preserved as the tie-breaker so
    ports stay deterministic."""
    if not results:
        return results
    intent = detect_retrieval_intent(query)
    if not intent.preference_query and not intent.concrete_fact_query:
        return results

    indexed: list[tuple[RetrievedChunk, int]] = []
    for i, r in enumerate(results):
        adjusted = RetrievedChunk(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            path=r.path,
            score=r.score * retrieval_intent_multiplier(intent, query, r),
            text=r.text,
            title=r.title,
            summary=r.summary,
            metadata=dict(r.metadata),
            bm25_rank=r.bm25_rank,
            vector_similarity=r.vector_similarity,
            rerank_score=r.rerank_score,
        )
        indexed.append((adjusted, i))

    indexed.sort(key=lambda pair: (-pair[0].score, pair[1]))
    rescored = [pair[0] for pair in indexed]
    return _diversify_composite_concrete_ranking(query, rescored)


@dataclass(slots=True)
class _CompositeFocusMatch:
    index: int = -1
    score: float = 0.0


def _diversify_composite_concrete_ranking(
    query: str, results: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    focuses = _filtered_phrase_probes(query)
    if len(results) < 3 or len(focuses) < 2 or not _is_composite_concrete_query(query):
        return results

    covered: set[int] = set()
    primary: list[RetrievedChunk] = []
    secondary: list[RetrievedChunk] = []
    near_misses: list[RetrievedChunk] = []
    duplicates: list[RetrievedChunk] = []
    for result in results:
        match = _best_composite_focus_match(retrieval_result_text(result), focuses)
        if match.index >= 0 and match.score >= 0.5:
            if match.index not in covered:
                primary.append(result)
                covered.add(match.index)
                continue
            duplicates.append(result)
            continue
        if match.index >= 0 and match.score >= 0.25:
            near_misses.append(result)
            continue
        secondary.append(result)
    return [*primary, *secondary, *near_misses, *duplicates]


def _is_composite_concrete_query(query: str) -> bool:
    lowered = query.strip().lower()
    if ENUMERATION_OR_TOTAL_QUERY_RE.search(lowered) is None:
        return False
    return " and " in lowered or " plus " in lowered or " or " in lowered


def _best_composite_focus_match(
    text: str, focuses: list[str]
) -> _CompositeFocusMatch:
    lowered_text = _normalise_focus_alignment_text(text)
    best = _CompositeFocusMatch()
    for index, focus in enumerate(focuses):
        score = _focus_alignment_score(lowered_text, focus)
        if score > best.score:
            best = _CompositeFocusMatch(index=index, score=score)
    return best
