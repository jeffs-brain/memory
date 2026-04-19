# SPDX-License-Identifier: Apache-2.0
"""English-only; non-English queries receive base RRF scores without
modification.

The regexes below are kept bit-for-bit identical to the TypeScript
reference in ``retrieval/hybrid.ts`` and the Go port in
``sdks/go/retrieval/intent.go``. Any drift breaks cross-SDK parity and
will be flagged by the conformance suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    r"\b(?:how long is my|what specific|which specific|what exact|which exact)\b",
    re.IGNORECASE,
)
FIRST_PERSON_FACT_LOOKUP_RE = re.compile(
    r"\b(?:did i|have i|was i|were i)\b",
    re.IGNORECASE,
)
FACT_LOOKUP_VERB_RE = re.compile(
    r"\b(?:pick(?:ed)? up|bought|ordered|spent|earned|sold|drove|travelled|traveled|watched|visited|completed|finished|submitted|booked)\b",
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
QUESTION_LIKE_NOTE_RE = re.compile(
    r"(?:^|\n)(?:what\s+(?:are|is|should|could)|which\s+(?:should|would)|how\s+(?:can|should|could|long)|can\s+you|could\s+you|should\s+i|would\s+you|when\s+did|where\s+(?:can|should)|why\s+(?:is|does|did))\b",
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
            or
            bool(ENUMERATION_OR_TOTAL_QUERY_RE.search(normalised))
            or (
                bool(FIRST_PERSON_FACT_LOOKUP_RE.search(normalised))
                and bool(FACT_LOOKUP_VERB_RE.search(normalised))
            )
        ),
    )


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


def concrete_fact_intent_multiplier(r: RetrievedChunk, text: str) -> float:
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


def retrieval_intent_multiplier(intent: RetrievalIntent, r: RetrievedChunk) -> float:
    multiplier = 1.0
    text = retrieval_result_text(r)
    if intent.preference_query:
        multiplier *= preference_intent_multiplier(r, text)
    if intent.concrete_fact_query:
        multiplier *= concrete_fact_intent_multiplier(r, text)
    return multiplier


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
            score=r.score * retrieval_intent_multiplier(intent, r),
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
    return [pair[0] for pair in indexed]
