# SPDX-License-Identifier: Apache-2.0
"""Extract durable knowledge from a session into memory files.

The ``EXTRACTION_PROMPT`` is ported verbatim from ``sdks/go/memory/extract.go``.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role
from ..query.temporal import (
    resolve_last_week,
    resolve_last_weekday,
    resolve_relative_day,
    resolve_relative_time,
)
from ._memstore import ListOpts, NotFoundError
from .paths import (
    base_name,
    memory_global_index,
    memory_global_prefix,
    memory_global_topic,
    memory_project_index,
    memory_project_prefix,
    memory_project_topic,
    project_slug as _project_slug,
)
from .store import parse_frontmatter
from .types import Message, TopicFile

if TYPE_CHECKING:
    from .contextualise import Contextualiser
    from .manager import MemoryManager

EXTRACT_MAX_TOKENS = 4096
EXTRACT_TEMPERATURE = 0
EXTRACT_MIN_MESSAGES = 2
EXTRACT_MAX_RECENT = 80
EXISTING_MEMORY_LIMIT = 24
EXISTING_MEMORY_PREVIEW_LIMIT = 400
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

DATE_TAG_RE = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")
WEEKDAY_TAG_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I
)
QUANTITY_TAG_RE = re.compile(r"\b\d{1,6}(?:\.\d+)?\b")
PROPER_NOUN_TAG_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")
MONEY_TAG_RE = re.compile(r"[\$£€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?")
UNIT_QUANTITY_TAG_RE = re.compile(
    r"\b(\d{1,6}(?:\.\d+)?)\s+"
    r"(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?|"
    r"km|kilometres?|miles?|metres?|meters?|kg|kilograms?|pounds?|lbs?|grams?|"
    r"percent|%|kbps|mbps|gbps|tbps|mb/s|gb/s|tb/s)\b",
    re.I,
)
WORD_UNIT_QUANTITY_TAG_RE = re.compile(
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?)\b",
    re.I,
)
MONTH_NAME_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    re.I,
)
DATE_INPUT_RE = re.compile(
    r"^(\d{4})[/-](\d{2})[/-](\d{2})"
    r"(?:\s+\([A-Za-z]{3}\))?"
    r"(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?$"
)
HEURISTIC_SESSION_DATE_RE = re.compile(
    r"\b\d{4}[/-]\d{2}[/-]\d{2}(?:\s+\([A-Za-z]{3}\))?(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b"
)
HEURISTIC_USER_FACT_LIMIT = 2
HEURISTIC_MILESTONE_FACT_LIMIT = 2
HEURISTIC_PREFERENCE_FACT_LIMIT = 2
HEURISTIC_PENDING_FACT_LIMIT = 3
HEURISTIC_EVENT_FACT_LIMIT = 2
FIRST_PERSON_FACT_RE = re.compile(
    r"\b(i|i'm|i’ve|i've|my|we|we're|we’ve|we've|our)\b", re.I
)
HEURISTIC_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")
HEURISTIC_ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.I,
)
HEURISTIC_ABBREVIATED_SENTENCE_END_RE = re.compile(
    r"\b(?:dr|mr|mrs|ms|prof)\.$",
    re.I,
)
HEURISTIC_MILESTONE_EVENT_RE = re.compile(
    r"\b(?:(?:just|recently)\s+)?"
    r"(?:completed|submitted|graduated|finished|started|joined|accepted|presented)\b",
    re.I,
)
HEURISTIC_MILESTONE_TIME_RE = re.compile(
    r"\b(?:today|yesterday|recently|just|last\s+"
    r"(?:week|month|year|summer|spring|fall|autumn|winter))\b",
    re.I,
)
HEURISTIC_MILESTONE_TOPIC_RE = re.compile(
    r"\b(?:degree|thesis|dissertation|paper|research|conference|course|class|project|internship|job|role|group|club|community|network|forum|association|society|linkedin)\b",
    re.I,
)
HEURISTIC_PREFERENCE_BESIDES_LIKE_RE = re.compile(
    r"\bbesides\s+([^.!?]+?),\s*i\s+(?:also\s+)?like\s+([^.!?]+?)(?:[.!?]|$)",
    re.I,
)
HEURISTIC_PREFERENCE_LIKE_RE = re.compile(
    r"\bi\s+(?:also\s+)?(?:like|love|prefer|enjoy)\s+([^.!?]+?)(?:[.!?]|$)",
    re.I,
)
HEURISTIC_PREFERENCE_COMPATIBLE_RE = re.compile(
    r"\bcompatible with (?:my|the)\s+([^.!?,\n]+)", re.I
)
HEURISTIC_PREFERENCE_DESIGNED_FOR_RE = re.compile(
    r"\bspecifically designed for\s+([^.!?,\n]+)", re.I
)
HEURISTIC_PREFERENCE_AS_USER_RE = re.compile(
    r"\bas a[n]?\s+([^.!?,\n]+?)\s+user\b", re.I
)
HEURISTIC_PREFERENCE_FIELD_RE = re.compile(r"\bfield of\s+([^.!?,\n]+)", re.I)
HEURISTIC_PREFERENCE_ADVANCED_RE = re.compile(
    r"\badvanced topics in\s+([^.!?,\n]+)", re.I
)
HEURISTIC_PREFERENCE_SKIP_BASICS_RE = re.compile(r"\bskip the basics\b", re.I)
HEURISTIC_PREFERENCE_WORKING_IN_FIELD_RE = re.compile(
    r"\b(?:i am|i'm)\s+working in the field\b", re.I
)
HEURISTIC_PENDING_ACTION_LEAD_RE = re.compile(
    r"^\s*(?:"
    r"i(?:'ve)?(?:\s+still)?\s+(?:need|have)\s+to|"
    r"i(?:'ve)?\s+got\s+to|"
    r"i\s+must|"
    r"i\s+should|"
    r"i\s+need\s+to\s+remember\s+to|"
    r"remember\s+to|"
    r"don't\s+let\s+me\s+forget\s+to"
    r")\s+([^.!?]+)",
    re.I,
)
HEURISTIC_PENDING_ACTION_START_RE = re.compile(
    r"^(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|call|email|pay|renew|cancel|buy|send|post|fix|follow\s+up)\b",
    re.I,
)
HEURISTIC_APPOINTMENT_RE = re.compile(
    r"\b(?:appointment|check-?up|consultation|follow-?up|therapy session|scan|surgery|dentist|doctor|gp|"
    r"dermatologist|orthodontist|hygienist|therapist|counsellor|counselor|psychiatrist|psychologist|"
    r"physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|"
    r"gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian)\b",
    re.I,
)
HEURISTIC_MEDICAL_ENTITY_RE = re.compile(
    r"\b(?:gp|doctor|dentist|dermatologist|orthodontist|hygienist|therapist|counsellor|counselor|"
    r"psychiatrist|psychologist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|"
    r"pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian)\b",
    re.I,
)
HEURISTIC_EVENT_RE = re.compile(
    r"\b(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar|service|mass|worship|prayer)\b",
    re.I,
)
HEURISTIC_EVENT_ATTENDANCE_RE = re.compile(
    r"\b(?:attend(?:ed|ing)?|went to|go(?:ing)? to|joined|join(?:ing)?|participat(?:ed|ing)|"
    r"volunteer(?:ed|ing)|present(?:ed|ing)|watch(?:ed|ing)|listen(?:ed|ing)\s+to|"
    r"got back from|completed)\b",
    re.I,
)
HEURISTIC_EVENT_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'/-]*(?:\s+[A-Z][A-Za-z0-9&'/-]*){0,6}\s+"
    r"(?:Workshop|Conference|Concert|Festival|Meetup|Show|Screening|Class|Course|Webinar|Lecture|Seminar))\b"
)
HEURISTIC_RELIGIOUS_SERVICE_RE = re.compile(
    r"\battend(?:ed|ing)?\s+([^,.!?]+?\s+service(?:\s+at\s+[^,.!?]+)?)\b",
    re.I,
)
HEURISTIC_WITH_PERSON_RE = re.compile(r"\bwith\s+(Dr\.?\s+[A-Z][a-zA-Z'-]+)\b")
HEURISTIC_RELATIVE_DATE_RE = re.compile(
    r"\b(?:today|tomorrow|tonight|this morning|this afternoon|this evening|this weekend|next weekend|"
    r"next week|next month|coming week|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"coming\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.I,
)
HEURISTIC_CLOCK_TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b", re.I
)
HEURISTIC_DURATION_FACT_RE = re.compile(
    r"\b(?:\d{1,4}-day|[a-z]+-day|[a-z]+-week|[a-z]+-month|[a-z]+-year|week-long|month-long|year-long)\b",
    re.I,
)
HEURISTIC_CADENCE_FACT_RE = re.compile(
    r"\b(?:"
    r"(?:every|each)\s+(?:day|morning|afternoon|evening|night|weekday|weekend|week|month|year|"
    r"other\s+week|two\s+weeks?|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|(?:once|twice)\s+(?:a|per)\s+(?:day|week|month|year)"
    r"|\d+\s+times?\s+(?:a|per)\s+(?:day|week|month|year)"
    r"|daily|weekly|monthly|yearly|annually|usually|normally|bi-?weekly|fortnightly"
    r")\b",
    re.I,
)
HEURISTIC_LOCATION_STORAGE_FACT_RE = re.compile(
    r"\b(?:i|i'm|i’ve|i've|i have)\s+(?:been\s+)?(?:keep(?:ing)?|kept|stor(?:e|ing|ed)|stash(?:ed|ing)?|leave|left|put|placed)\b[^.!?\n]*\b(?:under|inside|in|on|at|behind|beside|next to)\b",
    re.I,
)
HEURISTIC_SESSION_ID_RE = re.compile(
    r"(?im)\bsession[_ ]id\s*[:=]\s*([A-Za-z0-9._-]+)\b"
)
HEURISTIC_FILENAME_DATED_RE = re.compile(
    r"^(user-(?:fact|preference))-(\d{4}-\d{2}-\d{2})-(.+)\.md$"
)
HEURISTIC_FILENAME_RE = re.compile(r"^(user-(?:fact|preference))-(.+)\.md$")
HEURISTIC_RECOMMENDATION_REQUEST_RE = re.compile(
    r"\b(?:recommend|suggest|looking for|look for|what should i|which should i|"
    r"where should i stay|what to watch|what to read|what to serve)\b",
    re.I,
)
HEURISTIC_RECOMMENDATION_UNDER_RE = re.compile(
    r"\bunder\s+(\d{1,4}(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?|pages?|£|€|\$))\b",
    re.I,
)
HEURISTIC_RECOMMENDATION_NOT_TOO_RE = re.compile(
    r"\b(?:nothing|not)\s+too\s+([^,.!?;\n]+)", re.I
)
HEURISTIC_RECOMMENDATION_WITHOUT_RE = re.compile(
    r"\bwithout\s+([^,.!?;\n]+)", re.I
)
HEURISTIC_RECOMMENDATION_FAMILY_RE = re.compile(
    r"\b(?:family-friendly|kid-friendly)\b", re.I
)
HEURISTIC_RECOMMENDATION_LIGHT_RE = re.compile(
    r"\b(?:light-hearted|feel-good|cosy|cozy)\b", re.I
)
RELATIVE_TEMPORAL_TAG_RE = re.compile(
    r"\b(?:today|tomorrow|tonight|this morning|this afternoon|this evening|this weekend|next weekend|"
    r"next week|next month|coming week|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"coming\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.I,
)
CLOCK_TIME_TAG_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.I
)
PENDING_ACTION_TAG_RE = re.compile(
    r"\b(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|renew|cancel|follow\s+up)\b",
    re.I,
)
MEDICAL_TAG_RE = re.compile(
    r"\b(?:appointment|check-?up|consultation|follow-?up|doctor|gp|dentist|dermatologist|orthodontist|"
    r"hygienist|therapist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|"
    r"pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|"
    r"vet|veterinarian|clinic|hospital|prescription)\b",
    re.I,
)
EVENT_TAG_RE = re.compile(
    r"\b(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar)\b",
    re.I,
)
ENTERTAINMENT_TAG_RE = re.compile(
    r"\b(?:film|movie|show|series|book|novel|game|podcast|cinema)\b", re.I
)
HEURISTIC_STOPWORDS = {
    "been",
    "city",
    "definitely",
    "feels",
    "following",
    "getting",
    "have",
    "just",
    "last",
    "lately",
    "miles",
    "months",
    "really",
    "routine",
    "sticking",
    "their",
    "weeks",
}
AUTO_TAG_STOP_NOUNS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "when",
    "where",
    "what",
    "who",
    "why",
    "how",
    "observed",
    "date",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun",
    "user",
    "assistant",
}
SUMMARY_STOPWORDS = {
    "a",
    "an",
    "and",
    "appointment",
    "event",
    "has",
    "is",
    "task",
    "the",
    "this",
    "to",
    "user",
    "with",
}


EXTRACTION_PROMPT = """You are a memory extraction agent. Analyse the recent conversation messages below and determine what durable knowledge should be saved to the persistent memory system.

You MUST respond with ONLY a JSON object. Do NOT call tools, do NOT write prose. Just output the JSON.

Both speakers contribute durable knowledge. Treat user turns and assistant turns as equally valid sources of facts. Capture everything the user stated AND everything the assistant provided: recommendations (restaurants, hotels, shops, books), specific named suggestions, recipes, itineraries, enumerated lists or rankings the assistant gave, answers the assistant produced, corrections the assistant issued, plans the assistant proposed, colours or attributes the assistant described, and any quantities or dates the assistant cited. If the assistant enumerated items (a list of jobs, options, steps, or candidates), save the full enumeration verbatim including positions where relevant. When in doubt, extract both sides.

Preserve structured assistant outputs when they contain durable facts. If the assistant gives a roster, timetable, schedule, table, comparison, shortlist, or direct factual answer, keep the exact names, positions, shifts, prices, speeds, sizes, counts, and other concrete attributes rather than flattening them into a vague summary.

Preserve concrete historical facts exactly when they matter. Keep explicit user experiences, measurements, comparisons, relatives, places, and time references in the memory content instead of flattening them into a vague preference or goal. Examples:
- "My car was getting 30 miles per gallon in the city a few months ago." should preserve the 30 miles per gallon fact and timeframe.
- "I went on a two-week trip to Europe with my parents and younger brother last month." should preserve the trip, relatives, destination, and timeframe.
- "I've been sticking to my daily tidying routine for 4 weeks." should preserve the duration as a concrete user fact.
- If the conversation also reveals a broader preference, keep the concrete event as well rather than replacing it.

When a user states a concrete personal measurement, duration, past event, or status update, create a separate user memory for that fact even if the rest of the session is mostly recommendations, troubleshooting, or planning.

Memory types:
- user: User's role, preferences, knowledge level, working style
- feedback: Corrections or confirmations about approach (what to avoid or keep doing)
- project: Non-obvious context about ongoing work, goals, decisions, deadlines (includes assistant recommendations and enumerations worth recalling later)
- reference: Pointers to external systems, URLs, project names, named entities the assistant surfaced (restaurants, hotels, businesses, books, product names)

Memory scopes:
- global (~/.config/jeff/memory/): Cross-project knowledge. Types: user, feedback
- project (project memory directory): Project-specific knowledge. Types: project, reference

When deciding scope:
- user preferences, working style, general corrections \u2192 global
- project architecture, project-specific decisions, external system pointers, assistant recommendations and enumerations \u2192 project
- default to "project" if unsure

Examples of assistant-turn facts that MUST be captured:
- "I recommend Roscioli for romantic Italian in Rome." \u2192 create a reference memory naming the restaurant, cuisine, city.
- "Here are seven work-from-home jobs for seniors: 1. Virtual Assistant, 2. ..., 7. Transcriptionist." \u2192 save the full numbered list so later recall can reconstruct any position.
- "The Plesiosaur in the children's book had a blue scaly body." \u2192 save the attribute with its subject.
- "Sunday roster: Admon, 8 am - 4 pm (Day Shift)." \u2192 save the person's name, shift, and exact hours.
- "You upgraded your internet plan to 500 Mbps." \u2192 save the exact plan value, not a vague note about faster internet.

Updates and quantitative facts that MUST be captured:
- When the user gives a new count, total, amount, ratio, progress update, milestone, or outcome, save it even if an older memory on the same topic already exists.
- Prefer an update with supersedes when the new statement revises prior state.
- Stable personal facts like favourite ratios, purchase amounts, fundraising outcomes, reading progress, completed counts, and milestone dates are durable memory.
- Do not discard a later update just because it seems small. A new number often replaces an older one.
- When a later message changes a recurring cadence, schedule, count, price, bandwidth, screen size, or other exact attribute, preserve the new value explicitly and supersede the older one when appropriate.
- Do not round away specific attributes such as 55-inch, 500 Mbps, 8 am - 4 pm, or edition counts. Keep the exact value in the memory content.

Examples of user-turn updates that MUST be captured:
- "I just finished my fifth issue of National Geographic." \u2192 update the reading-progress memory and supersede the older "finished three issues" state when applicable.
- "I initially aimed to raise $200 and ended up raising $250." \u2192 save both the goal and the achieved amount so later questions can compute the difference.
- "I settled on a 3:1 gin-to-vermouth ratio for a classic martini." \u2192 save this as a durable user preference.
- "I spent $200 on the designer handbag and $500 on skincare." \u2192 save the concrete amounts, not just the product categories.

Do NOT save:
- Code patterns, architecture, or file paths derivable from the codebase
- Git history or recent changes (use git log for those)
- Debugging solutions (the fix is in the code)
- Ephemeral task details or in-progress work
- Anything already in the existing memories listed below

For each memory worth saving, output:
- action: "create" (new file) or "update" (modify existing)
- filename: e.g. "feedback_testing.md" (kebab-case, descriptive)
- name: human-readable name
- description: one-line description (used for future recall)
- type: user | feedback | project | reference
- scope: "global" or "project" (default to "project" if unsure)
- content:
  - for user and reference memories: direct factual prose that preserves the exact people, places, dates, relative time phrases, quantities, and historical events from the conversation. Prefer concrete statements over generic advice.
  - for feedback and project memories: structured with Why: and How to apply: lines
- index_entry: one-line entry for MEMORY.md (under 150 chars)
- supersedes (optional): when the user has corrected, updated, or contradicted an earlier stated fact for the same topic, set this to the filename of the earlier memory so it is retired. Only fill when you are confident the new fact replaces a specific older one; prefer leaving empty when unsure.

If nothing is worth saving, return: {"memories": []}

Respond with ONLY valid JSON: {"memories": [...]}"""


@dataclass(slots=True)
class ExtractedMemory:
    """A single memory extracted from a conversation."""

    action: str = ""
    filename: str = ""
    name: str = ""
    description: str = ""
    type: str = ""
    content: str = ""
    index_entry: str = ""
    scope: str = ""
    supersedes: str = ""
    tags: list[str] = field(default_factory=list)
    session_id: str = ""
    observed_on: str = ""
    session_date: str = ""
    context_prefix: str = ""
    modified_override: str = ""


@dataclass(slots=True)
class HeuristicPreferenceCandidate:
    summary: str
    evidence: str


@dataclass(slots=True)
class ExistingMemorySummary:
    path: str
    scope: str
    name: str
    description: str
    type: str
    modified: str = ""
    content: str = ""


class Extractor:
    """Manages background memory extraction."""

    def __init__(self, mem: "MemoryManager") -> None:
        self._mem = mem
        self._lock = threading.Lock()
        self._last_cursor = 0
        self._in_progress = False
        self._ctx: "Contextualiser | None" = None

    def set_contextualiser(self, ctx: "Contextualiser | None") -> None:
        self._ctx = ctx

    def reset_cursor(self) -> None:
        with self._lock:
            self._last_cursor = 0

    async def maybe_extract(
        self,
        provider: Provider,
        model: str,
        project_path: str,
        messages: list[Message],
        *,
        session_id: str = "",
        session_date: str = "",
    ) -> None:
        with self._lock:
            if self._in_progress:
                return
            self._in_progress = True
            cursor = self._last_cursor

        try:
            if len(messages) - cursor < EXTRACT_MIN_MESSAGES:
                return

            slug = _project_slug(project_path)

            phys_hints: list[str] = []
            gp = self._mem.store.local_path(memory_global_prefix())
            if gp:
                phys_hints.append(gp)
            pp = self._mem.store.local_path(memory_project_prefix(slug))
            if pp:
                phys_hints.append(pp)

            if has_memory_writes(messages[cursor:], *phys_hints):
                with self._lock:
                    self._last_cursor = len(messages)
                return

            recent = messages[cursor:]
            if len(recent) > EXTRACT_MAX_RECENT:
                recent = recent[-EXTRACT_MAX_RECENT:]

            try:
                existing_memories = list_existing_memories(self._mem, slug)
            except Exception:
                existing_memories = []

            user_prompt = extract_user_prompt(recent, existing_memories)

            try:
                resp = await provider.complete(
                    CompleteRequest(
                        model=model,
                        messages=[
                            LLMMessage(role=Role.SYSTEM, content=EXTRACTION_PROMPT),
                            LLMMessage(role=Role.USER, content=user_prompt),
                        ],
                        max_tokens=EXTRACT_MAX_TOKENS,
                        temperature=EXTRACT_TEMPERATURE,
                    )
                )
            except Exception:
                return

            parsed, parse_ok = _parse_extraction_result_payload(
                resp.text,
                default_scope="project",
                session_id=session_id,
                session_date=session_date,
            )
            if not parse_ok:
                parsed = []
            result = _post_process_session_extractions(
                recent,
                parsed,
                session_id=session_id,
                session_date=session_date,
            )
            if not result:
                with self._lock:
                    self._last_cursor = len(messages)
                return

            if self._ctx is not None and self._ctx.enabled():
                summary = extract_session_summary(recent)
                for em in result:
                    prefix = await self._ctx.build_prefix_async(
                        session_id, summary, em.content
                    )
                    if prefix:
                        em.context_prefix = prefix

            try:
                apply_extractions(self._mem, slug, result)
            except Exception:
                return

            with self._lock:
                self._last_cursor = len(messages)
        finally:
            with self._lock:
                self._in_progress = False


async def extract_from_messages(
    provider: Provider,
    model: str,
    mem: "MemoryManager",
    project_path: str,
    messages: list[Message],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    if len(messages) < 2:
        return []

    resolved_session_date = derive_session_date(messages, session_date)
    recent = messages
    if len(recent) > EXTRACT_MAX_RECENT:
        recent = recent[-EXTRACT_MAX_RECENT:]

    slug = _project_slug(project_path)
    try:
        existing_memories = list_existing_memories(mem, slug)
    except Exception:
        existing_memories = []
    user_prompt = extract_user_prompt(recent, existing_memories)

    resp = await provider.complete(
        CompleteRequest(
            model=model,
            messages=[
                LLMMessage(role=Role.SYSTEM, content=EXTRACTION_PROMPT),
                LLMMessage(role=Role.USER, content=user_prompt),
            ],
            max_tokens=EXTRACT_MAX_TOKENS,
            temperature=EXTRACT_TEMPERATURE,
        )
    )
    parsed, parse_ok = _parse_extraction_result_payload(
        resp.text,
        default_scope="project",
        session_id=session_id,
        session_date=resolved_session_date,
    )
    if not parse_ok:
        parsed = []
    return _post_process_session_extractions(
        recent,
        parsed,
        session_id=session_id,
        session_date=resolved_session_date,
    )


def extract_user_prompt(
    messages: list[Message],
    existing_memories: list[ExistingMemorySummary],
) -> str:
    parts: list[str] = []
    if existing_memories:
        parts.extend(["## Existing memories", ""])
        for memory in existing_memories:
            parts.append(f"### [{memory.scope}] {base_name(memory.path)}")
            if memory.name != "":
                parts.append(f"name: {memory.name}")
            if memory.description != "":
                parts.append(f"description: {memory.description}")
            if memory.type != "":
                parts.append(f"type: {memory.type}")
            if memory.modified != "":
                parts.append(f"modified: {memory.modified}")
            if memory.content != "":
                parts.append(f"content: {memory.content}")
            parts.append("")
    parts.append("## Recent conversation\n")
    for m in messages:
        role = _role_value(m.role)
        content = m.content
        if len(content) > 2000:
            content = content[:2000] + "\n[...truncated]"
        if m.role == Role.TOOL:
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"[{role} ({m.name})]: {content}\n")
            continue
        parts.append(f"[{role}]: {content}\n")
    return "\n".join(parts)


def list_existing_memories(
    mem: "MemoryManager",
    project_slug: str,
) -> list[ExistingMemorySummary]:
    summaries: list[ExistingMemorySummary] = []
    for scope, prefix in (
        ("global", memory_global_prefix()),
        ("project", memory_project_prefix(project_slug)),
    ):
        for entry in mem.store.list(prefix, ListOpts(recursive=True)):
            if entry.is_dir:
                continue
            filename = base_name(entry.path)
            if not filename.endswith(".md") or filename == "MEMORY.md":
                continue
            try:
                raw = mem.store.read(entry.path)
            except NotFoundError:
                continue
            frontmatter, body = parse_frontmatter(raw.decode("utf-8"))
            summaries.append(
                ExistingMemorySummary(
                    path=entry.path,
                    scope=scope,
                    name=frontmatter.name.strip(),
                    description=frontmatter.description.strip(),
                    type=frontmatter.type.strip(),
                    modified=frontmatter.modified.strip(),
                    content=truncate_prompt_content(body),
                )
            )
    summaries.sort(
        key=lambda summary: (
            -memory_summary_timestamp(summary.modified),
            summary.path,
        )
    )
    return summaries[:EXISTING_MEMORY_LIMIT]


def truncate_prompt_content(content: str) -> str:
    collapsed = " ".join(content.split())
    if len(collapsed) <= EXISTING_MEMORY_PREVIEW_LIMIT:
        return collapsed
    return f"{collapsed[:EXISTING_MEMORY_PREVIEW_LIMIT]}..."


def memory_summary_timestamp(value: str) -> float:
    parsed = parse_rfc3339(value)
    return parsed.timestamp() if parsed is not None else 0.0


def parse_extraction_result(
    content: str,
    *,
    default_scope: str = "project",
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    memories, parse_ok = _parse_extraction_result_payload(
        content,
        default_scope=default_scope,
        session_id=session_id,
        session_date=session_date,
    )
    return memories if parse_ok else []


def _parse_extraction_result_payload(
    content: str,
    *,
    default_scope: str = "project",
    session_id: str = "",
    session_date: str = "",
) -> tuple[list[ExtractedMemory], bool]:
    raw_memories, parse_ok = _parse_raw_extraction_memories(content)
    if not parse_ok:
        return [], False
    out: list[ExtractedMemory] = []
    for item in raw_memories:
        out.append(
            normalise_extracted_memory(
                item,
                default_scope=default_scope,
                session_id=session_id,
                session_date=session_date,
            )
        )
    return out, True


def _parse_raw_extraction_memories(content: str) -> tuple[list[dict[str, Any]], bool]:
    for candidate in _extraction_json_candidates(content):
        parsed = _decode_extraction_candidate(candidate)
        if parsed[1]:
            return parsed
        repaired = _repair_extraction_json_candidate(candidate)
        if repaired == candidate:
            continue
        repaired_parsed = _decode_extraction_candidate(repaired)
        if repaired_parsed[1]:
            return repaired_parsed
    return [], False


def _extraction_json_candidates(content: str) -> list[str]:
    trimmed = content.strip()
    if trimmed == "":
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        value = candidate.strip()
        if value == "" or value in seen:
            return
        seen.add(value)
        out.append(value)

    add(trimmed)
    add(_bracket_slice(trimmed, "{", "}"))
    add(_bracket_slice(trimmed, "[", "]"))
    return out


def _bracket_slice(content: str, open_char: str, close_char: str) -> str:
    start = content.find(open_char)
    end = content.rfind(close_char)
    if start < 0 or end <= start:
        return ""
    return content[start : end + 1]


def _repair_extraction_json_candidate(content: str) -> str:
    repaired = content
    while True:
        next_value = TRAILING_COMMA_RE.sub(r"\1", repaired)
        if next_value == repaired:
            return repaired
        repaired = next_value


def _decode_extraction_candidate(content: str) -> tuple[list[dict[str, Any]], bool]:
    trimmed = content.strip()
    if trimmed == "":
        return [], False
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return [], False

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)], True

    if not isinstance(parsed, dict):
        return [], False

    memories = parsed.get("memories")
    if isinstance(memories, list):
        return [item for item in memories if isinstance(item, dict)], True

    memory = parsed.get("memory")
    if isinstance(memory, dict):
        return [memory], True

    if _looks_like_raw_extracted_memory(parsed):
        return [parsed], True
    return [], True


def _looks_like_raw_extracted_memory(value: dict[str, Any]) -> bool:
    for key in ("filename", "content", "name", "description", "indexEntry", "index_entry"):
        if isinstance(value.get(key), str):
            return True
    return False


def normalise_extracted_memory(
    raw: dict[str, Any],
    *,
    default_scope: str,
    session_id: str,
    session_date: str,
) -> ExtractedMemory:
    resolved_session_id = (
        _string_field(raw, "sessionId")
        or _string_field(raw, "session_id")
        or session_id
    ).strip()
    scope = _string_field(raw, "scope")
    if scope not in {"global", "project"}:
        scope = default_scope

    action = _string_field(raw, "action")
    if action not in {"create", "update"}:
        action = "create"

    memory_type = _string_field(raw, "type")
    if memory_type not in {"user", "feedback", "project", "reference"}:
        memory_type = "project"

    tags_value = raw.get("tags", [])
    tags = [tag for tag in tags_value if isinstance(tag, str)] if isinstance(tags_value, list) else []

    return ExtractedMemory(
        action=action,
        filename=rewrite_heuristic_filename_for_session(
            _string_field(raw, "filename"),
            resolved_session_id,
        ),
        name=_string_field(raw, "name"),
        description=_string_field(raw, "description"),
        type=memory_type,
        content=_string_field(raw, "content"),
        index_entry=_string_field(raw, "indexEntry") or _string_field(raw, "index_entry"),
        scope=scope,
        supersedes=_string_field(raw, "supersedes"),
        tags=tags,
        session_id=resolved_session_id,
        observed_on=_string_field(raw, "observedOn") or _string_field(raw, "observed_on"),
        session_date=_string_field(raw, "sessionDate")
        or _string_field(raw, "session_date")
        or session_date,
        context_prefix=_string_field(raw, "contextPrefix")
        or _string_field(raw, "context_prefix"),
        modified_override=_string_field(raw, "modifiedOverride")
        or _string_field(raw, "modified_override"),
    )


def _string_field(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def has_memory_writes(messages: list[Message], *mem_dirs: str) -> bool:
    for m in messages:
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            if tc.name not in ("write", "edit"):
                continue
            args = tc.arguments
            for d in mem_dirs:
                if d and d in args:
                    return True
            if "memory/" in args:
                return True
    return False


def extract_session_summary(messages: list[Message]) -> str:
    for m in messages:
        if m.role == Role.SYSTEM and m.content.strip():
            return _truncate_one_line(m.content, 240)
    for m in messages:
        if m.role == Role.USER and m.content.strip():
            return _truncate_one_line(m.content, 240)
    return ""


def derive_session_date(messages: list[Message], session_date: str) -> str:
    if session_date.strip() != "" and parse_date_input(session_date.strip()) is not None:
        return session_date
    for message in messages:
        if message.role != Role.SYSTEM:
            continue
        matched = HEURISTIC_SESSION_DATE_RE.search(message.content)
        if matched is None:
            continue
        candidate = matched.group(0).strip()
        if parse_date_input(candidate) is not None:
            return candidate
    return session_date


def resolve_heuristic_session_id(
    messages: list[Message],
    extracted: list[ExtractedMemory],
    session_id: str,
) -> str:
    direct = session_id.strip()
    if direct != "":
        return direct
    for memory in extracted:
        candidate = memory.session_id.strip()
        if candidate != "":
            return candidate
    for message in messages:
        if message.role != Role.SYSTEM:
            continue
        matched = HEURISTIC_SESSION_ID_RE.search(message.content)
        if matched is None:
            continue
        candidate = matched.group(1).strip()
        if candidate != "":
            return candidate
    return ""


def resolve_heuristic_session_metadata(
    messages: list[Message],
    session_date: str,
) -> tuple[str, str]:
    direct = session_date.strip()
    if direct != "":
        parsed = parse_date_input(direct)
        if parsed is not None:
            return direct, parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for message in messages:
        if message.role != Role.SYSTEM:
            continue
        matched = HEURISTIC_SESSION_DATE_RE.search(message.content)
        if matched is None:
            continue
        candidate = matched.group(0).strip()
        parsed = parse_date_input(candidate)
        if parsed is not None:
            return candidate, parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return "", ""


def _truncate_one_line(s: str, n: int) -> str:
    s = s.replace("\r\n", " ").replace("\n", " ")
    s = " ".join(s.split())
    if n > 0 and len(s) > n:
        s = s[:n] + "..."
    return s


def _post_process_session_extractions(
    messages: list[Message],
    extracted: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    resolved_session_id = resolve_heuristic_session_id(messages, extracted, session_id)
    resolved_session_date, modified_override = resolve_heuristic_session_metadata(
        messages,
        session_date,
    )
    combined = list(extracted)
    combined.extend(
        derive_heuristic_user_facts(
            messages,
            combined,
            session_id=resolved_session_id,
            session_date=resolved_session_date,
        )
    )
    combined.extend(
        derive_heuristic_preference_facts(
            messages,
            combined,
            session_id=resolved_session_id,
            session_date=resolved_session_date,
        )
    )
    combined.extend(
        derive_heuristic_pending_facts(
            messages,
            combined,
            session_id=resolved_session_id,
            session_date=resolved_session_date,
        )
    )
    combined.extend(
        derive_heuristic_event_facts(
            messages,
            combined,
            session_id=resolved_session_id,
            session_date=resolved_session_date,
        )
    )
    combined.extend(
        derive_heuristic_milestone_facts(
            messages,
            combined,
            session_id=resolved_session_id,
            session_date=resolved_session_date,
        )
    )
    combined.extend(
        derive_heuristic_assistant_table_facts(
            messages,
            combined,
            session_id=resolved_session_id,
            session_date=resolved_session_date,
        )
    )
    if not combined:
        return []

    session_date_iso = short_iso_date(modified_override)
    date_tokens = build_date_tokens(modified_override)

    out: list[ExtractedMemory] = []
    for memory in combined:
        shaped = shape_extracted_memory(memory)
        next_session_id = shaped.session_id.strip() or resolved_session_id
        content = shaped.content
        if resolved_session_date and content.strip() and not content.startswith("[Date:"):
            content = f"{date_tokens}[Observed on {resolved_session_date}]\n\n{content}"
        tags = merge_tags(shaped.tags, auto_fact_tags(content))
        out.append(
            replace(
                shaped,
                filename=rewrite_heuristic_filename_for_session(
                    shaped.filename,
                    next_session_id,
                ),
                content=content,
                session_id=next_session_id,
                modified_override=shaped.modified_override or modified_override,
                observed_on=shaped.observed_on or modified_override,
                session_date=shaped.session_date or session_date_iso,
                tags=tags,
            )
        )
    return out


def shape_extracted_memory(memory: ExtractedMemory) -> ExtractedMemory:
    if memory.content.strip() == "":
        return memory
    summary = infer_searchable_summary(memory.content)
    if summary == "":
        return memory
    description = choose_more_specific_summary(memory.description, summary)
    index_entry = choose_more_specific_index_entry(memory.index_entry, summary)
    return replace(memory, description=description, index_entry=index_entry)


def infer_searchable_summary(content: str) -> str:
    text = strip_search_prefixes(content)
    if text == "":
        return ""
    preference = infer_heuristic_preference(text)
    return (
        infer_pending_task_summary(text)
        or infer_appointment_summary(text)
        or infer_event_summary(text)
        or (preference.summary if preference is not None else "")
    )

def strip_search_prefixes(content: str) -> str:
    return re.sub(r"^\[Observed on [^\]]+\]\n\n", "", re.sub(r"^\[Date:[^\]]+\]\n\n", "", content)).strip()


def choose_more_specific_summary(current: str, derived: str) -> str:
    cleaned_derived = _truncate_one_line(derived, 140)
    cleaned_current = current.strip()
    if cleaned_current == "":
        return cleaned_derived
    return cleaned_derived if is_less_specific_summary(cleaned_current, derived) else cleaned_current


def choose_more_specific_index_entry(current: str, derived: str) -> str:
    cleaned_derived = _truncate_one_line(derived, 140)
    cleaned_current = current.strip()
    if cleaned_current == "":
        return cleaned_derived
    if not is_less_specific_summary(cleaned_current, derived):
        return cleaned_current
    colon_index = cleaned_current.find(":")
    if cleaned_current.startswith("-") and colon_index > 0:
        return _truncate_one_line(
            f"{cleaned_current[: colon_index + 1]} {strip_trailing_full_stop(derived)}",
            140,
        )
    return cleaned_derived


def is_less_specific_summary(current: str, derived: str) -> bool:
    current_tokens = informative_summary_tokens(current)
    derived_tokens = informative_summary_tokens(derived)
    if not derived_tokens:
        return False
    if not current_tokens:
        return True
    missing = sum(1 for token in derived_tokens if token not in current_tokens)
    return missing >= max(2, (len(derived_tokens) + 1) // 2)


def informative_summary_tokens(value: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", value.lower()):
        if token in SUMMARY_STOPWORDS:
            continue
        out.add(token)
    return out


def merge_tags(existing: list[str], inferred: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in [*(existing or []), *inferred]:
        tag = raw.strip()
        if tag == "" or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def build_date_tokens(rfc3339: str) -> str:
    parsed = parse_rfc3339(rfc3339)
    iso = short_iso_date(rfc3339)
    if parsed is None or iso == "":
        return ""
    return f"[Date: {iso} {weekday_name(parsed)} {month_name(parsed)} {parsed.year}]\n\n"


def short_iso_date(rfc3339: str) -> str:
    parsed = parse_rfc3339(rfc3339)
    return parsed.strftime("%Y-%m-%d") if parsed is not None else ""


def parse_session_date_rfc3339(value: str) -> str:
    parsed = parse_date_input(value.strip())
    if parsed is None:
        return ""
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date_input(value: str) -> datetime | None:
    if value == "":
        return None
    matched = DATE_INPUT_RE.match(value)
    if matched is not None:
        year_raw, month_raw, day_raw, hour_raw, minute_raw, second_raw = matched.groups()
        return datetime(
            int(year_raw),
            int(month_raw),
            int(day_raw),
            int(hour_raw or "0"),
            int(minute_raw or "0"),
            int(second_raw or "0"),
            tzinfo=timezone.utc,
        )
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_rfc3339(value: str) -> datetime | None:
    if value.strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_heuristic_observed_on(
    sentence: str,
    anchor: datetime | None,
) -> str:
    if anchor is None:
        return ""

    matched = DATE_TAG_RE.search(sentence)
    if matched is not None:
        parsed = parse_date_input(matched.group(0).replace("-", "/"))
        if parsed is not None:
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")

    month_named = parse_month_name_date(sentence, anchor)
    if month_named is not None:
        return month_named.strftime("%Y-%m-%dT%H:%M:%SZ")

    for annotation in (
        resolve_relative_time(sentence, anchor),
        resolve_relative_day(sentence, anchor),
        resolve_last_week(sentence, anchor),
        resolve_last_weekday(sentence, anchor),
    ):
        if annotation is not None:
            observed = annotation.range_start.astimezone(timezone.utc).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            return observed.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ""


def parse_month_name_date(text: str, anchor: datetime) -> datetime | None:
    matched = MONTH_NAME_DATE_RE.search(text)
    if matched is None:
        return None
    raw = matched.group(0).replace(",", "")
    parts = raw.split()
    if len(parts) < 2:
        return None

    try:
        day = int(re.sub(r"(st|nd|rd|th)$", "", parts[1], flags=re.I))
    except ValueError:
        return None
    year = anchor.year
    if len(parts) >= 3:
        try:
            year = int(parts[2])
        except ValueError:
            return None

    try:
        month = datetime.strptime(parts[0], "%B").month
    except ValueError:
        return None

    resolved = datetime(year, month, day, tzinfo=timezone.utc)
    if len(parts) < 3 and resolved > anchor.replace(hour=0, minute=0, second=0, microsecond=0):
        resolved = resolved.replace(year=resolved.year - 1)
    return resolved


def with_observed_date_prefix(content: str, observed_on: str) -> str:
    trimmed = content.strip()
    if trimmed == "" or observed_on == "":
        return trimmed
    prefix = build_date_tokens(observed_on)
    return f"{prefix}{trimmed}" if prefix else trimmed


def auto_fact_tags(content: str) -> list[str]:
    if content == "":
        return []
    body = content[:4096]
    seen: set[str] = set()
    out: list[str] = []

    def add(value: str) -> None:
        tag = value.strip()
        if tag == "" or tag in seen:
            return
        seen.add(tag)
        out.append(tag)

    for match in DATE_TAG_RE.findall(body):
        add(match)
        parsed = parse_date_input(match.replace("-", "/"))
        if parsed is not None:
            add(weekday_name(parsed))
            add(month_name(parsed))
    for match in WEEKDAY_TAG_RE.findall(body):
        add(match[:1].upper() + match[1:].lower())
    for match in RELATIVE_TEMPORAL_TAG_RE.findall(body):
        add(match.lower())
    for match in CLOCK_TIME_TAG_RE.findall(body):
        add(match.lower())
    for match in MONEY_TAG_RE.findall(body):
        add(match)
    for quantity, unit in UNIT_QUANTITY_TAG_RE.findall(body):
        add(f"{quantity} {unit}")
    for match in QUANTITY_TAG_RE.findall(body):
        add(match)
    for match in PROPER_NOUN_TAG_RE.findall(body):
        if len(match) >= 3 and match.lower() not in AUTO_TAG_STOP_NOUNS:
            add(match)
    for match in PENDING_ACTION_TAG_RE.findall(body):
        add(match.lower())
    for match in MEDICAL_TAG_RE.findall(body):
        add(match.lower())
    for match in EVENT_TAG_RE.findall(body):
        add(match.lower())
    for match in ENTERTAINMENT_TAG_RE.findall(body):
        add(match.lower())
    if HEURISTIC_PENDING_ACTION_LEAD_RE.search(body) or re.search(
        r"\b(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|renew|cancel|follow\s+up)\b",
        body,
        re.I,
    ):
        add("pending")
        add("task")
    if HEURISTIC_APPOINTMENT_RE.search(body):
        add("appointment")
    if HEURISTIC_MEDICAL_ENTITY_RE.search(body) or re.search(
        r"\b(?:clinic|hospital|prescription)\b",
        body,
        re.I,
    ):
        add("medical")
    if HEURISTIC_EVENT_RE.search(body):
        add("event")
    if HEURISTIC_RECOMMENDATION_REQUEST_RE.search(body):
        add("recommendation")
    if re.search(
        r"\b(?:film|movie|show|series|book|novel|game|podcast|cinema)\b",
        body,
        re.I,
    ):
        add("entertainment")
    return out


def build_existing_memory_text_set(existing: list[ExtractedMemory]) -> set[str]:
    return {
        normalise_memory_text(value)
        for memory in existing
        for value in (
            memory.content,
            memory.description,
            memory.index_entry,
            infer_searchable_summary(memory.content),
        )
        if normalise_memory_text(value) != ""
    }


def normalise_memory_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def extract_pending_actions(sentence: str) -> list[str]:
    if sentence.strip().endswith("?"):
        return []
    matched = HEURISTIC_PENDING_ACTION_LEAD_RE.search(sentence)
    fragment = matched.group(1).strip() if matched is not None else ""
    if fragment == "":
        return []

    parts = [part.strip() for part in re.split(r"\s*(?:,|;|\bthen\b|\band\b)\s*", fragment, flags=re.I) if part.strip()]
    out: list[str] = []
    current = ""
    for part in parts:
        if HEURISTIC_PENDING_ACTION_START_RE.search(part) is not None:
            if current != "":
                out.append(current)
            current = part
            continue
        if current != "":
            current = f"{current} {part}".strip()

    if current != "":
        out.append(current)
    if out:
        return [clean_pending_action_clause(part) for part in out if clean_pending_action_clause(part) != ""]

    cleaned = clean_pending_action_clause(fragment)
    return [cleaned] if cleaned != "" else []


def clean_pending_action_clause(value: str) -> str:
    return " ".join(value.strip().strip(",:; ").split())


def build_pending_task_summary(action: str) -> str:
    return ensure_trailing_full_stop(
        f"The user still needs to {strip_trailing_full_stop(action)}"
    )


def infer_pending_task_summary(text: str) -> str | None:
    for sentence in split_into_fact_sentences(text):
        actions = extract_pending_actions(sentence)
        if actions:
            return build_pending_task_summary(actions[0])
    return None


def infer_appointment_summary(text: str) -> str | None:
    for sentence in split_into_fact_sentences(text):
        if (
            HEURISTIC_APPOINTMENT_RE.search(sentence) is None
            or HEURISTIC_PENDING_ACTION_LEAD_RE.search(sentence) is not None
            or sentence.strip().endswith("?")
        ):
            continue
        medical_entity = capture_whole_match(HEURISTIC_MEDICAL_ENTITY_RE, sentence)
        with_person = capture_group(HEURISTIC_WITH_PERSON_RE, sentence)
        temporal = extract_temporal_anchor(sentence)
        subject = (
            f"{with_indefinite_article(medical_entity)} {medical_entity} appointment"
            if medical_entity
            else "a medical appointment"
        )
        parts = [f"The user has {subject}"]
        if with_person:
            parts.append(f"with {with_person}")
        if temporal:
            parts.append(temporal)
        return ensure_trailing_full_stop(" ".join(parts))
    return None


def infer_event_summary(text: str) -> str | None:
    for sentence in split_into_fact_sentences(text):
        religious = infer_religious_service_summary(sentence)
        if religious is not None:
            return religious
        if (
            FIRST_PERSON_FACT_RE.search(sentence) is None
            or HEURISTIC_EVENT_RE.search(sentence) is None
            or HEURISTIC_PENDING_ACTION_LEAD_RE.search(sentence) is not None
            or HEURISTIC_EVENT_ATTENDANCE_RE.search(sentence) is None
            or sentence.strip().endswith("?")
        ):
            continue
        title = capture_group(HEURISTIC_EVENT_TITLE_RE, sentence) or extract_loose_event_phrase(sentence)
        if not title:
            continue
        parts = [f"The user attended {prefix_event_phrase(title)}"]
        temporal = extract_temporal_anchor(sentence)
        if temporal:
            parts.append(temporal)
        return ensure_trailing_full_stop(" ".join(parts))
    return None


def infer_religious_service_summary(sentence: str) -> str | None:
    if (
        FIRST_PERSON_FACT_RE.search(sentence) is None
        or HEURISTIC_PENDING_ACTION_LEAD_RE.search(sentence) is not None
        or sentence.strip().endswith("?")
    ):
        return None
    service = capture_group(HEURISTIC_RELIGIOUS_SERVICE_RE, sentence)
    if not service:
        return None
    return ensure_trailing_full_stop(
        f"The user attended {prefix_event_phrase(service)}"
    )


def extract_loose_event_phrase(sentence: str) -> str | None:
    matched = re.search(
        r"\b(?:a|an|the)\s+([^,.!?]+?\s+(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar|service|mass|worship|prayer))\b",
        sentence,
        re.I,
    )
    if matched is None:
        return None
    return matched.group(0).strip() or None


def prefix_event_phrase(value: str) -> str:
    trimmed = value.strip()
    if re.match(r"^(?:a|an|the)\b", trimmed, re.I):
        return trimmed
    return f"the {trimmed}"


def extract_temporal_anchor(text: str) -> str:
    date_anchor = capture_whole_match(DATE_TAG_RE, text) or capture_whole_match(
        HEURISTIC_RELATIVE_DATE_RE, text
    )
    time_anchor = capture_group(HEURISTIC_CLOCK_TIME_RE, text)
    parts: list[str] = []
    if date_anchor:
        parts.append(f"on {date_anchor}")
    if time_anchor:
        parts.append(f"at {time_anchor}")
    return " ".join(parts)


def capture_whole_match(pattern: re.Pattern[str], text: str) -> str:
    matched = pattern.search(text)
    return matched.group(0).strip() if matched is not None else ""


def ensure_trailing_full_stop(value: str) -> str:
    trimmed = value.strip()
    if trimmed.endswith((".", "!", "?")):
        return trimmed
    return f"{trimmed}."


def with_indefinite_article(value: str) -> str:
    trimmed = value.strip().lower()
    if trimmed[:1] in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def derive_heuristic_user_facts(
    messages: list[Message],
    existing: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    out: list[ExtractedMemory] = []
    seen = build_existing_memory_text_set(existing)
    iso = short_iso_date(parse_session_date_rfc3339(session_date))
    anchor = parse_date_input(session_date)
    for message in messages:
        if message.role != Role.USER:
            continue
        for sentence in heuristic_user_fact_candidates(message.content):
            canonical = sentence.lower()
            if (
                not FIRST_PERSON_FACT_RE.search(sentence)
                or not has_heuristic_user_fact(sentence)
                or canonical in seen
            ):
                continue
            slug = heuristic_fact_slug(sentence)
            if slug == "":
                continue
            observed_on = resolve_heuristic_observed_on(sentence, anchor)
            out.append(
                ExtractedMemory(
                    action="create",
                    filename=build_heuristic_session_filename(
                        "user-fact",
                        iso,
                        session_id,
                        slug,
                    ),
                    name=f"User Fact: {to_title_case(slug.replace('-', ' '))}",
                    description=_truncate_one_line(sentence, 140),
                    type="user",
                    content=with_observed_date_prefix(sentence, observed_on),
                    index_entry=_truncate_one_line(sentence, 140),
                    scope="global",
                    observed_on=observed_on,
                )
            )
            seen.add(canonical)
            if len(out) >= HEURISTIC_USER_FACT_LIMIT:
                return out
    return out


def derive_heuristic_milestone_facts(
    messages: list[Message],
    existing: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    out: list[ExtractedMemory] = []
    seen = build_existing_memory_text_set(existing)
    iso = short_iso_date(parse_session_date_rfc3339(session_date))
    anchor = parse_date_input(session_date)
    for message in messages:
        if message.role != Role.USER:
            continue
        for sentence in split_into_fact_sentences(message.content):
            canonical = sentence.lower()
            if not has_milestone_fact(sentence) or canonical in seen:
                continue
            slug = heuristic_fact_slug(sentence)
            if slug == "":
                continue
            observed_on = resolve_heuristic_observed_on(sentence, anchor)
            out.append(
                ExtractedMemory(
                    action="create",
                    filename=build_heuristic_session_filename(
                        "user-fact",
                        iso,
                        session_id,
                        f"milestone-{slug}",
                    ),
                    name=f"User Fact: {to_title_case(slug.replace('-', ' '))}",
                    description=_truncate_one_line(sentence, 140),
                    type="user",
                    content=with_observed_date_prefix(sentence, observed_on),
                    index_entry=_truncate_one_line(sentence, 140),
                    scope="global",
                    observed_on=observed_on,
                )
            )
            seen.add(canonical)
            if len(out) >= HEURISTIC_MILESTONE_FACT_LIMIT:
                return out
    return out


def derive_heuristic_pending_facts(
    messages: list[Message],
    existing: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    out: list[ExtractedMemory] = []
    seen = build_existing_memory_text_set(existing)
    iso = short_iso_date(parse_session_date_rfc3339(session_date))
    for message in messages:
        if message.role != Role.USER:
            continue
        for sentence in split_into_fact_sentences(message.content):
            for action in extract_pending_actions(sentence):
                summary = build_pending_task_summary(action)
                canonical = normalise_memory_text(summary)
                if canonical in seen:
                    continue
                slug = heuristic_fact_slug(summary)
                if slug == "":
                    continue
                out.append(
                    ExtractedMemory(
                        action="create",
                        filename=build_heuristic_session_filename(
                            "user-fact",
                            iso,
                            session_id,
                            f"task-{slug}",
                        ),
                        name=f"User Task: {to_title_case(action)}",
                        description=_truncate_one_line(summary, 140),
                        type="user",
                        content=f"{summary}\n\nEvidence: {sentence}",
                        index_entry=_truncate_one_line(summary, 140),
                        scope="global",
                    )
                )
                seen.add(canonical)
                if len(out) >= HEURISTIC_PENDING_FACT_LIMIT:
                    return out
    return out


def derive_heuristic_event_facts(
    messages: list[Message],
    existing: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    out: list[ExtractedMemory] = []
    seen = build_existing_memory_text_set(existing)
    iso = short_iso_date(parse_session_date_rfc3339(session_date))
    anchor = parse_date_input(session_date)
    for message in messages:
        if message.role != Role.USER:
            continue
        for sentence in split_into_fact_sentences(message.content):
            summary = infer_appointment_summary(sentence) or infer_event_summary(sentence)
            if summary is None:
                continue
            canonical = normalise_memory_text(summary)
            if canonical in seen:
                continue
            slug = heuristic_fact_slug(summary)
            if slug == "":
                continue
            observed_on = resolve_heuristic_observed_on(sentence, anchor)
            out.append(
                ExtractedMemory(
                    action="create",
                    filename=build_heuristic_session_filename(
                        "user-fact",
                        iso,
                        session_id,
                        f"event-{slug}",
                    ),
                    name=f"User Event: {to_title_case(slug.replace('-', ' '))}",
                    description=_truncate_one_line(summary, 140),
                    type="user",
                    content=with_observed_date_prefix(
                        f"{summary}\n\nEvidence: {sentence}",
                        observed_on,
                    ),
                    index_entry=_truncate_one_line(summary, 140),
                    scope="global",
                    observed_on=observed_on,
                )
            )
            seen.add(canonical)
            if len(out) >= HEURISTIC_EVENT_FACT_LIMIT:
                return out
    return out


def derive_heuristic_preference_facts(
    messages: list[Message],
    existing: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    out: list[ExtractedMemory] = []
    seen = build_existing_memory_text_set(existing)
    for memory in existing:
        if is_heuristic_preference_fact(memory):
            summary = heuristic_preference_summary(memory.content).lower()
            if summary:
                seen.add(summary)
    iso = short_iso_date(parse_session_date_rfc3339(session_date))
    for message in messages:
        if message.role != Role.USER:
            continue
        for candidate in build_heuristic_preference_candidates(message.content):
            canonical = candidate.summary.lower()
            if canonical in seen:
                continue
            slug = heuristic_fact_slug(candidate.summary)
            if slug == "":
                continue
            out.append(
                ExtractedMemory(
                    action="create",
                    filename=build_heuristic_session_filename(
                        "user-preference",
                        iso,
                        session_id,
                        slug,
                    ),
                    name=f"User Preference: {to_title_case(slug.replace('-', ' '))}",
                    description=_truncate_one_line(candidate.summary, 140),
                    type="user",
                    content=build_heuristic_preference_content(candidate),
                    index_entry=_truncate_one_line(candidate.summary, 140),
                    scope="global",
                )
            )
            seen.add(canonical)
            if len(out) >= HEURISTIC_PREFERENCE_FACT_LIMIT:
                return out
    return out


def derive_heuristic_assistant_table_facts(
    messages: list[Message],
    existing: list[ExtractedMemory],
    *,
    session_id: str = "",
    session_date: str = "",
) -> list[ExtractedMemory]:
    out: list[ExtractedMemory] = []
    seen = build_existing_memory_text_set(existing)
    iso = short_iso_date(parse_session_date_rfc3339(session_date))
    observed_on = parse_session_date_rfc3339(session_date)

    for message in messages:
        if message.role != Role.ASSISTANT:
            continue
        for table_lines in extract_markdown_table_blocks(message.content):
            parsed = parse_markdown_table(table_lines)
            if parsed is None:
                continue
            headers, rows = parsed
            if not rows:
                continue
            for row in rows:
                summary = build_weekday_table_row_summary(headers, row)
                if summary == "":
                    continue
                canonical = normalise_memory_text(summary)
                if canonical in seen:
                    continue
                slug = heuristic_fact_slug(summary)
                if slug == "":
                    continue
                out.append(
                    ExtractedMemory(
                        action="create",
                        filename=build_heuristic_filename("reference", iso, slug),
                        name=f"Reference: {to_title_case(slug.replace('-', ' '))}",
                        description=_truncate_one_line(summary, 140),
                        type="reference",
                        content=with_observed_date_prefix(summary, observed_on),
                        index_entry=_truncate_one_line(summary, 140),
                        scope="project",
                        observed_on=observed_on,
                    )
                )
                seen.add(canonical)
    return out


def extract_markdown_table_blocks(content: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if stripped == "":
            if current:
                blocks.append(current)
                current = []
            continue
        if "|" in stripped:
            current.append(stripped)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return [block for block in blocks if len(block) >= 3 and any(is_markdown_table_separator(line) for line in block)]


def parse_markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    for index in range(len(lines) - 1):
        header = split_markdown_table_row(lines[index])
        if len(header) < 2:
            continue
        if not is_markdown_table_separator(lines[index + 1]):
            continue
        rows = [split_markdown_table_row(line) for line in lines[index + 2 :]]
        rows = [row for row in rows if row]
        if not rows:
            return None
        width = max(len(header), max(len(row) for row in rows))
        return (
            pad_markdown_table_cells(header, width),
            [pad_markdown_table_cells(row, width) for row in rows],
        )
    return None


def split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells if any(cell != "" for cell in cells) else []


def pad_markdown_table_cells(cells: list[str], width: int) -> list[str]:
    if len(cells) >= width:
        return cells[:width]
    return [*cells, *[""] * (width - len(cells))]


def is_markdown_table_separator(line: str) -> bool:
    cells = split_markdown_table_row(line)
    if len(cells) < 2:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells)


def build_weekday_table_row_summary(headers: list[str], row: list[str]) -> str:
    weekday = canonical_weekday_label(row[0])
    if weekday == "":
        return ""
    if not is_weekday_table_lead_header(headers[0]):
        return ""

    row_cells = row[1:]
    header_cells = headers[1:]
    if not row_cells:
        return ""

    labelled_cells: list[str] = []
    for header, cell in zip(header_cells, row_cells):
        formatted = format_weekday_table_cell(header, cell)
        if formatted == "":
            continue
        labelled_cells.append(formatted)

    if not labelled_cells:
        return ""

    return ensure_trailing_full_stop(f"{weekday} roster: {'; '.join(labelled_cells)}")


def format_weekday_table_cell(header: str, value: str) -> str:
    header_clean = " ".join(header.split()).strip()
    value_clean = " ".join(value.split()).strip()
    if value_clean in {"", "-"}:
        return ""
    if header_clean == "":
        return value_clean
    return f"{value_clean}, {header_clean}"


def canonical_weekday_label(value: str) -> str:
    cleaned = " ".join(value.split()).strip().lower().rstrip(".,")
    mapping = {
        "mon": "Monday",
        "monday": "Monday",
        "tue": "Tuesday",
        "tues": "Tuesday",
        "tuesday": "Tuesday",
        "wed": "Wednesday",
        "wednesday": "Wednesday",
        "thu": "Thursday",
        "thur": "Thursday",
        "thurs": "Thursday",
        "thursday": "Thursday",
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return mapping.get(cleaned, "")


def is_weekday_table_lead_header(value: str) -> bool:
    cleaned = " ".join(value.split()).strip().lower().rstrip(".,")
    return cleaned == "" or re.fullmatch(
        r"(?:day(?:\s+of\s+week)?|weekday|week\s+day|schedule|roster|date)",
        cleaned,
    ) is not None


def split_into_fact_sentences(content: str) -> list[str]:
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")
    parts = [
        part.strip()
        for part in re.split(r"[\n]+|(?<=[.!?])\s+", normalised)
        if part.strip()
    ]
    out: list[str] = []
    index = 0
    while index < len(parts):
        sentence = parts[index]
        while (
            index + 1 < len(parts)
            and HEURISTIC_ABBREVIATED_SENTENCE_END_RE.search(sentence) is not None
        ):
            index += 1
            sentence = f"{sentence} {parts[index]}".strip()
        out.append(sentence)
        index += 1
    return out


def has_quantified_fact(sentence: str) -> bool:
    return bool(
        UNIT_QUANTITY_TAG_RE.search(sentence)
        or WORD_UNIT_QUANTITY_TAG_RE.search(sentence)
        or DATE_TAG_RE.search(sentence)
        or MONTH_NAME_DATE_RE.search(sentence)
        or HEURISTIC_DURATION_FACT_RE.search(sentence)
        or QUANTITY_TAG_RE.search(sentence)
        or HEURISTIC_ORDINAL_RE.search(sentence)
    )


def has_heuristic_user_fact(sentence: str) -> bool:
    return bool(
        not sentence.strip().endswith("?")
        and (
            has_quantified_fact(sentence)
            or HEURISTIC_CADENCE_FACT_RE.search(sentence)
            or HEURISTIC_LOCATION_STORAGE_FACT_RE.search(sentence)
        )
    )


def heuristic_user_fact_candidates(content: str) -> list[str]:
    candidates = split_into_fact_sentences(content)
    trimmed = content.strip()
    if trimmed == "" or (
        HEURISTIC_CADENCE_FACT_RE.search(trimmed) is None
        and HEURISTIC_LOCATION_STORAGE_FACT_RE.search(trimmed) is None
    ):
        return candidates
    canonical = trimmed.lower()
    if any(candidate.strip().lower() == canonical for candidate in candidates):
        return candidates
    return [*candidates, trimmed]


def has_milestone_fact(sentence: str) -> bool:
    return bool(
        FIRST_PERSON_FACT_RE.search(sentence)
        and not has_quantified_fact(sentence)
        and HEURISTIC_MILESTONE_TOPIC_RE.search(sentence)
        and (
            HEURISTIC_MILESTONE_EVENT_RE.search(sentence)
            or HEURISTIC_MILESTONE_TIME_RE.search(sentence)
        )
    )


def heuristic_fact_slug(sentence: str) -> str:
    words = HEURISTIC_WORD_RE.findall(sentence.lower())
    kept = [word for word in words if word not in HEURISTIC_STOPWORDS][:5]
    return "-".join(kept)


def build_heuristic_filename(prefix: str, iso: str, slug: str) -> str:
    parts = [prefix]
    if iso:
        parts.append(iso)
    parts.append(slug)
    return "-".join(parts) + ".md"


def build_heuristic_session_filename(
    prefix: str,
    iso: str,
    session_id: str,
    slug: str,
) -> str:
    return rewrite_heuristic_filename_for_session(
        build_heuristic_filename(prefix, iso, slug),
        session_id,
    )


def build_heuristic_preference_candidates(content: str) -> list[HeuristicPreferenceCandidate]:
    seen: set[str] = set()
    out: list[HeuristicPreferenceCandidate] = []
    for candidate in [normalise_preference_text(content), *split_into_fact_sentences(content)]:
        if candidate == "":
            continue
        inferred = infer_heuristic_preference(candidate)
        if inferred is None or inferred.summary.lower() in seen:
            continue
        seen.add(inferred.summary.lower())
        out.append(inferred)
    return out


def infer_heuristic_preference(content: str) -> HeuristicPreferenceCandidate | None:
    text = normalise_preference_text(content)
    if text == "":
        return None
    return (
        infer_explicit_preference(text)
        or infer_compatibility_preference(text)
        or infer_constraint_preference(text)
        or infer_advanced_preference(text)
    )


def infer_explicit_preference(text: str) -> HeuristicPreferenceCandidate | None:
    matched = HEURISTIC_PREFERENCE_BESIDES_LIKE_RE.search(text)
    if matched is not None:
        first = clean_preference_fragment(matched.group(1))
        second = clean_preference_fragment(matched.group(2))
        if first and second:
            hotel_match = re.match(r"^hotels?\s+with\s+(.+)$", second, re.I)
            summary = (
                f"The user prefers hotels with {first} and {clean_preference_fragment(hotel_match.group(1))}."
                if hotel_match is not None
                else f"The user prefers {first} and {second}."
            )
            return HeuristicPreferenceCandidate(summary=summary, evidence=text)
    matched = HEURISTIC_PREFERENCE_LIKE_RE.search(text)
    if matched is not None:
        fragment = clean_preference_fragment(matched.group(1))
        if fragment:
            return HeuristicPreferenceCandidate(
                summary=f"The user prefers {fragment}.",
                evidence=text,
            )
    return None


def infer_compatibility_preference(text: str) -> HeuristicPreferenceCandidate | None:
    raw_subject = capture_group(HEURISTIC_PREFERENCE_COMPATIBLE_RE, text)
    raw_subject = raw_subject or capture_group(HEURISTIC_PREFERENCE_DESIGNED_FOR_RE, text)
    raw_subject = raw_subject or capture_group(HEURISTIC_PREFERENCE_AS_USER_RE, text)
    if raw_subject == "":
        return None
    subject = clean_preference_fragment(raw_subject)
    if subject == "":
        return None
    return HeuristicPreferenceCandidate(
        summary=f"The user prefers {infer_preference_category(text)} compatible with their {subject}.",
        evidence=text,
    )


def infer_constraint_preference(text: str) -> HeuristicPreferenceCandidate | None:
    category = infer_recommendation_category(text)
    if category == "" or not HEURISTIC_RECOMMENDATION_REQUEST_RE.search(text):
        return None
    constraints = collect_recommendation_constraints(text)
    if not constraints:
        return None
    return HeuristicPreferenceCandidate(
        summary=f"The user prefers {category} with these constraints: {'; '.join(constraints)}.",
        evidence=text,
    )


def infer_advanced_preference(text: str) -> HeuristicPreferenceCandidate | None:
    topic = capture_group(HEURISTIC_PREFERENCE_FIELD_RE, text) or capture_group(
        HEURISTIC_PREFERENCE_ADVANCED_RE, text
    )
    if topic == "":
        return None
    if not (
        HEURISTIC_PREFERENCE_SKIP_BASICS_RE.search(text)
        or HEURISTIC_PREFERENCE_WORKING_IN_FIELD_RE.search(text)
        or "advanced" in text.lower()
    ):
        return None
    cleaned = clean_preference_fragment(topic)
    if cleaned == "":
        return None
    return HeuristicPreferenceCandidate(
        summary=f"The user prefers advanced publications, papers, and conferences on {cleaned} rather than introductory material.",
        evidence=text,
    )


def capture_group(pattern: re.Pattern[str], text: str) -> str:
    matched = pattern.search(text)
    return matched.group(1).strip() if matched is not None else ""


def infer_preference_category(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ("camera", "photography", "lens", "flash", "tripod", "camera bag", "gear")):
        return "photography accessories and gear"
    if any(word in lower for word in ("phone", "iphone", "screen protector", "power bank")):
        return "phone accessories"
    return infer_recommendation_category(text) or "accessories and options"


def infer_recommendation_category(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ("film", "movie", "cinema")):
        return "films"
    if any(word in lower for word in ("show", "series", "tv")):
        return "shows"
    if any(word in lower for word in ("book", "novel", "read", "reading")):
        return "books"
    if any(word in lower for word in ("hotel", "accommodation", "stay")):
        return "hotels"
    if any(word in lower for word in ("restaurant", "dinner", "lunch")):
        return "restaurants"
    if "game" in lower:
        return "games"
    if "podcast" in lower:
        return "podcasts"
    return ""


def collect_recommendation_constraints(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(value: str) -> None:
        cleaned = clean_preference_fragment(value)
        canonical = cleaned.lower()
        if cleaned == "" or canonical in seen:
            return
        seen.add(canonical)
        out.append(cleaned)

    if HEURISTIC_RECOMMENDATION_FAMILY_RE.search(text):
        add("family-friendly")
    matched = HEURISTIC_RECOMMENDATION_LIGHT_RE.search(text)
    if matched is not None:
        add(matched.group(0))
    not_too = capture_group(HEURISTIC_RECOMMENDATION_NOT_TOO_RE, text)
    if not_too:
        add(f"not too {not_too}")
    under = capture_group(HEURISTIC_RECOMMENDATION_UNDER_RE, text)
    if under:
        add(f"under {under}")
    without = capture_group(HEURISTIC_RECOMMENDATION_WITHOUT_RE, text)
    if without:
        add(f"without {without}")
    return out


def normalise_preference_text(value: str) -> str:
    return " ".join(value.split()).strip()


def clean_preference_fragment(value: str) -> str:
    return " ".join(value.strip(" ,:;").split()).strip(" ,:;")


def build_heuristic_preference_content(candidate: HeuristicPreferenceCandidate) -> str:
    return f"{candidate.summary}\n\nEvidence: {candidate.evidence}"


def heuristic_preference_summary(content: str) -> str:
    marker = "\n\nEvidence:"
    split = content.find(marker)
    return (content[:split] if split >= 0 else content).strip()


def is_heuristic_preference_fact(memory: ExtractedMemory) -> bool:
    return (
        memory.scope == "global"
        and memory.type == "user"
        and memory.filename.startswith("user-preference-")
    )


def sanitise_heuristic_file_segment(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9._-]", "-", value)).strip("-")


def rewrite_heuristic_filename_for_session(filename: str, session_id: str) -> str:
    session_segment = sanitise_heuristic_file_segment(session_id)
    if session_segment == "":
        return filename

    dated = HEURISTIC_FILENAME_DATED_RE.fullmatch(filename)
    if dated is not None:
        prefix, iso, rest = dated.groups()
        if rest == session_segment or rest.startswith(f"{session_segment}-"):
            return filename
        return f"{prefix}-{iso}-{session_segment}-{rest}.md"

    plain = HEURISTIC_FILENAME_RE.fullmatch(filename)
    if plain is not None:
        prefix, rest = plain.groups()
        if rest == session_segment or rest.startswith(f"{session_segment}-"):
            return filename
        return f"{prefix}-{session_segment}-{rest}.md"

    return filename


def to_title_case(value: str) -> str:
    return " ".join(part[:1].upper() + part[1:] for part in value.split() if part)


def strip_trailing_full_stop(value: str) -> str:
    return re.sub(r"[.!?]+$", "", value)


def weekday_name(value: datetime) -> str:
    return value.strftime("%A")


def month_name(value: datetime) -> str:
    return value.strftime("%B")


def build_manifest(topics: list[TopicFile]) -> str:
    lines: list[str] = []
    for t in topics:
        line = "- "
        if t.scope == "global":
            line += f"[global:{t.type}] "
        elif t.type:
            line += f"[{t.type}] "
        if _has_tag(t.tags, "heuristic"):
            conf = t.confidence or "low"
            line += f"[heuristic:{conf}] "
        line += base_name(t.path)
        if t.description:
            line += ": " + t.description
        lines.append(line)
    return "\n".join(lines).strip()


def build_manifests(project_topics: list[TopicFile], global_topics: list[TopicFile]) -> str:
    parts: list[str] = []
    pm = build_manifest(project_topics)
    if pm:
        parts.append("## Project memory files\n\n" + pm)
    gm = build_manifest(global_topics)
    if gm:
        parts.append("## Global memory files\n\n" + gm)
    return "\n\n".join(parts)


def _has_tag(tags: list[str], tag: str) -> bool:
    target = tag.lower()
    return any(t.lower() == target for t in tags)


def sanitise_filename(name: str) -> str:
    for sep in ("/", "\\"):
        idx = name.rfind(sep)
        if idx >= 0:
            name = name[idx + 1 :]
    return name


def apply_contextual_prefix(prefix: str, body: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return body
    return f"Context: {prefix}\n\n{body}"


def build_topic_file_content(em: ExtractedMemory) -> bytes:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    modified = em.modified_override or now
    created = em.modified_override or now

    lines: list[str] = ["---"]
    if em.name:
        lines.append(f"name: {em.name}")
    if em.description:
        lines.append(f"description: {em.description}")
    if em.type:
        lines.append(f"type: {em.type}")
    if em.action == "create":
        lines.append(f"created: {created}")
    lines.append(f"modified: {modified}")
    lines.append("source: session")
    if em.supersedes:
        lines.append(f"supersedes: {em.supersedes}")
    if em.session_id:
        lines.append(f"session_id: {em.session_id}")
    if em.observed_on:
        lines.append(f"observed_on: {em.observed_on}")
    if em.session_date:
        lines.append(f"session_date: {em.session_date}")
    if em.tags:
        lines.append(f"tags: [{', '.join(em.tags)}]")
    lines.append("---")
    lines.append("")
    body = apply_contextual_prefix(em.context_prefix, em.content)
    lines.append(body)
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")


def apply_extractions(
    mem: "MemoryManager", project_slug: str, memories: list[ExtractedMemory]
) -> None:
    project_entries: list[str] = []
    global_entries: list[str] = []

    pending: list[tuple[str, bytes]] = []
    for em in memories:
        if not em.filename or not em.content:
            continue

        filename = sanitise_filename(em.filename)
        if not filename.endswith(".md"):
            filename += ".md"
        slug = filename[: -len(".md")]

        if em.scope == "global":
            path = memory_global_topic(slug)
        else:
            path = memory_project_topic(project_slug, slug)

        pending.append((path, build_topic_file_content(em)))

        if em.index_entry:
            if em.scope == "global":
                global_entries.append(em.index_entry)
            else:
                project_entries.append(em.index_entry)

    if not pending:
        return

    def _run(b) -> None:
        for path, content in pending:
            b.write(path, content)

        for em in memories:
            if not em.supersedes:
                continue
            old_file = sanitise_filename(em.supersedes)
            if not old_file.endswith(".md"):
                old_file += ".md"
            old_slug = old_file[: -len(".md")]
            new_file = sanitise_filename(em.filename)
            if not new_file.endswith(".md"):
                new_file += ".md"
            if em.scope == "global":
                old_path = memory_global_topic(old_slug)
                fallback_old_path = memory_project_topic(project_slug, old_slug)
            else:
                old_path = memory_project_topic(project_slug, old_slug)
                fallback_old_path = memory_global_topic(old_slug)
            if not _stamp_superseded_by(b, old_path, new_file):
                _stamp_superseded_by(b, fallback_old_path, new_file)

        if project_entries:
            _append_index_entries(b, memory_project_index(project_slug), project_entries)
        if global_entries:
            _append_index_entries(b, memory_global_index(), global_entries)

    mem.store.batch(_run)


def _stamp_superseded_by(b, old_path: str, new_file: str) -> bool:
    try:
        raw = b.read(old_path)
    except NotFoundError:
        return False
    content = raw.decode("utf-8")
    lines = content.split("\n")
    if len(lines) < 2 or lines[0].strip() != "---":
        return False
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:
        return False

    replaced = False
    for i in range(1, close_idx):
        if lines[i].strip().startswith("superseded_by:"):
            lines[i] = f"superseded_by: {new_file}"
            replaced = True
            break
    if not replaced:
        lines = lines[:close_idx] + [f"superseded_by: {new_file}"] + lines[close_idx:]

    b.write(old_path, "\n".join(lines).encode("utf-8"))
    return True


def _append_index_entries(b, index_path: str, entries: list[str]) -> None:
    content = ""
    try:
        existing = b.read(index_path)
        content = existing.decode("utf-8").strip()
    except NotFoundError:
        content = ""
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if entry in content:
            continue
        if content:
            content += "\n"
        content += entry
    b.write(index_path, (content + "\n").encode("utf-8"))


def _role_value(role) -> str:
    if isinstance(role, Role):
        return role.value
    return str(role)
