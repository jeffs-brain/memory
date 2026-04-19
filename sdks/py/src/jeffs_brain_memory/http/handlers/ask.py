# SPDX-License-Identifier: Apache-2.0
"""`/ask` SSE endpoint: retrieve -> answer_delta* -> citation* -> done."""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, AsyncIterator

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ... import retrieval
from ...augmented_reader import resolve_deterministic_augmented_answer
from ...llm.types import CompleteRequest, Message, Role
from ...search.frontmatter import parse_memory_frontmatter
from ..problem import validation_error
from ._shared import decode_json_body, resolve_brain
from .search import filters_from_body, path_matches_filters, search_opts

_ASK_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

_ASK_SYSTEM = (
    "You are Jeffs Brain, a helpful assistant. When evidence is supplied, "
    "ground your answer in it and cite the path of any source you rely on. "
    "When no evidence is supplied, answer concisely from general knowledge."
)

# LME CoT reader settings, matching the Go runner's reader.go constants.
_AUGMENTED_TEMPERATURE = 0.0
_AUGMENTED_MAX_TOKENS = 800
_BASIC_TEMPERATURE = 0.2
_BASIC_MAX_TOKENS = 1024

# Reader template ported verbatim from sdks/go/eval/lme/reader.go so the
# augmented Python /ask matches the benchmark harness byte for byte.
_AUGMENTED_READER_TEMPLATE = (
    "I will give you several history chats between you and a user. "
    "Please answer the question based on the relevant chat history. "
    "Answer the question step by step: first extract all the relevant "
    "information, and then reason over the information to get the answer.\n"
    "\n"
    "Resolving conflicting information:\n"
    "- Each fact is tagged with a date. When the same topic appears with "
    "different values on different dates, prefer the value from the most "
    "recent session date.\n"
    "- Treat explicit supersession phrases as hard overrides regardless of "
    "how often the old value appears: \"now\", \"currently\", \"most "
    "recently\", \"actually\", \"correction\", \"I updated\", \"I "
    "changed\", \"no longer\".\n"
    "- Do not vote by frequency. One later correction outweighs any number "
    "of earlier mentions.\n"
    "- Never use a fact dated after the current date.\n"
    "- When the question names a specific item, event, place, or "
    "descriptor, prefer the fact that matches that target most directly. "
    "Do not substitute a broader category match or a different example "
    "from the same topic.\n"
    "- A direct statement of the full usual value outranks a newer note "
    "about only one segment, leg, or example from that routine unless "
    "the newer note explicitly says the full value changed.\n"
    "- For habit and routine questions (\"usually\", \"normally\", "
    "\"every week\", \"on Saturdays\", \"on weekdays\"), prefer "
    "explicit habitual statements over isolated single-day examples.\n"
    "- Do not let an example note about a narrower segment override the "
    "whole routine. For example, a \"30-minute morning commute\" note "
    "does not replace a direct statement of a \"45-minute daily "
    "commute to work\".\n"
    "- When one fact names the event and another fact gives the "
    "associated submission, booking, or join date for that same event "
    "or venue, combine them if the connection is explicit in the "
    "retrieved facts.\n"
    "\n"
    "Enumeration and counting:\n"
    "- When the question asks to list, count, enumerate, or total (\"how "
    "many\", \"list\", \"which\", \"what are all\", \"total\", \"in "
    "total\"), return every matching item you find across the retrieved "
    "facts, one per line, each tagged with its session date. Then state "
    "the count or total explicitly at the end.\n"
    "- Do not summarise into a single sentence when the question demands a "
    "list.\n"
    "- Add numeric values across sessions when the question asks for a "
    "total (hours, days, money, items). Show the arithmetic.\n"
    "- When both atomic event facts and retrospective roll-up summaries "
    "are present, prefer the atomic event facts and avoid double "
    "counting the roll-up.\n"
    "- Treat first-person past-tense purchases, gifts, sales, earnings, "
    "completions, or submissions as confirmed historical events even "
    "when they appear inside a planning or advice conversation. Exclude "
    "only clearly hypothetical or planned amounts.\n"
    "- If a spending or earnings question does not explicitly restrict "
    "the timeframe (\"today\", \"this time\", \"most recent\", "
    "\"current\"), include all confirmed historical amounts for the "
    "same subject across sessions.\n"
    "- For totals over named items, sum only the facts that match those "
    "named items directly. Do not add alternative purchases, adjacent "
    "examples, or broader category summaries unless the note clearly "
    "says they refer to the same item.\n"
    "- When a total names multiple specific items, people, or occasions, "
    "every named part must be supported directly. If any named part is "
    "missing or lacks an amount, do not return a partial total. State "
    "that the information provided is not enough.\n"
    "- When the question names a singular item plus another category, "
    "choose the single best-matching fact for that singular item. Do "
    "not combine multiple different handbags, flights, meals, or other "
    "same-category purchases unless the question explicitly asks for "
    "all of them.\n"
    "- When multiple notes appear to describe the same purchase, gift, "
    "booking, or transaction, count it once. Prefer the most direct "
    "transactional fact over recap notes, budget summaries, tracker "
    "entries, or assistant bookkeeping.\n"
    "- For \"spent\", \"cost\", and \"total amount\" questions, prefer "
    "direct transactional facts over plans, budgets, broad summaries, "
    "or calculations that only restate the same purchase.\n"
    "\n"
    "Preference-sensitive questions:\n"
    "- When the user asks for ideas, advice, inspiration, or recommendations, "
    "anchor the answer in explicit prior preferences, recent projects, "
    "recurring habits, and stated dislikes from the retrieved facts.\n"
    "- Avoid generic suggestions when the history already contains concrete "
    "tastes or recent examples. Reuse those specifics directly in the "
    "answer.\n"
    "- Infer durable preferences from concrete desired features or liked "
    "attributes even when the earlier example was tied to a different "
    "city, venue, or product.\n"
    "- When concrete amenities or features are present, prefer them over "
    "generic travel style or budget signals.\n"
    "- Ignore unrelated hostel, budget, or solo-travel examples when the "
    "retrieved facts already contain a clearer accommodation-feature "
    "preference and the question does not ask about price.\n"
    "- When the question asks for a specific or exact previously "
    "recommended item, answer with the narrowest directly supported set "
    "from the retrieved facts. Do not widen the answer with adjacent "
    "frameworks, resource catalogues, or loosely related examples.\n"
    "\n"
    "Unanswerable questions:\n"
    "- If the retrieved facts do not directly answer the question, state "
    "that clearly in the first sentence.\n"
    "- Keep the extraction step brief and limited to the missing "
    "subject. Do not narrate your search process.\n"
    "- Do not pad the answer with near-miss facts about a different "
    "city, person, product, or date unless they directly explain why "
    "the requested fact is unavailable.\n"
    "- End with a direct abstention that the information provided is not "
    "enough to answer the question.\n"
    "\n"
    "Temporal reasoning:\n"
    "- Today is {today_anchor} (this is the current date). Resolve "
    "relative references (\"recently\", \"last week\", \"a few days ago\", "
    "\"this month\") against this anchor.\n"
    "- For date-arithmetic questions (\"how many days between X and Y\"), "
    "first extract each event's ISO date from the fact tags, then compute "
    "the difference in days.\n"
    "\n"
    "History Chats:\n"
    "\n"
    "{content}\n"
    "\n"
    "Current Date: {current_date}\n"
    "Question: {question}\n"
    "Answer (step by step):"
)

# Surface forms accepted by both the Go runner and the LME dataset.
_QUESTION_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)

def _reader_today_anchor(question_date: str) -> str:
    """Render the temporal grounding line as ``YYYY-MM-DD (Weekday)``.

    Mirrors ``readerTodayAnchor`` in the Go reader. Falls back to
    ``"unknown"`` for empty input or the raw string when no layout
    parses.
    """
    s = question_date.strip()
    if not s:
        return "unknown"
    for fmt in _QUESTION_DATE_FORMATS:
        try:
            parsed = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return f"{parsed.strftime('%Y-%m-%d')} ({parsed.strftime('%A')})"
    # ISO-8601 last so the cheaper layouts win first.
    try:
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    return f"{parsed.strftime('%Y-%m-%d')} ({parsed.strftime('%A')})"


def _format_event(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def _chunk_to_wire(chunk: retrieval.RetrievedChunk) -> dict[str, Any]:
    return {
        "chunkId": chunk.chunk_id,
        "documentId": chunk.document_id,
        "path": chunk.path,
        "score": chunk.score,
        "text": chunk.text,
        "title": chunk.title,
        "summary": chunk.summary,
    }


async def _retrieve(
    br: Any,
    question: str,
    top_k: int,
    mode: str,
    question_date: str,
    candidate_k: int,
    rerank_top_n: int,
    filters: retrieval.Filters | None = None,
) -> list[retrieval.RetrievedChunk]:
    filters = filters or retrieval.Filters()
    try:
        selected_mode = retrieval.Mode(mode) if mode else retrieval.Mode.AUTO
    except ValueError:
        selected_mode = retrieval.Mode.AUTO
    req = retrieval.Request(
        query=question,
        question_date=question_date,
        top_k=top_k,
        mode=selected_mode,
        brain_id=br.id,
        filters=filters,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
    )
    try:
        resp = await br.retriever.retrieve(req)
    except Exception:  # noqa: BLE001
        resp = None
    chunks = (
        [chunk for chunk in resp.chunks if path_matches_filters(chunk.path, filters)]
        if resp
        else []
    )

    # BM25 fallback for empty retrievals, mirroring the Go daemon.
    if not chunks:
        try:
            hits = br.search_index.search_bm25(
                retrieval.augment_query_with_temporal(question, question_date),
                top_k=top_k,
                opts=search_opts(top_k, filters),
            )
        except Exception:  # noqa: BLE001
            hits = []
        for h in hits:
            if not path_matches_filters(h.path, filters):
                continue
            score = 1.0 / (1.0 + abs(h.score)) if h.score else 0.0
            chunks.append(
                retrieval.RetrievedChunk(
                    chunk_id=h.chunk_id or h.path,
                    document_id=h.document_id or h.path,
                    path=h.path,
                    score=score,
                    text=h.content or h.snippet,
                    title=h.title,
                    summary=h.summary,
                    metadata=dict(h.metadata),
                )
            )

    if len(chunks) > top_k:
        chunks = chunks[:top_k]
    return chunks


def _format_chunks(chunks: list[retrieval.RetrievedChunk]) -> str:
    """Render chunks as ``### title (path)\\n{text or summary}`` blocks."""
    out: list[str] = []
    for chunk in chunks:
        title = chunk.title or chunk.path
        body = chunk.text or chunk.summary
        out.append(f"### {title} ({chunk.path})\n{body}\n")
    return "\n".join(out)


def _metadata_value(chunk: retrieval.RetrievedChunk, *keys: str) -> str:
    for key in keys:
        value = chunk.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@lru_cache(maxsize=256)
def _parse_chunk_body(body: str) -> tuple[str, str, str, str, str]:
    stripped = body.strip()
    if not stripped.startswith("---"):
        return "", "", "", "", stripped
    frontmatter, parsed_body = parse_memory_frontmatter(body)
    return (
        frontmatter.session_id.strip(),
        frontmatter.session_date.strip(),
        frontmatter.observed_on.strip(),
        frontmatter.modified.strip(),
        parsed_body.strip(),
    )


def _chunk_session_id(chunk: retrieval.RetrievedChunk) -> str:
    metadata_value = _metadata_value(chunk, "session_id", "sessionId")
    if metadata_value:
        return metadata_value
    session_id, _, _, _, _ = _parse_chunk_body(chunk.text or chunk.summary)
    return session_id


def _chunk_date(chunk: retrieval.RetrievedChunk) -> str:
    metadata_value = _metadata_value(
        chunk,
        "session_date",
        "sessionDate",
        "observed_on",
        "observedOn",
        "modified",
    )
    if metadata_value:
        return metadata_value
    _, session_date, observed_on, modified, _ = _parse_chunk_body(
        chunk.text or chunk.summary
    )
    return session_date or observed_on or modified


def _display_body(chunk: retrieval.RetrievedChunk) -> str:
    _, _, _, _, body = _parse_chunk_body(chunk.text or chunk.summary)
    return body


def _cluster_chunks_by_session(
    chunks: list[retrieval.RetrievedChunk],
) -> list[retrieval.RetrievedChunk]:
    if len(chunks) <= 1:
        return list(chunks)
    order: list[str] = []
    groups: dict[str, list[retrieval.RetrievedChunk]] = {}
    for index, chunk in enumerate(chunks):
        session_id = _chunk_session_id(chunk) or f"__solo_{index}__"
        if session_id not in groups:
            order.append(session_id)
            groups[session_id] = []
        groups[session_id].append(chunk)
    ordered: list[retrieval.RetrievedChunk] = []
    for session_id in order:
        ordered.extend(groups.get(session_id, []))
    return ordered


def _source_tag(path: str) -> str:
    name = PurePosixPath(path).name
    return name[:-3] if name.endswith(".md") else name


def _format_augmented_chunks(
    question: str,
    chunks: list[retrieval.RetrievedChunk],
    question_date: str,
) -> str:
    ordered = _cluster_chunks_by_session(chunks)
    if not ordered:
        return ""
    parts: list[str] = []
    temporal_hint = retrieval.resolved_temporal_hint_line(question, question_date)
    if temporal_hint is not None:
        parts.append(temporal_hint)
        parts.append("")
    parts.append(f"Retrieved facts ({len(ordered)}):")
    parts.append("")
    for index, chunk in enumerate(ordered, start=1):
        labels = [f"[{_chunk_date(chunk) or 'unknown'}]"]
        session_id = _chunk_session_id(chunk)
        if session_id:
            labels.append(f"[session={session_id}]")
        source = _source_tag(chunk.path)
        if source:
            labels.append(f"[{source}]")
        parts.append(f"{index:2d}. {' '.join(labels)}")
        parts.append(_display_body(chunk))
        parts.append("")
    return "\n".join(parts).strip()


def _build_basic_prompt(
    question: str, chunks: list[retrieval.RetrievedChunk]
) -> str:
    parts: list[str] = []
    if chunks:
        parts.append("## Evidence\n")
        parts.append(_format_chunks(chunks))
    parts.append("## Question\n")
    parts.append(question)
    return "\n".join(parts)


def _build_augmented_prompt(
    question: str,
    chunks: list[retrieval.RetrievedChunk],
    question_date: str,
) -> str:
    """Render the LME CoT reader prompt with recency, enumeration, and
    temporal guidance. Empty retrieval is rendered as a blank History
    Chats block so the template's structure stays stable for the model.
    """
    content = _format_augmented_chunks(question, chunks, question_date)
    return _build_augmented_prompt_from_content(question, content, question_date)


def _build_augmented_prompt_from_content(
    question: str,
    content: str,
    question_date: str,
) -> str:
    today_anchor = _reader_today_anchor(question_date)
    current_date = question_date.strip() or "unknown"
    return _AUGMENTED_READER_TEMPLATE.format(
        today_anchor=today_anchor,
        content=content,
        current_date=current_date,
        question=question,
    )


async def ask(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 256 * 1024)
    if isinstance(body, Response):
        return body
    question = body.get("question")
    if not isinstance(question, str) or not question:
        return validation_error("question required")
    top_k_raw = body.get("topK")
    top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else 5
    candidate_k_raw = body.get("candidateK")
    if not isinstance(candidate_k_raw, int) or candidate_k_raw <= 0:
        candidate_k_raw = body.get("candidate_k")
    candidate_k = (
        candidate_k_raw if isinstance(candidate_k_raw, int) and candidate_k_raw > 0 else 0
    )
    rerank_top_n_raw = body.get("rerankTopN")
    if not isinstance(rerank_top_n_raw, int) or rerank_top_n_raw <= 0:
        rerank_top_n_raw = body.get("rerank_top_n")
    rerank_top_n = (
        rerank_top_n_raw
        if isinstance(rerank_top_n_raw, int) and rerank_top_n_raw > 0
        else 0
    )
    mode = body.get("mode") or ""
    model = body.get("model") or ""

    reader_mode_raw = body.get("reader_mode") or body.get("readerMode") or "basic"
    reader_mode = (
        str(reader_mode_raw).strip().lower() if reader_mode_raw else "basic"
    )
    if reader_mode not in ("basic", "augmented"):
        return validation_error(
            "reader_mode must be 'basic' or 'augmented'"
        )
    question_date_raw = (
        body.get("question_date") or body.get("questionDate") or ""
    )
    question_date = (
        str(question_date_raw) if isinstance(question_date_raw, str) else ""
    )
    filters = filters_from_body(
        body.get("filters") if isinstance(body.get("filters"), dict) else body
    )

    # Run retrieval before opening the stream so retrieval-time errors
    # still surface as Problem+JSON. Once the headers flush, every
    # failure rides the stream as an `error` event.
    chunks = await _retrieve(
        br,
        question,
        top_k,
        str(mode),
        question_date,
        candidate_k,
        rerank_top_n,
        filters,
    )
    daemon = request.app.state.daemon  # type: ignore[attr-defined]
    provider = daemon.llm

    async def event_stream() -> AsyncIterator[bytes]:
        retrieve_payload = {
            "chunks": [_chunk_to_wire(c) for c in chunks],
            "topK": top_k,
            "mode": mode,
        }
        yield _format_event("retrieve", json.dumps(retrieve_payload))

        if reader_mode == "augmented":
            rendered_evidence = _format_augmented_chunks(question, chunks, question_date)
            deterministic_answer = resolve_deterministic_augmented_answer(
                question, rendered_evidence
            )
            if deterministic_answer is not None:
                yield _format_event(
                    "answer_delta",
                    json.dumps({"text": deterministic_answer}),
                )
                for chunk in chunks:
                    yield _format_event(
                        "citation",
                        json.dumps(
                            {
                                "chunkId": chunk.chunk_id,
                                "path": chunk.path,
                                "title": chunk.title,
                                "score": chunk.score,
                            }
                        ),
                    )
                yield _format_event("done", json.dumps({"ok": True}))
                return

            prompt = _build_augmented_prompt_from_content(
                question,
                rendered_evidence,
                question_date,
            )
            # Augmented prompt embeds its own system-style preamble; the
            # Go reader sends a single user turn, so we mirror that.
            messages = [Message(role=Role.USER, content=prompt)]
            temperature = _AUGMENTED_TEMPERATURE
            max_tokens = _AUGMENTED_MAX_TOKENS
        else:
            prompt = _build_basic_prompt(question, chunks)
            messages = [
                Message(role=Role.SYSTEM, content=_ASK_SYSTEM),
                Message(role=Role.USER, content=prompt),
            ]
            temperature = _BASIC_TEMPERATURE
            max_tokens = _BASIC_MAX_TOKENS

        complete_request = CompleteRequest(
            model=str(model),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            stream = provider.complete_stream(complete_request)
            # The Fake provider returns an async iterator through
            # `await`; accommodate both shapes.
            if hasattr(stream, "__await__"):
                stream = await stream  # type: ignore[assignment]
            async for chunk in stream:  # type: ignore[async-for]
                if await request.is_disconnected():
                    return
                if chunk.delta_text:
                    yield _format_event(
                        "answer_delta",
                        json.dumps({"text": chunk.delta_text}),
                    )
                if chunk.stop is not None:
                    break
        except Exception as exc:  # noqa: BLE001
            yield _format_event("error", json.dumps({"message": str(exc)}))
            yield _format_event("done", json.dumps({"ok": False}))
            return

        for chunk in chunks:
            yield _format_event(
                "citation",
                json.dumps(
                    {
                        "chunkId": chunk.chunk_id,
                        "path": chunk.path,
                        "title": chunk.title,
                        "score": chunk.score,
                    }
                ),
            )
        yield _format_event("done", json.dumps({"ok": True}))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_ASK_SSE_HEADERS,
    )
