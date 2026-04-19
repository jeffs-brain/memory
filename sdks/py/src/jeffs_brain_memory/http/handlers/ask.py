# SPDX-License-Identifier: Apache-2.0
"""`/ask` SSE endpoint: retrieve -> answer_delta* -> citation* -> done."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncIterator

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ... import retrieval, search as search_pkg
from ...llm.types import CompleteRequest, Message, Role
from ..problem import validation_error
from ._shared import decode_json_body, resolve_brain

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
    br: Any, question: str, top_k: int, mode: str
) -> list[retrieval.RetrievedChunk]:
    try:
        selected_mode = retrieval.Mode(mode) if mode else retrieval.Mode.AUTO
    except ValueError:
        selected_mode = retrieval.Mode.AUTO
    req = retrieval.Request(
        query=question,
        top_k=top_k,
        mode=selected_mode,
        brain_id=br.id,
    )
    try:
        resp = await br.retriever.retrieve(req)
    except Exception:  # noqa: BLE001
        resp = None
    chunks: list[retrieval.RetrievedChunk] = list(resp.chunks) if resp else []

    # BM25 fallback for empty retrievals, mirroring the Go daemon.
    if not chunks:
        try:
            hits = br.search_index.search_bm25(
                question,
                top_k=top_k,
                opts=search_pkg.SearchOpts(max_results=top_k),
            )
        except Exception:  # noqa: BLE001
            hits = []
        for h in hits:
            score = 1.0 / (1.0 + abs(h.score)) if h.score else 0.0
            chunks.append(
                retrieval.RetrievedChunk(
                    chunk_id=h.chunk_id or h.path,
                    document_id=h.document_id or h.path,
                    path=h.path,
                    score=score,
                    text=h.snippet,
                    title=h.title,
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
    content = _format_chunks(chunks)
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

    # Run retrieval before opening the stream so retrieval-time errors
    # still surface as Problem+JSON. Once the headers flush, every
    # failure rides the stream as an `error` event.
    chunks = await _retrieve(br, question, top_k, str(mode))
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
            prompt = _build_augmented_prompt(question, chunks, question_date)
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
