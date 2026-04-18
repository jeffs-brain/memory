# SPDX-License-Identifier: Apache-2.0
"""`/ask` SSE endpoint: retrieve -> answer_delta* -> citation* -> done."""

from __future__ import annotations

import json
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

    # BM25 fallback for empty retrievals — mirrors the Go daemon.
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


def _build_prompt(question: str, chunks: list[retrieval.RetrievedChunk]) -> str:
    parts: list[str] = []
    if chunks:
        parts.append("## Evidence\n")
        for chunk in chunks:
            title = chunk.title or chunk.path
            path = chunk.path
            body = chunk.text or chunk.summary
            parts.append(f"### {title} ({path})\n{body}\n")
    parts.append("## Question\n")
    parts.append(question)
    return "\n".join(parts)


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

        prompt = _build_prompt(question, chunks)
        complete_request = CompleteRequest(
            model=str(model),
            messages=[
                Message(role=Role.SYSTEM, content=_ASK_SYSTEM),
                Message(role=Role.USER, content=prompt),
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        try:
            stream = provider.complete_stream(complete_request)
            # The Fake provider returns an async iterator through
            # `await` — accommodate both shapes.
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
