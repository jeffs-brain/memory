# SPDX-License-Identifier: Apache-2.0
"""Search handler. Delegates to the retriever, falls back to BM25."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ... import retrieval, search as search_pkg
from ..problem import internal_error, validation_error
from ._shared import decode_json_body, ok_json, resolve_brain


def _filters_from_body(raw: Any) -> retrieval.Filters:
    if not isinstance(raw, dict):
        return retrieval.Filters()
    tags_raw = raw.get("tags") or []
    tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
    return retrieval.Filters(
        path_prefix=str(raw.get("pathPrefix") or raw.get("path_prefix") or ""),
        tags=tags,
        scope=str(raw.get("scope") or ""),
        project=str(raw.get("project") or ""),
    )


def _trace_to_wire(trace: retrieval.Trace) -> dict[str, Any]:
    payload = asdict(trace)
    for key in ("requested_mode", "effective_mode"):
        value = payload.get(key)
        if hasattr(value, "value"):
            payload[key] = value.value
    return payload


def _attempt_to_wire(attempt: retrieval.Attempt) -> dict[str, Any]:
    payload = asdict(attempt)
    value = payload.get("mode")
    if hasattr(value, "value"):
        payload["mode"] = value.value
    return payload


def _chunk_to_wire(chunk: retrieval.RetrievedChunk) -> dict[str, Any]:
    return {
        "chunkId": chunk.chunk_id,
        "documentId": chunk.document_id,
        "path": chunk.path,
        "score": chunk.score,
        "text": chunk.text,
        "title": chunk.title,
        "summary": chunk.summary,
        "metadata": chunk.metadata,
        "bm25Rank": chunk.bm25_rank,
        "vectorSimilarity": chunk.vector_similarity,
        "rerankScore": chunk.rerank_score,
    }


async def search(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 256 * 1024)
    if isinstance(body, Response):
        return body

    query = body.get("query")
    if not isinstance(query, str) or not query:
        return validation_error("query required")
    top_k_raw = body.get("topK")
    top_k = top_k_raw if isinstance(top_k_raw, int) and top_k_raw > 0 else 10
    mode_raw = body.get("mode") or ""
    filters = _filters_from_body(body.get("filters"))

    started = time.perf_counter()
    try:
        mode = retrieval.Mode(str(mode_raw)) if mode_raw else retrieval.Mode.AUTO
    except ValueError:
        mode = retrieval.Mode.AUTO

    req = retrieval.Request(
        query=query,
        top_k=top_k,
        mode=mode,
        brain_id=br.id,
        filters=filters,
    )

    chunks: list[dict[str, Any]] = []
    trace: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []

    try:
        resp = await br.retriever.retrieve(req)
        if resp.chunks:
            chunks = [_chunk_to_wire(c) for c in resp.chunks]
        trace = _trace_to_wire(resp.trace)
        attempts = [_attempt_to_wire(a) for a in resp.attempts]
        took_ms = resp.took_ms or int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))

    # Fall back to raw BM25 when the retriever produced nothing.
    if not chunks:
        try:
            hits = br.search_index.search_bm25(
                query,
                top_k=top_k,
                opts=search_pkg.SearchOpts(max_results=top_k),
            )
        except Exception:  # noqa: BLE001
            hits = []
        for h in hits:
            score = 1.0 / (1.0 + abs(h.score)) if h.score else 0.0
            chunks.append(
                {
                    "chunkId": h.chunk_id or h.path,
                    "documentId": h.document_id or h.path,
                    "path": h.path,
                    "score": score,
                    "text": h.snippet,
                    "title": h.title,
                    "summary": "",
                    "metadata": {},
                    "bm25Rank": 0,
                    "vectorSimilarity": 0.0,
                    "rerankScore": 0.0,
                }
            )
        took_ms = int((time.perf_counter() - started) * 1000)

    payload: dict[str, Any] = {
        "chunks": chunks,
        "tookMs": took_ms,
    }
    if trace is not None:
        payload["trace"] = trace
    if attempts:
        payload["attempts"] = attempts
    return ok_json(payload)
