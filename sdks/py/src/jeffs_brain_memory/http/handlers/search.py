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


def _path_list_from_raw(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            continue
        trimmed = value.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return out


def filters_from_body(raw: Any) -> retrieval.Filters:
    if not isinstance(raw, dict):
        return retrieval.Filters()
    tags_raw = raw.get("tags") or []
    tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
    return retrieval.Filters(
        path_prefix=str(raw.get("pathPrefix") or raw.get("path_prefix") or ""),
        paths=_path_list_from_raw(
            raw.get("paths") or raw.get("pathList") or raw.get("path_list")
        ),
        tags=tags,
        scope=str(raw.get("scope") or ""),
        project=str(raw.get("project") or ""),
    )


def search_opts(top_k: int, filters: retrieval.Filters) -> search_pkg.SearchOpts:
    filter_map: dict[str, Any] = {}
    if filters.scope:
        filter_map["scope"] = filters.scope
    if filters.project:
        filter_map["project_slug"] = filters.project
    if filters.path_prefix:
        filter_map["path_prefix"] = filters.path_prefix
    if filters.paths:
        filter_map["paths"] = list(filters.paths)
    if filters.tags:
        filter_map["tags"] = list(filters.tags)
    return search_pkg.SearchOpts(max_results=top_k, filters=filter_map)


def path_matches_filters(path: str, filters: retrieval.Filters) -> bool:
    return filters.matches_path(path)


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
    mode_raw = body.get("mode") or ""
    filters = filters_from_body(
        body.get("filters") if isinstance(body.get("filters"), dict) else body
    )
    question_date_raw = body.get("question_date") or body.get("questionDate") or ""
    question_date = (
        str(question_date_raw) if isinstance(question_date_raw, str) else ""
    )

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
        question_date=question_date,
        filters=filters,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
    )

    chunks: list[dict[str, Any]] = []
    trace: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []

    try:
        resp = await br.retriever.retrieve(req)
        if resp.chunks:
            chunks = [
                _chunk_to_wire(c)
                for c in resp.chunks
                if path_matches_filters(c.path, filters)
            ]
        trace = _trace_to_wire(resp.trace)
        attempts = [_attempt_to_wire(a) for a in resp.attempts]
        took_ms = resp.took_ms or int((time.perf_counter() - started) * 1000)
    except Exception:  # noqa: BLE001
        resp = None

    # Fall back to raw BM25 when the retriever produced nothing.
    if not chunks:
        try:
            hits = br.search_index.search_bm25(
                retrieval.augment_query_with_temporal(query, question_date),
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
                {
                    "chunkId": h.chunk_id or h.path,
                    "documentId": h.document_id or h.path,
                    "path": h.path,
                    "score": score,
                    "text": h.content or h.snippet,
                    "title": h.title,
                    "summary": h.summary,
                    "metadata": h.metadata,
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
