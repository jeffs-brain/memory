# SPDX-License-Identifier: Apache-2.0
"""Hybrid retrieval orchestrator.

Mirrors ``go/retrieval/retrieval.go``: mode resolution, BM25 leg
with retry ladder, vector leg, RRF fusion, English intent reweight,
optional rerank pass, unanimity shortcut.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Any, Awaitable, Callable

from ..llm.provider import Embedder
from .intent import (
    detect_retrieval_intent,
    reweight_shared_memory_ranking,
)
from .reranker import Reranker
from .retry import (
    TrigramIndex,
    build_trigram_index,
    query_tokens,
    sanitise_query,
    strongest_term,
)
from .rrf import RRF_DEFAULT_K, RRFCandidate, reciprocal_rank_fusion
from .source import BM25Hit, Source, TrigramChunk, VectorHit
from .temporal import (
    build_bm25_query_plan,
    compile_bm25_fanout_query,
    resolved_date_hints,
    temporal_query_variants,
)
from ..query.temporal import parse_question_date
from .types import (
    Attempt,
    Filters,
    Mode,
    Request,
    Response,
    RetrievedChunk,
    Trace,
)

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 60
DEFAULT_RERANK_TOP_N = 20
PHRASE_PROBE_MIN_TOKENS = 2
PHRASE_PROBE_MAX_TOKENS = 4
BM25_FANOUT_PRIMARY_WINDOW = 10
BM25_FANOUT_MIN_OVERLAP = 2
UNANIMITY_WINDOW = 3
UNANIMITY_AGREE_MIN = 2
RECENCY_QUERY_RE = re.compile(
    r"\b(?:most recent|latest|last time|current(?:ly)?|now|newest)\b",
    re.IGNORECASE,
)
EARLIEST_QUERY_RE = re.compile(
    r"\b(?:earliest|first|initial|original|at first)\b",
    re.IGNORECASE,
)
INLINE_DATE_TAG_RE = re.compile(r"\[(?:observed on|date):?\s*([^\]]+)\]", re.IGNORECASE)
FRONTMATTER_DATE_KEYS = (
    "observed_on",
    "observedOn",
    "session_date",
    "sessionDate",
    "modified",
)


# Stopwords type kept permissive so callers can pass the search package's
# stopwords without requiring this module to import it.
Stopwords = object


def _compile_to_fts(q: str) -> str:
    """Lightweight query compiler. The Python search stub does not yet
    ship a real parser; for the retrieval tests we just pass trimmed
    text through so the fake source can score it."""
    return " ".join(q.split()) if q.strip() else ""


def _pick_id(id_: str, path: str) -> str:
    return id_ if id_ else path


def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata) if metadata else {}


def _join_bm25_attempt_query(queries: list[str]) -> str:
    compiled: list[str] = []
    seen: set[str] = set()
    for query in queries:
        expr = _compile_to_fts(query)
        if not expr or expr in seen:
            continue
        seen.add(expr)
        compiled.append(expr)
    return " || ".join(compiled)


def _bm25_fanout_overlap(
    primary: list[RRFCandidate], secondary: list[RRFCandidate]
) -> int:
    if not primary or not secondary:
        return 0
    primary_ids = {
        candidate.id
        for candidate in primary[:BM25_FANOUT_PRIMARY_WINDOW]
        if candidate.id
    }
    if not primary_ids:
        return 0
    overlap = 0
    for candidate in secondary[:BM25_FANOUT_PRIMARY_WINDOW]:
        if candidate.id not in primary_ids:
            continue
        overlap += 1
        if overlap >= BM25_FANOUT_MIN_OVERLAP:
            return overlap
    return overlap


def _should_bypass_bm25_fanout_overlap_gate(expr: str) -> bool:
    if any(ch.isdigit() for ch in expr):
        return True
    terms = 0
    for token in expr.split():
        if token in {"AND", "OR", "NOT"}:
            continue
        terms += 1
    return PHRASE_PROBE_MIN_TOKENS <= terms <= PHRASE_PROBE_MAX_TOKENS


def _resolve_mode(requested: Mode, has_embedder: bool) -> tuple[Mode, bool]:
    if not has_embedder and requested in (
        Mode.AUTO,
        Mode.HYBRID,
        Mode.SEMANTIC,
        Mode.HYBRID_RERANK,
    ):
        fell_back = requested != Mode.AUTO
        return Mode.BM25, fell_back
    if requested == Mode.AUTO:
        return Mode.HYBRID, False
    return requested, False


def _unanimity_shortcut(
    bm: list[RRFCandidate], vec: list[RRFCandidate], window: int, min_agree: int
) -> tuple[int, bool]:
    if len(bm) < window or len(vec) < window:
        return 0, False
    agreements = 0
    for i in range(window):
        if bm[i].id and bm[i].id == vec[i].id:
            agreements += 1
    return agreements, agreements >= min_agree


def _single_list(cands: list[RRFCandidate], k: int) -> list[RetrievedChunk]:
    if not cands:
        return []
    safe_k = k if k > 0 else RRF_DEFAULT_K
    out: list[RetrievedChunk] = []
    for i, c in enumerate(cands):
        chunk = RetrievedChunk(
            chunk_id=c.id,
            document_id=c.id,
            path=c.path,
            score=1.0 / float(safe_k + i + 1),
            text=c.content,
            title=c.title,
            summary=c.summary,
            metadata=_copy_metadata(c.metadata),
        )
        if c.have_bm25_rank:
            chunk.bm25_rank = c.bm25_rank
        if c.have_vector_sim:
            chunk.vector_similarity = c.vector_similarity
        out.append(chunk)
    return out


class Retriever:
    """Hybrid retrieval surface.

    Construct via keyword args so callers can omit optional dependencies
    without positional drift.
    """

    def __init__(
        self,
        *,
        source: Source,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = RRF_DEFAULT_K,
        stopwords: Stopwords | None = None,
        trigram_chunks: list[TrigramChunk] | None = None,
    ) -> None:
        if source is None:
            raise ValueError("retrieval: Retriever requires a non-nil source")
        self._source = source
        self._embedder = embedder
        self._reranker = reranker
        self._rrf_k = rrf_k if rrf_k > 0 else RRF_DEFAULT_K
        self._stopwords = stopwords
        self._trigram_source = trigram_chunks
        self._trigram_lock = asyncio.Lock()
        self._trigram_built = False
        self._trigram_idx: TrigramIndex | None = None

    async def retrieve(self, req: Request) -> Response:
        started = time.perf_counter()

        top_k = req.top_k if req.top_k > 0 else DEFAULT_TOP_K
        candidate_k = req.candidate_k if req.candidate_k > 0 else DEFAULT_CANDIDATE_K
        rerank_top_n = req.rerank_top_n if req.rerank_top_n > 0 else DEFAULT_RERANK_TOP_N

        requested_mode = req.mode if req.mode else Mode.AUTO
        mode, fell_back = _resolve_mode(requested_mode, self._embedder is not None)

        trace = Trace(
            requested_mode=requested_mode,
            effective_mode=mode,
            rrf_k=self._rrf_k,
            candidate_k=candidate_k,
            rerank_top_n=rerank_top_n,
            fell_back_to_bm25=fell_back,
        )
        attempts: list[Attempt] = []

        # BM25 leg with retry ladder on zero hits.
        bm_candidates, bm_attempts, used_retry = await self._run_bm25_leg(
            req, candidate_k
        )
        attempts.extend(bm_attempts)
        trace.used_retry = used_retry
        trace.bm25_hits = len(bm_candidates)

        # Vector leg (only when the mode requests it and an embedder
        # is configured).
        vec_candidates: list[RRFCandidate] = []
        if self._embedder is not None and mode in (
            Mode.HYBRID,
            Mode.SEMANTIC,
            Mode.HYBRID_RERANK,
        ):
            hits = await self._run_vector_leg(req, candidate_k)
            if hits:
                trace.embedder_used = True
                vec_candidates = hits
        trace.vector_hits = len(vec_candidates)

        # Fuse according to mode.
        fused = self._fuse(mode, bm_candidates, vec_candidates)
        trace.fused_hits = len(fused)

        # Intent-aware reweighting (English-only).
        intent = detect_retrieval_intent(req.query)
        trace.intent = intent.label()
        fused = reweight_shared_memory_ranking(req.query, fused)

        # Optional rerank pass.
        final = await self._maybe_rerank(
            req, mode, fused, bm_candidates, vec_candidates, rerank_top_n, trace
        )
        final = _reweight_temporal_ranking(req.query, req.question_date, final)

        if len(final) > top_k:
            final = final[:top_k]

        took_ms = int((time.perf_counter() - started) * 1000)
        return Response(chunks=final, took_ms=took_ms, trace=trace, attempts=attempts)

    async def _run_bm25_leg(
        self, req: Request, candidate_k: int
    ) -> tuple[list[RRFCandidate], list[Attempt], bool]:
        attempts: list[Attempt] = []

        initial_plan = build_bm25_query_plan(req.query, req.question_date)
        initial_queries = initial_plan.queries
        initial_expr = _join_bm25_attempt_query(initial_queries)
        candidates = await self._run_bm25_queries(
            initial_queries,
            initial_plan.phrase_probes,
            candidate_k,
            req.filters,
        )
        attempts.append(
            Attempt(
                rung=0,
                mode=Mode.BM25,
                top_k=candidate_k,
                reason="initial",
                query=initial_expr,
                chunks=len(candidates),
            )
        )

        if candidates or req.skip_retry_ladder:
            return candidates, attempts, False

        # Rung 1: strongest term. Skipped silently when strongest
        # matches the lowered trimmed raw query.
        lowered_raw = req.query.strip().lower()
        strongest = strongest_term(req.query)
        if strongest and strongest != lowered_raw:
            strongest_plan = build_bm25_query_plan(strongest, req.question_date)
            expr = _join_bm25_attempt_query(strongest_plan.queries)
            hits = await self._run_bm25_queries(
                strongest_plan.queries,
                strongest_plan.phrase_probes,
                candidate_k,
                req.filters,
            )
            attempts.append(
                Attempt(
                    rung=1,
                    mode=Mode.BM25,
                    top_k=candidate_k,
                    reason="strongest_term",
                    query=expr,
                    chunks=len(hits),
                )
            )
            if hits:
                return hits, attempts, True

        # Rung 2: force-refresh pass-through. No trace row emitted.
        _force_refresh_index()

        # Rung 3: refreshed sanitised.
        sanitised = sanitise_query(req.query)
        if sanitised:
            sanitised_plan = build_bm25_query_plan(sanitised, req.question_date)
            expr = _join_bm25_attempt_query(sanitised_plan.queries)
            hits = await self._run_bm25_queries(
                sanitised_plan.queries,
                sanitised_plan.phrase_probes,
                candidate_k,
                req.filters,
            )
            attempts.append(
                Attempt(
                    rung=3,
                    mode=Mode.BM25,
                    top_k=candidate_k,
                    reason="refreshed_sanitised",
                    query=expr,
                    chunks=len(hits),
                )
            )
            if hits:
                return hits, attempts, True

        # Rung 4: refreshed strongest term.
        strongest_of_sanitised = strongest_term(sanitised)
        if strongest_of_sanitised:
            strongest_plan = build_bm25_query_plan(
                strongest_of_sanitised, req.question_date
            )
            expr = _join_bm25_attempt_query(strongest_plan.queries)
            hits = await self._run_bm25_queries(
                strongest_plan.queries,
                strongest_plan.phrase_probes,
                candidate_k,
                req.filters,
            )
            attempts.append(
                Attempt(
                    rung=4,
                    mode=Mode.BM25,
                    top_k=candidate_k,
                    reason="refreshed_strongest",
                    query=expr,
                    chunks=len(hits),
                )
            )
            if hits:
                return hits, attempts, True

        # Rung 5: trigram fuzzy fallback.
        tokens = query_tokens(req.query)
        if tokens:
            idx = await self._ensure_trigram_index()
            if idx is not None:
                trigram_limit = max(candidate_k * 10, 200)
                fuzzy = idx.search(tokens, trigram_limit)
                fuzzy_cands: list[RRFCandidate] = []
                for i, h in enumerate(fuzzy):
                    if not req.filters.matches_path(h.path):
                        continue
                    fuzzy_cands.append(
                        RRFCandidate(
                            id=h.id,
                            path=h.path,
                            title=h.title,
                            summary=h.summary,
                            content=h.content,
                            bm25_rank=i,
                            have_bm25_rank=True,
                        )
                    )
                attempts.append(
                    Attempt(
                        rung=5,
                        mode=Mode.BM25,
                        top_k=candidate_k,
                        reason="trigram_fuzzy",
                        query=" ".join(tokens),
                        chunks=len(fuzzy_cands),
                    )
                )
                if fuzzy_cands:
                    return fuzzy_cands, attempts, True

        return [], attempts, True

    async def _run_bm25(
        self, expr: str, k: int, filters: Filters
    ) -> list[RRFCandidate]:
        if not expr:
            return []
        hits: list[BM25Hit] = await self._source.search_bm25(expr, k, filters)
        out: list[RRFCandidate] = []
        for i, h in enumerate(hits):
            out.append(
                RRFCandidate(
                    id=_pick_id(h.id, h.path),
                    path=h.path,
                    title=h.title,
                    summary=h.summary,
                    content=h.content,
                    metadata=_copy_metadata(h.metadata),
                    bm25_rank=i,
                    have_bm25_rank=True,
                )
            )
        return out

    async def _run_bm25_queries(
        self, queries: list[str], phrase_probes: list[str], k: int, filters: Filters
    ) -> list[RRFCandidate]:
        if not queries:
            return []
        primary_hits = await self._run_bm25(
            _compile_to_fts(compile_bm25_fanout_query(queries[0], phrase_probes)),
            k,
            filters,
        )
        lists: list[list[RRFCandidate]] = [primary_hits] if primary_hits else []
        for query in queries[1:]:
            expr = _compile_to_fts(compile_bm25_fanout_query(query, phrase_probes))
            hits = await self._run_bm25(expr, k, filters)
            if not hits:
                continue
            if (
                not primary_hits
                or _should_bypass_bm25_fanout_overlap_gate(expr)
                or _bm25_fanout_overlap(primary_hits, hits) >= BM25_FANOUT_MIN_OVERLAP
            ):
                lists.append(hits)
        if not lists:
            return []
        if len(lists) == 1:
            return lists[0]
        fused = reciprocal_rank_fusion(lists, self._rrf_k)
        out: list[RRFCandidate] = []
        for i, chunk in enumerate(fused):
            out.append(
                RRFCandidate(
                    id=_pick_id(chunk.chunk_id, chunk.path),
                    path=chunk.path,
                    title=chunk.title,
                    summary=chunk.summary,
                    content=chunk.text,
                    metadata=_copy_metadata(chunk.metadata),
                    bm25_rank=i,
                    have_bm25_rank=True,
                )
            )
        return out

    async def _run_vector_leg(
        self, req: Request, k: int
    ) -> list[RRFCandidate]:
        if self._embedder is None:
            return []
        vectors = await self._embedder.embed([req.query])
        if not vectors or not vectors[0]:
            return []
        hits: list[VectorHit] = await self._source.search_vector(
            vectors[0], k, req.filters
        )
        out: list[RRFCandidate] = []
        for i, h in enumerate(hits):
            out.append(
                RRFCandidate(
                    id=_pick_id(h.id, h.path),
                    path=h.path,
                    title=h.title,
                    summary=h.summary,
                    content=h.content,
                    metadata=_copy_metadata(h.metadata),
                    vector_similarity=h.similarity,
                    have_vector_sim=True,
                    bm25_rank=i,
                    have_bm25_rank=True,
                )
            )
        return out

    def _fuse(
        self,
        mode: Mode,
        bm: list[RRFCandidate],
        vec: list[RRFCandidate],
    ) -> list[RetrievedChunk]:
        if mode == Mode.BM25:
            return _single_list(bm, self._rrf_k)
        if mode == Mode.SEMANTIC:
            return _single_list(vec, self._rrf_k)
        lists: list[list[RRFCandidate]] = []
        if bm:
            lists.append(bm)
        if vec:
            lists.append(vec)
        if not lists:
            return []
        return reciprocal_rank_fusion(lists, self._rrf_k)

    async def _maybe_rerank(
        self,
        req: Request,
        mode: Mode,
        fused: list[RetrievedChunk],
        bm: list[RRFCandidate],
        vec: list[RRFCandidate],
        rerank_top_n: int,
        trace: Trace,
    ) -> list[RetrievedChunk]:
        if not fused:
            trace.rerank_skip_reason = "empty_candidates"
            return fused
        if self._reranker is None:
            trace.rerank_skip_reason = "no_reranker"
            return fused
        if mode != Mode.HYBRID_RERANK:
            trace.rerank_skip_reason = "mode_off"
            return fused

        agreements, shortcut = _unanimity_shortcut(
            bm, vec, UNANIMITY_WINDOW, UNANIMITY_AGREE_MIN
        )
        if shortcut:
            trace.rerank_skip_reason = "unanimity"
            trace.unanimity_skipped = True
            trace.agreements = agreements
            return fused

        n = min(rerank_top_n, len(fused))
        head = fused[:n]
        tail = fused[n:]

        try:
            reranked = await self._reranker.rerank(req.query, head)
        except Exception:
            trace.rerank_skip_reason = "rerank_failed"
            return fused
        if not reranked:
            trace.rerank_skip_reason = "rerank_failed"
            return fused

        trace.reranked = True
        trace.rerank_provider = _reranker_name(self._reranker)
        return list(reranked) + list(tail)

    async def _ensure_trigram_index(self) -> TrigramIndex | None:
        async with self._trigram_lock:
            if self._trigram_built:
                return self._trigram_idx
            self._trigram_built = True
            if self._trigram_source is not None:
                self._trigram_idx = build_trigram_index(self._trigram_source)
                return self._trigram_idx
            try:
                chunks = await self._source.chunks()
            except Exception:
                # Best-effort fallback: leave the index None so the
                # rung silently skips rather than failing the whole
                # retrieval call.
                return None
            self._trigram_idx = build_trigram_index(chunks)
            return self._trigram_idx


def _reranker_name(reranker: Reranker) -> str:
    name_fn: Callable[[], str] | None = getattr(reranker, "name", None)
    if callable(name_fn):
        try:
            return name_fn()
        except Exception:
            return "custom"
    return "custom"


def _reweight_temporal_ranking(
    query: str,
    question_date: str,
    results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not results:
        return list(results)

    anchor_time = _parse_candidate_time(question_date)
    filtered_results = list(results)
    if anchor_time is not None:
        filtered_results = []
        for chunk in results:
            candidate_time = _extract_candidate_time(chunk)
            if candidate_time is not None and candidate_time > anchor_time:
                continue
            filtered_results.append(chunk)
        if not filtered_results:
            return []

    wants_recency = RECENCY_QUERY_RE.search(query) is not None
    wants_earliest = not wants_recency and EARLIEST_QUERY_RE.search(query) is not None
    hint_times = _dedupe_datetimes(
        [_parse_candidate_time(hint) for hint in resolved_date_hints(query, question_date)]
    )
    hint_times = [value for value in hint_times if value is not None]
    if not wants_recency and not wants_earliest and not hint_times:
        return filtered_results

    candidate_times = [_extract_candidate_time(result) for result in filtered_results]
    dated = [value for value in candidate_times if value is not None]
    if not dated:
        return filtered_results

    min_time = min(dated)
    max_time = max(dated)

    scored: list[tuple[RetrievedChunk, int]] = []
    for index, chunk in enumerate(filtered_results):
        candidate_time = candidate_times[index]
        multiplier = 1.0
        if candidate_time is not None and hint_times:
            multiplier *= _temporal_hint_multiplier(candidate_time, hint_times)
        if candidate_time is not None and max_time > min_time:
            norm = (candidate_time.timestamp() - min_time.timestamp()) / (
                max_time.timestamp() - min_time.timestamp()
            )
            if wants_recency:
                multiplier *= 1.0 + 0.25 * norm
            if wants_earliest:
                multiplier *= 1.0 + 0.25 * (1.0 - norm)
        elif candidate_time is None and (wants_recency or wants_earliest):
            multiplier *= 0.95
        scored.append(
            (
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    path=chunk.path,
                    score=chunk.score * multiplier,
                    text=chunk.text,
                    title=chunk.title,
                    summary=chunk.summary,
                    metadata=dict(chunk.metadata),
                    bm25_rank=chunk.bm25_rank,
                    vector_similarity=chunk.vector_similarity,
                    rerank_score=chunk.rerank_score,
                ),
                index,
            )
        )

    scored.sort(key=lambda item: (-item[0].score, item[1]))
    return [chunk for chunk, _ in scored]


def _temporal_hint_multiplier(candidate_time: datetime, hint_times: list[datetime]) -> float:
    nearest_days = min(
        abs((candidate_time - hint).total_seconds()) / 86_400.0 for hint in hint_times
    )
    if nearest_days <= 1:
        return 1.35
    if nearest_days <= 7:
        return 1.20
    if nearest_days <= 30:
        return 1.08
    return 0.92


def _extract_candidate_time(chunk: RetrievedChunk) -> datetime | None:
    metadata_time = _extract_metadata_time(chunk.metadata)
    if metadata_time is not None:
        return metadata_time
    return _extract_time_from_text(chunk.text)


def _extract_metadata_time(metadata: dict[str, Any]) -> datetime | None:
    for key in FRONTMATTER_DATE_KEYS:
        value = metadata.get(key)
        if not isinstance(value, str):
            continue
        parsed = _parse_candidate_time(value)
        if parsed is not None:
            return parsed
    return None


def _extract_time_from_text(text: str) -> datetime | None:
    match = INLINE_DATE_TAG_RE.search(text)
    if match is not None:
        parsed = _parse_candidate_time(match.group(1))
        if parsed is not None:
            return parsed
    for key in FRONTMATTER_DATE_KEYS:
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if match is None:
            continue
        parsed = _parse_candidate_time(match.group(1))
        if parsed is not None:
            return parsed
    return None


def _parse_candidate_time(value: str) -> datetime | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        return parse_question_date(trimmed)
    except ValueError:
        return None


def _dedupe_datetimes(values: list[datetime]) -> list[datetime]:
    seen: set[datetime] = set()
    out: list[datetime] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _force_refresh_index() -> None:
    """Re-export the retry-module helper under the retriever module so
    orchestration-level callers have a local reference. Kept as a wrapper
    so the no-op stays traceable in stack traces."""
    # Local reference so the Rung-2 call site reads naturally.
    from .retry import force_refresh_index

    force_refresh_index()


__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_TOP_K",
    "Retriever",
    "UNANIMITY_AGREE_MIN",
    "UNANIMITY_WINDOW",
]


# Imported Awaitable stays referenced for typing even if not used directly.
_ = Awaitable
