# SPDX-License-Identifier: Apache-2.0
"""Hybrid retrieval orchestrator.

Mirrors ``sdks/go/retrieval/retrieval.go``: mode resolution, BM25 leg
with retry ladder, vector leg, RRF fusion, English intent reweight,
optional rerank pass, unanimity shortcut.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

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
UNANIMITY_WINDOW = 3
UNANIMITY_AGREE_MIN = 2


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

        if len(final) > top_k:
            final = final[:top_k]

        took_ms = int((time.perf_counter() - started) * 1000)
        return Response(chunks=final, took_ms=took_ms, trace=trace, attempts=attempts)

    async def _run_bm25_leg(
        self, req: Request, candidate_k: int
    ) -> tuple[list[RRFCandidate], list[Attempt], bool]:
        attempts: list[Attempt] = []

        initial_expr = _compile_to_fts(req.query)
        candidates = await self._run_bm25(initial_expr, candidate_k, req.filters)
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
            expr = _compile_to_fts(strongest)
            hits = await self._run_bm25(expr, candidate_k, req.filters)
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
            expr = _compile_to_fts(sanitised)
            hits = await self._run_bm25(expr, candidate_k, req.filters)
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
            expr = _compile_to_fts(strongest_of_sanitised)
            hits = await self._run_bm25(expr, candidate_k, req.filters)
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
                fuzzy = idx.search(tokens, candidate_k)
                fuzzy_cands: list[RRFCandidate] = []
                for i, h in enumerate(fuzzy):
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
