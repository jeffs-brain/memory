# SPDX-License-Identifier: Apache-2.0
"""Vector backfill for the HTTP daemon.

Mirrors ``go/cmd/memory/daemon_vectors.go``. Embed every
FTS-indexed path that lacks a vector under the configured embedding
model and persist the result on the shared search index. Runs on a
detached :class:`asyncio.Task` so brain open is not blocked by remote
embed round-trips; the first ``/search`` request is served by BM25
while vectors trickle in.

Idempotent across restarts: only un-embedded paths are processed, so
re-running the coroutine after a partial run picks up the remainder.
Per-batch failures are logged at WARN and skipped without aborting
the run, so one bad document cannot stall a backfill.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol, runtime_checkable

from .. import search
from ..llm.provider import Embedder

_log = logging.getLogger(__name__)


@runtime_checkable
class _ReadableStore(Protocol):
    """Minimal store surface the backfill needs.

    Matches :class:`PassthroughStore.read`; declared here so tests can
    pass lightweight fakes without pulling the full passthrough in.
    """

    async def read(self, path: str) -> bytes: ...


async def backfill_vectors(
    *,
    brain_id: str,
    store: _ReadableStore,
    index: search.Index,
    embedder: Embedder | None,
    model: str,
    logger: logging.Logger | None = None,
) -> int:
    """Embed every FTS-indexed path that lacks a vector under ``model``.

    Returns the number of ``knowledge_embeddings`` rows written across
    all batches. A zero return is also emitted when the index is
    already up to date, when the embedder is ``None``, or when
    ``model`` is empty (vector indexing disabled). The caller decides
    how to treat the result; nothing raises on the happy path.

    Batches are sized to :data:`search.EMBED_BATCH_SIZE` (100) and each
    document is capped at :data:`search.EMBED_TEXT_MAX` characters
    (8192) before the embed call. Both constants match the Go SDK so
    cross-language behaviour stays lockstep when comparing LongMemEval
    runs.
    """
    log = logger or _log
    if embedder is None or not model:
        return 0

    try:
        paths = index.list_indexed_paths()
    except Exception as exc:  # noqa: BLE001
        log.warning("vectors: list indexed paths failed brain=%s err=%s", brain_id, exc)
        return 0
    if not paths:
        return 0

    try:
        have = index.paths_with_vectors(model)
    except Exception as exc:  # noqa: BLE001
        log.debug("vectors: loading existing brain=%s err=%s", brain_id, exc)
        have = set()

    to_embed = [p for p in paths if p not in have]
    if not to_embed:
        log.info(
            "vectors: up to date brain=%s model=%s total=%d",
            brain_id,
            model,
            len(paths),
        )
        return 0

    log.info(
        "vectors: backfill start brain=%s model=%s count=%d have=%d",
        brain_id,
        model,
        len(to_embed),
        len(have),
    )
    started = time.perf_counter()
    embedded = 0

    for start in range(0, len(to_embed), search.EMBED_BATCH_SIZE):
        batch = to_embed[start : start + search.EMBED_BATCH_SIZE]

        texts: list[str] = []
        kept_paths: list[str] = []
        for path in batch:
            try:
                data = await store.read(path)
            except Exception:  # noqa: BLE001 - a missing source doc is skipped
                continue
            text = data.decode("utf-8", errors="replace")
            if len(text) > search.EMBED_TEXT_MAX:
                text = text[: search.EMBED_TEXT_MAX]
            texts.append(text)
            kept_paths.append(path)
        if not texts:
            continue

        try:
            vectors = await embedder.embed(texts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "vectors: embed batch failed brain=%s batch_start=%d err=%s",
                brain_id,
                start,
                exc,
            )
            continue

        if len(vectors) != len(kept_paths):
            log.warning(
                "vectors: embedder returned mismatched count brain=%s got=%d want=%d",
                brain_id,
                len(vectors),
                len(kept_paths),
            )
            continue

        items: list[tuple[str, list[float]]] = []
        for path, vec in zip(kept_paths, vectors, strict=False):
            if not vec:
                continue
            items.append((path, vec))
        if not items:
            continue
        try:
            written = index.upsert_embeddings(items, model=model)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "vectors: store batch failed brain=%s err=%s",
                brain_id,
                exc,
            )
            continue
        embedded += written
        log.debug(
            "vectors: batch stored brain=%s done=%d total=%d",
            brain_id,
            embedded,
            len(to_embed),
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "vectors: backfill done brain=%s model=%s embedded=%d duration_ms=%d",
        brain_id,
        model,
        embedded,
        duration_ms,
    )
    return embedded


__all__ = ["backfill_vectors"]
