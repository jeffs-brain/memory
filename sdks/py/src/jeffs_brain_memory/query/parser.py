# SPDX-License-Identifier: Apache-2.0
"""Top-level orchestrator: normalise, annotate, optionally distill."""

from __future__ import annotations

from . import normalise as _norm
from . import temporal as _temporal
from .distill import Distiller
from .types import Options, Query, Result, Trace

__all__ = ["parse"]


# Module-level distiller cache, keyed by ``id(provider)``. A fresh provider
# gets its own LRU; repeat calls with the same provider reuse the cache so
# every ``parse`` invocation isn't a cold start.
_distillers: dict[int, Distiller] = {}


def _distiller_for(options: Options) -> Distiller | None:
    if options.provider is None:
        return None
    key = id(options.provider)
    existing = _distillers.get(key)
    if existing is None:
        existing = Distiller(options.provider)
        _distillers[key] = existing
    return existing


async def parse(query: str, options: Options | None = None) -> Result:
    """Normalise, annotate and optionally distill ``query``.

    Pipeline (mirrors Go ``query.Distill`` but with the simpler Python
    surface):

    1. Normalise: strip, fold whitespace, lowercase for the cache copy.
    2. Tokenise and compute the significant-term list.
    3. Apply temporal recognisers if an anchor is configured.
    4. If ``options.distill`` is set and a provider is configured, call
       :class:`Distiller` and cache the result (when ``options.cache``).
    """

    opts = options if options is not None else Options()
    normalised = _norm.normalise_for_cache(query)
    tokens = _norm.tokenise(normalised)
    sig_terms = _norm.significant_terms(normalised)

    temporal = _temporal.annotate(query, opts.anchor)

    distilled: str | None = None
    used_cache = False
    did_distill = False

    if opts.distill and opts.provider is not None and query.strip():
        distiller = _distiller_for(opts)
        assert distiller is not None  # narrow for mypy
        # Inspect the cache before dispatch so the trace reports accurately.
        key = (opts.model, query.strip().lower())
        if opts.cache and distiller._cache_get(key) is not None:  # noqa: SLF001
            used_cache = True
        distilled = await distiller.distill(query, model=opts.model)
        did_distill = True
        if not opts.cache:
            # Cache bypass: drop the entry the distiller just recorded.
            distiller._cache.pop(key, None)  # noqa: SLF001

    parsed = Query(
        raw=query,
        normalised=normalised,
        tokens=tokens,
        significant_terms=sig_terms,
        temporal=temporal,
        distilled=distilled,
    )
    return Result(
        query=parsed,
        trace=Trace(used_cache=used_cache, distilled=did_distill),
    )
