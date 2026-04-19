# SPDX-License-Identifier: Apache-2.0
"""Starlette app factory for `memory serve`.

Mirrors the Go reference: a single `Daemon` instance owns the shared
LLM / embedder / per-brain resources; every handler pulls the daemon
off `app.state`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..llm.provider import Embedder, Provider
from .. import __version__
from .daemon import Daemon
from .handlers import ask as ask_mod
from .handlers import brains as brains_mod
from .handlers import documents as documents_mod
from .handlers import events as events_mod
from .handlers import ingest as ingest_mod
from .handlers import memory as memory_mod
from .handlers import search as search_mod
from .middleware import AuthMiddleware, SizeLimitMiddleware


async def _healthz(_: Request) -> Response:
    return JSONResponse({"ok": True})


async def _version(_: Request) -> Response:
    return JSONResponse({"version": __version__})


def _build_routes() -> list[Route]:
    prefix = "/v1/brains/{brain_id}"
    return [
        Route("/healthz", _healthz, methods=["GET"]),
        Route("/version", _version, methods=["GET"]),
        # Brain management.
        Route("/v1/brains", brains_mod.list_brains, methods=["GET"]),
        Route("/v1/brains", brains_mod.create_brain, methods=["POST"]),
        Route("/v1/brains/{brain_id}", brains_mod.get_brain, methods=["GET"]),
        Route("/v1/brains/{brain_id}", brains_mod.delete_brain, methods=["DELETE"]),
        # Documents.
        Route(f"{prefix}/documents/read", documents_mod.doc_read, methods=["GET"]),
        Route(f"{prefix}/documents/stat", documents_mod.doc_stat, methods=["GET"]),
        Route(f"{prefix}/documents/append", documents_mod.doc_append, methods=["POST"]),
        Route(f"{prefix}/documents/rename", documents_mod.doc_rename, methods=["POST"]),
        Route(f"{prefix}/documents/batch-ops", documents_mod.doc_batch, methods=["POST"]),
        Route(
            f"{prefix}/documents",
            documents_mod.doc_list_or_head,
            methods=["GET", "HEAD"],
        ),
        Route(f"{prefix}/documents", documents_mod.doc_put, methods=["PUT"]),
        Route(f"{prefix}/documents", documents_mod.doc_delete, methods=["DELETE"]),
        # Search + ask.
        Route(f"{prefix}/search", search_mod.search, methods=["POST"]),
        Route(f"{prefix}/ask", ask_mod.ask, methods=["POST"]),
        # Ingest.
        Route(f"{prefix}/ingest/file", ingest_mod.ingest_file, methods=["POST"]),
        Route(f"{prefix}/ingest/url", ingest_mod.ingest_url, methods=["POST"]),
        # Memory stages.
        Route(f"{prefix}/remember", memory_mod.remember, methods=["POST"]),
        Route(f"{prefix}/recall", memory_mod.recall, methods=["POST"]),
        Route(f"{prefix}/extract", memory_mod.extract, methods=["POST"]),
        Route(f"{prefix}/reflect", memory_mod.reflect, methods=["POST"]),
        Route(f"{prefix}/consolidate", memory_mod.consolidate, methods=["POST"]),
        # Events SSE.
        Route(f"{prefix}/events", events_mod.events, methods=["GET"]),
    ]


def create_app(
    *,
    daemon: Daemon | None = None,
    root: Path | str | None = None,
    auth_token: str | None = None,
    llm: Provider | None = None,
    embedder: Embedder | None = None,
    contextualise: bool | None = None,
    contextualise_cache_dir: str | None = None,
) -> Starlette:
    """Build the Starlette ASGI app, wiring every protocol endpoint.

    When `daemon` is None the app factory resolves one from env on
    startup. Tests pass a pre-built daemon so they control the LLM
    provider deterministically.
    """
    if daemon is None:
        resolved_root = root
        if resolved_root is None:
            resolved_root = os.environ.get("JB_HOME") or None
        resolved_token = auth_token or os.environ.get("JB_AUTH_TOKEN")
    else:
        resolved_token = daemon.auth_token or auth_token or os.environ.get("JB_AUTH_TOKEN")

    lifespan = _build_lifespan(
        preset=daemon,
        root=resolved_root if daemon is None else None,
        token=resolved_token,
        llm=llm,
        embedder=embedder,
        contextualise=contextualise,
        contextualise_cache_dir=contextualise_cache_dir,
    )

    app = Starlette(debug=False, routes=_build_routes(), lifespan=lifespan)

    # Expose daemon via app.state so handlers can fetch it cheaply.
    app.state.daemon = daemon  # type: ignore[attr-defined]

    # Middleware: auth first (outer) so unauthenticated requests never
    # reach the size-limit layer; size-limit second so overflow 413s
    # surface before handlers parse bodies.
    token_for_middleware = resolved_token if resolved_token else None
    app.add_middleware(SizeLimitMiddleware)
    app.add_middleware(AuthMiddleware, token=token_for_middleware)

    return app


def _build_lifespan(
    *,
    preset: Daemon | None,
    root: Path | str | None,
    token: str | None,
    llm: Provider | None,
    embedder: Embedder | None,
    contextualise: bool | None,
    contextualise_cache_dir: str | None,
):
    """Return an async lifespan context manager for the app."""

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        owned = False
        daemon: Daemon | None = getattr(app.state, "daemon", None)  # type: ignore[attr-defined]
        if daemon is None and preset is None:
            daemon = await Daemon.create(
                root=root,
                auth_token=token,
                llm=llm,
                embedder=embedder,
                contextualise=contextualise,
                contextualise_cache_dir=contextualise_cache_dir,
            )
            owned = True
            app.state.daemon = daemon  # type: ignore[attr-defined]
        try:
            yield
        finally:
            if owned and daemon is not None:
                await daemon.close()

    return lifespan


__all__ = ["create_app"]
