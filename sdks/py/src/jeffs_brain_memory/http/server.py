# SPDX-License-Identifier: Apache-2.0
"""Starlette app factory for `memory serve`."""

from __future__ import annotations

from starlette.applications import Starlette

from .routes import build_routes


def create_app() -> Starlette:
    """Build the Starlette ASGI app."""
    return Starlette(debug=False, routes=build_routes())
