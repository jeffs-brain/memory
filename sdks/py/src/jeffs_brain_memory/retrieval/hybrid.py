# SPDX-License-Identifier: Apache-2.0
"""Backwards-compatible shim — the hybrid pipeline now lives in
:mod:`retriever`.

Kept so callers that imported ``retrieval.hybrid.retrieve`` during the
phase-4 stub era keep compiling. New code should construct
:class:`Retriever` directly.
"""

from __future__ import annotations

from .retriever import Retriever
from .types import Request, Response

__all__ = ["Retriever", "Request", "Response"]
