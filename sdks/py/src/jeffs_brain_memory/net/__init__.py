# SPDX-License-Identifier: Apache-2.0
"""Outbound network helpers. Use :func:`safe_fetch` for any URL a user
or a model supplied."""

from .safe_fetch import (
    DEFAULT_SAFE_FETCH_MAX_BYTES,
    DEFAULT_SAFE_FETCH_MAX_REDIRECTS,
    DEFAULT_SAFE_FETCH_TIMEOUT,
    FetchFailedError,
    Resolver,
    SafeFetchResult,
    UnsafeUrlError,
    is_blocked_address,
    resolve_public_address,
    safe_fetch,
)

__all__ = [
    "DEFAULT_SAFE_FETCH_MAX_BYTES",
    "DEFAULT_SAFE_FETCH_MAX_REDIRECTS",
    "DEFAULT_SAFE_FETCH_TIMEOUT",
    "FetchFailedError",
    "Resolver",
    "SafeFetchResult",
    "UnsafeUrlError",
    "is_blocked_address",
    "resolve_public_address",
    "safe_fetch",
]
