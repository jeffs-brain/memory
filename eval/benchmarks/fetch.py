# SPDX-License-Identifier: Apache-2.0
"""Shared fetch, cache, and verification helpers for benchmark datasets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from benchmarks.base import FetchResult


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_and_verify(
    *,
    url: str,
    cache_dir: Path,
    benchmark: str,
    filename: str,
    expected_sha256: str | None = None,
    revision: str | None = None,
    timeout_s: float = 120.0,
) -> FetchResult:
    """Fetch a dataset into the local cache and verify it when a SHA is pinned."""
    cache_key = expected_sha256[:12] if expected_sha256 else (revision or "unpinned")[:12]
    target_dir = cache_dir / benchmark / cache_key
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    if target.exists():
        actual = sha256_file(target)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(f"cached {benchmark} dataset SHA256 mismatch: {actual}")
        return FetchResult(local_path=target, sha256=actual, revision=revision)

    last_error: httpx.HTTPError | None = None
    for attempt in range(2):
        try:
            with httpx.stream("GET", url, timeout=timeout_s, follow_redirects=True) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if chunk:
                            handle.write(chunk)
            break
        except httpx.HTTPError as exc:
            last_error = exc
            if target.exists():
                target.unlink()
            if attempt == 1:
                raise
    if last_error is not None and not target.exists():
        raise last_error

    actual = sha256_file(target)
    if expected_sha256 is not None and actual != expected_sha256:
        target.unlink(missing_ok=True)
        raise ValueError(f"{benchmark} dataset SHA256 mismatch: {actual}")
    return FetchResult(local_path=target, sha256=actual, revision=revision)
