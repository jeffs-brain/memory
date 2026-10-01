# SPDX-License-Identifier: Apache-2.0
"""Confinement for ``POST /ingest/file`` requests that name a
server-side path instead of sending ``contentBase64``.

Path ingest is disabled unless the daemon was started with an ingest
root, and a permitted path must resolve, after symlinks, inside that
root. Mirrors ``go/cmd/memory/handler_ingest.go`` and
``sdks/ts/memory/src/http/ingest-root.ts``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "IngestPathInvalidError",
    "PathIngestRefusedError",
    "resolve_ingest_path",
    "resolve_ingest_root",
]


class PathIngestRefusedError(Exception):
    """The request named a path the daemon refuses to read. Maps to 403."""


class IngestPathInvalidError(Exception):
    """The named path is missing or is not a regular file. Maps to 400."""


def resolve_ingest_root(root: str | Path | None) -> Path | None:
    """Validate the configured ingest root and return it as an absolute
    path. Blank input disables path ingest and yields ``None``. Raises when
    the root is missing or not a directory."""
    text = str(root).strip() if root is not None else ""
    if not text:
        return None
    abs_root = Path(os.path.abspath(text))
    if not abs_root.resolve(strict=True).is_dir():
        raise NotADirectoryError(f"ingest root: {abs_root} is not a directory")
    return abs_root


def _outside_root() -> PathIngestRefusedError:
    return PathIngestRefusedError("path resolves outside the configured ingest root")


def _nearest_existing_ancestor(path: Path) -> Path | None:
    """The resolved closest directory above ``path`` that exists."""
    for parent in path.parents:
        try:
            return parent.resolve(strict=True)
        except FileNotFoundError:
            continue
        except OSError:
            return None
    return None


def resolve_ingest_path(root: Path | None, requested: str) -> Path:
    """Resolve ``requested`` inside ``root``.

    A relative path is taken relative to the root. Containment is decided
    on the fully resolved path, so a symlink cannot point the read
    elsewhere and any spelling of a path inside the root is accepted. A
    path that does not exist is judged by its nearest existing ancestor,
    so a path outside the root is refused the same way whether or not it
    exists. Returns the real path of a regular file inside the root.
    """
    if root is None:
        raise PathIngestRefusedError(
            "server-side path ingest is disabled; send contentBase64 or start "
            "the daemon with an ingest root"
        )
    abs_root = os.path.abspath(root)
    real_root = Path(abs_root).resolve(strict=True)
    candidate = Path(os.path.normpath(os.path.join(abs_root, requested)))
    try:
        real = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        ancestor = _nearest_existing_ancestor(candidate)
        if ancestor is None or not ancestor.is_relative_to(real_root):
            raise _outside_root() from exc
        raise IngestPathInvalidError("file not found") from exc
    except OSError as exc:
        raise _outside_root() from exc
    if not real.is_relative_to(real_root):
        raise _outside_root()
    if not real.is_file():
        raise IngestPathInvalidError("path is not a regular file")
    return real
