# SPDX-License-Identifier: Apache-2.0
"""Shared filesystem layout helpers used by the fs and git stores.

Mirrors `brain/storeutil/` in the Go reference. Pure functions over
logical `BrainPath` values; no I/O state retained between calls.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from ..errors import ErrInvalidPath
from ..path import BrainPath, is_generated, validate_path
from . import FileInfo, ListOpts

__all__ = [
    "relative",
    "logical_from_rel",
    "resolve",
    "should_skip_dir",
    "should_skip_file",
    "atomic_write",
    "list_dir",
    "path_under",
    "glob_match_base",
]


def relative(path: BrainPath | str) -> str:
    """Map a logical path to its on-disk relative path under the store root.

    Layout matches the historical fsstore rules shared by both the fs and
    git backends so the stores are interoperable on the same directory.
    """
    lp = str(path)
    if lp in ("kb-schema.md", "schema-version.yaml"):
        return lp
    if lp.startswith("memory/global/"):
        return "memory/" + lp[len("memory/global/") :]
    if lp == "memory/global":
        return "memory"
    if lp.startswith("memory/project/"):
        rest = lp[len("memory/project/") :]
        slash = rest.find("/")
        if slash == -1:
            return f"projects/{rest}/memory"
        slug = rest[:slash]
        tail = rest[slash + 1 :]
        return f"projects/{slug}/memory/{tail}"
    if lp == "memory/project":
        return "projects"
    if lp == "wiki" or lp.startswith("wiki/"):
        return lp
    if lp == "raw" or lp.startswith("raw/"):
        return lp
    if lp == "memory" or lp.startswith("memory/"):
        # Treat bare `memory/...` paths as global memory for the fs layout.
        return lp
    raise ErrInvalidPath(f"brain: invalid path: unknown root in {lp!r}")


def logical_from_rel(rel: str) -> BrainPath:
    """Inverse of `relative` — map an on-disk relative path to its logical form."""
    rel = rel.replace(os.sep, "/")
    if rel in ("kb-schema.md", "schema-version.yaml"):
        return BrainPath(rel)
    if rel == "memory":
        return BrainPath("memory/global")
    if rel.startswith("memory/"):
        return BrainPath("memory/global/" + rel[len("memory/") :])
    if rel == "projects":
        return BrainPath("memory/project")
    if rel.startswith("projects/"):
        rest = rel[len("projects/") :]
        slash = rest.find("/")
        if slash == -1:
            return BrainPath(f"memory/project/{rest}")
        slug = rest[:slash]
        tail = rest[slash + 1 :]
        if tail == "memory":
            return BrainPath(f"memory/project/{slug}")
        if tail.startswith("memory/"):
            return BrainPath(f"memory/project/{slug}/" + tail[len("memory/") :])
        return BrainPath(f"memory/project/{slug}/{tail}")
    if rel == "wiki" or rel.startswith("wiki/"):
        return BrainPath(rel)
    if rel == "raw" or rel.startswith("raw/"):
        return BrainPath(rel)
    return BrainPath(rel)


def resolve(root: str | Path, path: BrainPath | str) -> str:
    """Turn a logical path into an absolute on-disk path under root."""
    validate_path(str(path))
    rel = relative(path)
    return str(Path(root) / rel)


def should_skip_dir(name: str) -> bool:
    """True for directories that must never appear in a list output."""
    if name == ".git":
        return True
    if name.startswith(".brain-staging"):
        return True
    return False


def should_skip_file(name: str) -> bool:
    """True for files that must never appear in a list output."""
    if name == ".git":
        return True
    if name.startswith(".brain-tmp-"):
        return True
    return False


def atomic_write(abs_path: str | Path, content: bytes, perm: int = 0o644) -> None:
    """POSIX atomic write: temp file in the same directory, fsync, rename.

    Callers either see the old content or the new content — never a torn
    write. A crash at any point leaves either the old file or no file at
    all. The parent directory is fsync'd on best-effort.
    """
    target = Path(abs_path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".brain-tmp-", dir=str(parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, perm)
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
    try:
        dir_fd = os.open(str(parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def list_dir(
    abs_root: str | Path,
    abs_dir: str | Path,
    logical_dir: BrainPath | str,
    opts: ListOpts,
) -> list[FileInfo]:
    """Walk the working tree under `abs_dir` and return matching entries.

    Mirrors Go's `storeutil.List`. Results are sorted by logical path.
    """
    abs_root_s = str(abs_root)
    abs_dir_p = Path(abs_dir)
    if not abs_dir_p.exists():
        return []
    if not abs_dir_p.is_dir():
        raise ValueError(f"list {logical_dir}: not a directory")

    out: list[FileInfo] = []

    def visit(path: Path, is_dir: bool) -> None:
        name = path.name
        if is_dir and should_skip_dir(name):
            return
        if not is_dir and should_skip_file(name):
            return
        try:
            rel = path.relative_to(abs_root_s)
        except ValueError:
            return
        logical = logical_from_rel(str(rel))
        if not is_dir and not opts.include_generated and is_generated(logical):
            return
        if opts.glob:
            if not fnmatch.fnmatchcase(name, opts.glob):
                return
        try:
            stat = path.stat()
        except OSError:
            return
        out.append(
            FileInfo(
                path=logical,
                size=stat.st_size if not is_dir else 0,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                is_dir=is_dir,
            )
        )

    if opts.recursive:
        for dirpath, dirnames, filenames in os.walk(abs_dir_p):
            # Prune hidden/metadata dirs in-place so os.walk skips them.
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
            for f in filenames:
                visit(Path(dirpath) / f, is_dir=False)
    else:
        for child in abs_dir_p.iterdir():
            visit(child, is_dir=child.is_dir())

    out.sort(key=lambda fi: fi.path)
    return out


def path_under(path: str, directory: str, recursive: bool) -> bool:
    """Report whether `path` falls under `directory` with shallow/recursive rules."""
    if directory == "":
        return True
    if path == directory:
        return True
    if not path.startswith(directory + "/"):
        return False
    if recursive:
        return True
    rest = path[len(directory) + 1 :]
    return "/" not in rest


def glob_match_base(pattern: str, path: str) -> bool:
    """Match `pattern` against the base name of `path` using fnmatch rules."""
    return fnmatch.fnmatchcase(posixpath.basename(path), pattern)
