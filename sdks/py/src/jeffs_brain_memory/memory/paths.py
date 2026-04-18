# SPDX-License-Identifier: Apache-2.0
"""Canonical logical path helpers for the memory layer.

Mirrors the Go `brain` package's path helpers so memory code can build
stable identifiers without string concatenation.
"""

from __future__ import annotations

import os
import posixpath
import re
import subprocess
import threading
from pathlib import Path
from typing import NewType
from urllib.parse import urlparse

BrainPath = NewType("BrainPath", str)

_MEMORY_ROOT = "memory"


def memory_global_index() -> BrainPath:
    return BrainPath(f"{_MEMORY_ROOT}/global/MEMORY.md")


def memory_global_topic(slug: str) -> BrainPath:
    return BrainPath(posixpath.join(_MEMORY_ROOT, "global", f"{slug}.md"))


def memory_global_prefix() -> BrainPath:
    return BrainPath(f"{_MEMORY_ROOT}/global")


def memory_project_index(project_slug: str) -> BrainPath:
    return BrainPath(posixpath.join(_MEMORY_ROOT, "project", project_slug, "MEMORY.md"))


def memory_project_topic(project_slug: str, slug: str) -> BrainPath:
    return BrainPath(posixpath.join(_MEMORY_ROOT, "project", project_slug, f"{slug}.md"))


def memory_project_prefix(project_slug: str) -> BrainPath:
    return BrainPath(posixpath.join(_MEMORY_ROOT, "project", project_slug))


def memory_projects_prefix() -> BrainPath:
    return BrainPath(f"{_MEMORY_ROOT}/project")


def memory_buffer_global() -> BrainPath:
    return BrainPath(f"{_MEMORY_ROOT}/buffer/global.md")


def memory_buffer_project(project_slug: str) -> BrainPath:
    return BrainPath(posixpath.join(_MEMORY_ROOT, "buffer", "project", f"{project_slug}.md"))


def base_name(path: str) -> str:
    idx = path.rfind("/")
    if idx >= 0:
        return path[idx + 1 :]
    return path


_slug_state_lock = threading.Lock()
_slug_state: dict[str, object] = {"map": None}


def _ensure_slug_map() -> "SlugMap":
    with _slug_state_lock:
        sm = _slug_state.get("map")
        if sm is None:
            sm = SlugMap("")
            sm.load()
            _slug_state["map"] = sm
        return sm  # type: ignore[return-value]


def set_slug_map_for_test(path: str):
    """Install a slug map backed by the given file; returns a cleanup fn."""
    with _slug_state_lock:
        prev = _slug_state.get("map")
        sm = SlugMap(path)
        sm.load()
        _slug_state["map"] = sm

    def _restore() -> None:
        with _slug_state_lock:
            _slug_state["map"] = prev

    return _restore


def project_slug(project_path: str) -> str:
    """Return a safe directory name derived from the project path."""
    sm = _ensure_slug_map()

    try:
        abs_path = os.path.abspath(project_path)
    except (TypeError, ValueError):
        abs_path = project_path

    existing = sm.lookup(abs_path)
    if existing is not None:
        return existing

    root = _git_root(project_path)
    canonical = ""
    if root:
        remote = _git_remote_url(root)
        if remote:
            canonical = _canonical_slug_from_remote(remote)

    if not canonical:
        base = root or abs_path
        canonical = _path_based_slug(base)

    try:
        sm.register(abs_path, canonical)
    except OSError:
        pass
    return canonical


def _path_based_slug(root: str) -> str:
    slug = root.replace(os.sep, "/")
    slug = slug.replace("/", "-")
    return slug.lstrip("-")


def _canonical_slug_from_remote(remote_url: str) -> str:
    remote_url = remote_url.strip()
    if not remote_url:
        return ""

    if "://" not in remote_url:
        if ":" in remote_url:
            path_part = remote_url.split(":", 1)[1]
        else:
            return ""
    else:
        try:
            parsed = urlparse(remote_url)
        except ValueError:
            return ""
        if not parsed.path:
            return ""
        path_part = parsed.path

    if path_part.endswith(".git"):
        path_part = path_part[:-4]
    path_part = path_part.strip("/")
    if not path_part:
        return ""

    parts = path_part.split("/")
    return "-".join(parts).lower()


def _git_root(project_path: str) -> str:
    if not project_path:
        return ""
    try:
        abs_path = os.path.abspath(project_path)
    except (TypeError, ValueError):
        return ""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=abs_path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_remote_url(repo_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def brain_root() -> str:
    override = os.environ.get("JEFFS_BRAIN_ROOT", "")
    if override:
        return override
    home = os.path.expanduser("~")
    if not home or home == "~":
        return "."
    return str(Path(home) / ".config" / "jeffs-brain")


# ---- re-export for __init__ convenience ----
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


# Deferred circular-free import.
from .slugmap import SlugMap  # noqa: E402
