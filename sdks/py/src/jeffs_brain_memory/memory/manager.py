# SPDX-License-Identifier: Apache-2.0
"""MemoryManager: top-level orchestration over a memory store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._memstore import ListOpts, NotFoundError, Store
from .paths import (
    base_name,
    memory_global_index,
    memory_global_prefix,
    memory_project_index,
    memory_project_prefix,
    project_slug as _project_slug,
)
from .store import parse_frontmatter
from .types import TopicFile

if TYPE_CHECKING:
    from .extract import ExtractedMemory

_MAX_INDEX_LINES = 200


class MemoryManager:
    """Entry point for all memory operations backed by a :class:`Store`."""

    def __init__(self, store: Store) -> None:
        self._store = store

    @property
    def store(self) -> Store:
        return self._store

    # ---- index ----

    def load_index_at(self, path: str) -> str:
        try:
            data = self._store.read(path)
        except NotFoundError:
            return ""
        content = data.decode("utf-8")
        lines = content.split("\n")
        if len(lines) > _MAX_INDEX_LINES:
            lines = lines[:_MAX_INDEX_LINES] + ["[...truncated]"]
        return "\n".join(lines).strip()

    def load_project_index(self, project_path: str) -> str:
        slug = _project_slug(project_path)
        return self.load_index_at(memory_project_index(slug))

    def load_global_index(self) -> str:
        return self.load_index_at(memory_global_index())

    # ---- topics ----

    def _list_topics_under(self, prefix: str, scope: str) -> list[TopicFile]:
        try:
            entries = self._store.list(prefix, ListOpts(include_generated=True))
        except NotFoundError:
            return []

        topics: list[TopicFile] = []
        for entry in entries:
            if entry.is_dir:
                continue
            name = base_name(entry.path)
            if not name.endswith(".md"):
                continue
            if name.lower() == "memory.md":
                continue
            try:
                data = self._store.read(entry.path)
            except NotFoundError:
                continue
            fm, _ = parse_frontmatter(data.decode("utf-8"))
            topic_name = fm.name or name[: -len(".md")]
            topics.append(
                TopicFile(
                    name=topic_name,
                    description=fm.description,
                    type=fm.type,
                    path=entry.path,
                    created=fm.created,
                    modified=fm.modified,
                    tags=list(fm.tags),
                    confidence=fm.confidence,
                    source=fm.source,
                    scope=scope,
                )
            )
        return topics

    def list_project_topics(self, project_path: str) -> list[TopicFile]:
        slug = _project_slug(project_path)
        return self._list_topics_under(memory_project_prefix(slug), "project")

    def list_global_topics(self) -> list[TopicFile]:
        return self._list_topics_under(memory_global_prefix(), "global")

    def read_topic(self, path: str) -> str:
        data = self._store.read(path)
        return data.decode("utf-8")

    # ---- prompt helpers ----

    def build_memory_prompt_for(self, project_path: str) -> str:
        global_index = self.load_global_index()
        project_index = self.load_project_index(project_path)
        if not global_index and not project_index:
            return ""

        parts: list[str] = []
        if global_index:
            label = "memory/global"
            phys = self._store.local_path(memory_global_prefix())
            if phys:
                label = f"memory/global ({phys})"
            parts.append("# Global Memory\n\n" + global_index)
            parts.append(f"**Global memory directory:** {label}\n")
        if project_index:
            slug = _project_slug(project_path)
            label = f"memory/project/{slug}"
            phys = self._store.local_path(memory_project_prefix(slug))
            if phys:
                label = f"memory/project/{slug} ({phys})"
            parts.append("# Project Memory\n\n" + project_index)
            parts.append(f"**Project memory directory:** {label}")
        return "\n\n".join(parts).strip()

    # ---- wikilinks ----

    def resolve_wikilink(self, link: str, project_path: str) -> str:
        """Resolve ``[[topic]]`` or ``[[global:topic]]`` to a brain path."""
        from .wikilink import resolve_wikilink as _impl

        return _impl(self, link, project_path)

    def resolve_all_wikilinks(self, content: str, project_path: str) -> list[str]:
        from .wikilink import resolve_all_wikilinks as _impl

        return _impl(self, content, project_path)

    def exists(self, path: str) -> bool:
        return self._store.exists(path)

    # ---- deferred bridges ----

    def apply_extractions(
        self, project_slug: str, memories: list["ExtractedMemory"]
    ) -> None:
        from .extract import apply_extractions as _impl

        _impl(self, project_slug, memories)

    def apply_heuristics(self, project_slug: str, heuristics) -> None:
        from .heuristic import apply_heuristics as _impl

        _impl(self, project_slug, heuristics)


# Convenience alias so callers used to Go naming find it.
Memory = MemoryManager
