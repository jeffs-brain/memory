# SPDX-License-Identifier: Apache-2.0
"""`[[topic]]` wikilink parsing and resolution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .paths import (
    memory_global_prefix,
    memory_project_prefix,
    project_slug as _project_slug,
)

if TYPE_CHECKING:
    from .manager import MemoryManager

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def extract_wikilinks(content: str) -> list[str]:
    return WIKILINK_PATTERN.findall(content)


def normalise_topic(topic: str) -> str:
    topic = topic.strip().lower().replace(" ", "-")
    return topic


def _resolve_in(mem: "MemoryManager", topic: str, prefix: str) -> str:
    normalised = normalise_topic(topic)
    if not normalised:
        return ""
    candidate = f"{prefix}/{normalised}.md"
    try:
        if mem.exists(candidate):
            return candidate
    except Exception:
        return ""
    return ""


def resolve_wikilink(mem: "MemoryManager", link: str, project_path: str) -> str:
    if "|" in link:
        link = link.split("|", 1)[0]
    link = link.strip()
    if not link:
        return ""

    slug = _project_slug(project_path)

    if link.startswith("global:"):
        topic = link[len("global:") :].strip()
        return _resolve_in(mem, topic, memory_global_prefix())

    resolved = _resolve_in(mem, link, memory_project_prefix(slug))
    if resolved:
        return resolved
    return _resolve_in(mem, link, memory_global_prefix())


def resolve_all_wikilinks(
    mem: "MemoryManager", content: str, project_path: str
) -> list[str]:
    links = extract_wikilinks(content)
    if not links:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for link in links:
        resolved = resolve_wikilink(mem, link, project_path)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out
