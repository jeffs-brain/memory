# SPDX-License-Identifier: Apache-2.0
"""Heuristic memory file management (reinforcement + listing)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ._memstore import ListOpts, NotFoundError
from .paths import (
    base_name,
    memory_global_index,
    memory_global_prefix,
    memory_global_topic,
    memory_project_index,
    memory_project_prefix,
    memory_project_topic,
)
from .reflect import Heuristic
from .store import parse_frontmatter
from .types import TopicFile

if TYPE_CHECKING:
    from .manager import MemoryManager

STOP_WORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "for",
    "to",
    "of",
    "with",
    "when",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "not",
    "do",
}

_NON_ALPHA_NUM = re.compile(r"[^a-z0-9 ]")


class HeuristicSummary:
    def __init__(
        self,
        name: str,
        path: str,
        confidence: str,
        tags: list[str],
        scope: str,
        is_anti: bool,
    ) -> None:
        self.name = name
        self.path = path
        self.confidence = confidence
        self.tags = tags
        self.scope = scope
        self.is_anti = is_anti


def apply_heuristics(
    mem: "MemoryManager", project_slug: str, heuristics: list[Heuristic]
) -> None:
    project_entries: list[str] = []
    global_entries: list[str] = []
    writes: list[tuple[str, bytes]] = []

    for h in heuristics:
        if not h.rule:
            continue
        if h.scope == "global":
            prefix = memory_global_prefix()
        else:
            prefix = memory_project_prefix(project_slug)
        existing_path, existing_content, found = _find_existing_heuristic(
            mem, h, prefix
        )
        if found:
            content = merge_heuristic(existing_content, h)
            path = existing_path
        else:
            content = build_heuristic_content(h)
            fname_no_ext = heuristic_filename(h)[: -len(".md")]
            if h.scope == "global":
                path = memory_global_topic(fname_no_ext)
            else:
                path = memory_project_topic(project_slug, fname_no_ext)

        writes.append((path, content.encode("utf-8")))

        entry = f"- [heuristic] {base_name(path)}: {truncate(h.rule, 100)}"
        if h.scope == "global":
            global_entries.append(entry)
        else:
            project_entries.append(entry)

    if not writes:
        return

    def _run(b) -> None:
        for p, c in writes:
            b.write(p, c)
        if project_entries:
            _append_index_entries(b, memory_project_index(project_slug), project_entries)
        if global_entries:
            _append_index_entries(b, memory_global_index(), global_entries)

    mem.store.batch(_run)


def _find_existing_heuristic(
    mem: "MemoryManager", h: Heuristic, prefix: str
) -> tuple[str, str, bool]:
    try:
        entries = mem.store.list(prefix, ListOpts(include_generated=True))
    except NotFoundError:
        return "", "", False

    candidate_words = significant_words(heuristic_filename(h))
    for entry in entries:
        if entry.is_dir:
            continue
        name = base_name(entry.path)
        if not name.endswith(".md") or name.lower() == "memory.md":
            continue
        try:
            data = mem.store.read(entry.path)
        except NotFoundError:
            continue
        fm, _ = parse_frontmatter(data.decode("utf-8"))
        if not has_tag(fm.tags, "heuristic"):
            continue
        if not has_tag(fm.tags, h.category):
            continue
        existing_words = significant_words(name)
        if jaccard_similarity(candidate_words, existing_words) > 0.5:
            return entry.path, data.decode("utf-8"), True
    return "", "", False


def heuristic_filename(h: Heuristic) -> str:
    words = significant_words(h.rule)
    slug = h.category or "general"
    prefix = "heuristic-anti" if h.anti_pattern else "heuristic"
    limit = min(2, len(words))
    parts = [prefix, slug] + words[:limit]
    return "-".join(parts) + ".md"


def significant_words(text: str) -> list[str]:
    cleaned = _NON_ALPHA_NUM.sub(" ", text.lower())
    result: list[str] = []
    for w in cleaned.split():
        if w not in STOP_WORDS and len(w) > 1:
            result.append(w)
    return result


def build_heuristic_content(h: Heuristic) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    heading = first_n_words(h.rule, 5)
    lines = ["---"]
    lines.append(f'name: "{capitalise(h.category)}: {heading}"')
    lines.append(f'description: "{truncate(h.rule, 100)}"')
    lines.append("type: feedback")
    lines.append(f"created: {now}")
    lines.append(f"modified: {now}")
    lines.append(f"confidence: {h.confidence}")
    lines.append("source: reflection")
    lines.append("tags:")
    lines.append("  - heuristic")
    lines.append(f"  - {h.category}")
    if h.anti_pattern:
        lines.append("  - anti-pattern")
    lines.append("---")
    lines.append("")
    if h.anti_pattern:
        lines.append(f"## Anti-pattern: {heading}")
        lines.append("")
        lines.append(f"**Don't:** {h.rule}")
        lines.append("")
        alt = extract_alternative(h.rule)
        if alt:
            lines.append(f"**Instead:** {alt}")
            lines.append("")
        lines.append("**Why:** Observed during reflection")
        lines.append("")
        lines.append(f"**Confidence:** {h.confidence} (1 observation)")
    else:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(h.rule)
        lines.append("")
        if h.context:
            lines.append(f"**Context:** {h.context}")
            lines.append("")
        lines.append("**Why:** Observed during reflection")
        lines.append("")
        lines.append(f"**Confidence:** {h.confidence} (1 observation)")
    return "\n".join(lines) + "\n"


def merge_heuristic(existing_content: str, h: Heuristic) -> str:
    fm, body = parse_frontmatter(existing_content)
    count = count_sections(body)
    new_confidence = confidence_from_observations(count + 1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    heading = first_n_words(h.rule, 5)
    section_lines: list[str] = [""]
    if h.anti_pattern:
        section_lines.append(f"## Anti-pattern: {heading}")
        section_lines.append("")
        section_lines.append(f"**Don't:** {h.rule}")
        section_lines.append("")
        section_lines.append("**Why:** Observed during reflection")
        section_lines.append("")
        section_lines.append(f"**Confidence:** {new_confidence} ({count + 1} observations)")
    else:
        section_lines.append(f"## {heading}")
        section_lines.append("")
        section_lines.append(h.rule)
        section_lines.append("")
        if h.context:
            section_lines.append(f"**Context:** {h.context}")
            section_lines.append("")
        section_lines.append("**Why:** Observed during reflection")
        section_lines.append("")
        section_lines.append(f"**Confidence:** {new_confidence} ({count + 1} observations)")

    lines = ["---"]
    if fm.name:
        lines.append(f'name: "{fm.name}"')
    if fm.description:
        lines.append(f'description: "{fm.description}"')
    if fm.type:
        lines.append(f"type: {fm.type}")
    if fm.created:
        lines.append(f"created: {fm.created}")
    lines.append(f"modified: {now}")
    lines.append(f"confidence: {new_confidence}")
    if fm.source:
        lines.append(f"source: {fm.source}")
    if fm.tags:
        lines.append("tags:")
        for tag in fm.tags:
            lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    lines.extend(section_lines)
    return "\n".join(lines)


def confidence_from_observations(count: int) -> str:
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def has_tag(tags: list[str], tag: str) -> bool:
    target = tag.lower()
    return any(t.lower() == target for t in tags)


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def count_sections(body: str) -> int:
    count = 0
    for line in body.split("\n"):
        if line.strip().startswith("## "):
            count += 1
    return count


def first_n_words(text: str, n: int) -> str:
    words = text.split()
    if len(words) > n:
        words = words[:n]
    return " ".join(words)


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len < 4:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def capitalise(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]


def extract_alternative(rule: str) -> str:
    lower = rule.lower()
    for marker in ("instead ", "use ", "prefer "):
        idx = lower.find(marker)
        if idx >= 0:
            return rule[idx:].strip()
    return ""


def _append_index_entries(b, index_path: str, entries: list[str]) -> None:
    content = ""
    try:
        existing = b.read(index_path)
        content = existing.decode("utf-8").strip()
    except NotFoundError:
        content = ""
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if entry in content:
            continue
        if content:
            content += "\n"
        content += entry
    b.write(index_path, (content + "\n").encode("utf-8"))


def list_heuristics_in(mem: "MemoryManager", project_path: str) -> list[HeuristicSummary]:
    summaries: list[HeuristicSummary] = []
    for t in mem.list_project_topics(project_path):
        if has_tag(t.tags, "heuristic"):
            summaries.append(
                HeuristicSummary(
                    name=t.name,
                    path=t.path,
                    confidence=t.confidence,
                    tags=list(t.tags),
                    scope="project",
                    is_anti=has_tag(t.tags, "anti-pattern"),
                )
            )
    for t in mem.list_global_topics():
        if has_tag(t.tags, "heuristic"):
            summaries.append(
                HeuristicSummary(
                    name=t.name,
                    path=t.path,
                    confidence=t.confidence,
                    tags=list(t.tags),
                    scope="global",
                    is_anti=has_tag(t.tags, "anti-pattern"),
                )
            )
    return summaries
