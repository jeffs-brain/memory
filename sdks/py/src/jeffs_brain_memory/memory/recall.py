# SPDX-License-Identifier: Apache-2.0
"""Memory recall: pick relevant memories, format with date headers.

``RECALL_SELECTOR_PROMPT`` is ported verbatim from
``sdks/go/memory/recall.go``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role
from .extract import build_manifest
from .heuristic import has_tag
from .paths import base_name
from .store import parse_frontmatter
from .types import TopicFile
from .wikilink import extract_wikilinks

if TYPE_CHECKING:
    from .manager import MemoryManager

MAX_RECALL_TOPICS = 5
MAX_SCAN_FILES = 200
MAX_MEMORY_LINES = 200
MAX_MEMORY_BYTES = 4096
MAX_LINKED_MEMORIES = 2
RECALL_MAX_TOKENS = 256
RECALL_TEMPERATURE = 0.0

GLOBAL_PLENTIFUL_THRESHOLD = 5
ASSISTANT_PROJECT_CAP = 10


RECALL_SELECTOR_PROMPT = """You are selecting memories that will be useful to an AI assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected" array of filenames for the memories that will clearly be useful (up to 5). Only include memories you are certain will be helpful based on their name and description.

- If unsure whether a memory is relevant, do not include it. Be selective.
- If no memories are relevant, return an empty array.

Memories may be project-scoped (specific to this codebase) or global (cross-project knowledge about the user, their preferences and history).
Both can be useful \u2014 prefer project memories when the query is about this specific codebase, and global memories when the query is about general patterns, personal context, or user preferences.

Memories tagged [heuristic] are learned patterns from past sessions. Prefer high-confidence heuristics when they match the task.

Respond with ONLY valid JSON, no other text. Example: {"selected": ["feedback_testing.md", "project_auth.md"]}"""

RECALL_HINT_ASSISTANT = "\n\nThe user is currently in assistant mode \u2014 a conversational personal assistant session. Prefer global/personal memories over project-specific ones unless the query is clearly about a specific codebase."

RECALL_HINT_CODING = "\n\nThe user is currently in coding mode \u2014 an AI coding harness session. Prefer project-specific memories over global ones unless the query is clearly about general user preferences."


@dataclass(slots=True)
class SurfacedMemory:
    path: str = ""
    content: str = ""
    topic: TopicFile = field(default_factory=TopicFile)
    linked_from: str = ""


@dataclass(slots=True)
class RecallWeights:
    global_weight: float = 0.0
    project_weight: float = 0.0


async def recall(
    mem: "MemoryManager",
    provider: Provider,
    model: str,
    project_path: str,
    user_query: str,
    surfaced: set[str] | None,
    weights: RecallWeights,
) -> list[SurfacedMemory]:
    surfaced = surfaced or set()
    project_topics = mem.list_project_topics(project_path)
    global_topics = mem.list_global_topics()

    project_topics = [t for t in project_topics if t.path not in surfaced]
    global_topics = [t for t in global_topics if t.path not in surfaced]

    candidates: list[TopicFile] = []
    if weights.global_weight > weights.project_weight:
        candidates.extend(global_topics)
        if len(global_topics) < GLOBAL_PLENTIFUL_THRESHOLD:
            cap = min(ASSISTANT_PROJECT_CAP, len(project_topics))
            candidates.extend(project_topics[:cap])
    elif weights.project_weight > weights.global_weight:
        candidates.extend(project_topics)
        candidates.extend(global_topics)
    else:
        candidates.extend(project_topics)
        candidates.extend(global_topics)

    if not candidates:
        return []
    if len(candidates) > MAX_SCAN_FILES:
        candidates = candidates[:MAX_SCAN_FILES]

    manifest = build_manifest(candidates)
    user_prompt = f"Query: {user_query}\n\nAvailable memories:\n{manifest}"

    system_prompt = RECALL_SELECTOR_PROMPT
    if weights.global_weight > weights.project_weight:
        system_prompt += RECALL_HINT_ASSISTANT
    elif weights.project_weight > weights.global_weight:
        system_prompt += RECALL_HINT_CODING

    try:
        resp = await provider.complete(
            CompleteRequest(
                model=model,
                messages=[
                    LLMMessage(role=Role.SYSTEM, content=system_prompt),
                    LLMMessage(role=Role.USER, content=user_prompt),
                ],
                max_tokens=RECALL_MAX_TOKENS,
                temperature=RECALL_TEMPERATURE,
            )
        )
    except Exception:
        return []

    selected = parse_selected_memories(resp.text)
    if not selected:
        return []

    topic_by_file = {base_name(t.path): t for t in candidates}
    memories: list[SurfacedMemory] = []
    for filename in selected:
        topic = topic_by_file.get(filename)
        if topic is None:
            continue
        try:
            content = read_capped_topic(mem, topic.path)
        except Exception:
            continue
        memories.append(SurfacedMemory(path=topic.path, content=content, topic=topic))
        if len(memories) >= MAX_RECALL_TOPICS:
            break

    linked = follow_wikilinks(mem, memories, surfaced, project_path)
    memories.extend(linked)
    return memories


def follow_wikilinks(
    mem: "MemoryManager",
    memories: list[SurfacedMemory],
    surfaced: set[str],
    project_path: str,
) -> list[SurfacedMemory]:
    loaded: set[str] = {m.path for m in memories}
    linked: list[SurfacedMemory] = []
    for mem_ in memories:
        if len(linked) >= MAX_LINKED_MEMORIES:
            break
        for link in extract_wikilinks(mem_.content):
            if len(linked) >= MAX_LINKED_MEMORIES:
                break
            resolved = mem.resolve_wikilink(link, project_path)
            if not resolved or resolved in loaded or resolved in surfaced:
                continue
            try:
                content = read_capped_topic(mem, resolved)
            except Exception:
                continue
            link_target = link.split("|", 1)[0].strip() if "|" in link else link
            linked.append(
                SurfacedMemory(
                    path=resolved,
                    content=content,
                    topic=topic_from_path(mem, resolved),
                    linked_from=link_target,
                )
            )
            loaded.add(resolved)
    return linked


def topic_from_path(mem: "MemoryManager", path: str) -> TopicFile:
    try:
        content = mem.read_topic(path)
    except Exception:
        return TopicFile(path=path, scope=infer_scope(path))
    fm, _ = parse_frontmatter(content)
    name = fm.name or base_name(path)[: -len(".md")]
    return TopicFile(
        name=name,
        description=fm.description,
        type=fm.type,
        path=path,
        created=fm.created,
        modified=fm.modified,
        tags=list(fm.tags),
        confidence=fm.confidence,
        source=fm.source,
        scope=infer_scope(path),
    )


def infer_scope(path: str) -> str:
    if path.startswith("memory/global/") or path == "memory/global":
        return "global"
    return "project"


def parse_selected_memories(content: str) -> list[str]:
    content = content.strip()
    start = content.find("{")
    if start >= 0:
        end = content.rfind("}")
        if end > start:
            content = content[start : end + 1]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    selected = parsed.get("selected", [])
    if not isinstance(selected, list):
        return []
    out = [str(s) for s in selected]
    if len(out) > MAX_RECALL_TOPICS:
        out = out[:MAX_RECALL_TOPICS]
    return out


def read_capped_topic(mem: "MemoryManager", path: str) -> str:
    content = mem.read_topic(path)
    if len(content) > MAX_MEMORY_BYTES:
        content = content[:MAX_MEMORY_BYTES] + "\n[...truncated]"
    lines = content.split("\n")
    if len(lines) > MAX_MEMORY_LINES:
        lines = lines[:MAX_MEMORY_LINES] + ["[...truncated]"]
        content = "\n".join(lines)
    return content


# ---- formatting ----


def format_recalled_memories(memories: list[SurfacedMemory]) -> str:
    return format_recalled_memories_with_context(memories, datetime.now(timezone.utc))


def format_recalled_memories_with_context(
    memories: list[SurfacedMemory], now: datetime
) -> str:
    if not memories:
        return ""
    parts: list[str] = []
    for m in memories:
        age = topic_age(m.topic.modified, now)
        label = memory_label(m)
        block = "<system-reminder>\n"
        block += f"{label} (saved {age}): {base_name(m.path)}\n"
        header = date_header(m.topic.modified, now)
        if header:
            block += header + "\n"
        block += "\n"
        block += m.content
        block += "\n</system-reminder>\n"
        parts.append(block)
    return "".join(parts).strip()


def sort_memories_chronologically(
    memories: list[SurfacedMemory],
) -> list[SurfacedMemory]:
    if not memories:
        return []
    dated: list[tuple[int, datetime, SurfacedMemory]] = []
    undated: list[tuple[int, SurfacedMemory]] = []
    for i, m in enumerate(memories):
        t = parse_topic_time(m.topic.modified)
        if t is None:
            undated.append((i, m))
        else:
            dated.append((i, t, m))
    dated.sort(key=lambda x: (x[1], x[0]))
    out: list[SurfacedMemory] = [d[2] for d in dated]
    for _, m in undated:
        out.append(m)
    return out


def parse_topic_time(modified: str) -> datetime | None:
    if not modified:
        return None
    if modified.endswith("Z"):
        s = modified[:-1] + "+00:00"
    else:
        s = modified
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def date_header(modified: str, now: datetime) -> str:
    t = parse_topic_time(modified)
    if t is None:
        return ""
    iso = t.astimezone(timezone.utc).strftime("%Y-%m-%d")
    rel = relative_time_string(t, now)
    if not rel:
        return f"=== {iso} ==="
    return f"=== {iso} ({rel}) ==="


def topic_age(modified: str, now: datetime) -> str:
    t = parse_topic_time(modified)
    if t is None:
        return "unknown time ago"
    delta = now - t
    days = int(delta.total_seconds() // 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def relative_time_string(then: datetime, now: datetime) -> str:
    diff = now - then
    total = diff.total_seconds()
    if total < 0:
        return ""
    days = int(total // 86400)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days <= 6:
        return f"{days} days ago"
    if days <= 27:
        weeks = (days + 3) // 7
        if weeks == 1:
            return "1 week ago"
        return f"{weeks} weeks ago"
    if days <= 364:
        months = (days + 15) // 30
        if months < 1:
            months = 1
        if months == 1:
            return "1 month ago"
        return f"{months} months ago"
    years = days // 365
    if years == 1:
        return "1 year ago"
    return f"{years} years ago"


def memory_label(m: SurfacedMemory) -> str:
    if m.linked_from:
        return f"Linked memory (via [[{m.linked_from}]])"
    if has_tag(m.topic.tags, "heuristic"):
        conf = m.topic.confidence or "low"
        return f"Learned heuristic ({conf} confidence)"
    if m.topic.scope == "global":
        return "Global memory"
    return "Memory"
