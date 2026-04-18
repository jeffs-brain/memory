# SPDX-License-Identifier: Apache-2.0
"""Extract durable knowledge from a session into memory files.

The ``EXTRACTION_PROMPT`` is ported verbatim from ``sdks/go/memory/extract.go``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role
from ._memstore import ListOpts, NotFoundError
from .paths import (
    base_name,
    memory_global_index,
    memory_global_prefix,
    memory_global_topic,
    memory_project_index,
    memory_project_prefix,
    memory_project_topic,
    project_slug as _project_slug,
)
from .store import parse_frontmatter
from .types import Message, TopicFile, messages_as_llm

if TYPE_CHECKING:
    from .contextualise import Contextualiser
    from .manager import MemoryManager

EXTRACT_MAX_TOKENS = 4096
EXTRACT_TEMPERATURE = 0.2
EXTRACT_MIN_MESSAGES = 6
EXTRACT_MAX_RECENT = 40


EXTRACTION_PROMPT = """You are a memory extraction agent. Analyse the recent conversation messages below and determine what durable knowledge should be saved to the persistent memory system.

You MUST respond with ONLY a JSON object. Do NOT call tools, do NOT write prose. Just output the JSON.

Both speakers contribute durable knowledge. Treat user turns and assistant turns as equally valid sources of facts. Capture everything the user stated AND everything the assistant provided: recommendations (restaurants, hotels, shops, books), specific named suggestions, recipes, itineraries, enumerated lists or rankings the assistant gave, answers the assistant produced, corrections the assistant issued, plans the assistant proposed, colours or attributes the assistant described, and any quantities or dates the assistant cited. If the assistant enumerated items (a list of jobs, options, steps, or candidates), save the full enumeration verbatim including positions where relevant. When in doubt, extract both sides.

Memory types:
- user: User's role, preferences, knowledge level, working style
- feedback: Corrections or confirmations about approach (what to avoid or keep doing)
- project: Non-obvious context about ongoing work, goals, decisions, deadlines (includes assistant recommendations and enumerations worth recalling later)
- reference: Pointers to external systems, URLs, project names, named entities the assistant surfaced (restaurants, hotels, businesses, books, product names)

Memory scopes:
- global (~/.config/jeff/memory/): Cross-project knowledge. Types: user, feedback
- project (project memory directory): Project-specific knowledge. Types: project, reference

When deciding scope:
- user preferences, working style, general corrections \u2192 global
- project architecture, project-specific decisions, external system pointers, assistant recommendations and enumerations \u2192 project
- default to "project" if unsure

Examples of assistant-turn facts that MUST be captured:
- "I recommend Roscioli for romantic Italian in Rome." \u2192 create a reference memory naming the restaurant, cuisine, city.
- "Here are seven work-from-home jobs for seniors: 1. Virtual Assistant, 2. ..., 7. Transcriptionist." \u2192 save the full numbered list so later recall can reconstruct any position.
- "The Plesiosaur in the children's book had a blue scaly body." \u2192 save the attribute with its subject.

Do NOT save:
- Code patterns, architecture, or file paths derivable from the codebase
- Git history or recent changes (use git log for those)
- Debugging solutions (the fix is in the code)
- Ephemeral task details or in-progress work
- Anything already in the existing memories listed below

For each memory worth saving, output:
- action: "create" (new file) or "update" (modify existing)
- filename: e.g. "feedback_testing.md" (kebab-case, descriptive)
- name: human-readable name
- description: one-line description (used for future recall)
- type: user | feedback | project | reference
- scope: "global" or "project" (default to "project" if unsure)
- content: the memory content (structured with Why: and How to apply: lines for feedback/project types)
- index_entry: one-line entry for MEMORY.md (under 150 chars)
- supersedes (optional): when the user has corrected, updated, or contradicted an earlier stated fact for the same topic, set this to the filename of the earlier memory so it is retired. Only fill when you are confident the new fact replaces a specific older one; prefer leaving empty when unsure.

If nothing is worth saving, return: {"memories": []}

Respond with ONLY valid JSON: {"memories": [...]}"""


@dataclass(slots=True)
class ExtractedMemory:
    """A single memory extracted from a conversation."""

    action: str = ""
    filename: str = ""
    name: str = ""
    description: str = ""
    type: str = ""
    content: str = ""
    index_entry: str = ""
    scope: str = ""
    supersedes: str = ""
    tags: list[str] = field(default_factory=list)
    session_id: str = ""
    observed_on: str = ""
    session_date: str = ""
    context_prefix: str = ""
    modified_override: str = ""


class Extractor:
    """Manages background memory extraction."""

    def __init__(self, mem: "MemoryManager") -> None:
        self._mem = mem
        self._lock = threading.Lock()
        self._last_cursor = 0
        self._in_progress = False
        self._ctx: "Contextualiser | None" = None

    def set_contextualiser(self, ctx: "Contextualiser | None") -> None:
        self._ctx = ctx

    def reset_cursor(self) -> None:
        with self._lock:
            self._last_cursor = 0

    async def maybe_extract(
        self,
        provider: Provider,
        model: str,
        project_path: str,
        messages: list[Message],
    ) -> None:
        with self._lock:
            if self._in_progress:
                return
            self._in_progress = True
            cursor = self._last_cursor

        try:
            if len(messages) - cursor < EXTRACT_MIN_MESSAGES:
                return

            slug = _project_slug(project_path)

            phys_hints: list[str] = []
            gp = self._mem.store.local_path(memory_global_prefix())
            if gp:
                phys_hints.append(gp)
            pp = self._mem.store.local_path(memory_project_prefix(slug))
            if pp:
                phys_hints.append(pp)

            if has_memory_writes(messages[cursor:], *phys_hints):
                with self._lock:
                    self._last_cursor = len(messages)
                return

            recent = messages[cursor:]
            if len(recent) > EXTRACT_MAX_RECENT:
                recent = recent[-EXTRACT_MAX_RECENT:]

            project_topics = self._mem.list_project_topics(project_path)
            global_topics = self._mem.list_global_topics()
            manifest = build_manifests(project_topics, global_topics)

            mem_dir_display = memory_project_prefix(slug)
            if phys_hints:
                mem_dir_display = phys_hints[-1]
            user_prompt = extract_user_prompt(recent, manifest, mem_dir_display)

            try:
                resp = await provider.complete(
                    CompleteRequest(
                        model=model,
                        messages=[
                            LLMMessage(role=Role.SYSTEM, content=EXTRACTION_PROMPT),
                            LLMMessage(role=Role.USER, content=user_prompt),
                        ],
                        max_tokens=EXTRACT_MAX_TOKENS,
                        temperature=EXTRACT_TEMPERATURE,
                    )
                )
            except Exception:
                return

            result = parse_extraction_result(resp.text)
            if not result:
                with self._lock:
                    self._last_cursor = len(messages)
                return

            if self._ctx is not None and self._ctx.enabled():
                summary = extract_session_summary(recent)
                for em in result:
                    prefix = self._ctx.build_prefix("", summary, em.content)
                    if prefix:
                        em.context_prefix = prefix

            try:
                apply_extractions(self._mem, slug, result)
            except Exception:
                pass

            with self._lock:
                self._last_cursor = len(messages)
        finally:
            with self._lock:
                self._in_progress = False


async def extract_from_messages(
    provider: Provider,
    model: str,
    mem: "MemoryManager",
    project_path: str,
    messages: list[Message],
) -> list[ExtractedMemory]:
    if len(messages) < 2:
        return []

    recent = messages
    if len(recent) > EXTRACT_MAX_RECENT:
        recent = recent[-EXTRACT_MAX_RECENT:]

    project_topics = mem.list_project_topics(project_path)
    global_topics = mem.list_global_topics()
    manifest = build_manifests(project_topics, global_topics)

    slug = _project_slug(project_path)
    mem_dir_display = memory_project_prefix(slug)
    user_prompt = extract_user_prompt(recent, manifest, mem_dir_display)

    resp = await provider.complete(
        CompleteRequest(
            model=model,
            messages=[
                LLMMessage(role=Role.SYSTEM, content=EXTRACTION_PROMPT),
                LLMMessage(role=Role.USER, content=user_prompt),
            ],
            max_tokens=EXTRACT_MAX_TOKENS,
            temperature=EXTRACT_TEMPERATURE,
        )
    )
    return parse_extraction_result(resp.text)


def extract_user_prompt(
    messages: list[Message], existing_manifest: str, mem_dir_display: str
) -> str:
    parts: list[str] = []
    if existing_manifest:
        parts.append("## Existing memory files\n\n" + existing_manifest)
        parts.append(
            "Check this list before writing \u2014 update an existing file rather than creating a duplicate.\n"
        )
    parts.append("## Recent conversation\n")
    lines: list[str] = []
    for m in messages:
        role = _role_value(m.role)
        content = m.content
        if len(content) > 2000:
            content = content[:2000] + "\n[...truncated]"
        if m.role == Role.TOOL:
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"[{role} ({m.name})]: {content}\n")
            continue
        lines.append(f"[{role}]: {content}\n")
    parts.append("\n".join(lines))
    parts.append(f"\nMemory directory: {mem_dir_display}\n")
    return "\n".join(parts)


def parse_extraction_result(content: str) -> list[ExtractedMemory]:
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
    memories = parsed.get("memories", [])
    if not isinstance(memories, list):
        return []
    out: list[ExtractedMemory] = []
    for item in memories:
        if not isinstance(item, dict):
            continue
        out.append(
            ExtractedMemory(
                action=str(item.get("action", "")),
                filename=str(item.get("filename", "")),
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                type=str(item.get("type", "")),
                content=str(item.get("content", "")),
                index_entry=str(item.get("index_entry", "")),
                scope=str(item.get("scope", "")),
                supersedes=str(item.get("supersedes", "")),
                tags=[str(t) for t in item.get("tags", [])],
            )
        )
    return out


def has_memory_writes(messages: list[Message], *mem_dirs: str) -> bool:
    for m in messages:
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            if tc.name not in ("write", "edit"):
                continue
            args = tc.arguments
            for d in mem_dirs:
                if d and d in args:
                    return True
            if "memory/" in args:
                return True
    return False


def extract_session_summary(messages: list[Message]) -> str:
    for m in messages:
        if m.role == Role.SYSTEM and m.content.strip():
            return _truncate_one_line(m.content, 240)
    for m in messages:
        if m.role == Role.USER and m.content.strip():
            return _truncate_one_line(m.content, 240)
    return ""


def _truncate_one_line(s: str, n: int) -> str:
    s = s.replace("\r\n", " ").replace("\n", " ")
    s = " ".join(s.split())
    if n > 0 and len(s) > n:
        s = s[:n] + "..."
    return s


def build_manifest(topics: list[TopicFile]) -> str:
    lines: list[str] = []
    for t in topics:
        line = "- "
        if t.scope == "global":
            line += f"[global:{t.type}] "
        elif t.type:
            line += f"[{t.type}] "
        if _has_tag(t.tags, "heuristic"):
            conf = t.confidence or "low"
            line += f"[heuristic:{conf}] "
        line += base_name(t.path)
        if t.description:
            line += ": " + t.description
        lines.append(line)
    return "\n".join(lines).strip()


def build_manifests(project_topics: list[TopicFile], global_topics: list[TopicFile]) -> str:
    parts: list[str] = []
    pm = build_manifest(project_topics)
    if pm:
        parts.append("## Project memory files\n\n" + pm)
    gm = build_manifest(global_topics)
    if gm:
        parts.append("## Global memory files\n\n" + gm)
    return "\n\n".join(parts)


def _has_tag(tags: list[str], tag: str) -> bool:
    target = tag.lower()
    return any(t.lower() == target for t in tags)


def sanitise_filename(name: str) -> str:
    for sep in ("/", "\\"):
        idx = name.rfind(sep)
        if idx >= 0:
            name = name[idx + 1 :]
    return name


def apply_contextual_prefix(prefix: str, body: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return body
    return f"Context: {prefix}\n\n{body}"


def build_topic_file_content(em: ExtractedMemory) -> bytes:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    modified = em.modified_override or now
    created = em.modified_override or now

    lines: list[str] = ["---"]
    if em.name:
        lines.append(f"name: {em.name}")
    if em.description:
        lines.append(f"description: {em.description}")
    if em.type:
        lines.append(f"type: {em.type}")
    if em.action == "create":
        lines.append(f"created: {created}")
    lines.append(f"modified: {modified}")
    lines.append("source: session")
    if em.supersedes:
        lines.append(f"supersedes: {em.supersedes}")
    if em.session_id:
        lines.append(f"session_id: {em.session_id}")
    if em.observed_on:
        lines.append(f"observed_on: {em.observed_on}")
    if em.session_date:
        lines.append(f"session_date: {em.session_date}")
    if em.tags:
        lines.append(f"tags: [{', '.join(em.tags)}]")
    lines.append("---")
    lines.append("")
    body = apply_contextual_prefix(em.context_prefix, em.content)
    lines.append(body)
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")


def apply_extractions(
    mem: "MemoryManager", project_slug: str, memories: list[ExtractedMemory]
) -> None:
    project_entries: list[str] = []
    global_entries: list[str] = []

    pending: list[tuple[str, bytes]] = []
    for em in memories:
        if not em.filename or not em.content:
            continue

        filename = sanitise_filename(em.filename)
        if not filename.endswith(".md"):
            filename += ".md"
        slug = filename[: -len(".md")]

        if em.scope == "global":
            path = memory_global_topic(slug)
        else:
            path = memory_project_topic(project_slug, slug)

        pending.append((path, build_topic_file_content(em)))

        if em.index_entry:
            if em.scope == "global":
                global_entries.append(em.index_entry)
            else:
                project_entries.append(em.index_entry)

    if not pending:
        return

    def _run(b) -> None:
        for path, content in pending:
            b.write(path, content)

        for em in memories:
            if not em.supersedes:
                continue
            old_file = sanitise_filename(em.supersedes)
            if not old_file.endswith(".md"):
                old_file += ".md"
            old_slug = old_file[: -len(".md")]
            new_file = sanitise_filename(em.filename)
            if not new_file.endswith(".md"):
                new_file += ".md"
            if em.scope == "global":
                old_path = memory_global_topic(old_slug)
            else:
                old_path = memory_project_topic(project_slug, old_slug)
            _stamp_superseded_by(b, old_path, new_file)

        if project_entries:
            _append_index_entries(b, memory_project_index(project_slug), project_entries)
        if global_entries:
            _append_index_entries(b, memory_global_index(), global_entries)

    mem.store.batch(_run)


def _stamp_superseded_by(b, old_path: str, new_file: str) -> None:
    try:
        raw = b.read(old_path)
    except NotFoundError:
        return
    content = raw.decode("utf-8")
    lines = content.split("\n")
    if len(lines) < 2 or lines[0].strip() != "---":
        return
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:
        return

    replaced = False
    for i in range(1, close_idx):
        if lines[i].strip().startswith("superseded_by:"):
            lines[i] = f"superseded_by: {new_file}"
            replaced = True
            break
    if not replaced:
        lines = lines[:close_idx] + [f"superseded_by: {new_file}"] + lines[close_idx:]

    b.write(old_path, "\n".join(lines).encode("utf-8"))


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


def _role_value(role) -> str:
    if isinstance(role, Role):
        return role.value
    return str(role)
