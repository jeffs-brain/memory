# SPDX-License-Identifier: Apache-2.0
"""Consolidation: index rebuild, dedup, staleness, reinforcement.

``DEDUPLICATION_SYSTEM_PROMPT`` is ported verbatim from
``sdks/go/memory/consolidate.go``.
"""

from __future__ import annotations

import json
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role
from ._memstore import ListOpts, NotFoundError
from .heuristic import (
    confidence_from_observations,
    count_sections,
    has_tag,
    jaccard_similarity,
    significant_words,
)
from .paths import (
    base_name,
    memory_global_prefix,
    memory_projects_prefix,
)
from .store import parse_frontmatter
from .types import Frontmatter, TopicFile

if TYPE_CHECKING:
    from .manager import MemoryManager


STALENESS_THRESHOLD_DAYS = 90
DEDUPLICATION_JACCARD_CUTOFF = 0.3
DEDUPLICATION_MAX_TOKENS = 512
DEDUPLICATION_TEMPERATURE = 0.0


DEDUPLICATION_SYSTEM_PROMPT = """You are analysing two memory files for overlap. Determine whether they cover the same topic or are distinct.

Respond with ONLY a JSON object:
{
  "verdict": "keep_first" | "keep_second" | "merge" | "distinct",
  "reason": "brief explanation"
}

- "distinct": files cover different topics, keep both
- "keep_first": files overlap, the first is more complete \u2014 delete the second
- "keep_second": files overlap, the second is more complete \u2014 delete the first
- "merge": files have complementary information \u2014 combine into one

Respond with ONLY valid JSON, no other text."""


@dataclass(slots=True)
class ConsolidationReport:
    episodes_reviewed: int = 0
    memories_merged: int = 0
    heuristics_updated: int = 0
    indexes_rebuilt: int = 0
    stale_memories_flagged: int = 0
    insights_promoted: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class Consolidator:
    """Maintains memory health: rebuild indexes, dedup, flag stale, reinforce."""

    def __init__(
        self,
        provider: Provider | None,
        model: str,
        mem: "MemoryManager",
    ) -> None:
        self._provider = provider
        self._model = model
        self._mem = mem
        self._lock = threading.Lock()
        self._in_progress = False

    async def run_full(self) -> ConsolidationReport:
        with self._lock:
            if self._in_progress:
                raise RuntimeError("consolidation already in progress")
            self._in_progress = True
        start = _time.monotonic()
        report = ConsolidationReport()
        try:
            for prefix in self._scope_prefixes():
                await self._run_scope_full(prefix, report)
        finally:
            report.duration_seconds = _time.monotonic() - start
            with self._lock:
                self._in_progress = False
        return report

    async def run_quick(self) -> ConsolidationReport:
        with self._lock:
            if self._in_progress:
                raise RuntimeError("consolidation already in progress")
            self._in_progress = True
        start = _time.monotonic()
        report = ConsolidationReport()
        try:
            for prefix in self._scope_prefixes():
                self._run_scope_quick(prefix, report)
        finally:
            report.duration_seconds = _time.monotonic() - start
            with self._lock:
                self._in_progress = False
        return report

    def _scope_prefixes(self) -> list[str]:
        prefixes: list[str] = [memory_global_prefix()]
        try:
            entries = self._mem.store.list(
                memory_projects_prefix(), ListOpts(include_generated=True)
            )
        except NotFoundError:
            return prefixes
        for e in entries:
            if e.is_dir:
                prefixes.append(e.path)
        return prefixes

    def _run_scope_quick(self, prefix: str, report: ConsolidationReport) -> None:
        self._detect_staleness_in(prefix, report)

        def _run(b) -> None:
            err = self._rebuild_index_in_batch(b, prefix)
            if err:
                report.errors.append(f"rebuilding index {prefix}: {err}")
            else:
                report.indexes_rebuilt += 1
            updated, errs = self._reinforce_heuristics_in_batch(b, prefix)
            report.heuristics_updated += updated
            report.errors.extend(errs)

        try:
            self._mem.store.batch(_run)
        except Exception as e:
            report.errors.append(f"consolidate batch {prefix}: {e}")

    async def _run_scope_full(self, prefix: str, report: ConsolidationReport) -> None:
        self._detect_staleness_in(prefix, report)

        topics = self._list_topics_for_dedup(prefix)
        merges: list[tuple[str, str, str]] = []
        if self._provider is not None and len(topics) >= 2:
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    words_a = significant_words(base_name(topics[i].path))
                    words_b = significant_words(base_name(topics[j].path))
                    if jaccard_similarity(words_a, words_b) < DEDUPLICATION_JACCARD_CUTOFF:
                        continue
                    try:
                        content_a = self._mem.store.read(topics[i].path).decode("utf-8")
                        content_b = self._mem.store.read(topics[j].path).decode("utf-8")
                    except NotFoundError:
                        continue
                    prompt = (
                        f"## File 1: {base_name(topics[i].path)}\n\n{content_a}\n\n"
                        f"---\n\n## File 2: {base_name(topics[j].path)}\n\n{content_b}"
                    )
                    try:
                        resp = await self._provider.complete(
                            CompleteRequest(
                                model=self._model,
                                messages=[
                                    LLMMessage(
                                        role=Role.SYSTEM,
                                        content=DEDUPLICATION_SYSTEM_PROMPT,
                                    ),
                                    LLMMessage(role=Role.USER, content=prompt),
                                ],
                                max_tokens=DEDUPLICATION_MAX_TOKENS,
                                temperature=DEDUPLICATION_TEMPERATURE,
                            )
                        )
                    except Exception as e:
                        report.errors.append(f"dedup LLM call: {e}")
                        continue
                    verdict = parse_deduplication_result(resp.text)
                    if verdict == "keep_first":
                        merges.append(("delete", topics[j].path, ""))
                    elif verdict == "keep_second":
                        merges.append(("delete", topics[i].path, ""))
                    elif verdict == "merge":
                        merges.append(("merge", topics[i].path, topics[j].path))
        elif self._provider is None:
            report.errors.append("deduplication skipped: no LLM provider")

        def _run(b) -> None:
            err = self._rebuild_index_in_batch(b, prefix)
            if err:
                report.errors.append(f"rebuilding index {prefix}: {err}")
            else:
                report.indexes_rebuilt += 1

            for op in merges:
                action, a_path, b_path = op
                if action == "delete":
                    try:
                        b.delete(a_path)
                        report.memories_merged += 1
                    except Exception as e:
                        report.errors.append(f"removing {a_path}: {e}")
                elif action == "merge":
                    try:
                        self._merge_topics_in_batch(b, a_path, b_path)
                        report.memories_merged += 1
                    except Exception as e:
                        report.errors.append(
                            f"merging {base_name(a_path)} + {base_name(b_path)}: {e}"
                        )

            if merges:
                err = self._rebuild_index_in_batch(b, prefix)
                if err:
                    report.errors.append(f"rebuilding index post-merge {prefix}: {err}")

            updated, errs = self._reinforce_heuristics_in_batch(b, prefix)
            report.heuristics_updated += updated
            report.errors.extend(errs)

        try:
            self._mem.store.batch(_run)
        except Exception as e:
            report.errors.append(f"consolidate batch {prefix}: {e}")

    # ---- helpers ----

    def _list_topics_for_dedup(self, prefix: str) -> list[TopicFile]:
        try:
            entries = self._mem.store.list(prefix, ListOpts(include_generated=True))
        except NotFoundError:
            return []
        topics: list[TopicFile] = []
        for entry in entries:
            if entry.is_dir:
                continue
            name = base_name(entry.path)
            if not name.endswith(".md") or name.lower() == "memory.md":
                continue
            try:
                data = self._mem.store.read(entry.path)
            except NotFoundError:
                continue
            fm, _ = parse_frontmatter(data.decode("utf-8"))
            topics.append(
                TopicFile(
                    name=fm.name or name[: -len(".md")],
                    description=fm.description,
                    type=fm.type,
                    path=entry.path,
                    created=fm.created,
                    modified=fm.modified,
                    tags=list(fm.tags),
                    confidence=fm.confidence,
                    source=fm.source,
                )
            )
        return topics

    def _rebuild_index_in_batch(self, b, prefix: str) -> str:
        try:
            entries = b.list(prefix, ListOpts(include_generated=True))
        except NotFoundError:
            return ""
        except Exception as e:
            return str(e)

        lines: list[str] = []
        for entry in entries:
            if entry.is_dir:
                continue
            name = base_name(entry.path)
            if not name.endswith(".md") or name.lower() == "memory.md":
                continue
            try:
                data = b.read(entry.path)
            except NotFoundError:
                continue
            fm, _ = parse_frontmatter(data.decode("utf-8"))
            display_name = fm.name or name[: -len(".md")]
            desc = fm.description or fm.type or "no description"
            lines.append(f"- [{display_name}]({name}) \u2014 {desc}")

        index_path = f"{prefix}/MEMORY.md"
        content = "\n".join(lines)
        if content:
            content += "\n"
        try:
            b.write(index_path, content.encode("utf-8"))
        except Exception as e:
            return str(e)
        return ""

    def _detect_staleness_in(self, prefix: str, report: ConsolidationReport) -> None:
        threshold = datetime.now(timezone.utc) - timedelta(days=STALENESS_THRESHOLD_DAYS)
        try:
            entries = self._mem.store.list(prefix, ListOpts(include_generated=True))
        except NotFoundError:
            return
        except Exception as e:
            report.errors.append(f"reading prefix for staleness: {prefix}: {e}")
            return
        for entry in entries:
            if entry.is_dir:
                continue
            name = base_name(entry.path)
            if not name.endswith(".md") or name.lower() == "memory.md":
                continue
            mtime = self._modified_time(entry.path)
            if mtime is not None and mtime < threshold:
                report.stale_memories_flagged += 1

    def _modified_time(self, path: str) -> datetime | None:
        try:
            data = self._mem.store.read(path)
        except NotFoundError:
            return None
        fm, _ = parse_frontmatter(data.decode("utf-8"))
        if fm.modified:
            try:
                return _parse_rfc3339(fm.modified)
            except ValueError:
                pass
        try:
            info = self._mem.store.stat(path)
        except NotFoundError:
            return None
        return info.mod_time

    def _merge_topics_in_batch(self, b, path_a: str, path_b: str) -> None:
        mod_a = self._modified_time(path_a) or datetime.fromtimestamp(0, tz=timezone.utc)
        mod_b = self._modified_time(path_b) or datetime.fromtimestamp(0, tz=timezone.utc)
        keeper, donor = path_a, path_b
        if mod_b > mod_a:
            keeper, donor = path_b, path_a

        keeper_data = b.read(keeper).decode("utf-8")
        donor_data = b.read(donor).decode("utf-8")
        _, donor_body = parse_frontmatter(donor_data)

        combined = (
            keeper_data.strip()
            + "\n\n---\n\n"
            + f"*Merged from {base_name(donor)}:*\n\n"
            + donor_body.strip()
            + "\n"
        )
        b.write(keeper, combined.encode("utf-8"))
        b.delete(donor)

    def _reinforce_heuristics_in_batch(self, b, prefix: str) -> tuple[int, list[str]]:
        try:
            entries = b.list(prefix, ListOpts(include_generated=True))
        except NotFoundError:
            return 0, []
        except Exception as e:
            return 0, [f"reading prefix for heuristics: {prefix}: {e}"]

        updated = 0
        errs: list[str] = []
        for entry in entries:
            if entry.is_dir:
                continue
            name = base_name(entry.path)
            if not name.endswith(".md") or name.lower() == "memory.md":
                continue
            try:
                data = b.read(entry.path)
            except NotFoundError:
                continue
            fm, body = parse_frontmatter(data.decode("utf-8"))
            if not has_tag(fm.tags, "heuristic"):
                continue
            count = count_sections(body)
            new_confidence = confidence_from_observations(count)
            if new_confidence == fm.confidence:
                continue
            rebuilt = rebuild_with_updated_confidence(fm, body, new_confidence)
            try:
                b.write(entry.path, rebuilt.encode("utf-8"))
            except Exception as e:
                errs.append(f"updating heuristic {name}: {e}")
                continue
            updated += 1
        return updated, errs


def parse_deduplication_result(content: str) -> str:
    content = content.strip()
    start = content.find("{")
    if start >= 0:
        end = content.rfind("}")
        if end > start:
            content = content[start : end + 1]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return "distinct"
    return str(parsed.get("verdict", "distinct"))


def rebuild_with_updated_confidence(
    fm: Frontmatter, body: str, new_confidence: str
) -> str:
    lines = ["---"]
    if fm.name:
        lines.append(f'name: "{fm.name}"')
    if fm.description:
        lines.append(f'description: "{fm.description}"')
    if fm.type:
        lines.append(f"type: {fm.type}")
    if fm.created:
        lines.append(f"created: {fm.created}")
    if fm.modified:
        lines.append(f"modified: {fm.modified}")
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
    return "\n".join(lines)


def _parse_rfc3339(s: str) -> datetime:
    # Python's fromisoformat accepts "...+00:00" but not "...Z" before 3.11.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)
