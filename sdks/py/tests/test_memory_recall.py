# SPDX-License-Identifier: Apache-2.0
"""Recall formatting, weights, manifest."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.memory import (
    MAX_RECALL_TOPICS,
    MemoryManager,
    MemStore,
    RecallWeights,
    SurfacedMemory,
    TopicFile,
    build_manifest,
    date_header,
    format_recalled_memories_with_context,
    memory_global_topic,
    memory_project_topic,
    parse_selected_memories,
    read_capped_topic,
    recall,
    relative_time_string,
    set_slug_map_for_test,
    sort_memories_chronologically,
    topic_age,
)


FIXED_NOW = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)


def days_ago(n: int) -> str:
    return (FIXED_NOW - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def iso(tmp_path: Path):
    restore = set_slug_map_for_test(str(tmp_path / "slug-map.yaml"))
    yield
    restore()


def test_build_manifest_empty():
    assert build_manifest([]) == ""


def test_build_manifest_entries():
    topics = [
        TopicFile(
            name="Auth",
            description="Auth migration notes",
            type="project",
            path="memory/project/x/project_auth.md",
        ),
        TopicFile(
            name="Style",
            description="",
            type="feedback",
            path="memory/project/x/feedback_style.md",
        ),
    ]
    out = build_manifest(topics)
    assert "[project] project_auth.md: Auth migration notes" in out
    assert "[feedback] feedback_style.md" in out
    assert "feedback_style.md:" not in out


def test_parse_selected_valid():
    sel = parse_selected_memories('{"selected": ["a.md", "b.md"]}')
    assert sel == ["a.md", "b.md"]


def test_parse_selected_caps():
    sel = parse_selected_memories(
        '{"selected": ["a.md","b.md","c.md","d.md","e.md","f.md","g.md"]}'
    )
    assert len(sel) == MAX_RECALL_TOPICS


def test_parse_selected_invalid():
    assert parse_selected_memories("not json") == []


def test_parse_selected_empty_array():
    assert parse_selected_memories('{"selected": []}') == []


def test_parse_selected_wrapped():
    assert parse_selected_memories('```json\n{"selected": ["a.md"]}\n```') == ["a.md"]


def test_read_capped_small(iso):
    store = MemStore()
    mem = MemoryManager(store)
    p = memory_global_topic("t")
    store.write(p, b"short content")
    assert read_capped_topic(mem, p) == "short content"


def test_read_capped_truncates_long(iso):
    store = MemStore()
    mem = MemoryManager(store)
    p = memory_global_topic("big")
    big = "x" * 5000
    store.write(p, big.encode())
    out = read_capped_topic(mem, p)
    assert "[...truncated]" in out


def test_topic_age_today():
    assert topic_age(FIXED_NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), FIXED_NOW) == "today"


def test_topic_age_empty():
    assert topic_age("", FIXED_NOW) == "unknown time ago"


def test_topic_age_yesterday():
    assert topic_age(days_ago(1), FIXED_NOW) == "yesterday"


@pytest.mark.parametrize(
    "days,want",
    [
        (0, "today"),
        (1, "yesterday"),
        (3, "3 days ago"),
        (7, "1 week ago"),
        (21, "3 weeks ago"),
        (30, "1 month ago"),
        (180, "6 months ago"),
        (400, "1 year ago"),
    ],
)
def test_relative_time_buckets(days, want):
    then = FIXED_NOW - timedelta(days=days)
    assert relative_time_string(then, FIXED_NOW) == want


def test_relative_time_future_empty():
    then = FIXED_NOW + timedelta(days=2)
    assert relative_time_string(then, FIXED_NOW) == ""


def test_sort_memories_empty():
    assert sort_memories_chronologically([]) == []


def test_sort_memories_oldest_first():
    recent = SurfacedMemory(
        path="memory/global/recent.md",
        topic=TopicFile(modified=days_ago(1), scope="global"),
    )
    ancient = SurfacedMemory(
        path="memory/global/ancient.md",
        topic=TopicFile(modified=days_ago(400), scope="global"),
    )
    mid = SurfacedMemory(
        path="memory/global/mid.md",
        topic=TopicFile(modified=days_ago(30), scope="global"),
    )
    out = sort_memories_chronologically([recent, ancient, mid])
    assert out[0].path.endswith("ancient.md")
    assert out[-1].path.endswith("recent.md")


def test_sort_memories_undated_last():
    dated = SurfacedMemory(
        path="memory/global/dated.md",
        topic=TopicFile(modified=days_ago(1)),
    )
    undated = SurfacedMemory(
        path="memory/global/undated.md",
        topic=TopicFile(),
    )
    out = sort_memories_chronologically([undated, dated])
    assert out[0].path.endswith("dated.md")
    assert out[1].path.endswith("undated.md")


def test_format_recalled_empty():
    assert format_recalled_memories_with_context([], FIXED_NOW) == ""


def test_format_recalled_injects_header():
    mems = [
        SurfacedMemory(
            path="memory/global/dated.md",
            content="dated body",
            topic=TopicFile(name="Dated", scope="global", modified=days_ago(21)),
        )
    ]
    out = format_recalled_memories_with_context(mems, FIXED_NOW)
    assert "3 weeks ago" in out
    assert "dated body" in out
    assert "Global memory (saved" in out


def test_format_recalled_no_timestamp():
    mems = [
        SurfacedMemory(
            path="memory/global/x.md",
            content="content",
            topic=TopicFile(name="Nots", scope="project"),
        )
    ]
    out = format_recalled_memories_with_context(mems, FIXED_NOW)
    assert "=== " not in out
    assert "Memory (saved unknown time ago)" in out


def test_format_recalled_heuristic_label():
    mems = [
        SurfacedMemory(
            path="memory/global/h.md",
            content="body",
            topic=TopicFile(
                tags=["heuristic"], confidence="high", scope="global", modified=days_ago(1)
            ),
        )
    ]
    out = format_recalled_memories_with_context(mems, FIXED_NOW)
    assert "Learned heuristic (high confidence)" in out


def test_format_recalled_linked_label():
    mems = [
        SurfacedMemory(
            path="memory/project/p.md",
            content="body",
            topic=TopicFile(scope="project", modified=days_ago(1)),
            linked_from="auth",
        )
    ]
    out = format_recalled_memories_with_context(mems, FIXED_NOW)
    assert "Linked memory (via [[auth]])" in out


def test_date_header_fields():
    h = date_header(days_ago(21), FIXED_NOW)
    assert h.startswith("=== ")
    assert "3 weeks ago" in h


@pytest.mark.asyncio
async def test_recall_selects_and_returns(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = __import__(
        "jeffs_brain_memory.memory", fromlist=["project_slug"]
    ).project_slug("/p")
    store.write(
        memory_project_topic(slug, "auth"),
        b"---\nname: Auth\ndescription: Auth notes\ntype: project\n---\n\nbody.",
    )
    store.write(
        memory_global_topic("style"),
        b"---\nname: Style\ndescription: Style notes\ntype: user\n---\n\nstyle.",
    )
    provider = FakeProvider(['{"selected": ["auth.md"]}'])
    out = await recall(
        mem,
        provider,
        "m",
        "/p",
        "auth question",
        None,
        RecallWeights(project_weight=1.0),
    )
    assert len(out) == 1
    assert out[0].path.endswith("auth.md")
