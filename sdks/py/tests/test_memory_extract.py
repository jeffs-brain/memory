# SPDX-License-Identifier: Apache-2.0
"""Extraction: prompt shape, parse_extraction_result, apply_extractions."""

from __future__ import annotations

from pathlib import Path

import pytest

from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.llm.types import Role
from jeffs_brain_memory.memory import (
    EXTRACTION_PROMPT,
    ExtractedMemory,
    Extractor,
    MemoryManager,
    MemStore,
    Message,
    ToolCall,
    apply_extractions,
    build_topic_file_content,
    extract_from_messages,
    has_memory_writes,
    memory_global_index,
    memory_global_topic,
    memory_project_index,
    memory_project_topic,
    parse_extraction_result,
    project_slug,
    sanitise_filename,
    set_slug_map_for_test,
)


@pytest.fixture
def iso(tmp_path: Path):
    restore = set_slug_map_for_test(str(tmp_path / "slug-map.yaml"))
    yield
    restore()


def test_parse_extraction_result_valid():
    raw = '{"memories": [{"action": "create", "filename": "test.md", "name": "Test", "description": "A test", "type": "project", "content": "hello", "index_entry": "- [Test](test.md)"}]}'
    result = parse_extraction_result(raw)
    assert len(result) == 1
    assert result[0].filename == "test.md"
    assert result[0].content == "hello"


def test_parse_extraction_result_empty():
    assert parse_extraction_result('{"memories": []}') == []


def test_parse_extraction_result_invalid():
    assert parse_extraction_result("not json") == []


def test_parse_extraction_result_wrapped():
    raw = '```json\n{"memories": [{"filename": "x.md", "content": "hi"}]}\n```'
    assert len(parse_extraction_result(raw)) == 1


def test_has_memory_writes_none():
    msgs = [
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="hi"),
    ]
    assert not has_memory_writes(msgs, "/some/memory")


def test_has_memory_writes_write_to_memory():
    msgs = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    name="write",
                    arguments='{"file_path": "/home/jeff/projects/test/memory/feedback.md"}',
                )
            ],
        )
    ]
    assert has_memory_writes(msgs, "/home/jeff/projects/test/memory", "/home/jeff/memory")


def test_has_memory_writes_elsewhere():
    msgs = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(name="write", arguments='{"file_path": "/tmp/other.md"}')
            ],
        )
    ]
    assert not has_memory_writes(msgs, "/home/jeff/memory")


def test_sanitise_filename_strips_traversal():
    assert sanitise_filename("../../etc/passwd") == "passwd"
    assert sanitise_filename("evil\\..\\file.md") == "file.md"
    assert sanitise_filename("clean.md") == "clean.md"


def test_apply_extractions_creates_files(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/example/project")

    memories = [
        ExtractedMemory(
            action="create",
            filename="arch.md",
            name="Arch",
            description="Arch notes",
            type="project",
            content="Body content.",
            index_entry="- [Arch](arch.md) -- Arch notes",
            scope="project",
        ),
        ExtractedMemory(
            action="create",
            filename="prefs.md",
            name="Prefs",
            description="User prefs",
            type="user",
            content="User likes typed code.",
            index_entry="- [Prefs](prefs.md) -- User prefs",
            scope="global",
        ),
    ]
    apply_extractions(mem, slug, memories)

    p = store.read(memory_project_topic(slug, "arch")).decode("utf-8")
    assert "Body content." in p
    assert "type: project" in p
    g = store.read(memory_global_topic("prefs")).decode("utf-8")
    assert "User likes typed code." in g
    idx = store.read(memory_project_index(slug)).decode("utf-8")
    assert "Arch" in idx
    gidx = store.read(memory_global_index()).decode("utf-8")
    assert "Prefs" in gidx


def test_apply_extractions_supersession(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/x")
    # Seed the old memory.
    store.write(
        memory_project_topic(slug, "old"),
        b"---\nname: Old\ntype: project\n---\n\nOld body.\n",
    )
    apply_extractions(
        mem,
        slug,
        [
            ExtractedMemory(
                action="create",
                filename="new.md",
                name="New",
                description="New version",
                type="project",
                content="New body",
                scope="project",
                supersedes="old.md",
            )
        ],
    )
    old = store.read(memory_project_topic(slug, "old")).decode("utf-8")
    assert "superseded_by: new.md" in old


def test_apply_extractions_skips_empty():
    store = MemStore()
    mem = MemoryManager(store)
    # nothing should raise; nothing written.
    apply_extractions(mem, "slug", [])
    assert not store.exists(memory_project_index("slug"))


def test_build_topic_file_content_session_date():
    em = ExtractedMemory(
        action="create",
        filename="fact.md",
        name="Test",
        description="desc",
        type="project",
        content="body",
        scope="project",
        session_id="sess-42",
        session_date="2024-03-25",
        observed_on="2024-03-25T09:00:00Z",
    )
    data = build_topic_file_content(em).decode("utf-8")
    assert "session_date: 2024-03-25" in data
    assert "session_id: sess-42" in data


def test_build_topic_file_content_prefix():
    em = ExtractedMemory(
        action="create",
        filename="f.md",
        name="T",
        type="project",
        content="User bought a blue car.",
        scope="project",
        context_prefix="On Monday the user discussed cars.",
    )
    data = build_topic_file_content(em).decode("utf-8")
    assert "Context: On Monday the user discussed cars.\n\nUser bought a blue car." in data


@pytest.mark.asyncio
async def test_extractor_maybe_extract_writes(iso):
    store = MemStore()
    mem = MemoryManager(store)
    ex = Extractor(mem)

    reply = '{"memories": [{"action": "create", "filename": "facts.md", "name": "Facts", "description": "facts", "type": "project", "scope": "project", "content": "A fact.", "index_entry": "- facts"}]}'
    provider = FakeProvider([reply])

    messages = [
        Message(role=Role.USER, content="tell me"),
        Message(role=Role.ASSISTANT, content="here is info"),
        Message(role=Role.USER, content="and more"),
        Message(role=Role.ASSISTANT, content="more info"),
        Message(role=Role.USER, content="thanks"),
        Message(role=Role.ASSISTANT, content="welcome"),
    ]
    await ex.maybe_extract(provider, "test-model", "/p", messages)
    slug = project_slug("/p")
    data = store.read(memory_project_topic(slug, "facts")).decode("utf-8")
    assert "A fact." in data


@pytest.mark.asyncio
async def test_extract_from_messages_returns_parsed(iso):
    store = MemStore()
    mem = MemoryManager(store)
    reply = '{"memories": [{"filename": "a.md", "content": "x", "scope": "project"}]}'
    provider = FakeProvider([reply])
    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b"),
        ],
    )
    assert len(out) == 1
    assert out[0].content == "x"


def test_extraction_prompt_includes_scope_rules():
    # Basic sanity that the verbatim prompt is preserved.
    assert "Memory types:" in EXTRACTION_PROMPT
    assert "You MUST respond with ONLY a JSON object." in EXTRACTION_PROMPT


@pytest.mark.asyncio
async def test_extract_from_messages_adds_heuristic_user_fact_for_quantified_update(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/07/15 (Sat) 22:42.",
            ),
            Message(
                role=Role.USER,
                content="I've been reading about the Amazon rainforest and its indigenous communities in National Geographic, and I just finished my fifth issue.",
            ),
            Message(role=Role.ASSISTANT, content="That sounds fascinating."),
        ],
        session_id="session-a",
        session_date="2023/07/15 (Sat) 22:42",
    )

    heuristic = next((memory for memory in out if "fifth issue" in memory.content), None)
    assert heuristic is not None
    assert heuristic.scope == "global"
    assert heuristic.type == "user"
    assert "user-fact-2023-07-15-session-a" in heuristic.filename


@pytest.mark.asyncio
async def test_extract_from_messages_adds_heuristic_milestone_fact(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2022/11/17 (Thu) 15:34.",
            ),
            Message(
                role=Role.USER,
                content="I'm planning to start learning about deep learning. By the way, I just completed my undergraduate degree in computer science.",
            ),
            Message(role=Role.ASSISTANT, content="That foundation will help."),
        ],
        session_id="session-b",
        session_date="2022/11/17 (Thu) 15:34",
    )

    heuristic = next(
        (memory for memory in out if "completed my undergraduate degree" in memory.content),
        None,
    )
    assert heuristic is not None
    assert "milestone" in heuristic.filename


@pytest.mark.asyncio
async def test_extract_from_messages_adds_month_name_date_fact(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/07/07 (Fri) 04:44.",
            ),
            Message(
                role=Role.USER,
                content="My close friend Rachel got engaged on May 15th, and we're already planning her bachelorette party.",
            ),
            Message(role=Role.ASSISTANT, content="That sounds exciting."),
        ],
        session_id="session-rachel",
        session_date="2023/07/07 (Fri) 04:44",
    )

    heuristic = next((memory for memory in out if "May 15th" in memory.content), None)
    assert heuristic is not None
    assert heuristic.content.startswith("[Date: 2023-05-15")
    assert heuristic.observed_on == "2023-05-15T00:00:00Z"


@pytest.mark.asyncio
async def test_extract_from_messages_adds_group_join_milestone_with_resolved_date(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/05/25 (Thu) 01:50.",
            ),
            Message(
                role=Role.USER,
                content='I just joined a new book club group called "Page Turners" last week, where we discuss our favourite novels and share recommendations.',
            ),
            Message(role=Role.ASSISTANT, content="That sounds like a great group."),
        ],
        session_id="session-page-turners",
        session_date="2023/05/25 (Thu) 01:50",
    )

    heuristic = next((memory for memory in out if "Page Turners" in memory.content), None)
    assert heuristic is not None
    assert heuristic.content.startswith("[Date: 2023-05-18")
    assert heuristic.observed_on == "2023-05-18T00:00:00Z"


@pytest.mark.asyncio
async def test_extract_from_messages_adds_relative_time_number_word_fact(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/05/26 (Fri) 18:20.",
            ),
            Message(
                role=Role.USER,
                content="I'm actually planning to buy a new phone charger, since I lost my old one at the gym about two weeks ago.",
            ),
            Message(role=Role.ASSISTANT, content="Buying a new charger makes sense."),
        ],
        session_id="session-charger",
        session_date="2023/05/26 (Fri) 18:20",
    )

    heuristic = next(
        (
            memory
            for memory in out
            if "lost my old one at the gym about two weeks ago" in memory.content
        ),
        None,
    )
    assert heuristic is not None
    assert heuristic.observed_on == "2023-05-12T00:00:00Z"


@pytest.mark.asyncio
async def test_extract_from_messages_adds_airbnb_booking_lead_time_fact(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/05/27 (Sat) 03:04.",
            ),
            Message(
                role=Role.USER,
                content="I've had a great experience with Airbnb in the past, like when I stayed in Haight-Ashbury for my best friend's wedding and had to book three months in advance.",
            ),
            Message(role=Role.ASSISTANT, content="That sounds like a memorable trip."),
        ],
        session_id="session-airbnb",
        session_date="2023/05/27 (Sat) 03:04",
    )

    heuristic = next(
        (
            memory
            for memory in out
            if "Airbnb" in memory.content
            and "book three months in advance" in memory.content
        ),
        None,
    )
    assert heuristic is not None
    assert "session-airbnb" in heuristic.filename


@pytest.mark.asyncio
async def test_extract_from_messages_adds_pending_task_fact_without_event_drift(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/05/27 (Sat) 09:00.",
            ),
            Message(
                role=Role.USER,
                content="I still need to schedule a dentist appointment and pick up my prescription.",
            ),
            Message(role=Role.ASSISTANT, content="I can help you remember both tasks."),
        ],
        session_id="session-pending",
        session_date="2023/05/27 (Sat) 09:00",
    )

    assert any(
        "The user still needs to schedule a dentist appointment." in memory.content
        for memory in out
    )
    assert not any(
        "The user has a dentist appointment" in memory.content for memory in out
    )


@pytest.mark.asyncio
async def test_extract_from_messages_adds_heuristic_preference_fact(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/07/14 (Fri) 20:05.",
            ),
            Message(
                role=Role.USER,
                content="Can you recommend a family-friendly, light-hearted film for tonight, ideally under 100 minutes and without gore?",
            ),
            Message(
                role=Role.ASSISTANT,
                content="I will keep the suggestions gentle and short.",
            ),
        ],
        session_id="session-c",
        session_date="2023/07/14 (Fri) 20:05",
    )

    heuristic = next(
        (
            memory
            for memory in out
            if "user-preference-2023-07-14-session-c" in memory.filename
        ),
        None,
    )
    assert heuristic is not None
    assert "family-friendly" in heuristic.content
    assert "without gore" in heuristic.content


@pytest.mark.asyncio
async def test_extract_from_messages_adds_religious_service_event_fact(iso):
    store = MemStore()
    mem = MemoryManager(store)
    provider = FakeProvider(['{"memories": []}'])

    out = await extract_from_messages(
        provider,
        "m",
        mem,
        "/p",
        [
            Message(
                role=Role.SYSTEM,
                content="This conversation took place on 2023/04/06 (Thu) 05:36.",
            ),
            Message(
                role=Role.USER,
                content="I'm glad I got to attend the Maundy Thursday service at the Episcopal Church, it was a beautiful and moving experience.",
            ),
            Message(role=Role.ASSISTANT, content="That sounds meaningful."),
        ],
        session_id="session-service",
        session_date="2023/04/06 (Thu) 05:36",
    )

    heuristic = next(
        (
            memory
            for memory in out
            if "Maundy Thursday service at the Episcopal Church" in memory.content
        ),
        None,
    )
    assert heuristic is not None
    assert heuristic.observed_on == "2023-04-06T05:36:00Z"
