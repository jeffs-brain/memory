# SPDX-License-Identifier: Apache-2.0
"""Episode recorder: session keying, in-memory store."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.llm.types import Role
from jeffs_brain_memory.memory import (
    Episode,
    EpisodeRecorder,
    InMemoryEpisodeStore,
    Message,
    ToolCall,
    should_record_episode,
)


def test_should_record_requires_min_messages():
    msgs = [Message(role=Role.USER, content="x")] * 3
    assert not should_record_episode(msgs)


def test_should_record_requires_write_edit():
    msgs = [
        Message(role=Role.USER, content="x"),
        Message(role=Role.ASSISTANT, content="y"),
    ] * 4
    assert not should_record_episode(msgs)


def test_should_record_detects_write_tool():
    msgs = [
        Message(role=Role.USER, content="x"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(name="write", arguments="")],
        ),
    ] + [Message(role=Role.USER, content="y")] * 10
    assert should_record_episode(msgs)


@pytest.mark.asyncio
async def test_maybe_record_stores_significant_episode():
    store = InMemoryEpisodeStore()
    reply = '{"significant": true, "summary": "did it", "outcome": "success", "heuristics": [], "tags": []}'
    provider = FakeProvider([reply])
    recorder = EpisodeRecorder()
    msgs = [
        Message(role=Role.USER, content="do"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(name="edit", arguments="")],
        ),
        Message(role=Role.USER, content="more"),
        Message(role=Role.ASSISTANT, content="ok"),
        Message(role=Role.USER, content="yet"),
        Message(role=Role.ASSISTANT, content="done"),
        Message(role=Role.USER, content="bye"),
        Message(role=Role.ASSISTANT, content="bye"),
    ]
    await recorder.maybe_record(provider, "m", store, "/p", "sess-1", msgs)
    assert len(store.episodes) == 1
    assert store.episodes[0].session_id == "sess-1"
    assert store.episodes[0].outcome == "success"


@pytest.mark.asyncio
async def test_maybe_record_insignificant_does_nothing():
    store = InMemoryEpisodeStore()
    reply = '{"significant": false}'
    provider = FakeProvider([reply])
    recorder = EpisodeRecorder()
    msgs = [
        Message(role=Role.USER, content="do"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(name="edit", arguments="")],
        ),
    ] + [Message(role=Role.USER, content=".")] * 10
    await recorder.maybe_record(provider, "m", store, "/p", "sess-2", msgs)
    assert store.episodes == []


def test_episode_dataclass_defaults():
    ep = Episode()
    assert ep.session_id == ""
    assert ep.heuristics == []
