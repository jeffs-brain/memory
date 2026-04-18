# SPDX-License-Identifier: Apache-2.0
"""Reflection: should_reflect, analyse_session, parsing."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.llm.types import Role
from jeffs_brain_memory.memory import (
    Heuristic,
    MemoryManager,
    MemStore,
    Message,
    REFLECTION_SYSTEM_PROMPT,
    Reflector,
    ToolCall,
    analyse_session,
    count_assistant_tool_iterations,
    count_write_tool_calls,
    find_user_corrections,
    infer_outcome,
    parse_reflection_result,
    should_reflect,
)


def test_reflection_prompt_verbatim_marker():
    assert "You are a reflection agent." in REFLECTION_SYSTEM_PROMPT
    assert "Anti-pattern signals" in REFLECTION_SYSTEM_PROMPT


def test_should_reflect_too_few_messages():
    msgs = [Message(role=Role.USER, content="hi")] * 5
    assert not should_reflect(msgs, 0)


def test_should_reflect_no_write_tools():
    msgs = [
        Message(role=Role.USER, content="Read"),
        Message(
            role=Role.ASSISTANT,
            content="ok",
            tool_calls=[ToolCall(name="read", arguments='{"file_path": "/a"}')],
        ),
        Message(role=Role.TOOL, content="x", name="read"),
        Message(role=Role.ASSISTANT, content="ok"),
        Message(role=Role.USER, content="Thanks"),
        Message(role=Role.ASSISTANT, content="yw"),
        Message(role=Role.USER, content="What?"),
        Message(role=Role.ASSISTANT, content="hmm"),
    ]
    assert not should_reflect(msgs, 0)


def test_should_reflect_with_write_tools():
    msgs = [
        Message(role=Role.USER, content="fix"),
        Message(role=Role.ASSISTANT, content="k"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(name="edit", arguments='{"file_path": "/a"}')],
        ),
        Message(role=Role.TOOL, content="ok", name="edit"),
        Message(role=Role.ASSISTANT, content="done"),
        Message(role=Role.USER, content="great"),
        Message(role=Role.ASSISTANT, content="np"),
        Message(role=Role.USER, content="bye"),
    ]
    assert should_reflect(msgs, 0)


def test_find_user_corrections_detected():
    msgs = [
        Message(role=Role.USER, content="no, that's wrong"),
        Message(role=Role.ASSISTANT, content="sorry"),
    ]
    corrections = find_user_corrections(msgs)
    assert len(corrections) == 1


def test_count_write_tool_calls():
    msgs = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(name="write", arguments=""),
                ToolCall(name="edit", arguments=""),
                ToolCall(name="read", arguments=""),
            ],
        )
    ]
    assert count_write_tool_calls(msgs) == 2


def test_count_assistant_tool_iterations():
    msgs = [
        Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(name="x")]),
        Message(role=Role.ASSISTANT, content="plain"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(name="y")]),
    ]
    assert count_assistant_tool_iterations(msgs) == 2


def test_analyse_session_fields():
    msgs = [
        Message(role=Role.USER, content="Fix the bug"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(name="edit", arguments='{"file_path": "/x.go"}')],
        ),
        Message(role=Role.TOOL, content="ok", name="edit"),
        Message(role=Role.USER, content="thanks"),
    ]
    a = analyse_session(msgs)
    assert a.task_description == "Fix the bug"
    assert a.write_tool_calls == 1
    assert a.iteration_count == 1
    assert a.outcome == "success"


def test_infer_outcome_success():
    msgs = [
        Message(role=Role.ASSISTANT, content=""),
        Message(role=Role.USER, content="perfect, thanks"),
    ]
    assert infer_outcome(msgs) == "success"


def test_infer_outcome_failure():
    msgs = [
        Message(role=Role.ASSISTANT, content=""),
        Message(role=Role.USER, content="that's wrong"),
    ]
    assert infer_outcome(msgs) == "failure"


def test_infer_outcome_partial_default():
    msgs = [
        Message(role=Role.ASSISTANT, content=""),
        Message(role=Role.USER, content="hmm"),
    ]
    assert infer_outcome(msgs) == "partial"


def test_parse_reflection_result_basic():
    raw = """{
      "outcome": "success",
      "summary": "did a thing",
      "retry_feedback": "",
      "heuristics": [{"rule": "R", "confidence": "low", "category": "testing", "anti_pattern": false}],
      "should_record_episode": true
    }"""
    r = parse_reflection_result(raw)
    assert r.outcome == "success"
    assert len(r.heuristics) == 1
    assert r.heuristics[0].rule == "R"


def test_parse_reflection_result_invalid():
    r = parse_reflection_result("garbage")
    assert r.outcome == ""


@pytest.mark.asyncio
async def test_reflector_force_reflect_applies_heuristics():
    store = MemStore()
    mem = MemoryManager(store)
    reflector = Reflector(mem)

    reply = """{
      "outcome": "success",
      "summary": "s",
      "retry_feedback": "",
      "heuristics": [{"rule": "Always test", "confidence": "low", "category": "testing", "scope": "project", "anti_pattern": false}],
      "should_record_episode": false
    }"""
    provider = FakeProvider([reply])

    msgs = [
        Message(role=Role.USER, content="do X"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(name="edit", arguments='{"file_path": "/x"}')],
        ),
        Message(role=Role.TOOL, content="ok", name="edit"),
        Message(role=Role.USER, content="thanks"),
        Message(role=Role.ASSISTANT, content="ok"),
        Message(role=Role.USER, content="bye"),
        Message(role=Role.ASSISTANT, content="bye"),
        Message(role=Role.USER, content="."),
    ]
    out = await reflector.force_reflect(provider, "m", "/p", msgs)
    assert out is not None
    assert out.outcome == "success"


def test_heuristic_dataclass_defaults():
    h = Heuristic()
    assert h.rule == ""
    assert h.anti_pattern is False
