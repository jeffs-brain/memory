# SPDX-License-Identifier: Apache-2.0
"""Procedural detection."""

from __future__ import annotations

from jeffs_brain_memory.llm.types import Role
from jeffs_brain_memory.memory import (
    Message,
    ToolCall,
    detect_procedurals,
    format_procedural_record,
    infer_procedural_context,
    infer_tool_call_outcome,
)


def test_detect_skill_invocation():
    msgs = [
        Message(role=Role.USER, content="Deploy the staging environment"),
        Message(
            role=Role.ASSISTANT,
            content="Running deploy skill",
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    name="skill",
                    arguments='{"skill": "deploy", "args": "--env staging"}',
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            content="Deployment complete",
            tool_call_id="tc_1",
            name="skill",
        ),
    ]
    records = detect_procedurals(msgs)
    assert len(records) == 1
    r = records[0]
    assert r.tier == "skill"
    assert r.name == "deploy"
    assert r.outcome == "ok"
    assert r.task_context == "Deploy the staging environment"


def test_detect_agent_invocation():
    msgs = [
        Message(role=Role.USER, content="Review the PR"),
        Message(
            role=Role.ASSISTANT,
            content="Delegating",
            tool_calls=[
                ToolCall(
                    id="tc_2",
                    name="agent",
                    arguments='{"type": "reviewer", "prompt": "Review PR #42"}',
                )
            ],
        ),
        Message(
            role=Role.TOOL, content="Review complete", tool_call_id="tc_2", name="agent"
        ),
    ]
    records = detect_procedurals(msgs)
    assert len(records) == 1
    assert records[0].tier == "agent"
    assert records[0].name == "reviewer"
    assert records[0].task_context == "Review PR #42"


def test_detect_empty_for_no_tool_calls():
    msgs = [
        Message(role=Role.USER, content="q"),
        Message(role=Role.ASSISTANT, content="a"),
    ]
    assert detect_procedurals(msgs) == []


def test_detect_skip_empty_skill_name():
    msgs = [
        Message(role=Role.USER, content="x"),
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(name="skill", arguments='{"skill": ""}')],
        ),
    ]
    assert detect_procedurals(msgs) == []


def test_detect_malformed_args():
    msgs = [
        Message(role=Role.USER, content="x"),
        Message(
            role=Role.ASSISTANT, tool_calls=[ToolCall(name="skill", arguments="{bad}")]
        ),
    ]
    assert detect_procedurals(msgs) == []


def test_infer_outcome_error():
    msgs = [
        Message(role=Role.ASSISTANT, content="running"),
        Message(
            role=Role.TOOL,
            content="error: permission denied",
            tool_call_id="t",
            name="skill",
        ),
    ]
    assert infer_tool_call_outcome(msgs, 0, "t") == "error"


def test_infer_outcome_success():
    msgs = [
        Message(role=Role.ASSISTANT, content="running"),
        Message(
            role=Role.TOOL,
            content="All good",
            tool_call_id="t",
            name="skill",
        ),
    ]
    assert infer_tool_call_outcome(msgs, 0, "t") == "ok"


def test_infer_outcome_partial_no_result():
    msgs = [Message(role=Role.ASSISTANT, content="running")]
    assert infer_tool_call_outcome(msgs, 0, "t") == "partial"


def test_infer_context_truncation():
    msgs = [Message(role=Role.USER, content="a" * 200)]
    assert len(infer_procedural_context(msgs, 1)) == 160


def test_infer_context_no_user():
    msgs = [Message(role=Role.ASSISTANT, content="x")]
    assert infer_procedural_context(msgs, 0) == ""


def test_format_procedural_record_basic():
    from jeffs_brain_memory.memory import ProceduralRecord

    r = ProceduralRecord(
        tier="skill",
        name="deploy",
        task_context="Deploy staging",
        outcome="ok",
        tool_calls=["skill"],
        tags=["procedural", "skill", "deploy"],
    )
    out = format_procedural_record(r)
    assert "name: deploy" in out
    assert "type: procedural" in out
    assert "tier: skill" in out
    assert "## Context" in out
    assert "## Tool sequence" in out
