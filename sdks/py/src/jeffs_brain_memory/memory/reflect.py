# SPDX-License-Identifier: Apache-2.0
"""Session reflection to extract generalisable heuristics.

``REFLECTION_SYSTEM_PROMPT`` is ported verbatim from
``sdks/go/memory/reflect.go``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role
from .paths import project_slug as _project_slug
from .types import Message

if TYPE_CHECKING:
    from .manager import MemoryManager

REFLECT_MAX_TOKENS = 4096
REFLECT_TEMPERATURE = 0.3
REFLECT_MIN_MESSAGES = 8
REFLECT_MAX_RECENT = 60


REFLECTION_SYSTEM_PROMPT = """You are a reflection agent. You analyse completed coding sessions to extract lasting wisdom.

Your job is NOT to summarise what happened \u2014 it is to identify GENERALISABLE PATTERNS.

Good heuristic: "When working on Go projects with generated code, check for //go:generate directives before modifying generated files."
Bad heuristic: "The file cmd/server/main.go has a bug on line 42." (Too specific.)

## Output format
Respond with ONLY valid JSON:
{
  "outcome": "success|partial|failure",
  "summary": "one paragraph",
  "retry_feedback": "what to do differently if retrying this specific task",
  "heuristics": [
    {
      "rule": "imperative, actionable pattern",
      "context": "when this applies (language, framework, problem type)",
      "confidence": "low|medium|high",
      "category": "approach|debugging|architecture|testing|communication",
      "scope": "project|global",
      "anti_pattern": false
    }
  ],
  "should_record_episode": true
}

## When to produce heuristics
- User corrected the agent \u2192 HIGH confidence (possibly anti_pattern=true)
- Multiple approaches tried before success \u2192 MEDIUM confidence
- Non-obvious error encountered \u2192 LOW confidence
- Routine session \u2192 empty array is fine

## Anti-pattern signals
Look for: "no", "don't", "stop", "instead", "that's wrong", "not like that", agent backtracking, multiple failed attempts"""


CORRECTION_PATTERNS = [
    "no,",
    "no ",
    "don't",
    "do not",
    "stop",
    "instead",
    "that's wrong",
    "not like that",
    "not what i",
    "please revert",
    "undo that",
    "try again",
]


@dataclass(slots=True)
class Heuristic:
    rule: str = ""
    context: str = ""
    confidence: str = ""
    category: str = ""
    scope: str = ""
    anti_pattern: bool = False


@dataclass(slots=True)
class ReflectionResult:
    outcome: str = ""
    summary: str = ""
    retry_feedback: str = ""
    heuristics: list[Heuristic] = field(default_factory=list)
    should_record_episode: bool = False


@dataclass(slots=True)
class SessionAnalysis:
    task_description: str = ""
    tool_call_summary: str = ""
    errors_encountered: list[str] = field(default_factory=list)
    user_corrections: list[str] = field(default_factory=list)
    iteration_count: int = 0
    write_tool_calls: int = 0
    outcome: str = ""


class Reflector:
    """Manages background session reflection."""

    def __init__(self, mem: "MemoryManager") -> None:
        self._mem = mem
        self._lock = threading.Lock()
        self.in_progress = False
        self.last_cursor = 0

    async def maybe_reflect(
        self,
        provider: Provider,
        model: str,
        project_path: str,
        messages: list[Message],
    ) -> ReflectionResult | None:
        with self._lock:
            if self.in_progress:
                return None
            self.in_progress = True
            cursor = self.last_cursor
        try:
            if not should_reflect(messages, cursor):
                return None
            result = await self._reflect(provider, model, messages, cursor)
            if result is None:
                return None
            if result.heuristics:
                slug = _project_slug(project_path)
                try:
                    self._mem.apply_heuristics(slug, result.heuristics)
                except Exception:
                    pass
            with self._lock:
                self.last_cursor = len(messages)
            return result
        finally:
            with self._lock:
                self.in_progress = False

    async def force_reflect(
        self,
        provider: Provider,
        model: str,
        project_path: str,
        messages: list[Message],
    ) -> ReflectionResult | None:
        with self._lock:
            cursor = self.last_cursor
        result = await self._reflect(provider, model, messages, cursor)
        if result is None:
            return None
        if result.heuristics:
            slug = _project_slug(project_path)
            try:
                self._mem.apply_heuristics(slug, result.heuristics)
            except Exception:
                pass
        with self._lock:
            self.last_cursor = len(messages)
        return result

    async def _reflect(
        self,
        provider: Provider,
        model: str,
        messages: list[Message],
        cursor: int,
    ) -> ReflectionResult | None:
        recent = messages[cursor:]
        if len(recent) > REFLECT_MAX_RECENT:
            recent = recent[-REFLECT_MAX_RECENT:]

        analysis = analyse_session(recent)
        user_prompt = build_reflection_prompt(analysis, recent)

        try:
            resp = await provider.complete(
                CompleteRequest(
                    model=model,
                    messages=[
                        LLMMessage(role=Role.SYSTEM, content=REFLECTION_SYSTEM_PROMPT),
                        LLMMessage(role=Role.USER, content=user_prompt),
                    ],
                    max_tokens=REFLECT_MAX_TOKENS,
                    temperature=REFLECT_TEMPERATURE,
                )
            )
        except Exception:
            return None

        result = parse_reflection_result(resp.text)
        if not result.outcome:
            return None
        return result


def should_reflect(messages: list[Message], cursor: int) -> bool:
    new_messages = messages[cursor:]
    if len(new_messages) < REFLECT_MIN_MESSAGES:
        return False
    writes = count_write_tool_calls(new_messages)
    corrections = find_user_corrections(new_messages)
    iterations = count_assistant_tool_iterations(new_messages)
    if writes == 0 and iterations < 10 and not corrections:
        return False
    return True


def analyse_session(messages: list[Message]) -> SessionAnalysis:
    return SessionAnalysis(
        task_description=extract_task_description(messages),
        tool_call_summary=summarise_tool_calls(messages),
        errors_encountered=find_errors(messages),
        user_corrections=find_user_corrections(messages),
        iteration_count=count_assistant_tool_iterations(messages),
        write_tool_calls=count_write_tool_calls(messages),
        outcome=infer_outcome(messages),
    )


def extract_task_description(messages: list[Message]) -> str:
    for m in messages:
        if m.role != Role.USER:
            continue
        content = m.content.strip()
        if len(content) < 5:
            continue
        if len(content) > 500:
            content = content[:500] + "..."
        return content
    return ""


def summarise_tool_calls(messages: list[Message]) -> str:
    tools: dict[str, dict] = {}
    order: list[str] = []
    for m in messages:
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            name = tc.name
            info = tools.get(name)
            if info is None:
                info = {"count": 0, "files": []}
                tools[name] = info
                order.append(name)
            info["count"] += 1
            if name in ("write", "edit"):
                fp = extract_file_path(tc.arguments)
                if fp:
                    info["files"].append(fp)

    parts: list[str] = []
    for name in order:
        info = tools[name]
        line = f"{name}: {info['count']} calls"
        if info["files"]:
            seen: set[str] = set()
            uniq: list[str] = []
            for f in info["files"]:
                b = base_filename(f)
                if b not in seen:
                    seen.add(b)
                    uniq.append(b)
            line += f" ({', '.join(uniq)})"
        parts.append(line)
    return "\n".join(parts).strip()


def extract_file_path(args: str) -> str:
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        return ""
    if isinstance(parsed, dict):
        val = parsed.get("file_path", "")
        return str(val) if val else ""
    return ""


def base_filename(path: str) -> str:
    parts = path.split("/")
    return parts[-1] if parts else path


def find_user_corrections(messages: list[Message]) -> list[str]:
    corrections: list[str] = []
    for m in messages:
        if m.role != Role.USER:
            continue
        lower = m.content.lower()
        for pattern in CORRECTION_PATTERNS:
            if pattern in lower:
                content = m.content
                if len(content) > 200:
                    content = content[:200] + "..."
                corrections.append(content)
                break
    return corrections


def find_errors(messages: list[Message]) -> list[str]:
    errors: list[str] = []
    for m in messages:
        if m.role != Role.TOOL:
            continue
        lower = m.content.lower()
        if any(k in lower for k in ("error", "failed", "panic", "cannot", "fatal")):
            content = m.content
            if len(content) > 300:
                content = content[:300] + "..."
            errors.append(content)
        for block in m.blocks:
            if block.tool_result is not None and block.tool_result.is_error:
                content = block.tool_result.content
                if len(content) > 300:
                    content = content[:300] + "..."
                errors.append(content)
    return errors


def count_write_tool_calls(messages: list[Message]) -> int:
    count = 0
    for m in messages:
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            if tc.name in ("write", "edit"):
                count += 1
    return count


def count_assistant_tool_iterations(messages: list[Message]) -> int:
    count = 0
    for m in messages:
        if m.role == Role.ASSISTANT and m.tool_calls:
            count += 1
    return count


def infer_outcome(messages: list[Message]) -> str:
    if not messages:
        return "unknown"
    end = len(messages)
    start = max(end - 5, 0)
    for i in range(end - 1, start - 1, -1):
        m = messages[i]
        lower = m.content.lower()
        if m.role == Role.USER:
            if any(w in lower for w in ("thanks", "perfect", "great", "looks good")):
                return "success"
            if any(w in lower for w in ("that's wrong", "not working", "broken", "revert")):
                return "failure"
    return "partial"


def build_reflection_prompt(analysis: SessionAnalysis, messages: list[Message]) -> str:
    parts: list[str] = ["## Session analysis\n"]
    if analysis.task_description:
        parts.append(f"**Task:** {analysis.task_description}\n")
    if analysis.tool_call_summary:
        parts.append(f"**Tool usage:**\n{analysis.tool_call_summary}\n")
    parts.append(f"**Iterations:** {analysis.iteration_count} tool-call rounds")
    parts.append(f"**Write/edit calls:** {analysis.write_tool_calls}")
    parts.append(f"**Inferred outcome:** {analysis.outcome}\n")
    if analysis.errors_encountered:
        parts.append("**Errors encountered:**")
        for e in analysis.errors_encountered:
            parts.append(f"- {e}")
        parts.append("")
    if analysis.user_corrections:
        parts.append("**User corrections:**")
        for c in analysis.user_corrections:
            parts.append(f"- {c}")
        parts.append("")
    parts.append("## Conversation transcript\n")
    for m in messages:
        if m.role == Role.USER:
            content = m.content
            if len(content) > 1000:
                content = content[:1000] + "\n[...truncated]"
            parts.append(f"[user]: {content}\n")
        elif m.role == Role.ASSISTANT:
            content = m.content
            if len(content) > 1000:
                content = content[:1000] + "\n[...truncated]"
            if content:
                parts.append(f"[assistant]: {content}\n")
            for tc in m.tool_calls:
                args = tc.arguments
                if len(args) > 200:
                    args = args[:200] + "..."
                parts.append(f"[tool_call {tc.name}]: {args}\n")
        elif m.role == Role.TOOL:
            content = m.content
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"[tool_result {m.name}]: {content}\n")
    return "\n".join(parts)


def parse_reflection_result(content: str) -> ReflectionResult:
    content = content.strip()
    start = content.find("{")
    if start >= 0:
        end = content.rfind("}")
        if end > start:
            content = content[start : end + 1]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return ReflectionResult()

    heuristics: list[Heuristic] = []
    for h in parsed.get("heuristics", []) or []:
        if not isinstance(h, dict):
            continue
        heuristics.append(
            Heuristic(
                rule=str(h.get("rule", "")),
                context=str(h.get("context", "")),
                confidence=str(h.get("confidence", "")),
                category=str(h.get("category", "")),
                scope=str(h.get("scope", "")),
                anti_pattern=bool(h.get("anti_pattern", False)),
            )
        )
    return ReflectionResult(
        outcome=str(parsed.get("outcome", "")),
        summary=str(parsed.get("summary", "")),
        retry_feedback=str(parsed.get("retry_feedback", "")),
        heuristics=heuristics,
        should_record_episode=bool(parsed.get("should_record_episode", False)),
    )
