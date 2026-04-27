# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP helpers for daemon-backed eval runners."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import click
import httpx


@dataclass
class ScenarioOutcome:
    answer: str
    citations: list[dict[str, Any]]
    error: str | None


@dataclass(frozen=True)
class RequestSpec:
    path: str
    body: dict[str, Any]
    streaming: bool


def ask_path(brain: str) -> str:
    return f"/v1/brains/{quote(brain, safe='')}/ask"


def search_path(brain: str) -> str:
    return f"/v1/brains/{quote(brain, safe='')}/search"


def brain_path(brain: str) -> str:
    return f"/v1/brains/{quote(brain, safe='')}"


def ingest_file_path(brain: str) -> str:
    return f"{brain_path(brain)}/ingest/file"


def question_date(item: dict[str, Any]) -> str | None:
    for key in ("questionDate", "question_date"):
        value = item.get(key)
        if isinstance(value, str) and value != "":
            return value
    return None


def safe_doc_stem(raw: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in raw]
    stem = "".join(chars).strip("-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    return stem or "item"


def build_request_spec(
    *,
    brain: str,
    item: dict[str, Any],
    question: str,
    top_k: int,
    mode: str,
    scenario: str,
    candidate_k: int = 0,
    rerank_top_n: int = 0,
) -> RequestSpec:
    if scenario == "ask-basic":
        return RequestSpec(
            path=ask_path(brain),
            body={"question": question, "topK": top_k, "mode": mode},
            streaming=True,
        )
    if scenario == "ask-augmented":
        body: dict[str, Any] = {
            "question": question,
            "topK": top_k,
            "mode": mode,
            "readerMode": "augmented",
        }
        qdate = question_date(item)
        if qdate is not None:
            body["questionDate"] = qdate
        return RequestSpec(path=ask_path(brain), body=body, streaming=True)
    if scenario == "search-retrieve-only":
        body: dict[str, Any] = {"query": question, "topK": top_k, "mode": mode}
        qdate = question_date(item)
        if qdate is not None:
            body["questionDate"] = qdate
        if candidate_k > 0:
            body["candidateK"] = candidate_k
        if rerank_top_n > 0:
            body["rerankTopN"] = rerank_top_n
        return RequestSpec(
            path=search_path(brain),
            body=body,
            streaming=False,
        )
    raise click.ClickException(f"Unknown scenario: {scenario}")


def parse_sse_frame(raw: str) -> tuple[str, str] | None:
    """Parse a single SSE frame into (event, data_json_string)."""
    event = "message"
    data_lines: list[str] = []
    for line in raw.splitlines():
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if not data_lines:
        return None
    return event, "\n".join(data_lines)


def extract_delta(payload: dict[str, Any]) -> str:
    for key in ("delta", "text", "content", "token"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


async def ask_one(
    client: httpx.AsyncClient,
    *,
    spec: RequestSpec,
) -> ScenarioOutcome:
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    error: str | None = None
    final_answer_from_done: str | None = None

    try:
        if spec.streaming:
            async with client.stream(
                "POST",
                spec.path,
                json=spec.body,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                buffer = ""
                async for chunk in resp.aiter_text():
                    if chunk == "":
                        continue
                    buffer += chunk
                    while "\n\n" in buffer:
                        raw_frame, buffer = buffer.split("\n\n", 1)
                        parsed = parse_sse_frame(raw_frame)
                        if parsed is None:
                            continue
                        event, data_str = parsed
                        try:
                            payload = json.loads(data_str) if data_str else {}
                        except json.JSONDecodeError:
                            payload = {"_raw": data_str}
                        if event in ("answer_delta", "token"):
                            answer_parts.append(extract_delta(payload))
                        elif event == "citation":
                            citations.append(payload)
                        elif event == "done":
                            if isinstance(payload, dict):
                                candidate = payload.get("answer")
                                if isinstance(candidate, str):
                                    final_answer_from_done = candidate
                            buffer = ""
                            break
                        elif event == "error":
                            code = payload.get("code") if isinstance(payload, dict) else None
                            message = payload.get("message") if isinstance(payload, dict) else None
                            error = f"stream_error: code={code} message={message}"
                            buffer = ""
                            break
                    if final_answer_from_done is not None or error is not None:
                        break
        else:
            resp = await client.post(
                spec.path,
                json=spec.body,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                payload = {}
            chunks = payload.get("chunks")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    body = chunk.get("text")
                    if not isinstance(body, str) or body == "":
                        summary = chunk.get("summary")
                        body = summary if isinstance(summary, str) else ""
                    if body != "":
                        answer_parts.append(body)
                    citations.append(
                        {
                            key: chunk[key]
                            for key in ("chunkId", "path", "title", "score")
                            if key in chunk
                        }
                    )
    except httpx.HTTPStatusError as exc:
        try:
            body = await exc.response.aread()
            text = body.decode("utf-8", errors="replace")[:512]
        except Exception:
            text = "(body unread)"
        error = f"http_{exc.response.status_code}: {text}"
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"

    if spec.streaming:
        answer = (
            final_answer_from_done
            if final_answer_from_done is not None
            else "".join(answer_parts)
        )
    else:
        answer = "\n\n".join(part for part in answer_parts if part != "")
    return ScenarioOutcome(answer=answer, citations=citations, error=error)
