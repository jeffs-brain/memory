# SPDX-License-Identifier: Apache-2.0
"""Tests for the augmented LME-style reader prompt on the /ask handler.

Verifies that ``reader_mode="augmented"`` produces the ported CoT prompt
with recency, enumeration, and temporal-anchor guidance, and that the
LLM call uses the paper-faithful temperature/max-tokens. Basic mode
remains byte-identical to the original behaviour.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import socket
import threading
import time
from typing import AsyncIterator, Iterator

import httpx
import pytest
import uvicorn

from jeffs_brain_memory.http.handlers import ask as ask_handler
from jeffs_brain_memory.http import Daemon, create_app
from jeffs_brain_memory.http.handlers.ask import (
    _AUGMENTED_MAX_TOKENS,
    _AUGMENTED_TEMPERATURE,
    _BASIC_MAX_TOKENS,
    _BASIC_TEMPERATURE,
    _build_augmented_prompt,
    _build_basic_prompt,
    _retrieve,
    _reader_today_anchor,
)
from jeffs_brain_memory.llm.types import (
    CompleteRequest,
    CompleteResponse,
    StopReason,
    StreamChunk,
)
from jeffs_brain_memory.retrieval import Filters, RetrievedChunk


class _RecordingProvider:
    """Capture every CompleteRequest the handler dispatches.

    Mirrors the FakeProvider streaming shape but records the request so
    the test can assert on the prompt text and call params.
    """

    def __init__(self, response_text: str = "ok") -> None:
        self.response_text = response_text
        self.requests: list[CompleteRequest] = []
        self._closed = False

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        self.requests.append(req)
        return CompleteResponse(
            text=self.response_text,
            stop=StopReason.END_TURN,
            tokens_in=sum(len(m.content) for m in req.messages),
            tokens_out=len(self.response_text),
        )

    async def complete_stream(
        self, req: CompleteRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(req)
        return self._stream(self.response_text)

    @staticmethod
    async def _stream(text: str) -> AsyncIterator[StreamChunk]:
        for ch in text:
            yield StreamChunk(delta_text=ch)
        yield StreamChunk(stop=StopReason.END_TURN)

    async def close(self) -> None:
        self._closed = True


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@contextlib.contextmanager
def _run_app(app) -> Iterator[str]:  # type: ignore[no-untyped-def]
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                probe.connect(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _must_create_brain(client: httpx.Client, brain_id: str) -> None:
    resp = client.post("/v1/brains", json={"brainId": brain_id})
    if resp.status_code != 201:
        pytest.fail(f"create brain {brain_id}: {resp.status_code} {resp.text}")


def _drain_sse(stream) -> dict[str, list[str]]:  # type: ignore[no-untyped-def]
    events: dict[str, list[str]] = {}
    deadline = time.monotonic() + 5.0
    event_name = ""
    data_buf: list[str] = []
    for line in stream.iter_lines():
        if time.monotonic() > deadline:
            break
        if line == "":
            if event_name:
                events.setdefault(event_name, []).append("\n".join(data_buf))
            event_name = ""
            data_buf = []
            if "done" in events:
                break
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_buf.append(line[len("data:") :].lstrip())
    return events


# -- Unit tests on the prompt builder --------------------------------------


def test_today_anchor_lme_format() -> None:
    """Mirror the Go reader_test.go TestReaderTodayAnchor cases."""
    cases = [
        ("2023/05/26 (Fri) 02:28", "2023-05-26 (Friday)"),
        ("2023/05/26 02:28", "2023-05-26 (Friday)"),
        ("2023/05/26", "2023-05-26 (Friday)"),
        ("2023-05-26", "2023-05-26 (Friday)"),
        ("", "unknown"),
    ]
    for input_date, expected in cases:
        assert _reader_today_anchor(input_date) == expected, input_date


def test_augmented_prompt_contains_lme_guidance() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            path="raw/documents/notes.md",
            score=1.0,
            text="On 2024-03-01 the user said they like coffee.",
            title="notes",
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d2",
            path="raw/documents/update.md",
            score=0.9,
            text="On 2024-04-12 the user said they actually prefer tea now.",
            title="update",
        ),
    ]
    prompt = _build_augmented_prompt(
        question="What does the user drink?",
        chunks=chunks,
        question_date="2024-04-15",
    )
    lower = prompt.lower()
    # Recency guidance.
    assert "most recent session date" in lower
    assert "never use a fact dated after the current date" in lower
    # Enumeration / counting guidance.
    for kw in ("list", "count", "enumerat", "total", "one per line"):
        assert kw in lower, f"missing enumeration keyword {kw!r}"
    assert "avoid double counting the roll-up" in lower
    assert (
        "include all confirmed historical amounts for the same subject across sessions"
        in lower
    )
    assert "infer durable preferences from concrete desired features" in lower
    assert "ignore unrelated hostel, budget, or solo-travel examples" in lower
    # Temporal anchor and CoT directive.
    assert "today is 2024-04-15 (monday)" in lower
    assert "answer step by step" in lower or "step by step" in lower
    # Evidence blocks stay bounded and numbered.
    assert "Retrieved facts (2):" in prompt
    assert " 1. [unknown] [notes]" in prompt
    assert " 2. [unknown] [update]" in prompt
    # Question and current date trailer.
    assert "Current Date: 2024-04-15" in prompt
    assert "Question: What does the user drink?" in prompt


def test_augmented_prompt_unknown_anchor_when_no_date() -> None:
    prompt = _build_augmented_prompt(
        question="Hi",
        chunks=[],
        question_date="",
    )
    assert "Today is unknown" in prompt
    assert "Current Date: unknown" in prompt


def test_retrieve_fallback_uses_temporal_augmentation() -> None:
    class _EmptyRetriever:
        async def retrieve(self, _req):
            return None

    class _SearchHit:
        def __init__(self) -> None:
            self.path = "raw/lme/session-1.md"
            self.document_id = self.path
            self.chunk_id = self.path
            self.score = 1.0
            self.snippet = "Bought apples."
            self.content = "Bought apples on 2024/03/08."
            self.title = "session-1"
            self.summary = "Temporal hit"
            self.metadata = {"session_date": "2024-03-08"}

    class _RecordingIndex:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search_bm25(self, query: str, *, top_k: int, opts) -> list[_SearchHit]:
            del top_k, opts
            self.calls.append(query)
            return [_SearchHit()]

    class _Brain:
        id = "brain"

        def __init__(self) -> None:
            self.retriever = _EmptyRetriever()
            self.search_index = _RecordingIndex()

    brain = _Brain()
    chunks = asyncio.run(
        _retrieve(
            brain,
            "What did the user buy last Friday?",
            3,
            "bm25",
            "2024/03/13 (Wed) 10:00",
            0,
            0,
        )
    )
    assert chunks
    assert chunks[0].text == "Bought apples on 2024/03/08."
    assert any("2024/03/08" in call for call in brain.search_index.calls)
    assert any("2024-03-08" in call for call in brain.search_index.calls)


def test_augmented_prompt_clusters_session_hits_and_strips_frontmatter() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c2",
            document_id="d2",
            path="raw/lme/session-2.md",
            score=0.9,
            text="---\nsession_id: s2\nsession_date: 2024-02-20\n---\n[user]: The bike is blue now.",
            metadata={"session_id": "s2", "session_date": "2024-02-20"},
        ),
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            path="raw/lme/session-1-a.md",
            score=1.0,
            text="---\nsession_id: s1\nsession_date: 2024-01-10\n---\n[user]: I bought a bike.",
            metadata={"session_id": "s1", "session_date": "2024-01-10"},
        ),
        RetrievedChunk(
            chunk_id="c3",
            document_id="d3",
            path="raw/lme/session-1-b.md",
            score=0.8,
            text="---\nsession_id: s1\nsession_date: 2024-01-10\n---\n[user]: It was red at first.",
            metadata={"session_id": "s1", "session_date": "2024-01-10"},
        ),
    ]
    prompt = _build_augmented_prompt(
        question="What colour is the bike?",
        chunks=chunks,
        question_date="2024-04-15",
    )
    first = prompt.index("[session=s2]")
    second = prompt.index("[session=s1]")
    third = prompt.rindex("[session=s1]")
    assert first < second < third
    assert "session_id:" not in prompt
    assert "session_date:" not in prompt
    assert "[user]: I bought a bike." in prompt


def test_augmented_prompt_uses_frontmatter_fallback_for_session_and_date() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c2",
            document_id="d2",
            path="raw/lme/session-2.md",
            score=0.9,
            text="---\nsession_id: s2\nmodified: 2024-02-20\n---\n[user]: The bike is blue now.",
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            path="raw/lme/session-1-a.md",
            score=1.0,
            text="---\nsession_id: s1\nobserved_on: 2024-01-10\n---\n[user]: I bought a bike.",
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="c3",
            document_id="d3",
            path="raw/lme/session-1-b.md",
            score=0.8,
            text="---\nsession_id: s1\nsession_date: 2024-01-10\n---\n[user]: It was red at first.",
            metadata={},
        ),
    ]
    prompt = _build_augmented_prompt(
        question="What colour is the bike?",
        chunks=chunks,
        question_date="2024-04-15",
    )
    first = prompt.index("[session=s2]")
    second = prompt.index("[session=s1]")
    third = prompt.rindex("[session=s1]")
    assert first < second < third
    assert "[2024-01-10] [session=s1] [session-1-a]" in prompt
    assert "[2024-02-20] [session=s2] [session-2]" in prompt
    assert "session_id:" not in prompt
    assert "observed_on:" not in prompt
    assert "modified:" not in prompt
    assert "[user]: The bike is blue now." in prompt


def test_basic_prompt_unchanged() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            path="raw/documents/badger.md",
            score=1.0,
            text="Badgers are nocturnal.",
            title="badger",
        ),
    ]
    prompt = _build_basic_prompt("what are badgers", chunks)
    expected = (
        "## Evidence\n\n### badger (raw/documents/badger.md)\nBadgers are nocturnal.\n\n## Question\n\nwhat are badgers"
    )
    assert prompt == expected


# -- End-to-end: augmented mode dispatches the right CompleteRequest -------


def test_ask_augmented_dispatches_lme_prompt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    provider = _RecordingProvider("Tea.")

    async def build() -> Daemon:
        return await Daemon.create(root=tmp_path, llm=provider)

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "augask")
            ingest = client.post(
                "/v1/brains/augask/ingest/file",
                json={
                    "path": "drink.md",
                    "contentType": "text/markdown",
                    "contentBase64": base64.b64encode(
                        b"# drink\n\nThe user said they actually prefer tea now."
                    ).decode(),
                },
            )
            assert ingest.status_code == 200, ingest.text

            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/augask/ask",
                json={
                    "question": "What does the user drink?",
                    "topK": 2,
                    "reader_mode": "augmented",
                    "question_date": "2024-04-15",
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                events = _drain_sse(stream)

            assert "retrieve" in events
            assert "answer_delta" in events
            assert "done" in events

    # Wait briefly so the streamed call has finished landing in the
    # recorder before we assert. The handler runs in the uvicorn loop;
    # the SSE drain above already consumed `done`, so the call has
    # completed and the request is recorded.
    assert provider.requests, "handler did not call the LLM provider"
    req = provider.requests[-1]
    assert req.temperature == _AUGMENTED_TEMPERATURE
    assert req.max_tokens == _AUGMENTED_MAX_TOKENS
    assert len(req.messages) == 1, "augmented mode sends a single user turn"
    body = req.messages[0].content
    lower = body.lower()
    for phrase in (
        "most recent session date",
        "one per line",
        "if any named part is missing or lacks an amount",
        "count it once",
        "prefer direct transactional facts over plans, budgets",
        "today is 2024-04-15 (monday)",
        "step by step",
        "30-minute morning commute",
        "combine them if the connection is explicit",
        "state that clearly in the first sentence",
        "the information provided is not enough to answer the question",
    ):
        assert phrase in lower, f"missing augmented phrase {phrase!r}"
    assert "Retrieved facts (" in body
    assert "### " not in body
    assert "Current Date: 2024-04-15" in body
    assert "Question: What does the user drink?" in body

    asyncio.run(daemon.close())


def test_ask_augmented_uses_deterministic_resolver_before_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    provider = _RecordingProvider("Wrong answer.")

    async def build() -> Daemon:
        return await Daemon.create(root=tmp_path, llm=provider)

    async def fake_retrieve(*_args, **_kwargs) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="paper",
                document_id="paper",
                path="raw/lme/paper.md",
                score=1.0,
                text="I submitted my research paper on sentiment analysis to ACL.",
                metadata={"session_date": "2023-05-22", "session_id": "s1"},
            ),
            RetrievedChunk(
                chunk_id="acl-date",
                document_id="acl-date",
                path="raw/lme/acl-date.md",
                score=0.9,
                text="I'm reviewing for ACL, and their submission date was February 1st.",
                metadata={"session_date": "2023-02-01", "session_id": "s2"},
            ),
        ]

    monkeypatch.setattr(ask_handler, "_retrieve", fake_retrieve)

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "resolver")

            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/resolver/ask",
                json={
                    "question": "When did I submit my research paper on sentiment analysis?",
                    "topK": 2,
                    "reader_mode": "augmented",
                    "question_date": "2024-04-15",
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                events = _drain_sse(stream)

    assert "retrieve" in events
    assert "answer_delta" in events
    assert "citation" in events
    assert "done" in events
    answer = "".join(json.loads(delta)["text"] for delta in events["answer_delta"])
    assert "February 1st" in answer
    assert provider.requests == []

    asyncio.run(daemon.close())


def test_ask_augmented_accepts_live_eval_camel_case_request_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    provider = _RecordingProvider("Wrong answer.")
    captured: dict[str, object] = {}

    async def build() -> Daemon:
        return await Daemon.create(root=tmp_path, llm=provider)

    async def fake_retrieve(
        _brain,
        question: str,
        top_k: int,
        mode: str,
        question_date: str,
        candidate_k: int,
        rerank_top_n: int,
        filters=None,
    ) -> list[RetrievedChunk]:
        captured.update(
            question=question,
            top_k=top_k,
            mode=mode,
            question_date=question_date,
            candidate_k=candidate_k,
            rerank_top_n=rerank_top_n,
            filters=filters,
        )
        return [
            RetrievedChunk(
                chunk_id="paper",
                document_id="paper",
                path="raw/lme/paper.md",
                score=1.0,
                text="I submitted my research paper on sentiment analysis to ACL.",
                metadata={"session_date": "2023-05-22", "session_id": "s1"},
            ),
            RetrievedChunk(
                chunk_id="acl-date",
                document_id="acl-date",
                path="raw/lme/acl-date.md",
                score=0.9,
                text="I'm reviewing for ACL, and their submission date was February 1st.",
                metadata={"session_date": "2023-02-01", "session_id": "s2"},
            ),
        ]

    monkeypatch.setattr(ask_handler, "_retrieve", fake_retrieve)

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "evalshape")

            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/evalshape/ask",
                json={
                    "question": "When did I submit my research paper on sentiment analysis?",
                    "topK": 2,
                    "mode": "auto",
                    "readerMode": "augmented",
                    "questionDate": "2024/04/15 (Mon) 09:30",
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                events = _drain_sse(stream)

    assert captured == {
        "question": "When did I submit my research paper on sentiment analysis?",
        "top_k": 2,
        "mode": "auto",
        "question_date": "2024/04/15 (Mon) 09:30",
        "candidate_k": 0,
        "rerank_top_n": 0,
        "filters": Filters(),
    }
    assert "retrieve" in events
    assert "answer_delta" in events
    assert "citation" in events
    assert "done" in events
    answer = "".join(json.loads(delta)["text"] for delta in events["answer_delta"])
    assert "February 1st" in answer
    assert provider.requests == []

    asyncio.run(daemon.close())


def test_ask_basic_keeps_existing_call_params(tmp_path) -> None:  # type: ignore[no-untyped-def]
    provider = _RecordingProvider("Badgers are nocturnal.")

    async def build() -> Daemon:
        return await Daemon.create(root=tmp_path, llm=provider)

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "basicask")
            ingest = client.post(
                "/v1/brains/basicask/ingest/file",
                json={
                    "path": "badger.md",
                    "contentType": "text/markdown",
                    "contentBase64": base64.b64encode(
                        b"# badger\n\nBadgers are nocturnal mustelids."
                    ).decode(),
                },
            )
            assert ingest.status_code == 200, ingest.text

            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/basicask/ask",
                json={"question": "what are badgers", "topK": 1},
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                events = _drain_sse(stream)

            assert "answer_delta" in events
            assert "done" in events

    assert provider.requests, "handler did not call the LLM provider"
    req = provider.requests[-1]
    assert req.temperature == _BASIC_TEMPERATURE
    assert req.max_tokens == _BASIC_MAX_TOKENS
    # Basic mode still uses system + user turns.
    assert len(req.messages) == 2
    assert "## Evidence" in req.messages[1].content
    assert "## Question" in req.messages[1].content
    # Augmented guidance must NOT leak into basic mode.
    assert "most recent session date" not in req.messages[1].content.lower()

    asyncio.run(daemon.close())


def test_ask_rejects_invalid_reader_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path, llm=_RecordingProvider("ok")
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "badmode")
            resp = client.post(
                "/v1/brains/badmode/ask",
                json={"question": "hi", "reader_mode": "wat"},
            )
            assert resp.status_code == 400, resp.text
            assert "reader_mode" in resp.text

    asyncio.run(daemon.close())
