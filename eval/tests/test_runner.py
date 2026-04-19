# SPDX-License-Identifier: Apache-2.0
"""Runner unit tests.

Integration tests that spawn real SDK binaries are skipped — this suite
only validates CLI parsing, dataset loading, scorer selection, and the
`SdkRunner` lifecycle via mocks.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

import runner as runner_module
from runner import (
    EvalScore,
    QuestionResult,
    _ask_one,
    _ask_path,
    _build_request_spec,
    _extract_delta,
    _load_dataset,
    _parse_sse_frame,
    _search_path,
    main,
    write_result,
)
from sdks import get_runner
from sdks.base import SdkRunner, pick_free_port
from sdks.go import GoRunner
from sdks.py import PyRunner
from sdks.ts import TsRunner


class TestGetRunner:
    def test_returns_ts_runner(self) -> None:
        assert isinstance(get_runner("ts"), TsRunner)

    def test_returns_go_runner(self) -> None:
        assert isinstance(get_runner("go"), GoRunner)

    def test_returns_py_runner(self) -> None:
        assert isinstance(get_runner("py"), PyRunner)

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown sdk"):
            get_runner("rust")


class TestRunnerCommands:
    def test_ts_command_uses_node_and_port(self) -> None:
        cmd = TsRunner().build_command(4321)
        assert cmd[0] == "node"
        assert "127.0.0.1:4321" in cmd

    def test_go_command_uses_go_run(self) -> None:
        cmd = GoRunner().build_command(4321)
        assert cmd[:3] == ["go", "run", "./cmd/memory"]
        assert "127.0.0.1:4321" in cmd

    def test_py_command_uses_uv(self) -> None:
        cmd = PyRunner().build_command(4321)
        assert cmd[0] == "uv"
        assert "127.0.0.1:4321" in cmd

    def test_names_are_distinct(self) -> None:
        assert {r.name for r in (TsRunner(), GoRunner(), PyRunner())} == {"ts", "go", "py"}


class TestPickFreePort:
    def test_returns_positive_int(self) -> None:
        port = pick_free_port()
        assert isinstance(port, int)
        assert 1024 < port < 65536


class TestSdkRunnerLifecycle:
    def test_endpoint_requires_start(self) -> None:
        with pytest.raises(RuntimeError, match="not started"):
            _ = TsRunner().endpoint

    def test_stop_before_start_is_idempotent(self) -> None:
        TsRunner().stop()  # must not raise

    def test_start_invokes_subprocess_and_health_check(self) -> None:
        inst = TsRunner()

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None

        with (
            patch("sdks.base.subprocess.Popen", return_value=fake_proc) as popen,
            patch("sdks.base.httpx.get") as http_get,
        ):
            http_get.return_value = MagicMock(status_code=200)
            inst.start(port=54321)
            assert popen.called
            assert inst.endpoint == "http://127.0.0.1:54321"

        # stop() should SIGTERM then wait
        fake_proc.wait.return_value = 0
        inst.stop()
        fake_proc.send_signal.assert_called_once()


class TestLoadDataset:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "ds.jsonl"
        p.write_text(
            '{"id": "a", "question": "q1"}\n'
            "\n"
            "# a comment\n"
            '{"id": "b", "question": "q2"}\n',
            encoding="utf-8",
        )
        items = _load_dataset(p, limit=None)
        assert len(items) == 2
        assert items[0]["id"] == "a"

    def test_limit_caps(self, tmp_path: Path) -> None:
        p = tmp_path / "ds.jsonl"
        p.write_text(
            "\n".join(f'{{"id": "q{i}", "question": "?"}}' for i in range(10)),
            encoding="utf-8",
        )
        items = _load_dataset(p, limit=3)
        assert len(items) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        import click

        with pytest.raises(click.ClickException):
            _load_dataset(tmp_path / "nope.jsonl", limit=None)


class TestWriteResult:
    def test_creates_dated_dir_and_json(self, tmp_path: Path) -> None:
        score = EvalScore(
            sdk="ts",
            dataset="x.jsonl",
            scorer="exact",
            total=1,
            passed=1,
            pass_rate=1.0,
            mean_score=1.0,
            started_at="s",
            finished_at="e",
            scenario="ask-basic",
            mode="auto",
            questions=[
                QuestionResult(
                    id="a", question="q", answer="a", score=1.0, passed=True, latency_ms=1.0
                )
            ],
        )
        out = write_result(tmp_path, "ts", score)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["sdk"] == "ts"
        assert data["questions"][0]["id"] == "a"


class TestAskHelpers:
    def test_ask_path_encodes_brain(self) -> None:
        assert _ask_path("eval") == "/v1/brains/eval/ask"
        assert _ask_path("team alpha") == "/v1/brains/team%20alpha/ask"

    def test_search_path_encodes_brain(self) -> None:
        assert _search_path("eval") == "/v1/brains/eval/search"
        assert _search_path("team alpha") == "/v1/brains/team%20alpha/search"

    def test_build_request_spec_for_ask_basic(self) -> None:
        spec = _build_request_spec(
            brain="eval",
            item={},
            question="where?",
            top_k=3,
            mode="hybrid",
            scenario="ask-basic",
        )
        assert spec.path == "/v1/brains/eval/ask"
        assert spec.streaming is True
        assert spec.body == {"question": "where?", "topK": 3, "mode": "hybrid"}

    def test_build_request_spec_for_ask_augmented_uses_question_date(self) -> None:
        spec = _build_request_spec(
            brain="eval",
            item={"question_date": "2024/05/26 (Sun) 09:00"},
            question="where?",
            top_k=3,
            mode="hybrid",
            scenario="ask-augmented",
        )
        assert spec.path == "/v1/brains/eval/ask"
        assert spec.streaming is True
        assert spec.body == {
            "question": "where?",
            "topK": 3,
            "mode": "hybrid",
            "readerMode": "augmented",
            "questionDate": "2024/05/26 (Sun) 09:00",
        }

    def test_build_request_spec_for_search_retrieve_only(self) -> None:
        spec = _build_request_spec(
            brain="eval",
            item={"questionDate": "2024-05-26T09:00:00Z"},
            question="where?",
            top_k=4,
            mode="bm25",
            scenario="search-retrieve-only",
            candidate_k=80,
            rerank_top_n=40,
        )
        assert spec.path == "/v1/brains/eval/search"
        assert spec.streaming is False
        assert spec.body == {
            "query": "where?",
            "topK": 4,
            "mode": "bm25",
            "questionDate": "2024-05-26T09:00:00Z",
            "candidateK": 80,
            "rerankTopN": 40,
        }

    def test_hybrid_rerank_is_a_supported_search_mode(self) -> None:
        assert "hybrid-rerank" in runner_module.SEARCH_MODES

    def test_build_request_spec_for_ask_augmented_uses_camel_question_date(self) -> None:
        spec = _build_request_spec(
            brain="eval",
            item={"questionDate": "2024-05-26T09:00:00Z"},
            question="where?",
            top_k=3,
            mode="auto",
            scenario="ask-augmented",
        )
        assert spec.body["questionDate"] == "2024-05-26T09:00:00Z"

    def test_parse_sse_frame_reads_event_and_data(self) -> None:
        frame = "event: answer_delta\ndata: {\"delta\": \"Hi \"}"
        assert _parse_sse_frame(frame) == ("answer_delta", '{"delta": "Hi "}')

    def test_parse_sse_frame_skips_comments_and_empty(self) -> None:
        assert _parse_sse_frame(": keepalive\n\n") is None
        assert _parse_sse_frame("") is None

    def test_parse_sse_frame_defaults_event_to_message(self) -> None:
        assert _parse_sse_frame('data: {"ok": true}') == ("message", '{"ok": true}')

    def test_extract_delta_prefers_delta_then_falls_back(self) -> None:
        assert _extract_delta({"delta": "abc"}) == "abc"
        assert _extract_delta({"token": "xyz"}) == "xyz"
        assert _extract_delta({"text": "t"}) == "t"
        assert _extract_delta({}) == ""

    def test_ask_one_posts_augmented_ask_shape(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text=(
                    'event: retrieve\ndata: {"chunks":[]}\n\n'
                    'event: answer_delta\ndata: {"text":"Hello"}\n\n'
                    'event: done\ndata: {"answer":"Hello"}\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        async def run_once() -> object:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
                spec = _build_request_spec(
                    brain="eval",
                    item={"question_date": "2024/05/26 (Sun) 09:00"},
                    question="where?",
                    top_k=3,
                    mode="hybrid",
                    scenario="ask-augmented",
                )
                return await _ask_one(client, spec=spec)

        outcome = asyncio.run(run_once())
        assert captured["path"] == "/v1/brains/eval/ask"
        assert captured["body"] == {
            "question": "where?",
            "topK": 3,
            "mode": "hybrid",
            "readerMode": "augmented",
            "questionDate": "2024/05/26 (Sun) 09:00",
        }
        assert outcome.answer == "Hello"

    def test_ask_one_posts_search_retrieve_only_shape(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "chunks": [
                        {"chunkId": "c1", "path": "wiki/a.md", "title": "A", "score": 0.9, "text": "alpha"},
                        {"chunkId": "c2", "path": "wiki/b.md", "title": "B", "score": 0.7, "summary": "beta"},
                    ]
                },
            )

        async def run_once() -> object:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
                spec = _build_request_spec(
                    brain="eval",
                    item={"question_date": "2024/05/26 (Sun) 09:00"},
                    question="where?",
                    top_k=4,
                    mode="semantic",
                    scenario="search-retrieve-only",
                    candidate_k=80,
                    rerank_top_n=40,
                )
                return await _ask_one(client, spec=spec)

        outcome = asyncio.run(run_once())
        assert captured["path"] == "/v1/brains/eval/search"
        assert captured["body"] == {
            "query": "where?",
            "topK": 4,
            "mode": "semantic",
            "questionDate": "2024/05/26 (Sun) 09:00",
            "candidateK": 80,
            "rerankTopN": 40,
        }
        assert outcome.answer == "alpha\n\nbeta"
        assert outcome.citations == [
            {"chunkId": "c1", "path": "wiki/a.md", "title": "A", "score": 0.9},
            {"chunkId": "c2", "path": "wiki/b.md", "title": "B", "score": 0.7},
        ]


class TestCli:
    def test_help_exits_zero(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--sdk" in result.output
        assert "--scenario" in result.output
        assert "--candidate-k" in result.output
        assert "--rerank-top-n" in result.output
        assert "--scorer" in result.output
        assert "default: auto" in result.output

    def test_invalid_sdk_rejected(self) -> None:
        result = CliRunner().invoke(main, ["--sdk", "rust"])
        assert result.exit_code != 0

    def test_invalid_scorer_rejected(self) -> None:
        result = CliRunner().invoke(main, ["--sdk", "ts", "--scorer", "telepathy"])
        assert result.exit_code != 0

    def test_floor_failure_exits_nonzero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ds = tmp_path / "ds.jsonl"
        ds.write_text('{"id": "x", "question": "q"}\n', encoding="utf-8")

        class _StubRunner(SdkRunner):
            @property
            def name(self) -> str:
                return "stub"

            @property
            def workdir(self) -> Path:
                return tmp_path

            def build_command(self, port: int) -> list[str]:
                return ["true"]

            def start(self, *, port: int = 0) -> None:
                self._resolved_port = port or 1

            def stop(self) -> None:
                return None

        monkeypatch.setattr(runner_module, "get_runner", lambda _name: _StubRunner())

        def _fake_run_eval(**_: object) -> EvalScore:
            return EvalScore(
                sdk="ts",
                dataset=str(ds),
                scorer="exact",
                total=10,
                passed=5,
                pass_rate=0.5,
                mean_score=0.5,
                started_at="s",
                finished_at="e",
                scenario="ask-basic",
                mode="auto",
            )

        monkeypatch.setattr(runner_module, "run_eval", _fake_run_eval)

        result = CliRunner().invoke(
            main,
            [
                "--sdk",
                "ts",
                "--dataset",
                str(ds),
                "--scorer",
                "exact",
                "--output",
                str(tmp_path / "out"),
                "--floor",
                "0.9",
            ],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output or "FAIL" in (result.stderr_bytes or b"").decode()

    def test_run_eval_defaults_match_daemon_mode_semantics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = tmp_path / "ds.jsonl"
        ds.write_text('{"id": "x", "question": "q", "expected_substrings": ["ok"]}\n', encoding="utf-8")
        captured: dict[str, object] = {}

        async def _fake_run_eval_async(**kwargs: object) -> list[QuestionResult]:
            captured.update(kwargs)
            return [
                QuestionResult(
                    id="x",
                    question="q",
                    answer="ok",
                    score=1.0,
                    passed=True,
                    latency_ms=1.0,
                )
            ]

        monkeypatch.setattr(runner_module, "_run_eval_async", _fake_run_eval_async)

        score = runner_module.run_eval(
            endpoint="http://127.0.0.1:9999",
            dataset=ds,
            scorer_kind="exact",
            sdk="ts",
            limit=None,
        )

        assert captured["scenario"] == "ask-basic"
        assert captured["mode"] == "auto"
        assert captured["candidate_k"] == 0
        assert captured["rerank_top_n"] == 0
        assert score.scenario == "ask-basic"
        assert score.mode == "auto"

    def test_main_threads_retrieve_only_knobs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = tmp_path / "ds.jsonl"
        ds.write_text('{"id": "x", "question": "q"}\n', encoding="utf-8")
        captured: dict[str, object] = {}

        class _StubRunner(SdkRunner):
            @property
            def name(self) -> str:
                return "stub"

            @property
            def workdir(self) -> Path:
                return tmp_path

            def build_command(self, port: int) -> list[str]:
                return ["true"]

            def start(self, *, port: int = 0) -> None:
                self._resolved_port = port or 1

            def stop(self) -> None:
                return None

        def _fake_run_eval(**kwargs: object) -> EvalScore:
            captured.update(kwargs)
            return EvalScore(
                sdk="py",
                dataset=str(ds),
                scorer="exact",
                total=1,
                passed=1,
                pass_rate=1.0,
                mean_score=1.0,
                started_at="s",
                finished_at="e",
                scenario=str(kwargs["scenario"]),
                mode=str(kwargs["mode"]),
            )

        monkeypatch.setattr(runner_module, "get_runner", lambda _name: _StubRunner())
        monkeypatch.setattr(runner_module, "run_eval", _fake_run_eval)

        result = CliRunner().invoke(
            main,
            [
                "--sdk",
                "py",
                "--scenario",
                "search-retrieve-only",
                "--dataset",
                str(ds),
                "--scorer",
                "exact",
                "--output",
                str(tmp_path / "out"),
                "--floor",
                "0",
                "--top-k",
                "20",
                "--candidate-k",
                "80",
                "--rerank-top-n",
                "40",
            ],
        )

        assert result.exit_code == 0
        assert captured["scenario"] == "search-retrieve-only"
        assert captured["top_k"] == 20
        assert captured["candidate_k"] == 80
        assert captured["rerank_top_n"] == 40
