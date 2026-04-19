# SPDX-License-Identifier: Apache-2.0
"""Runner unit tests.

Integration tests that spawn real SDK binaries are skipped — this suite
only validates CLI parsing, dataset loading, scorer selection, and the
`SdkRunner` lifecycle via mocks.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import runner as runner_module
from runner import (
    EvalScore,
    QuestionResult,
    _ask_path,
    _extract_delta,
    _load_dataset,
    _parse_sse_frame,
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
            mode="direct",
            dataset="x.jsonl",
            scorer="exact",
            total=1,
            passed=1,
            pass_rate=1.0,
            mean_score=1.0,
            started_at="s",
            finished_at="e",
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


class TestCli:
    def test_help_exits_zero(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--sdk" in result.output
        assert "--scorer" in result.output

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
                mode="direct",
                dataset=str(ds),
                scorer="exact",
                total=10,
                passed=5,
                pass_rate=0.5,
                mean_score=0.5,
                started_at="s",
                finished_at="e",
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
