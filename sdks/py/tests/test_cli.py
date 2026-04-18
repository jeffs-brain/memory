# SPDX-License-Identifier: Apache-2.0
"""CLI smoke tests via click's test runner."""

from __future__ import annotations

from click.testing import CliRunner

from jeffs_brain_memory.cli.main import main


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.output


def test_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for sub in [
        "init",
        "ingest",
        "search",
        "ask",
        "serve",
        "remember",
        "recall",
        "reflect",
        "consolidate",
        "create-brain",
        "list-brains",
    ]:
        assert sub in result.output, f"expected {sub!r} in --help output"


def test_init_stub() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0
    assert "scaffold only" in result.output


def test_ingest_stub() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "./docs"])
    assert result.exit_code == 0
    assert "scaffold only" in result.output


def test_search_stub() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["search", "hello"])
    assert result.exit_code == 0
    assert "scaffold only" in result.output


def test_create_brain_stub() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["create-brain", "foo"])
    assert result.exit_code == 0
    assert "scaffold only" in result.output
