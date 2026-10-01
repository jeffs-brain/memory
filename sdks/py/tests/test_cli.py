# SPDX-License-Identifier: Apache-2.0
"""CLI smoke tests via click's test runner."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from jeffs_brain_memory.cli.main import main


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.output


PLANNED = [
    ["init"],
    ["ingest", "./docs"],
    ["search", "hello"],
    ["ask", "why?"],
    ["remember", "a note"],
    ["recall", "hello"],
    ["reflect"],
    ["consolidate"],
    ["create-brain", "foo"],
    ["list-brains"],
]


def test_help_lists_only_working_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    for args in PLANNED:
        assert f"  {args[0]} " not in result.output, f"{args[0]!r} should be hidden from --help"


@pytest.mark.parametrize("args", PLANNED, ids=lambda args: args[0])
def test_planned_commands_fail(args: list[str]) -> None:
    runner = CliRunner()
    result = runner.invoke(main, args)
    assert result.exit_code == 1
    assert "not implemented in the Python SDK yet" in result.output
