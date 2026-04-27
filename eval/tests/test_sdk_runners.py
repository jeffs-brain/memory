# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from sdks.go import _parse_go_version, _resolve_go_binary
from sdks.ts import _node_version, _resolve_node_binary


def test_ts_node_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("JB_EVAL_TS_NODE", "/custom/node")

    assert _resolve_node_binary() == "/custom/node"


def test_ts_node_resolver_prefers_highest_nvm_node_24(tmp_path: Path, monkeypatch) -> None:
    nvm_dir = tmp_path / ".nvm"
    for version in ("v24.1.0", "v24.13.0", "v22.22.0"):
        node = nvm_dir / "versions" / "node" / version / "bin" / "node"
        node.parent.mkdir(parents=True)
        node.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("JB_EVAL_TS_NODE", raising=False)
    monkeypatch.delenv("JB_EVAL_NODE", raising=False)
    monkeypatch.setenv("NVM_DIR", str(nvm_dir))

    assert _resolve_node_binary() == str(
        nvm_dir / "versions" / "node" / "v24.13.0" / "bin" / "node"
    )


def test_node_version_parsing() -> None:
    assert _node_version("v24.13.0") == (24, 13, 0)
    assert _node_version("v24.bad") == (24, 0, 0)


def test_go_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("JB_EVAL_GO", "/custom/go")

    assert _resolve_go_binary() == "/custom/go"


def test_go_resolver_prefers_cached_mod_toolchain(tmp_path: Path, monkeypatch) -> None:
    go = (
        tmp_path
        / "golang.org"
        / "toolchain@v0.0.1-go1.25.8.darwin-arm64"
        / "bin"
        / "go"
    )
    go.parent.mkdir(parents=True)
    go.write_text("#!/bin/sh\necho 'go version go1.25.8 darwin/arm64'\n", encoding="utf-8")
    go.chmod(0o755)

    monkeypatch.delenv("JB_EVAL_GO", raising=False)
    monkeypatch.delenv("JB_EVAL_GO_BIN", raising=False)
    monkeypatch.setenv("GOMODCACHE", str(tmp_path))

    assert _resolve_go_binary() == str(go)


def test_go_version_parsing() -> None:
    assert _parse_go_version("go version go1.26.2 darwin/arm64") == (1, 26, 2)
    assert _parse_go_version("go version go1.25 darwin/arm64") == (1, 25, 0)
    assert _parse_go_version("not go") is None
