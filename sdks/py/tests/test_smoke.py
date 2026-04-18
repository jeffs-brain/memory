# SPDX-License-Identifier: Apache-2.0
"""Smoke tests — import roundtrip and basic identity."""

from __future__ import annotations


def test_package_imports() -> None:
    import jeffs_brain_memory

    assert jeffs_brain_memory.__version__ == "0.0.1"


def test_submodule_imports() -> None:
    from jeffs_brain_memory import (  # noqa: F401
        cli,
        eval,
        http,
        ingest,
        knowledge,
        llm,
        memory,
        path,
        query,
        rerank,
        retrieval,
        search,
        store,
    )


def test_store_protocol_is_abstract() -> None:
    from jeffs_brain_memory.store import Store

    assert getattr(Store, "__abstractmethods__", None), "Store must expose abstract methods"


def test_spec_dir_reachable(spec_dir) -> None:
    """The shared spec directory must be co-located with the SDK."""
    assert (spec_dir / "PROTOCOL.md").is_file()
