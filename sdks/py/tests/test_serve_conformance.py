# SPDX-License-Identifier: Apache-2.0
"""Conformance harness replaying `spec/conformance/http-contract.json`.

Delegates to :mod:`jeffs_brain_memory.http.conformance`, which spins up
a fresh uvicorn server per case against a fresh app + daemon.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from jeffs_brain_memory.http import create_app
from jeffs_brain_memory.http.conformance import (
    SKIP_CASES,
    load_conformance_doc,
    replay_cases,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "spec"


def _make_app_factory():  # type: ignore[no-untyped-def]
    """Return a factory that mints a fresh app + tmpdir per call."""
    owned_dirs: list[tempfile.TemporaryDirectory] = []

    def factory():  # type: ignore[no-untyped-def]
        tmp = tempfile.TemporaryDirectory(prefix="jb-conformance-")
        owned_dirs.append(tmp)
        return create_app(root=tmp.name)

    factory._owned = owned_dirs  # type: ignore[attr-defined]
    return factory


def test_conformance_replay() -> None:
    """Run every case and assert the daemon clears the 28/29 bar."""
    factory = _make_app_factory()

    async def _run():
        return await replay_cases(factory, spec_dir=SPEC_DIR)

    report = asyncio.run(_run())
    try:
        if report.failed:
            pytest.fail(
                f"{report.failed} conformance case(s) failed:\n"
                + "\n".join(
                    f"  - {r.name}: {r.error}" for r in report.results if not r.ok
                )
            )
        # Minimum pass count mirrors the Go reference (28/29 plus the skip).
        assert report.passed >= 28, f"expected 28+ passes, got {report.passed}"
        assert len(SKIP_CASES) >= 1
    finally:
        for tmp in getattr(factory, "_owned", []):
            try:
                tmp.cleanup()
            except Exception:
                pass


def test_conformance_fixture_presence() -> None:
    doc = load_conformance_doc(SPEC_DIR)
    cases = list(doc.get("cases") or [])
    assert len(cases) >= 29, f"expected 29+ cases, got {len(cases)}"
