# SPDX-License-Identifier: Apache-2.0
"""SDK runner registry."""
from __future__ import annotations

from sdks.base import SdkRunner
from sdks.go import GoRunner
from sdks.py import PyRunner
from sdks.ts import TsRunner

__all__ = ["SdkRunner", "GoRunner", "PyRunner", "TsRunner", "get_runner"]


def get_runner(name: str) -> SdkRunner:
    if name == "ts":
        return TsRunner()
    if name == "go":
        return GoRunner()
    if name == "py":
        return PyRunner()
    raise ValueError(f"unknown sdk: {name}")
