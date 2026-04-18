# SPDX-License-Identifier: Apache-2.0
"""HTTP daemon — implements `spec/PROTOCOL.md`."""

from __future__ import annotations

from .server import create_app

__all__ = ["create_app"]
