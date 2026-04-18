# SPDX-License-Identifier: Apache-2.0
"""`memory` CLI — click entry points."""

from __future__ import annotations

from . import commands
from .main import main

__all__ = ["commands", "main"]
