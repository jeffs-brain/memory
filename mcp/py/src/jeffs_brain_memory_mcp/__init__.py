# SPDX-License-Identifier: Apache-2.0
"""jeffs-brain memory MCP server.

Stdio MCP server exposing the canonical eleven `memory_*` tools from the
Jeffs Brain memory surface. Runs against a local filesystem brain by
default; flips to the hosted API when ``JB_TOKEN`` is set.
"""

from __future__ import annotations

__all__ = ["__version__", "SERVER_NAME", "SERVER_VERSION"]

__version__ = "0.1.0"
SERVER_NAME = "jeffs-brain-memory-mcp"
SERVER_VERSION = __version__
