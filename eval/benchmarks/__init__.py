# SPDX-License-Identifier: Apache-2.0
"""Benchmark adapter registry."""
from __future__ import annotations

from benchmarks.base import BenchmarkAdapter
from benchmarks.bpi_bench import BpiBenchAdapter
from benchmarks.locomo import LoCoMoAdapter
from benchmarks.memory_agent_bench import MemoryAgentBenchAdapter

_ADAPTERS = {
    "bpi": BpiBenchAdapter,
    "bpi-bench": BpiBenchAdapter,
    "bpi_bench": BpiBenchAdapter,
    "locomo": LoCoMoAdapter,
    "memory-agent-bench": MemoryAgentBenchAdapter,
    "memory_agent_bench": MemoryAgentBenchAdapter,
    "mab": MemoryAgentBenchAdapter,
}


def get_adapter(name: str) -> BenchmarkAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        expected = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"unknown benchmark adapter {name!r}; expected one of: {expected}"
        ) from exc


__all__ = ["BpiBenchAdapter", "LoCoMoAdapter", "MemoryAgentBenchAdapter", "get_adapter"]
