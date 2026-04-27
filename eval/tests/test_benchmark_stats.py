# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from benchmarks.stats import bootstrap_ci, latency_percentile


def test_latency_percentile_uses_nearest_rank_without_mutating_input() -> None:
    latencies = [50, 10, 90, 30, 70, 20, 100, 40, 80, 60]
    original = latencies.copy()

    assert latency_percentile(latencies, 50) == 50
    assert latency_percentile(latencies, 95) == 100
    assert latency_percentile(latencies, 0) == 10
    assert latency_percentile(latencies, 100) == 100
    assert latencies == original


def test_latency_percentile_clamps_percentile_bounds() -> None:
    assert latency_percentile([10, 20, 30], -10) == 10
    assert latency_percentile([10, 20, 30], 110) == 30
    assert latency_percentile([], 50) == 0


def test_bootstrap_ci_is_deterministic_for_seed() -> None:
    outcomes = [True] * 30 + [False] * 20

    first = bootstrap_ci(outcomes, seed=123, resamples=500)
    second = bootstrap_ci(outcomes, seed=123, resamples=500)

    assert first == second
    assert first[0] <= 0.6 <= first[1]
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_bootstrap_ci_handles_edge_cases() -> None:
    assert bootstrap_ci([], seed=1, resamples=200) == (0.0, 0.0)
    assert bootstrap_ci([True] * 20, seed=1, resamples=200) == (1.0, 1.0)
    assert bootstrap_ci([False] * 20, seed=1, resamples=200) == (0.0, 0.0)


def test_bootstrap_ci_uses_default_for_non_positive_resamples() -> None:
    low, high = bootstrap_ci([True, False, True, True, False], seed=7, resamples=0)

    assert 0.0 <= low <= high <= 1.0
