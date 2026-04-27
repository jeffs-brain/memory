# SPDX-License-Identifier: Apache-2.0
"""Pure benchmark statistics helpers."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

DEFAULT_BOOTSTRAP_RESAMPLES = 1000


def bootstrap_ci(
    outcomes: Sequence[bool],
    *,
    seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if not outcomes:
        return (0.0, 0.0)
    if resamples <= 0:
        resamples = DEFAULT_BOOTSTRAP_RESAMPLES

    rng = random.Random(seed)
    count = len(outcomes)
    means: list[float] = []

    for _ in range(resamples):
        correct = 0
        for _ in range(count):
            if outcomes[rng.randrange(count)]:
                correct += 1
        means.append(correct / count)

    means.sort()
    return (_percentile_of_sorted(means, 2.5), _percentile_of_sorted(means, 97.5))


def latency_percentile(latencies: Sequence[int | float], pct: float) -> int | float:
    if not latencies:
        return 0

    clamped_pct = min(max(pct, 0), 100)
    sorted_latencies = sorted(latencies)
    rank = math.ceil((clamped_pct / 100) * len(sorted_latencies))
    rank = min(max(rank, 1), len(sorted_latencies))
    return sorted_latencies[rank - 1]


def _percentile_of_sorted(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0

    clamped_pct = min(max(pct, 0), 100)
    rank = math.ceil((clamped_pct / 100) * len(values))
    rank = min(max(rank, 1), len(values))
    return values[rank - 1]
