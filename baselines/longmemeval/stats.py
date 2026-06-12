"""Statistics helpers for benchmark reporting.

Implements the reporting standards from TeleMem's evaluation charter
(docs/evaluation.md): Wilson 95% intervals per category and multi-seed
mean ± std, so that score gaps can be read against their noise floor.
"""

import math


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_std(values):
    """Mean and sample standard deviation (n-1). std is 0.0 for n < 2."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var))


def intervals_overlap(a, b):
    """True if two (low, high) intervals overlap — i.e. the gap may be noise."""
    return a[0] <= b[1] and b[0] <= a[1]
