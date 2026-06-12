#!/usr/bin/env python3
"""Offline tests for the benchmark statistics helpers (baselines/longmemeval/stats.py)."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "baselines", "longmemeval"),
)

from stats import wilson_ci, mean_std, intervals_overlap  # noqa: E402


class TestWilsonCI(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 1.0))

    def test_bounds(self):
        lo, hi = wilson_ci(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_known_value(self):
        # 50/100 with z=1.96 -> approximately (0.404, 0.596)
        lo, hi = wilson_ci(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(hi, 0.596, places=2)

    def test_extremes_stay_in_unit_interval(self):
        lo, hi = wilson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        lo, hi = wilson_ci(10, 10)
        self.assertEqual(hi, 1.0)

    def test_small_n_wider_than_large_n(self):
        small = wilson_ci(8, 10)
        large = wilson_ci(800, 1000)
        self.assertGreater(small[1] - small[0], large[1] - large[0])


class TestMeanStd(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(mean_std([]), (0.0, 0.0))

    def test_single(self):
        self.assertEqual(mean_std([0.7]), (0.7, 0.0))

    def test_known(self):
        mean, std = mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(std, 2.138, places=3)


class TestIntervalsOverlap(unittest.TestCase):
    def test_overlapping(self):
        self.assertTrue(intervals_overlap((0.4, 0.6), (0.55, 0.7)))

    def test_disjoint(self):
        self.assertFalse(intervals_overlap((0.4, 0.5), (0.6, 0.7)))

    def test_touching(self):
        self.assertTrue(intervals_overlap((0.4, 0.5), (0.5, 0.6)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
