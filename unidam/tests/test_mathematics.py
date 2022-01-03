import pytest
import unittest
import numpy as np
from numpy.testing import assert_equal
from unidam.utils import mathematics


def test_to_borders():
    v = np.arange(10, dtype=float)
    b = mathematics.to_borders(v, 2.5, 5.5)
    assert_equal(b.min(), 2.5)
    assert_equal(b.max(), 5.5)


class MathTest(unittest.TestCase):

    def test_to_str(self):
        v = np.arange(3)
        self.assertEqual(mathematics.to_str(v), '0 1 2')

    def test_median_mad(self):
        v = np.ones(4)
        m, mm = mathematics.median_mad(v)
        self.assertEqual(m, 1)
        self.assertEqual(mm, 0)
        v = np.arange(10)
        m, mm = mathematics.median_mad(v)
        self.assertEqual(m, 4.5)
        assert_equal(mm, 3.709)

    def test_quantile(self):
        v = np.arange(20, dtype=float)
        assert_equal(mathematics.quantile(v, np.ones(20), 0.5), 9.5)
        assert_equal(mathematics.quantile(v, np.ones(20), 0.25), 4.5)
        assert_equal(mathematics.quantile(v, np.ones(20), [0.25, 0.75]), [4.5, 14.5])
        w = np.ones(20)
        w[:10] = 2
        assert_equal(mathematics.quantile(v, w, 0.5), 7.)

    def test_bin_estimate(self):
        v = np.arange(20)
        w = np.ones(20)
        w[:10] = 2.
        assert_equal(mathematics.bin_estimate(v)[1], 3)

    def test_move_to_end(self):
        a = [1, 2, 3, 4, 5]
        mathematics.move_to_end(a, 3)
        self.assertEqual(a, [1, 2, 4, 5, 3])
        a = [1, 2, 3, 4, 5]
        mathematics.move_to_end(a, 0)
        self.assertEqual(a, [1, 2, 3, 4, 5])
        a = [1, 2, 3, 3, 4, 5]
        mathematics.move_to_end(a, 3)
        self.assertEqual(a, [1, 2, 3, 4, 5, 3])

