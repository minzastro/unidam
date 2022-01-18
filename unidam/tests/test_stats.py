import unittest
import numpy as np
from numpy.testing import assert_equal
from unidam.utils import stats


class StatsTest(unittest.TestCase):
    def test_to_bins(self):
        arr = np.arange(3)
        assert_equal(stats.to_bins(arr), [-0.5, 0.5, 1.5, 2.5])

    def test_from_bins(self):
        arr = np.arange(3)
        assert_equal(stats.from_bins(arr), [0.5, 1.5])

    def test_min_count_bins(self):
        data = np.arange(100)
        min_count = 10
        bins = stats.min_count_bins(data, 1, min_count,
                                    minimum=None, maximum=None)
        assert_equal(len(bins), 10)
        bins = stats.min_count_bins(data, 20, min_count,
                                    minimum=None, maximum=None)
        assert_equal(len(bins), 6)
