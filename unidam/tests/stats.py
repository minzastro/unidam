from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
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

