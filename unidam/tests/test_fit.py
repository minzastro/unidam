import numpy as np
import unittest
from numpy.testing import assert_equal, assert_almost_equal
from unidam.fitters import FITTERS, find_best_fit2


class FitTest(unittest.TestCase):
    def test1(self):
        x = np.linspace(0, 3, 30)
        y = np.exp(-(x-1.)**2 / 2)
        y = y / (y.sum() * (x[1] - x[0]))
        fit = find_best_fit2(x, y)
        assert_equal(fit[0], 'T')
        assert_almost_equal(fit[1], [1., 1., 0., 3.], decimal=2)