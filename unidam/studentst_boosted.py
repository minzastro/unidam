import studentst_boost
import numpy as np
from scipy.stats import rv_continuous


class studentst_gen(rv_continuous):
    def pdf(self, x, loc, scale, shape, dmin, dmax):
        sn = studentst_boost.StudentsT(loc, np.abs(scale), np.abs(shape))
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            if xx < dmax and xx > dmin:
                result[ix] = sn.pdf(xx)
        return result.reshape(x_shape)

    def cdf(self, x, loc, scale, shape, dmin, dmax):
        sn = studentst_boost.StudentsT(loc, scale, shape)
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            if xx < dmax and xx > dmin:
                result[ix] = sn.cdf(xx)
        return result.reshape(x_shape)

    def ppf(self, x, loc, scale, shape, dmin, dmax):
        sn = studentst_boost.StudentsT(loc, scale, shape)
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            if xx >= 1.:
                result[ix] = np.inf
            elif xx <= 0.:
                result[ix] = -np.inf
            else:
                result[ix] = sn.ppf(xx)
        return result.reshape(x_shape)

studentst_boosted = studentst_gen(name="studentst", shapes="shape, dmin, dmax")
