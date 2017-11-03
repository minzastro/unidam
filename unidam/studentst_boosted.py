import studentst_boost
import numpy as np
from scipy.stats import rv_continuous


def _get_students_norm(st, dmin, dmax):
    cdfs = [st.cdf(dmax), st.cdf(dmin)]
    if cdfs[0] == cdfs[1]:
        return 1.
    else:
        return cdfs[0] - cdfs[1]


class studentst_gen(rv_continuous):
    def pdf(self, x, loc, scale, shape, dmin, dmax):
        sn = studentst_boost.StudentsT(loc, np.abs(scale), np.abs(shape))
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            if xx < dmax and xx > dmin:
                result[ix] = sn.pdf(xx)
        return result.reshape(x_shape) / _get_students_norm(sn, dmin, dmax)

    def cdf(self, x, loc, scale, shape, dmin, dmax):
        sn = studentst_boost.StudentsT(loc, np.abs(scale), np.abs(shape))
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            if xx <= dmax and xx >= dmin:
                result[ix] = (sn.cdf(xx) - sn.cdf(dmin)) \
                    / _get_students_norm(sn, dmin, dmax)
            elif xx > dmax:
                result[ix] = 1.
        return result.reshape(x_shape)

    def ppf(self, x, loc, scale, shape, dmin, dmax):
        sn = studentst_boost.StudentsT(loc, np.abs(scale), np.abs(shape))
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
