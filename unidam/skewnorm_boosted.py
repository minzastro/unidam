from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import skewnorm_boost
import numpy as np
from scipy.stats import norm, rv_continuous

# TODO: use vectorize?

class skewnorm_gen(rv_continuous):
    def pdf(self, x, shape, loc, scale):
        sn = skewnorm_boost.SkewNorm(loc, scale, shape)
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            result[ix] = sn.pdf(xx)
        return result.reshape(x_shape)

    def cdf(self, x, shape, loc, scale):
        sn = skewnorm_boost.SkewNorm(loc, scale, shape)
        x_shape = x.shape
        result = np.zeros(x.size)
        for ix, xx in enumerate(x.flatten()):
            result[ix] = sn.cdf(xx)
        return result.reshape(x_shape)

    def ppf(self, x, shape, loc, scale):
        sn = skewnorm_boost.SkewNorm(loc, scale, shape)
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

skewnorm_boosted = skewnorm_gen(name="skewnorm", shapes="shape")
