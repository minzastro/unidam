"""An implementation of the skew-t and skew-normal distributions for SciPy"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

# TODO: Implement skew-t

# Copyright (c) 2012 Oliver M. Haynold
# All rights reserved.

from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
import math
import numpy as np
from scipy.stats import norm, rv_continuous, chi2
from scipy.special import gamma

## Skew-normal distribution
class skewnorm_gen(rv_continuous):
    """The skew-normal distribution

    %(before_notes)s

    Notes
    -----
    This replicates the functionality of the sn distribution from the
    package of the same name in CRAN.
    See http://cran.r-project.org/web/packages/sn/ for more information.

    %(example)s

    """

    @staticmethod
    def _fui(h, i):
        return (h ** (2 * i)) / ((2 ** i) * gamma(i + 1))

    @staticmethod
    def _myconcat(a, b):
        if a.size > 0:
            if b.size > 0:
                return np.concatenate([a, b])
            else:
                return a
        else:
            return b

    @staticmethod
    def _tInt(h, a, jmax, cutPoint):
        seriesL = np.empty(0)
        seriesH = np.empty(0)
        i = np.arange(0, jmax + 1)
        low = h <= cutPoint
        hL = h[low]
        hH = h[np.logical_not(low)]
        L = hL.size
        atana = np.arctan(a)
        if L > 0:
            b = skewnorm_gen._fui(hL[:, np.newaxis], i)
            cumb = b.cumsum(axis=1)  # transposed compared to R code
            b1 = np.exp(-0.5 * hL ** 2)[:, np.newaxis] * cumb
            matr = np.ones((jmax + 1, L)) - b1.transpose()
            jk = ([1.0, -1.0] * jmax)[0:jmax + 1] / (2 * i + 1)
            matr = np.inner((jk[:, np.newaxis] * matr).transpose(), a ** (2 * i + 1.0))
            seriesL = (atana - matr.flatten()) / (2 * np.pi)
        if hH.size > 0:
            seriesH = (atana * np.exp(-0.5 * (hH * hH) * a / atana)
                        * (1 + 0.00868 * (hH ** 4) * a ** 4) / (2.0 * np.pi))
        series = np.empty(h.size)
        series[low] = seriesL
        series[np.logical_not(low)] = seriesH
        return series

    @staticmethod
    def _tOwen(h, a, jmax=50, cutPoint=6):
        aa = np.abs(a)
        ah = np.abs(h)
        if np.isnan(aa):
            raise ValueError("a is NaN")
        if np.isposinf(aa):
            return 0.5 * norm.cdf(-ah)
        if aa == 0.0:
            return np.zeros(h.size)
        na = np.isnan(h)
        inf = np.isposinf(ah)
        ah[np.logical_or(na, inf)] = 0
        ncdf = norm.cdf(ah)
        if aa <= 1:
            owen = skewnorm_gen._tInt(ah, aa, jmax, cutPoint)
        else:
            owen = (0.5 * ncdf + norm.cdf(aa * ah)
                    * (0.5 - ncdf) -
                    skewnorm_gen._tInt(aa * ah, (1.0 / aa), jmax, cutPoint))
        owen[np.isposinf(owen)] = 0
        return owen * np.sign(a)

    _cumulantsHalfNormCache = np.empty(0)
    @staticmethod
    def _cumulantsHalfNorm(n=4):
        if skewnorm_gen._cumulantsHalfNormCache.size < n:
            n = max(n, 2)
            n = int(2 * math.ceil(n * 0.5))
            halfN = n / 2
            m = np.arange(0, halfN)
            signs = np.array([1.0, -1.0] * halfN)[0:halfN]
            a = np.zeros(2 * halfN)
            a[2 * np.arange(0, halfN)] = signs * np.sqrt(2.0 / np.pi) / (gamma(m + 1) * 2 ** m * (2 * m + 1))
            coeff = np.array([a[0]] * n)
            for k in range(2, n + 1):
                ind = np.arange(0, k - 1)
                coeff[k - 1] = a[k - 1] - np.sum((ind + 1) * coeff[ind] * a[ind[::-1]] / k)
            kappa = coeff * gamma(np.arange(1, n + 1) + 1)
            kappa[1] = 1 + kappa[1]
            skewnorm_gen._cumulantsHalfNormCache = kappa
        return np.copy(skewnorm_gen._cumulantsHalfNormCache[:n])

    @staticmethod
    def _cumulants(shape, n=4):
        delta = shape / np.sqrt(1 + shape ** 2)
        n0 = n
        n = max(n, 2)
        kv = skewnorm_gen._cumulantsHalfNorm(n)
        kv = kv[0:n]
        kv[1] = kv[1] - 1
        kappa = delta ** np.arange(1, n + 1) * kv
        kappa[1] = kappa[1] + 1
        return kappa[0:n0]

    def _pdf(self, x, shape):
        return 2.0 * norm.pdf(x) * norm.cdf(x * shape)

    def _cdf(self, x, shape):
        if np.all(shape == shape[0]):
            res = norm.cdf(x) - 2 * skewnorm_gen._tOwen(x, shape[0])
        else:
            tow = np.vectorize(lambda x, shape: skewnorm_gen._tOwen(np.array([x]), shape))
            res = norm.cdf(x) - 2 * tow(x, shape)
        res = np.maximum(0.0, res)
        res = np.minimum(1.0, res)
        return res

    def _rvs(self, shape):
        u1 = norm.rvs(self._size)
        u2 = norm.rvs(self._size)
        idd = (u2 > shape * u1)
        u1[idd] = -u1[idd]
        return u1

    def _ppfInternal(self, q, shape):
        maxQ = np.sqrt(chi2.ppf(q, 1))
        minQ = -np.sqrt(chi2.ppf(1 - q, 1))
        if shape > 1e+5:
            return maxQ
        if shape < -1e+5:
            return minQ
        nan = np.isnan(q) | (q > 1) | (q < 0)
        zero = q == 0
        one = q == 1
        q[nan | zero | one] = 0.5
        cum = skewnorm_gen._cumulants(shape, 4)
        g1 = cum[2] / cum[1] ** (3 / 2.0)
        g2 = cum[3] / cum[1] ** 2
        x = norm.ppf(q)
        x = (x + (x * x - 1) * g1 / 6. + x * (x * x - 3) * g2 / 24. - x * (2 * x * x - 5) * g1 ** 2 / 36.)
        x = cum[0] + np.sqrt(cum[1]) * x
        tol = 1e-8
        maxErr = 1
        while maxErr > tol:
            #sn = skewnorm_local(shape)
            x1 = x - (skewnorm_local.cdf(x, shape) - q) / (skewnorm_local.pdf(x, shape))
            x1 = np.minimum(x1, maxQ)
            x1 = np.maximum(x1, minQ)
            maxErr = np.amax(np.abs(x1 - x) / (1 + np.abs(x)))
            x = x1
        x[nan] = np.NaN
        x[zero] = -np.Infinity
        x[one] = np.Infinity
        return x

    def _ppf(self, q, shape):
        if np.all(shape == shape[0]):
            return self._ppfInternal(q, shape[0])
        else:
            return np.vectorize(lambda q, shape: self._ppfInternal(np.array([q]), shape))(q, shape)

    def _argcheck(self, shape):
        return np.asarray(1)

skewnorm_local = skewnorm_gen(name="skewnorm", shapes="shape")
