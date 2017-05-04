"""An implementation of the truncated exponent with modulus distribution for SciPy"""
# All rights reserved.
import numpy as np
from scipy.stats import rv_continuous

class trunc_revexpon_gen(rv_continuous):
    """
    The truncated expoent distribution:
    P = exp(-|x-m|/sigma) if x > a && x < b, 0 otherwise
    """
    def _argcheck(self, a, b):
        """
        Convert from upper/lower boundaries to a and b.
        """
        self.a = a
        self.b = b
        self._nb = np.exp(a)
        self._na = np.exp(b)
        self._delta = np.abs(self._nb - self._na)
        self._logdelta = np.log(self._delta)
        return (a != b)

    def _pdf(self, x, a, b):
        """
        Calculate PDF.
        """
        result = np.zeros_like(x)
        mask = (x >= a) * (x <= b)
        result[mask] = np.exp(-np.abs(x[mask])) / self._delta
        return result

    def _cdf(self, x, a, b):
        result = np.zeros_like(x)
        a = a[0]
        b = b[0]
        if a < 0 and b > 0:
            norm = 2. - np.exp(a) - np.exp(-b)
            mask1 = (x >= a) * (x <= 0)
            mask2 = (x >= 0) * (x <= b)
            result[mask1] = np.exp(x[mask1]) - np.exp(a)
            result[mask2] = 2. - np.exp(a) - np.exp(-x[mask2])
        elif a >= 0:
            norm = np.exp(b) - np.exp(a)
            mask2 = (x >= a) * (x <= b)
            result[mask2] = np.exp(a) - np.exp(-x[mask2])
        else:  # b < 0
            norm = np.exp(b) - np.exp(a)
            mask2 = (x >= a) * (x <= b)
            result[mask2] = np.exp(x[mask2]) - np.exp(a)
        result[x <= b] /= norm
        result[x > b] = 1.
        return result

trunc_revexpon = trunc_revexpon_gen(name="trunc_revexpon", shapes='a,b')
