import numpy as np
from scipy.stats import t, rv_continuous


class trunc_t_gen(rv_continuous):
    """
    The truncated Student's t-distribution.
    """
    def _argcheck(self, df, a, b):
        self.a = a
        self.b = b
        self._nb = t._cdf(b, df)
        self._na = t._cdf(a, df)
        self._delta = self._nb - self._na
        self._logdelta = np.log(self._delta)
        return (a != b)

    def _pdf(self, x, df, a, b):
        result = np.zeros_like(x)
        mask = (x >= a) * (x <= b)
        result[mask] = t._pdf(x[mask], df) / self._delta
        return result

    def _cdf(self, x, df, a, b):
        result = np.zeros_like(x)
        mask = (x >= a) * (x <= b)
        result[mask] = (t._cdf(x[mask], df) - self._na) / self._delta
        result[x > b] = 1.
        return result


trunc_t = trunc_t_gen(name="trunc_t", shapes='df,a,b')
