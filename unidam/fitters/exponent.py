import numpy as np
from unidam.fitters import basic


def exponent(self, x, mu, sigma):
    """Re-normalized exponent distribution."""
    result = np.exp(-np.abs(x - mu) / sigma)
    return result / ((x[1] - x[0]) * result.sum())


class ExponentFit(basic.PdfFitter):
    LETTER = 'L'

    FUNC = exponent

    USE_TRF = True

    def is_solution_ok(self, popt, pcov):
        # If mu is outside the support interval,
        # then there is a degeneracy between mu and sigma.
        # In this case only sigma has to be constrained.
        if ((popt[0] > self.x.max() or popt[0] < self.x.min()) and
            np.sqrt(pcov[1, 1]) < 10. * np.abs(popt[1])) or \
                np.all(np.sqrt(np.abs(np.diag(pcov))) < 10. * np.abs(popt)):
            return True
        else:
            return False

    def is_applicable(self):
        return len(self.x) > 2
