import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from unidam.utils.mathematics import kl_divergence, wstatistics

warnings.filterwarnings("ignore", category=np.RankWarning)


class PdfFitter():

    USE_TRF = False
    FUNC = None
    LETTER = ''
    ONLY_POSITIVE = True

    def is_solution_ok(self, popt, pcov):
        if np.any(np.isinf(np.diag(pcov))):
            return False
        if np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt)):
            return False
        return True

    def __init__(self, x, y):
        self.x = x
        self.y = y
        if self.ONLY_POSITIVE:
            y_good = np.where(y > 0)[0]
            self.x = x[y_good[0]:y_good[-1] + 1]
            self.y = y[y_good[0]:y_good[-1] + 1]
        self.init_params = wstatistics(x, y, 2)
        self.y_range = (y.min(), y.max())
        self.bounds = (-np.inf, np.inf)

    def _fit(self):
        try:
            popt, pcov = curve_fit(self.FUNC,
                                   self.x,
                                   self.y,
                                   self.init_params,
                                   ftol=1e-5)
            passed = self.is_solution_ok(popt, pcov)
        except RuntimeError:
            passed = False
        if not passed:
            if self.USE_TRF:
                print('%s attempts TRF' % self.__class__)
                # Try something else...
                popt, pcov = curve_fit(self.FUNC,
                                       self.x,
                                       self.y,
                                       self.init_params,
                                       ftol=1e-5, method='trf',
                                       bounds=self.bounds)
                if not self.is_solution_ok(popt, pcov):
                    # Fit did not converge
                    return [self.init_params, 1e10]
            else:
                return [self.init_params, 1e10]
        return [popt, kl_divergence(self.x, self.FUNC, popt, self.y)]

    def fit(self):
        return [self.LETTER, self._fit()]
