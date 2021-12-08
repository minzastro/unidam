import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from unidam.utils.mathematics import kl_divergence, wstatistics

warnings.filterwarnings("ignore", category=np.RankWarning)


def is_solution_ok(popt, pcov):
    if np.any(np.isinf(np.diag(pcov))):
        return False
    if np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt)):
        return False
    return True


class PdfFitter():

    USE_TRF = False
    FUNC = None
    LETTER = ''

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.init_params = wstatistics(x, y, 2)
        self.y_range = (y.min(), y.max())
        self.bounds = None

    def _fit(self):
        popt, pcov = curve_fit(self.FUNC,
                               self.x,
                               self.y,
                               self.init_params,
                               ftol=1e-5)
        if is_solution_ok(popt, pcov):
            if self.USE_TRF:
                # Try something else...
                popt, pcov = curve_fit(self.FUNC,
                                       self.x,
                                       self.y,
                                       self.init_params,
                                       ftol=1e-5, method='trf',
                                       bounds=self.bounds)
                if is_solution_ok(popt, pcov):
                    # Fit did not converge
                    return [self.init_params, 1e10]
            else:
                return [self.init_params, 1e10]
        return [popt, kl_divergence(self.x, self.FUNC, popt, self.y)]

    def fit(self):
        return [self.LETTER, self._fit()]
