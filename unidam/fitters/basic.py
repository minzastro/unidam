import warnings
import numpy as np
from scipy.optimize import curve_fit
from unidam.utils.mathematics import kl_divergence, wstatistics

warnings.filterwarnings("ignore", category=np.RankWarning)


class PdfFitter():

    USE_TRF = False
    FUNC = None
    LETTER = ''
    ONLY_POSITIVE = True
    PADDING = False

    def __init__(self, x, y):
        self.x = x
        step = x[1] - x[0]
        self.y = y
        if self.ONLY_POSITIVE:
            y_good = np.where(y > 0)[0]
            self.x = x[y_good[0]:y_good[-1] + 1]
            self.y = y[y_good[0]:y_good[-1] + 1]
        self.init_params = wstatistics(x, y, 2)
        self.bounds = (-np.inf, np.inf)
        if self.PADDING:
            self.x = np.concatenate([np.arange(x[0] - step*10, x[0], step),
                                    x,
                                    np.arange(x[-1], x[-1] + step * 10, step)])
            self.y = np.pad(self.y, 10, mode='constant', constant_values=0)

    def is_applicable(self):
        """Returns:
            True if the fitter can be applied to the data.
        """
        return True

    def is_solution_ok(self, popt, pcov):
        """ Test if the current solution is acceptable.
        This is True if:
            a) Covariance matrix elements are all non-infinite AND
            b) Parameter uncertainties (from the covarinace matrix diagonal)
               are smaller than 10xparameter - this is to exclude
               very poor fits.

        Args:
            popt : array of optimal parameters
            pcov : covariance matrix.

        Returns:
            [type]: [description]
        """
        if np.any(np.isinf(np.diag(pcov))):
            return False
        if np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt)):
            return False
        return True

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
                # Try something else...
                try:
                    popt, pcov = curve_fit(self.FUNC,
                                           self.x,
                                           self.y,
                                           self.init_params,
                                           ftol=1e-5, method='trf',
                                           bounds=self.bounds)
                    if not self.is_solution_ok(popt, pcov):
                        # Fit did not converge
                        return [self.init_params, 1e10]
                except RuntimeError:
                    return [self.init_params, 2e10]
            else:
                return [self.init_params, 1e10]
        return [popt, kl_divergence(self.x, self.FUNC, popt, self.y)]

    def fit(self):
        return [self.LETTER, self._fit()]
