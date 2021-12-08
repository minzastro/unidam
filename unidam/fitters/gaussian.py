import numpy as np
from unidam.fitters import basic
from unidam.utils.extra_functions import unidam_extra_functions as uef


def tgauss(dummy, x, mu, sigma, lower, upper):
    """
    Proxy for truncated Gaussian.
    """
    return uef.trunc_normal(x, mu, sigma, lower, upper)


class TGaussianFit(basic.PdfFitter):
    USE_TRF = True

    LETTER = 'T'

    FUNC = tgauss

    def __init__(self, x, y):
        super(TGaussianFit, self).__init__(x, y)
        y_good = np.where(y > 0.1 * y.max())[0]
        lower = x[y_good[0]]
        upper = x[y_good[-1]]
        self.bounds = ((-np.inf, -np.inf, x.min(), x.min()),
                       (np.inf, np.inf, x.max(), x.max()))
        self.init_params.extend([lower, upper])
