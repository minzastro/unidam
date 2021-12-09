import numpy as np
from unidam.utils.extra_functions import unidam_extra_functions as uef
from unidam.fitters import basic
import warnings

warnings.filterwarnings("ignore", category=np.RankWarning)


def skew_gauss(self, x, a, b, c):
    return uef.skew_normal_pdf_arr(x, a, b, c)


ESTIMATE_CONST = np.power(0.5 * (4 - np.pi), 2. / 3.)


def estimate_skew(mu0, sigma0, mode0):
    """
    First estimate for the nu parameter of the Student's t-distribution.
    """
    s_est = (mu0 - mode0) / sigma0
    gamma = np.power(np.abs(s_est), 2. / 3.)
    delta = np.sqrt(np.pi * 0.5 * gamma / (gamma + ESTIMATE_CONST))
    if delta < 1.:
        return np.sign(s_est) * delta / np.sqrt(1. - delta * delta)
    return np.sign(s_est)


class SkewFit(basic.PdfFitter):
    LETTER = 'S'

    FUNC = skew_gauss

    USE_TRF = True

    def __init__(self, x, y):
        super(SkewFit, self).__init__(x, y)
        param = estimate_skew(self.init_params[0],
                              self.init_params[1], x[np.argmax(y)])
        self.init_params.append(param)

    def is_applicable(self):
        return len(self.x) > 4