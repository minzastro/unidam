from unidam.fitters import basic
from unidam.utils.extra_functions import unidam_extra_functions as uef


def tgauss(x, lower, upper, mu, sigma):
    """
    Proxy for truncated Gaussian.
    """
    return uef.trunc_normal(x, mu, sigma, lower, upper)


class TGaussianFit(basic.PdfFitter):
    USE_TRF = True

    LETTER = 'T'

    FUNC = tgauss

    def __init__(self, x, y):
        super(x, y)
