from .gaussian import TGaussianFit
from .linear import LinearFit
from .students import StudentsFit
from .exponent import ExponentFit
from .skew import SkewFit

FITTERS = [TGaussianFit, LinearFit, StudentsFit, ExponentFit, SkewFit]


def find_best_fit2(xdata, ydata, return_all=False):
    result = {}
    best = '-'
    best_value = 1e20
    for fitter_class in FITTERS:
        fitter = fitter_class(xdata, ydata)
        fit = fitter.fit()
        if fit[1][-1] < best_value:
            best_value = fit[1][-1]
            best = fitter.LETTER
        result[fitter.LETTER] = fit[-1]
    if return_all:
        return result
    return best, *result[best]
