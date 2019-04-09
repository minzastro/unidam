from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
import warnings
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.optimize import curve_fit
try:
    import skewnorm_boost

    def skew_gauss(x, mu, sigma, alpha):
        """
        Skewed Gaussian distribution.
        """
        skewn = skewnorm_boost.SkewNorm(mu, sigma, alpha)
        result = np.array([skewn.pdf(xx) for xx in x])
        return result
except ImportError:
    # No skewnorm_boost library available.
    # A local python-based version will be used,
    # which is considerably slower.
    from unidam.utils.skewnorm_local import skewnorm_local as sn

    def skew_gauss(x, mu, sigma, alpha):
        """
        Skewed Gaussian distribution.
        """
        result = np.array([sn.pdf(xx, loc=mu, scale=sigma, shape=alpha
                                  ) for xx in x])
        return result

try:
    import studentst_boost

    def t_student(x, mu, sigma, degrees_of_freedom):
        """
        Wrapper for Student's T-distribution,
        re-normalized and modified to allow negative parameters.
        """
        student = studentst_boost.StudentsT(mu, np.abs(sigma),
                                            np.abs(degrees_of_freedom))
        result = np.array([student.pdf(xx) for xx in x])
        return result / (result.sum() * (x[1] - x[0]))
except ImportError:
    # No studentst_boost library available.
    # A local python-based version will be used,
    # which is considerably slower.
    from scipy.stats import t

    def t_student(x, mu, sigma, degrees_of_freedom):
        result = t.pdf(x, np.abs(degrees_of_freedom), mu, np.abs(sigma))
        return result / (result.sum() * (x[1] - x[0]))


def trunc_line(xin, val, val2, lower, upper):
    """
    Trapezoidal distribution:
    f(x) = val * x + val2, if lower <= x <= upper,
           0, otherwise.
    """
    result = np.zeros_like(xin)
    mask = (xin >= lower) * (xin <= upper)
    result[mask] = np.polyval([val, val2], xin[mask])
    result[result < 0.] = 0.
    return result


def do_fit_linear(xin, yin):
    """
    Fit a trunc_line function.
    """
    best = np.polyfit(xin, yin, 1)
    lower = -1
    upper = len(xin)
    mode = yin.max()
    modepos = np.argmax(yin)
    if modepos > 0:
        residuals = np.ones(modepos) * np.inf
        while lower < modepos and yin[lower + 1] < mode * 0.2:
            # Increase lower bound gradually,
            # re-fitting at each step, while residuals decrease.
            lower += 1
            test = np.polyfit(xin[lower:upper], yin[lower:upper], 1)
            test_r = np.sum((yin[lower:upper] -
                             np.polyval(test, xin[lower:upper]))**2) + \
                     np.sum(yin[:lower]**2) + np.sum(yin[upper:]**2)
            residuals[lower] = test_r
        lower = np.argmin(residuals)
        if np.any(~np.isinf(residuals)):
            best = np.copy(test)
    if upper > modepos:
        residuals = np.ones(upper - modepos) * np.inf
        while upper >= modepos and yin[upper - 1] < mode * 0.2:
            # Decrease upper bound gradually,
            # re-fitting at each step, while residuals decrease.
            upper -= 1
            test = np.polyfit(xin[lower:upper], yin[lower:upper], 1)
            test_r = np.sum((yin[lower:upper] -
                             np.polyval(test, xin[lower:upper]))**2) + \
                     np.sum(yin[:lower]**2) + np.sum(yin[upper:]**2)
            residuals[upper - modepos - 1] = test_r
        if np.any(~np.isinf(residuals)):
            upper = np.argmin(residuals) + modepos
            best = np.copy(test)
    result = [best[0], best[1], xin[lower], xin[upper - 1]]
    return [result, kl_divergence(xin, trunc_line, result, yin)]


def truncate_gauss(x, mu, sigma, a, b):
    """
    Truncated Gaussian distribution.
    Modification to allow for negative sigma
    and a, b parameters on x scale.
    """
    sigma = np.abs(sigma)
    alpha = (a-mu)/sigma
    beta = (b-mu)/sigma
    if alpha > 37 or beta < -37:
        # We are too far in the tail of the Gaussian,
        # fit is not reliable.
        return np.zeros(x.shape)
    else:
        return truncnorm.pdf(x, loc=mu, scale=sigma, a=alpha, b=beta)


def exponent(x, mu, sigma):
    """Re-normalized exponent distribution."""
    result = np.exp(-np.abs(x - mu) / sigma)
    return result / ((x[1] - x[0])*result.sum())


def gauss(x, mu, sigma):
    """
    Scaled Gaussian distribtion.
    """
    return norm.pdf(x, loc=mu, scale=sigma)


def kl_divergence(x, func, par, y):
    """
    Symmetric Kullback-Leibler divergence value.
    """
    values = func(x, *par)
    mask = np.logical_and(y > 1e-50, values > 1e-50)
    if np.any(mask) and mask.sum() > 2:
        return (np.sum(values[mask]*np.log(values[mask] / y[mask])) +
                np.sum(y[mask]*np.log(y[mask]/values[mask]))) / mask.sum()
    else:
        return 2e10


def akaike(x, func, par, y):
    """
    Akaike information criteria.
    """
    values = func(x, *par)
    mask = np.logical_and(values > 1e-50, y > 0)
    if np.any(mask) and mask.sum() > 2:
        k = len(par)
        aic = 2.*k - np.log(np.sum((y - values)**2))
        aic_c = aic + 2. * k * (k + 1) / (len(y) - k - 1)
        return aic_c
    else:
        return 2e10


def do_fit_exponent(xdata, ydata, p0, lower, upper):
    """
    Fit the (truncated) peaked exponential.
    The equation for PDF is:
      f(x) = exp(-|x-mu|/sigma), a < x < b; 0 otherwise.
    """
    try:
        popt, pcov = curve_fit(exponent, xdata, ydata, p0,
                               ftol=1e-5)
    except RuntimeError:
        return [p0, 2e10]
    # If mu is outside the support interval,
    # then there is a degeneracy between mu and sigma.
    # In this case only sigma has to be constrained.
    if ((popt[0] > upper or popt[0] < lower) and
        np.sqrt(pcov[1, 1]) < 10.*np.abs(popt[1])) or \
        np.all(np.sqrt(np.abs(np.diag(pcov))) < 10.*np.abs(popt)):
        return [popt, kl_divergence(xdata, exponent, popt, ydata)]
    else:
        return [p0, 1e10]


def do_fit(xdata, ydata, func, p0, bounds=(-np.inf, np.inf),
           use_trf=True):
    """
    Fit given function to data.
    """
    init_param = p0
    try:
        popt, pcov = curve_fit(func, xdata, ydata, init_param,
                               ftol=1e-5)
        if np.any(np.isinf(np.diag(pcov))) or \
            np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt)):
            if use_trf:
                # Try something else...
                popt, pcov = curve_fit(func, xdata, ydata,
                                       init_param,
                                       ftol=1e-5, method='trf',
                                       bounds=bounds)
                if np.any(np.isinf(np.diag(pcov))) or \
                   np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt)):
                    # Fit did not converge
                    return [p0, 1e10]
            else:
                return [p0, 1e10]
        return [popt, kl_divergence(xdata, func, popt, ydata)]
    except (RuntimeError, TypeError):
        # Error while fitting
        return [p0, 1.5e10]


def do_fit_trunc(xdata, ydata, func, p0, minmax):
    """
    Fit truncated data.
    """
    attempt = do_fit(xdata, ydata, func, p0)
    if attempt[1] < 1e9:
        return attempt
    else:
        attempt = do_fit(xdata, ydata, func, (minmax[0], minmax[1]-minmax[0]))
    if attempt[1] < 1e9:
        return attempt
    else:
        attempt = do_fit(xdata, ydata, func, (minmax[1], minmax[1]-minmax[0]))
    return attempt


def do_fit_student(xdata, ydata, p0):
    """
    Fit Student's T-function.
    """
    try:
        popt, _ = curve_fit(t_student, xdata, ydata, p0, ftol=1e-5)
        return [popt, kl_divergence(xdata, t_student, popt, ydata)]
    except (RuntimeError, TypeError):
        # Error while fitting
        return [p0, 1.5e10]


ESTIMATE_CONST = np.power(0.5 * (4 - np.pi), 2./3.)


def estimate_skew(mu0, sigma0, mode0):
    """
    First estimate for the nu parameter of the Student's t-distribution.
    """
    s_est = (mu0 - mode0) / sigma0
    gamma = np.power(np.abs(s_est), 2./3.)
    delta = np.sqrt(np.pi * 0.5 * gamma / (gamma + ESTIMATE_CONST))
    if delta < 1.:
        return np.sign(s_est) * delta / np.sqrt(1. - delta * delta)
    return np.sign(s_est)


def pad_x_and_y(x, y, count):
    """
    Pad input data with zeros to the left and to the right.
    """
    xstep_left = x[1] - x[0]
    xstep_right = x[-1] - x[-2]
    xxdata = np.concatenate((
        np.arange(x[0] - xstep_left * count,
                  x[0] - xstep_left * 0.5, xstep_left),
        x,
        np.arange(x[-1] + xstep_right,
                  x[-1] + xstep_right * (0.5 + count), xstep_right)
        ))
    yydata = np.zeros_like(xxdata)
    yydata[10:-10] = y
    return xxdata, yydata


def find_best_fit(xdata, ydata, mu0, sigma0, return_all=False):
    """
    Finding best fit function.
    Trying to fit Gaussian, truncated Gaussian and skewed Gaussian.
    Choosing the one with lowest KL-divergence value.
    """
    lower = xdata[0] - 0.5*(xdata[1] - xdata[0])
    upper = xdata[-1] + 0.5*(xdata[1] - xdata[0])

    def tgauss(x, *p):
        """
        Proxy for truncated Gaussian.
        """
        return truncate_gauss(x, p[0], p[1], lower, upper)
    xxdata, yydata = pad_x_and_y(xdata, ydata, 10)
    # Empirical first estimate for the Student's parameter:
    nu0 = np.min([np.power(sigma0, -0.7), 1])
    y_good = np.where(ydata > 0)[0]
    lower = xdata[y_good[0]]
    upper = xdata[y_good[-1]]
    xdata = xdata[y_good[0]:y_good[-1] + 1]
    ydata = ydata[y_good[0]:y_good[-1] + 1]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        fits = {}
        fits['F'] = do_fit_linear(xdata, ydata)
        fits['L'] = do_fit_exponent(xdata, ydata, (mu0, sigma0),
                                    lower, upper)
        fits['P'] = do_fit_student(xdata, ydata, (mu0, sigma0, nu0))
        if np.abs(fits['P'][0][2]) > 50:
            # Exclude extremly bad shapes.
            # They are to be replaced by truncated gaussians.
            fits['P'][1] = 1e11
        # For truncated Gaussian we use padded data.
        fits['T'] = do_fit_trunc(xxdata, yydata, tgauss, (mu0, sigma0),
                                 (lower, upper))
        if (fits['L'][0][0] < upper and
            fits['L'][0][0] > lower) or \
                fits['L'][1] > 1e9:
            # Skipping Gaussian (for now)
            # fits['G'] = do_fit(xxdata, yydata, gauss, (mu0, sigma0))
            # if fits['G'][0][0] < lower or fits['G'][0][0] > upper:
            #     fits['G'][1] = 3e11
            param = estimate_skew(mu0, sigma0, xxdata[np.argmax(yydata)])
            fits['S'] = do_fit(xxdata, yydata, skew_gauss,
                               [mu0, sigma0, param])
            if np.abs(fits['S'][0][2]) > 50:
                # Exclude extremly skewed shapes.
                # They are to be replaced by truncated gaussians.
                fits['S'][1] = 1e11
    best = '-'
    best_value = 1e20
    for key, value in fits.items():
        if value[1] < best_value:
            best = key
            best_value = value[1]
        if key in 'TL':
            # We have to include limits for a truncated Gaussian
            fits[key][0] = np.insert(fits[key][0], 2, [lower, upper])
        elif key == 'P':
            fits[key][0] = np.insert(fits[key][0], 3, [lower, upper])
    if return_all:
        return fits
    return best, fits[best][0], fits[best][1]
