import numpy as np
import warnings
from scipy.stats import norm, truncnorm, t
from scipy.optimize import curve_fit
try:
    import skewnorm_boost
    def skew_gauss(x, mu, sigma, alpha):
        """
        Skewed Gaussian distribution.
        """
        sn = skewnorm_boost.SkewNorm(mu, sigma, alpha)
        result = np.array([sn.pdf(xx) for xx in x])
        return result
except ImportError:
    from unidam.utils.skewnorm_local import skewnorm_local as sn
    def skew_gauss(x, mu, sigma, alpha):
        """
        Skewed Gaussian distribution.
        """
        result = np.array([sn.pdf(xx) for xx in x])
        return result

def t_student(x, mu, sigma, df):
    return t.pdf(x, df, mu, sigma)


def truncate_gauss(x, mu, sigma, a, b):
    """
    Truncated Gaussian distribution.
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
    mask = np.logical_and(values > 1e-50, y > 0)
    if np.any(mask) and mask.sum() > 2:
        return np.sum(values[mask]*np.log(values[mask]/y[mask])) + \
               np.sum(y[mask]*np.log(y[mask]/values[mask]))
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
        return [p0, 2e10, None]
    # If mu is outside the support interval,
    # then there is a degeneracy between mu and sigma.
    # In this case only sigma has to be constrained.
    if ((popt[0] > upper or popt[0] < lower) and
        np.sqrt(pcov[1, 1]) < 10.*np.abs(popt[1])) or \
        np.all(np.sqrt(np.abs(np.diag(pcov))) < 10.*np.abs(popt)):
        return [popt, kl_divergence(xdata, exponent, popt, ydata), 2]
    else:
        return [p0, 1e10, None]


def do_fit(xdata, ydata, func, p0, bounds=(-np.inf, np.inf),
           use_trf=True, allow_first_bad=False):
    """
    Fit given function to data.
    """
    init_param = p0
    try:
        popt, pcov = curve_fit(func, xdata, ydata, init_param,
                               ftol=1e-5)
        if allow_first_bad:
            result = np.any(np.isinf(np.diag(pcov))[1:]) or \
            np.any((np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt))[1:])
        else:
            result = np.any(np.isinf(np.diag(pcov))) or \
            np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt))

        if result:
            if use_trf:
                # Try something else...
                popt, pcov = curve_fit(func, xdata, ydata,
                                       init_param,
                                       ftol=1e-5, method='trf',
                                       bounds=bounds)
                if np.any(np.isinf(np.diag(pcov))) or \
                   np.any(np.sqrt(np.abs(np.diag(pcov))) > 10. * np.abs(popt)):
                    # Fit did not converge
                    return [p0, 1e10, None]
            else:
                return [p0, 1e10, None]
        return [popt, kl_divergence(xdata, func, popt, ydata), 2]
    except (RuntimeError, TypeError):
        # Error while fitting
        return [p0, 1.5e10, None]


def do_fit_skew(xdata, ydata, p0):
    """
    Fit skew data.
    """
    param = estimate_skew(p0[0], p0[1], xdata[np.argmax(ydata)])
    attempt = do_fit(xdata, ydata, skew_gauss, list(p0) + [param],
                     bounds=((xdata.min(), -np.inf, -50., 0.),
                             (xdata.max(), np.inf, 50., np.inf)),
                     use_trf=False)
    if attempt[1] < 1e9:
        return attempt
    return attempt


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
    else:
        return np.sign(s_est)


def find_best_fit(xdata, ydata, mu0, sigma0):
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
    xstep_left = xdata[1] - xdata[0]
    xstep_right = xdata[-1] - xdata[-2]
    xxdata = np.concatenate((
            np.arange(xdata[0] - xstep_left * 10,
                      xdata[0] - xstep_left * 0.5, xstep_left),
            xdata,
            np.arange(xdata[-1] + xstep_right,
                      xdata[-1] + xstep_right * 10.5, xstep_right)
        ))
    yydata = np.zeros_like(xxdata)
    yydata[10:-10] = ydata
    # Empirical first estimate for the Student's parameter:
    nu0 = np.min([np.power(sigma0, -0.7), 1])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        fits = [['G'] + do_fit(xxdata, yydata, gauss, (mu0, sigma0)),
                ['T'] + do_fit_trunc(xxdata, yydata, tgauss, (mu0, sigma0),
                                     (lower, upper)),
                ['S'] + do_fit_skew(xxdata, yydata, (mu0, sigma0)),
                ['P'] + do_fit(xdata, ydata, t_student, (mu0, sigma0, nu0)),
                ['L'] + do_fit_exponent(xdata, ydata, (mu0, sigma0),
                                        lower, upper), ]
    if abs(fits[2][1][2]) > 50:
        # Exclude extremly skewed shapes.
        # They are to be replaced by truncated gaussians.
        fits[2][2] = 1e11
    if fits[0][1][0] < lower or fits[0][1][0] > upper:
        # Exclude extremly skewed shapes.
        # They are to be replaced by truncated gaussians.
        fits[0][2] = 3e11
    best = np.argmin([fit[2] for fit in fits])
    if fits[best][0] == 'T' or fits[best][0] == 'L':
        # We have to include limits for a truncated Gaussian
        fits[best][1] = np.insert(fits[best][1], 2,
                                  [lower, upper])
    elif fits[best][0] == 'P':
        fits[best][1] = np.insert(fits[best][1], 3,
                                  [lower, upper])
    return fits[best][0], fits[best][1], fits[best][2]
