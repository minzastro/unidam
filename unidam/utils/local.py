import numpy as np
from scipy.stats import norm, truncnorm
from unidam.utils.extra_functions import unidam_extra_functions as uef
skew_gauss = uef.skew_normal_pdf_arr
class studentst():
    @classmethod
    def pdf(cls, x, mu, sigma, degrees_of_freedom):
           result = uef.student_pdf(x, np.abs(degrees_of_freedom), mu, np.abs(sigma))
           return result / (result.sum() * (x[1] - x[0]))

class skewnorm():
    @classmethod
    def pdf(cls, x, mu, sigma, skew):
        result = uef.skew_normal_pdf_arr(x, mu, sigma, skew)
        return result

from unidam.utils.trunc_revexpon import trunc_revexpon
from unidam.utils.trunc_line import TruncLine

"""
A collection of tools to convert UniDAM output parameters to
PDF(x).

Usage (example):
row -- row of UniDAM output file.
x = AGE_RANGE
y = get_ydata('age', row, x)
"""


def get_param(fit, par):
    """
    Convert parameters from DB to distribution parameters.
      S: skew-normal distribution,
      F: trapezoidal distribution,
      G: Gaussian (normal) distribution,
      T: truncated normal distribution,
      P: truncated Student's t-distribution,
      L: truncated Laplacian (exponent) distribution.
    """
    if fit == 'S':
        return skewnorm, par[:3]
    elif fit == 'F':
        return TruncLine, par[:4]
    elif fit == 'G':
        return norm, [par[0], par[1]]
    elif fit == 'T':
        sigma = np.abs(par[1])
        alpha = (par[2] - par[0]) / par[1]
        beta = (par[3] - par[0]) / par[1]
        return truncnorm, [alpha, beta, par[0], sigma]
    elif fit == 'P':
        return studentst, par[:-2]
    elif fit == 'L':
        sigma = np.abs(par[1])
        if par[0] < par[2]:
            par[0] = par[2] - 1e-3
        elif par[0] > par[3]:
            par[0] = par[3] + 1e-3
        alpha = (par[2] - par[0]) / sigma
        beta = (par[3] - par[0]) / sigma
        return trunc_revexpon, [alpha, beta, par[0], sigma]
    else:
        raise ValueError('Unknown fit type: %s' % fit)
    return None


def get_ydata(name, row, binx):
    """
    Prepare Y-data from PDF fitting function.
    """
    if row['%s_fit' % name] in 'GSTPLF':
        func, par = get_param(row['%s_fit' % name], row['%s_par' % name])
        #print(row['%s_fit' % name], func, par)
        ydata = func.pdf(binx, *par)
        if np.any(ydata) < 0:
            print(name, ydata)
    else:
        ydata = np.zeros_like(binx)
    return ydata


def vargauss_filter1d(xinput, yinput, sigma):
    """
    Gaussian kernel smooth for unequally-spaced grid.
    """
    output = np.zeros_like(xinput)
    for ind, item in enumerate(yinput):
        local_sigma = (sigma * xinput[ind])**2
        add = np.exp(-0.5 * (xinput - xinput[ind])**2 / local_sigma)
        output += item * add / add.sum()
    return output
