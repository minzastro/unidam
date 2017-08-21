import numpy as np
from scipy.stats import norm, truncnorm
from unidam.skewnorm_boosted import skewnorm_boosted as skewnorm
from unidam.studentst_boosted import studentst_boosted as studentst
from unidam.utils.trunc_revexpon import trunc_revexpon
from unidam.utils.fit import trunc_line

class TruncLine(object):
    @classmethod
    def pdf(cls, *par):
        return trunc_line(*par)

def get_param(fit, par):
    """
    Convert parameters from DB to distribution parameters.
    """
    if fit == 'S':
        return skewnorm, [par[2], par[0], par[1]]
    elif fit == 'F':
        return TruncLine, par[:-1]
    elif fit == 'G':
        return norm, [par[0], par[1]]
    elif fit == 'T':
        sigma = np.abs(par[1])
        alpha = (par[2] - par[0]) / par[1]
        beta = (par[3] - par[0]) / par[1]
        return truncnorm, [alpha, beta, par[0], sigma]
    elif fit == 'P':
        return studentst, par
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


def vargauss_filter1d(xinput, yinput, sigma):
    output = np.zeros_like(xinput)
    for ind, item in enumerate(yinput):
        local_sigma = (sigma * xinput[ind])**2
        add = np.exp(-0.5 * (xinput - xinput[ind])**2 / local_sigma)
        output += item * add / add.sum()
    return output