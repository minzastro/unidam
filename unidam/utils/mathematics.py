# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:50:56 2015

@author: mints
"""
import numpy as np

INV_SQRT2 = 1./np.sqrt(2.)
INV_SQRT2Pi = 1./np.sqrt(2.*np.pi)
MAD_COEFF = 1.4836


def to_str(arr):
    """
    Convert array to string nicely.
    """
    return ' '.join(map(str, arr.flatten()))


def median_mad(data):
    """
    Median and median absolute deviation of data.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))*MAD_COEFF
    return median, mad


def quantile(data, weights, quantile=0.5):
    """
    Weighted quantile of a data.
    From https://github.com/nudomarinero/wquantiles.git
    """
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    cum_weights = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    num_data = (cum_weights - 0.5 * sorted_weights) / np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, num_data, sorted_data)


def bin_estimate(data, weights=None):
    """
    Estimate bin width and number of bins required to represent the data
    in histogram.
    """
    if weights is None:
        weights = np.ones_line(data)
    q25 = quantile(data, weights, 0.25)
    q75 = quantile(data, weights, 0.75)
    h = 2. * (q75 - q25) * np.power(len(weights), -1./3.)
    if h > 0:
        return h, int((data.max() - data.min()) / h) + 1
    else:
        return 0., len(data)


def statistics(data, best):
    """
    Calculate standard deviation and skew for data
    with respect to a given item (best).
    """
    std = np.sqrt(np.mean((data - data[best])**2))
    skew = np.mean(((data - data[best]) / std)**3)
    return std, skew


def wstatistics(data, weights, moments=4):
    """
    First 4 moments with of a dataset with weights.
    """
    mean = np.average(data, weights=weights)
    std = np.sqrt(np.average((data - mean)**2, weights=weights))
    if moments >= 3:
        skew = np.average(((data - mean)/std)**3, weights=weights)
        if moments >= 4:
            kurt = np.average(((data - mean)/std)**4, weights=weights)
        else:
            kurt = None
    else:
        skew = None
        kurt = None
    return mean, std, skew, kurt
