# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:50:56 2015

@author: mints
"""
import numpy as np

INV_SQRT2 = 1./np.sqrt(2.)
INV_SQRT2Pi = 1./np.sqrt(2.*np.pi)
MAD_COEFF = 1.4836


def kl_divergence(x, func, par, y):
    """
    Symmetric Kullback-Leibler divergence value.
    """
    values = func(x, *par)
    mask = np.logical_and(y > 1e-50, values > 1e-50)
    if np.any(mask) and mask.sum() > 2:
        return (np.sum(values[mask] * np.log(values[mask] / y[mask])) +
                np.sum(y[mask] * np.log(y[mask] / values[mask]))) / mask.sum()
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
        aic = 2. * k - np.log(np.sum((y - values) ** 2))
        aic_c = aic + 2. * k * (k + 1) / (len(y) - k - 1)
        return aic_c
    else:
        return 2e10


def to_borders(arr, min_value, max_value):
    """
    Force array to range between minimum and maximum values.
    """
    arr[np.where(arr < min_value)[0]] = min_value
    arr[np.where(arr > max_value)[0]] = max_value
    return arr


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
    if np.sum(sorted_weights) > 0:
        num_data = (cum_weights - 0.5 * sorted_weights) / \
            np.sum(sorted_weights)
    else:
        raise ValueError('Total weight is zero')
    # Get the value of the weighted median
    return np.interp(quantile, num_data, sorted_data)


def bin_estimate(data, weights=None):
    """
    Estimate bin width and number of bins required to represent the data
    in histogram.
    """
    if weights is None:
        weights = np.ones_like(data)
    q25 = quantile(data, weights, 0.25)
    q75 = quantile(data, weights, 0.75)
    width = 2. * (q75 - q25) * np.power(len(weights), -1./3.)
    if width > 0:
        return width, int((data.max() - data.min()) / width) + 1
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
    result = [mean, std]
    if moments >= 3:
        result.append(np.average(((data - mean)/std)**3, weights=weights))
        if moments >= 4:
            result.append(np.average(((data - mean)/std)**4, weights=weights))
    return result


def move_to_end(arr, item):
    """
    Move item to the end of array (if present).
    """
    if item in arr:
        arr.append(arr.pop(arr.index(item)))

def move_to_beginning(arr, item):
    """
    Move item to the end of array (if present).
    """
    if item in arr:
        arr.insert(0, arr.pop(arr.index(item)))
