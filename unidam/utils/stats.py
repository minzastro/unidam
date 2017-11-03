import numpy as np


def to_bins(arr):
    """
    Convert bin centers to bin boundaries.
    """
    result = np.zeros(len(arr)+1)
    result[1:-1] = 0.5 * (arr[1:] + arr[:-1])
    result[0] = arr[0] - 0.5*(arr[1] - arr[0])
    result[-1] = arr[-1] + 0.5*(arr[-1] - arr[-2])
    return result


def from_bins(bins):
    """
    Convert bin boundaries to bin centers.
    """
    return 0.5*(bins[1:] + bins[:-1])


def min_count_bins(data, min_step, min_count, minimum=None, maximum=None):
    """
    Create binning for the histogram such that each bin contains at least
    *min_count* points
    """
    if minimum is None:
        minimum = data.min()
    if maximum is None:
        maximum = data.max()
    bins = np.concatenate((np.arange(minimum, maximum, min_step), [maximum]))
    result = []
    histogram, _ = np.histogram(data, bins)
    sub_sum = 0
    result = [bins[0]]
    for i in xrange(len(histogram)):
        sub_sum += histogram[i]
        if sub_sum > min_count:
            sub_sum = 0
            result.append(bins[i + 1])
    result[-1] = bins[-1]
    return result


def norm(a, b):
    return b.sum() / a.sum()


def get_chi2(a, b):
    """
    Chi-squared probability for the difference between two arrays.
    """
    off = (a - b)**2
    return np.sqrt(np.sum(off)) / b.sum()


def get_mean_offset(a, b):
    off = np.abs(a - b)
    return np.sum(off) / b.sum()


def kl_divergence(a, b, lowlim=1e-10):
    mask = np.logical_and(b > lowlim, a > lowlim)
    return np.sum(a[mask] * np.log(a[mask] / b[mask]))


def kl_divergence_symmetric(a, b, lowlim=1e-10):
    return kl_divergence(a, b, lowlim) + kl_divergence(b, a, lowlim)
