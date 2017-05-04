import numpy as np
from scipy.signal import argrelextrema


def histogram_splitter(hist, bins, min_order=2, max_order=3, dip_depth=0.75,
                       use_spikes=True, spike_threshold=3,
                       spike_location=1.75, spike_spread=0.3):
    """
    Function to split histogram using local minima and local maxima.

    We detect local minima and maxima of the histogram.
    Local minima (or maxima) are defined as locations of bins that have
    lower (higher) value $h_i$ than all other bins within the window:
    $h_i = min\{h_j, i-n \leq j \leq i+n\}$ for a local minimum and
    $h_i = max\{h_j, i-n \leq j \leq i+n\}$ for a local maximum.
    Window size $n$ was taken to be 3 for maxima and 2 for minima.
    Differences in window sizes are caused by the need to locate minima with
    high precision and to avoid too many maxima in noisy data.
    Formally, it is possible to have more than one local minimum between
    two local maxima -- we split only by the lowest of them in this case.
    We split the sample at positions of local minima that are lower
    than 0.75 times the value of the smallest of the two enclosing maxima.
    """
    # Locate minima and maxima
    mins = argrelextrema(hist, np.less_equal, order=min_order)[0]
    # Pad right with zeros + wrap to get maxima on first/last bin correctly
    maxs = argrelextrema(np.append(hist, np.zeros(3)), np.greater,
                         order=max_order, mode='wrap')[0]
    if use_spikes:
        with np.errstate(all='ignore'):
            spikes = np.where(np.logical_and(
                hist[1:-1] / hist[:-2] > spike_threshold,
                hist[1:-1] / hist[2:] > spike_threshold))[0]
            if len(spikes) > 0:
                spikes = spikes + 1
                # Remove peaks already detected as maxima
                spikes = np.setdiff1d(spikes, maxs)
            if len(spikes) > 0:
                # Spikes might happen at masses around 1.75 Msol
                # (this is a feature, not a bug).
                # They have to be treated separately.
                spikes = np.where(np.abs(bins[spikes] - spike_location) <
                                  spike_spread)[0]
        if len(spikes) > 0:
            spikes = spikes + 1
            maxs = np.append(maxs, spikes)
            maxs = np.sort(np.unique(maxs))
            mins = np.append(mins, spikes - 1)
            mins = np.append(mins, spikes + 1)
            mins = np.sort(np.unique(mins))
    pos = 0.5 * (bins[1:] + bins[:-1])
    maximums = np.vstack((hist[maxs], pos[maxs]))
    minimums = np.vstack((hist[mins], pos[mins]))
    # Select only maxima bigger than 0.01 of the main maximum
    # ...removed, to get rid of "bad" fits.
    # maximums = maximums[:, maximums[0] > 0.01*hist.max()]
    for imax in xrange(maximums.shape[1] - 1):
        # Get list of minimums between this and next maximum
        between = minimums[:,
                           (minimums[1] > maximums[1, imax]) *
                           (minimums[1] < maximums[1, imax + 1])]
        # Drop minimums that are not low enough...
        between = between[:, between[0] < dip_depth *
                          np.min(maximums[0, imax:imax + 2])]
        if len(between[0]) == 0:
            # No minimum between maximums, merge maximums...
            continue
        # Split by lowest minimum...
        yield between[1, np.argmin(between[0])]
