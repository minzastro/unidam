from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np

ONE_SIGMA = 0.6827
THREE_SIGMA = 0.9973


def find_confidence(xdata, ydata, frac):
    """
    Get confidence intervals for histogram.
    Returns minimal range of xdata such that fraction *frac* of
    the integral is covered.
    """
    step = xdata[1] - xdata[0]
    total = np.sum(ydata)
    fraction = frac * total
    pos_left = np.argmax(ydata)
    pos_right = pos_left
    current_fraction = ydata[pos_left]
    x_left = xdata[pos_left] - 0.5 * step
    x_right = xdata[pos_right] + 0.5 * step
    while current_fraction < fraction:
        # Check if we should add from the left side
        if pos_right == len(ydata) - 1 or \
           (pos_left > 0 and ydata[pos_left - 1] > ydata[pos_right + 1]):
            # Check if with the next bin we have reached the goal
            if ydata[pos_left - 1] < fraction - current_fraction:
                # No, add next bin and proceed
                current_fraction = current_fraction + ydata[pos_left - 1]
                pos_left = pos_left - 1
                x_left = xdata[pos_left] - 0.5 * step
            else:
                # Yes. add part of the bin and exit
                x_left = xdata[pos_left] - \
                    step * (fraction - current_fraction) / ydata[pos_left - 1]
                x_right = xdata[pos_right] + 0.5 * step
                current_fraction = fraction
                break
        else:
            # Add from the right side
            # Check if with the next bin we have reached the goal
            if ydata[pos_right + 1] < fraction - current_fraction:
                # No, add next bin and proceed
                current_fraction = current_fraction + ydata[pos_right + 1]
                pos_right = pos_right + 1
                x_right = xdata[pos_right] + 0.5 * step
            else:
                # Yes. add part of the bin and exit
                x_left = xdata[pos_left] - 0.5 * step
                x_right = xdata[pos_right] + \
                    step * (fraction - current_fraction) / ydata[pos_right + 1]
                current_fraction = fraction
                break
    # Return the range:
    return np.array([x_left, x_right])
