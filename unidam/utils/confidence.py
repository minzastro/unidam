import numpy as np

ONE_SIGMA = 0.6827
THREE_SIGMA = 0.9973


def find_confidence(xdata, ydata, frac):
    """
    Get confidence intervals for histogram.
    Returns minimal range of xdata such that fraction *frac* of
    the integral is covered.
    """
    xstep = xdata[1] - xdata[0]
    total = np.sum(ydata)
    frac1 = frac * total
    pos_left = np.argmax(ydata)
    pos_right = pos_left
    part1 = ydata[pos_left]
    x_left = xdata[pos_left] - 0.5 * xstep
    x_right = xdata[pos_right] + 0.5 * xstep
    while part1 < frac1:
        if pos_right == len(ydata) - 1 or \
           (pos_left > 0 and ydata[pos_left - 1] > ydata[pos_right + 1]):
            if ydata[pos_left - 1] < frac1 - part1:
                part1 = part1 + ydata[pos_left - 1]
                pos_left = pos_left - 1
                x_left = xdata[pos_left] - 0.5 * xstep
            else:
                x_left = xdata[pos_left] - \
                    xstep * (frac1 - part1) / ydata[pos_left - 1]
                x_right = xdata[pos_right] + 0.5 * xstep
                part1 = frac1
        else:
            if ydata[pos_right + 1] < frac1 - part1:
                part1 = part1 + ydata[pos_right + 1]
                pos_right = pos_right + 1
                x_right = xdata[pos_right] + 0.5 * xstep
            else:
                x_left = xdata[pos_left] - 0.5 * xstep
                x_right = xdata[pos_right] + \
                    xstep * (frac1 - part1) / ydata[pos_right + 1]
                part1 = frac1
    return x_left, x_right
