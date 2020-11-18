import numpy as np
from unidam.core.model_fitter import model_fitter as mf  # pylint: disable=no-member
from unidam.utils.local import vargauss_filter1d
from scipy.ndimage.filters import gaussian_filter1d
from unidam.utils.stats import from_bins, to_bins
from unidam.utils.confidence import find_confidence, ONE_SIGMA, THREE_SIGMA
from unidam.utils.mathematics import wstatistics, quantile, bin_estimate, \
    to_borders
from unidam.utils.fit import find_best_fit

class HistogramAnalyzer():
    MINIMUM_STEP = {'age': 0.02,
                    'distance_modulus': 0.04,
                    'distance': 100.,
                    'extinction': 0.001,
                    'parallax': 0.00}


    def __init__(self, name, mode_data, weights, smooth=None,
                     extinction_data=None, dump=False):
        self.name = name
        self.mode_data = mode_data
        self.weights = weights
        self.smooth = smooth
        self.extinction_data = extinction_data
        self.m_min = self.mode_data.min()
        self.m_max = self.mode_data.max()
        self.dump = dump

    def _get_histogram_parts(self, bins):
        """
        Get the smoothed histogram for the case when parallax is known.
        In this case we need to distinguish between models with
        extinctions below and above the input extinction --
        these sets will need different smoothing.
        """
        part1 = self.mode_data[self.extinction_data < mf.extinction]
        weight1 = self.weights[self.extinction_data < mf.extinction]
        bin_centers = from_bins(bins)
        hist = np.zeros_like(bin_centers)
        if len(part1) > 0:
            bin_left = max(0, bins.searchsorted(part1.min()) - 1)
            bin_right = bins.searchsorted(part1.max()) + 1
            if bin_right - bin_left == 1:
                hist[bin_left] = weight1.sum()
            else:
                xbins = bins[bin_left:bin_right]
                hist1, _ = \
                    self._get_histogram(xbins, part1, weight1, self.smooth[0])
                hist[bin_left:bin_right - 1] = hist1
        # And now the second part with a different smoothing
        part2 = self.mode_data[self.extinction_data >= mf.extinction]
        weight2 = self.weights[self.extinction_data >= mf.extinction]
        if len(part2) > 0:
            hist2, _ = \
                self._get_histogram(bins, part2, weight2, self.smooth[1])
            hist += hist2
        return hist, bin_centers

    def _get_histogram(self, bins, mode_data, weights,
                       smooth=None):
        """
        Prepare a proper histogram for fitting.
        params:
            :name: parameter name, only for special treatment of
                some parameters. E.g. for ages we use pre-defined grid,

            :mode_data:
        """
        if self.name == 'age':
            # For ages we always use a fixed grid.
            bin_centers = self.age_grid
        else:
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # We need mode_data >= bins[0], otherwise numpy behaves
        # strangely sometimes.
        hist = np.histogram(mode_data[mode_data >= bins[0]],
                            bins,
                            weights=weights[mode_data >= bins[0]]
                            )[0]
        hist = hist / (hist.sum() * (bins[1:] - bins[:-1]))
        if smooth is not None:
            if self.name in ['distance_modulus', 'extinction']:
                hist = gaussian_filter1d(
                    hist,
                    smooth / (bins[1] - bins[0]),
                    mode='constant')
            else:
                hist = vargauss_filter1d(bin_centers, hist, smooth)
        return hist, bin_centers

    def _update_err(self, err, avg, bin_step):
        if self.smooth is not None:
            if self.name in ['distance_modulus', 'extinction']:
                err = np.sqrt(err ** 2 + np.sum(self.smooth ** 2))
            else:
                err = np.sqrt(err ** 2 + np.sum((avg * self.smooth) ** 2))
        if err == 0.:
            # This is done for the case of very low weights...
            # I guess it should be done otherwise, but...
            err = max(np.std(self.mode_data), bin_step * 0.2)
        elif err > 0.5 * (self.m_max - self.m_min):
            # This is for the case of very high smoothing values.
            err = 0.5 * (self.m_max - self.m_min)
        elif err < bin_step * 0.2:
            err = bin_step * 0.2
        return err


    def get_bin_count(self):
        """
        Estimate the number of bins needed to represent the data.
        """
        # We use different binning for different parameters.
        if self.name == 'mass':
            # Fixed step in mass, as masses are discrete
            # for some isochrones.
            bins = np.arange(self.m_min * 0.95, self.m_max * 1.1, 0.06)
        elif self.name == 'distance':
            # Fixed number of bins for distances
            bins = np.linspace(self.m_min * 0.95, self.m_max, 50)
        elif self.name == 'age':
            bins = to_bins(self.age_grid)
        else:
            if self.name == 'extinction':
                m_min = self.mode_data[self.mode_data > 0].min() / 2.
            else:
                m_min = self.m_min
            bin_size, _ = bin_estimate(self.mode_data, self.weights)
            if self.name in self.MINIMUM_STEP:
                bin_size = max(bin_size, self.MINIMUM_STEP[self.name])
            if bin_size < np.finfo(np.float32).eps:
                # In some (very ugly) cases h cannot be properly
                # determined...
                bin_count = 3
            else:
                bin_count = max(int((self.m_max - m_min) / bin_size) + 1, 3)
            bins = np.linspace(m_min, self.m_max, bin_count)
        return bins


    def process_mode(self):
        """
        Produce PDF representation for a given value of the mode.
        """
        bin_centers = None
        if len(self.mode_data) == 1 or \
                np.abs(self.m_min - self.m_max) < np.finfo(np.float32).eps:
            mode = self.mode_data[0]
            avg = self.mode_data[0]
            median = self.mode_data[0]
            err = 0.
            fit, par, kl_div = 'N', [], 1e10
        else:
            # Get a first guess on the number of bins needed
            bins = self.get_bin_count()
            median = quantile(self.mode_data, self.weights)
            avg, err = wstatistics(self.mode_data, self.weights, 2)
            err = self._update_err(err, avg, bins[1] - bins[0])
            if len(bins) <= 4 and self.name != 'age':
                # This happens sometimes, huh.
                mode = avg
                fit, par, kl_div = 'N', [], 1e10
            else:
                if self.name in ['distance_modulus', 'extinction'] and \
                        len(self.smooth) == 2:
                    hist, bin_centers = self._get_histogram_parts(bins)
                    avg, err = wstatistics(bin_centers, hist, 2)
                else:
                    hist, bin_centers = self._get_histogram(bins, self.mode_data,
                                                            self.weights, self.smooth)
                mode = bin_centers[np.argmax(hist)]
                if np.sum(hist > 0) < 4:
                    #  Less than 4 non-negative bins, impossible to fit.
                    mode = avg
                    fit, par, kl_div = 'N', [], 7e10
                else:
                    fit, par, kl_div = find_best_fit(bin_centers, hist,
                                                     avg, err)
                    if kl_div > 1e9:
                        # No fit converged.
                        fit = 'E'
        result = {'_mean': avg,
                  '_err': err,
                  '_mode': mode,
                  '_median': median,
                  '_fit': fit,
                  }
        if self.name == 'extinction':
            result['_zero'] = 1. - self.mode_data[self.mode_data > 0.].shape[0] \
                              / float(self.mode_data.shape[0])
        if self.dump and bin_centers is not None:
            result.update({'_bins_debug': bin_centers,
                           '_hist_debug': hist})
        if fit in 'GSTPLF':
            result.update(self._fix_fit_results(fit, par, bin_centers, hist))
        else:
            # If we have no fit, than report just mean and err.
            result.update(self._dummy_result(avg, err))
        if self.name in ['extinction', 'distance', 'parallax']:
            # These values cannot be negative...
            if result['_low_1sigma'] < 0.:
                result['_low_1sigma'] = 0.
            if result['_low_3sigma'] < 0.:
                result['_low_3sigma'] = 0.
        return {'%s%s' % (self.name, key): value
                for (key, value) in result.items()}

    def _fix_fit_results(self, fit, par, bin_centers, hist):
        result_par = np.array(list(par) + [0] * 5)[:5]
        if fit in 'TL':
            result_par[2] = self.m_min
            result_par[3] = self.m_max
        elif fit == 'P':
            result_par[3] = self.m_min
            result_par[4] = self.m_max
        if fit in 'GSTPL':
            result_par[1] = abs(result_par[1])
        # Find confidence intervals.
        sigma1 = to_borders(find_confidence(bin_centers, hist,
                                            ONE_SIGMA),
                            self.m_min, self.m_max)
        sigma3 = to_borders(find_confidence(bin_centers, hist,
                                            THREE_SIGMA),
                            self.m_min, self.m_max)
        return  {'_par': result_par,
                 '_low_1sigma': sigma1[0],
                 '_up_1sigma': sigma1[1],
                 '_low_3sigma': sigma3[0],
                 '_up_3sigma': sigma3[1]}

    def _dummy_result(self, avg, err):
        return {'_par': np.array([avg, err, 0., 0., 0.]),
                '_low_1sigma': avg - err,
                '_up_1sigma': avg + err,
                '_low_3sigma': avg - 3 * err,
                '_up_3sigma': avg + 3 * err}
