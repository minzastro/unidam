#!/usr/bin/env python
import os
import warnings
from collections import OrderedDict
from ConfigParser import ConfigParser
import numpy as np
import simplejson as json
from astropy.table import Table, Column
from astropy.io import fits
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter1d
from unidam.iso.model_fitter import model_fitter as mf  # pylint: disable=no-member
from unidam.iso.histogram_splitter import histogram_splitter
from unidam.utils.fit import find_best_fit
from unidam.utils.mathematics import wstatistics, quantile, bin_estimate, \
                                     to_borders, move_to_end
from unidam.utils.confidence import find_confidence, ONE_SIGMA, THREE_SIGMA
from unidam.utils.stats import to_bins, from_bins
from unidam.utils import constants
from unidam.utils.local import vargauss_filter1d


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    JSON encoder which supports numpy values.
    For debugging output.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_splitted(config, name):
    """
    Get array from config.
    """
    if not config.has_option('general', name):
        return []
    elif not config.get('general', name):
        return []
    else:
        return config.get('general', name).split(',')


class UniDAMTool(object):
    """
    Estimating distance and extinction
    from model fitting + 2MASS/AllWISE magnitudes.
    """
    DEFAULTS = {
        'distance_prior': '1',
        'use_magnitudes': '1',
        'fitted_columns': 'stage,age,mass,distance_modulus,extinction,distance,parallax',
        'max_param_err': '4',
        'parallax_known': '0',
        'allow_negative_extinction': '0',
        'dump_pdf': False
        }

    MIN_USPDF_WEIGHT = 0.03

    # These special distance-related columns are treated differently
    # by the FORTRAN module.
    SPECIAL = ['distance_modulus', 'extinction', 'distance', 'parallax']

    MINIMUM_STEP = {'age': 0.02,
                    'distance_modulus': 0.04,
                    'distance': 100.,
                    'extinction': 0.001,
                    'parallax': 0.00}

    def __init__(self, config_filename=None):
        self.mag = None
        self.mag_matrix = None
        self.mag_err = None
        self.mag_names = None
        self.abs_mag = None
        self.Rk = None
        self.param = None
        self.dump = False
        self.param_err = None
        self.id_column = None
        self.dump_pdf = False
        self.dump_pdf_file = None
        self.model_column_names = None
        self.config = {}
        config = ConfigParser()
        config.optionxform = str
        if config_filename is None:
            config_filename = os.path.join(os.path.dirname(__file__),
                                           'unidam.conf')
        if not os.path.exists(config_filename):
            raise Exception('Config file %s not found' % config_filename)
        config.read(config_filename)
        for key, value in self.DEFAULTS.iteritems():
            if not config.has_option('general', key):
                config.set('general', key, str(value))
        self.keep_columns = get_splitted(config, 'keep_columns')
        self.fitted_columns = get_splitted(config, 'fitted_columns')
        for item in self.SPECIAL:
            move_to_end(self.fitted_columns, item)
        # This array contains indices in the model table for input data
        self.model_columns = get_splitted(config, 'model_columns')
        self.default_bands = get_splitted(config, 'band_columns')
        # This is the index of column with output model weights.
        # In the output table (mf.model_data) there are columns for
        # fitted columns + columns for L_iso, L_sed and
        # at the very end - weight column
        self.w_column = len(self.fitted_columns) + 2
        self.dump_pdf = config.getboolean('general', 'dump_pdf')
        if self.dump_pdf:
            self.total_age_pdf = np.zeros(constants.AGE_RANGE.shape[0])
            self.total_2d_pdf = np.zeros((constants.DM_RANGE.shape[0],
                                          constants.AGE_RANGE.shape[0]))
        mf.max_param_err = config.getint('general', 'max_param_err')
        mf.use_model_weight = True
        mf.use_magnitude_probability = \
            config.getboolean('general', 'use_magnitudes')
        mf.distance_prior = config.getint('general', 'distance_prior')
        mf.allow_negative_extinction = config.getboolean(
            'general', 'allow_negative_extinction')
        for icolumn, column in enumerate(self.SPECIAL):
            mf.special_columns[icolumn] = column in self.fitted_columns
        mf.parallax_known = config.getboolean('general', 'parallax_known')
        if mf.parallax_known:
            self._update_config('parallax', config)
            self._update_config('extinction', config)
            self.MINIMUM_STEP['distance_modulus'] = 1e-5
        self.RK = {band: constants.R_FACTORS[band] / constants.R_FACTORS['K']
                   for band in constants.R_FACTORS}
        self.RV = {band: constants.R_FACTORS[band] / constants.R_FACTORS['V']
                   for band in constants.R_FACTORS}
        self._load_models(os.path.join(os.path.dirname(__file__),
                                       config.get('general', 'model_file')))

    def _names_to_indices(self, columns):
        """
        Convert a list of column names to name-index dictionary.
        """
        if self.model_column_names is None:
            raise Exception('Cannot convert names to indices - model file not loaded yet')
        result = []
        for name in columns:
            if name in self.model_column_names:
                result.append((name, self.model_column_names.index(name)))
            else:
                result.append((name, -1))
        return OrderedDict(result)

    def _update_config(self, name, config):
        """
        Transfer data from config file to local object config...
        """
        self.config[name] = config.get(name, 'column')
        self.config['%s_err' % name] = config.get(name, 'err_column')

    def _load_models(self, filename):
        """
        Load models from Numpy or fits file.
        Send them to FORTRAN module.
        """
        if not os.path.exists(filename):
            raise Exception('Model file %s is not found' % filename)
        table = fits.open(filename)
        self.model_data = None
        if os.path.exists(filename + '.npy'):
            if os.path.getmtime(filename + '.npy') > os.path.getmtime(filename):
                self.model_data = np.load(filename + '.npy')
        if self.model_data is None:
            self.model_data = np.asarray(table[1].data.tolist(), dtype=float)
            np.save(filename + '.npy', self.model_data)
        self.age_grid = np.asarray(table[2].data, dtype=float)
        self.model_column_names = [column.name for column in table[1].columns]
        self.fitted_columns = self._names_to_indices(self.fitted_columns)
        self.model_columns = self._names_to_indices(self.model_columns)
        self.default_bands = self._names_to_indices(self.default_bands)
        mf.alloc_models(self.model_data)

    def _apply_mask(self, mask):
        """
        Mask out missing magnitudes.
        """
        self.mag_names = self.mag_names[mask]
        self.mag = self.mag[mask]
        self.mag_err = self.mag_err[mask]
        self.abs_mag = self.abs_mag[mask]
        self.Rk = self.Rk[mask]

    def _validate_row(self, row_id):
        """
        Check if the row is good for processing.
        """
        if np.isnan(self.param).any():
            print('No spectral params for %s' % row_id)
            return {'id': row_id,
                    'error': 'No spectral params'}
        if np.any(self.param < -100.) or np.any(self.param_err <= 0):
            print('No spectral params or invalid params for %s' % row_id)
            return {'id': row_id,
                    'error': 'No spectral params or invalid params'}
        if self.mag.size == 0:
            print('No photometry for %s' % row_id)
            return {'id': row_id,
                    'error': 'No photometry'}
        return None

    def _push_to_fortran(self, row):
        """
        Push all data for FORTRAN module.
        """
        # This matrix is used to solve linear equations system for
        # distance modulus and extinctions.
        self.mag_matrix = [[np.sum(self.mag_err),
                            np.sum(self.mag_err * self.Rk)],
                           [np.sum(self.mag_err * self.Rk),
                            np.sum(self.mag_err * self.Rk * self.Rk)]]
        # Passing values to the module
        mf.matrix0 = self.mag_matrix
        mf.mask_models[:] = True
        if self.mag_err.size > 1:
            mf.matrix_det = 1. / np.linalg.det(self.mag_matrix)
        else:
            mf.matrix_det = 0.  # Will be unsused anyway
        mf.alloc_mag(self.mag, self.mag_err, self.Rk)
        mf.alloc_param(self.param, self.param_err)
        # Collect model-file column indices of fitted columns
        fitted = [item for item in self.fitted_columns.values() if item >= 0]
        mf.alloc_settings(self.abs_mag, self.model_columns.values(),
                          fitted)
        if mf.parallax_known:
            mf.parallax = row[self.config['parallax']]
            mf.parallax_error = row[self.config['parallax_err']]
            mf.extinction = row[self.config['extinction']]
            mf.extinction_error = row[self.config['extinction_err']]

    def get_estimates(self, row, dump=False):
        """
        Estimate distance and other parameters set in self.fitted_columns.
        """
        # Set maximum differnce between model and observation in units
        # of the observational error.
        self.dump = dump
        self.prepare_row(row)
        validate = self._validate_row(row[self.id_column])
        if validate is not None:
            return validate
        self._push_to_fortran(row)
        # HERE THINGS HAPPEN!
        m_count = mf.find_best()
        # Now deal with the result:
        if m_count == 0:
            print('No model fitting for %s' % row[self.id_column])
            return {'id': row[self.id_column],
                    'error': 'No model fitting'}
        model_params = mf.model_params[:m_count]
        stages = np.asarray(model_params[:, 0], dtype=int)
        uniq_stages = np.unique(stages)
        mode_weight = np.zeros(len(uniq_stages))
        for istage, stage in enumerate(uniq_stages):
            mode_weight[istage] = np.sum(model_params[stages == stage,
                                                      self.w_column])
        total_mode_weight = np.sum(mode_weight)
        if total_mode_weight == 0.:
            # Does this ever work?
            print('No model fitting (test) for %s' % row[self.id_column])
            return {'id': row[self.id_column],
                    'error': 'No model fitting'}
        try:
            mode_weight = mode_weight / total_mode_weight
        except ZeroDivisionError:
            print('Zero weight for %s' % row[self.id_column])
            return {'id': row[self.id_column],
                    'error': 'Zero weight'}
        # Setting best stage
        result = []
        for istage, stage in enumerate(uniq_stages):
            if mode_weight[istage] < self.MIN_USPDF_WEIGHT:
                # Ignore stages with a small weight
                continue
            stage_data = model_params[stages == stage]
            stage_data = stage_data[stage_data[:, self.w_column] > 0]
            # Split stage data into USPDFs
            for part_data in self.split_multimodal(stage_data):
                if np.sum(part_data[:, self.w_column]) / \
                   total_mode_weight < self.MIN_USPDF_WEIGHT:
                    # ignore USPDF with small weight
                    continue
                result.append(self.get_row(part_data, total_mode_weight))
        # Now enumerate USPDF priorities
        weights = [arow['uspdf_weight'] for arow in result]
        for ibest, best in enumerate(np.argsort(weights)[::-1]):
            result[best]['uspdf_priority'] = ibest
        for item in result:
            item.update({'total_uspdfs': len(result),
                         'id': row[self.id_column]})
            for keep in self.keep_columns:
                item[keep] = row[keep]
        # Sort by priority
        result.sort(key=lambda x: x['uspdf_priority'])
        result = self.assign_quality(result)
        if self.dump:
            # Store results into a json-file.
            self.dump_results(model_params, row, result)
            # Delete *debug* keys in the results
            # They are not to be exported to final fits table.
            for rrow in result:
                todel = []
                for key in rrow:
                    if key.endswith('debug'):
                        todel.append(key)
                for key in todel:
                    del rrow[key]
        return result

    def prepare_row(self, row):
        """
        Prepare magnitudes and spectral parameters
        for a given row.
        """
        self.mag_names = np.array(self.default_bands.keys(), dtype=str)
        self.mag = np.zeros(len(self.mag_names))
        self.mag_err = np.zeros(len(self.mag_names))
        for iband, band in enumerate(self.mag_names):
            self.mag[iband] = row['%smag' % band]
            # Storing the inverse uncertainty squared
            # for computational efficiency.
            self.mag_err[iband] = 1. / (row['e_%smag' % band])**2
        self.Rk = np.array([self.RK[band] for band in self.mag_names])
        self.abs_mag = np.array(self.default_bands.values(), dtype=int)
        # Filter out bad data:
        self._apply_mask(~(np.isnan(self.mag_err) + np.isnan(self.mag)))
        self.param = np.array([row[name] for name in self.model_columns])
        self.param_err = np.array([float(row['d%s' % name])
                                   for name in self.model_columns])

    def split_mass(self, stage_data):
        """
        Simplest approach to splitting histogram into peaks,
        separated by valleys.
        Special treatment for masses - PDF can contain a sharp peak.
        """
        if 'mass' not in self.fitted_columns:
            # We cannot fit in mass, if mass is not fitted for
            yield stage_data
            return
        mass_column = self.fitted_columns.keys().index('mass')
        # Make a histogram in log-masses
        xbins = np.logspace(np.log10(stage_data[:, mass_column].min() * 0.9),
                            np.log10(stage_data[:, mass_column].max() + 0.5),
                            20)
        if xbins[1] - xbins[0] < 0.05:
            # There is a low-mass-end
            spacing = np.log10(xbins[0] + 0.051) - np.log10(xbins[0])
            steps = int((np.log10(xbins[-1]) -
                         np.log10(xbins[0])) / spacing) + 1
            if steps > 10:
                xbins = np.logspace(np.log10(xbins[0]), np.log10(xbins[-1]),
                                    steps)
            else:
                xbins = np.arange(xbins[0], xbins[-1], 0.075)
        h = np.histogram(stage_data[:, mass_column], bins=xbins,
                         weights=stage_data[:, self.w_column])
        highest_positive_bin = np.nonzero(h[0])[0][-1]
        if highest_positive_bin < len(h[0]) - 1:
            # Right-hand side of the histogram goes to zero in some cases,
            # because weights might be too small.
            # In such case we remove low-weight rows from the sample.
            stage_data = stage_data[stage_data[:, mass_column] <=
                                    xbins[highest_positive_bin + 1]]
        for split in histogram_splitter(h[0], h[1]):
            yield stage_data[stage_data[:, mass_column] <= split]
            stage_data = stage_data[stage_data[:, mass_column] > split]
        # Do not forget to yield last item
        yield stage_data

    def split_other(self, stage_data, param='distance_modulus', dip=0.33):
        """
        Split in "other" params (distance modulus or age).
        """
        if param not in self.fitted_columns:
            #  We cannot split in parameter than is not fitted for.
            yield stage_data
            return
        split_column = self.fitted_columns.keys().index(param)
        if param == 'age':
            xbins = to_bins(self.age_grid)
            max_order = 10
        else:
            step, _ = bin_estimate(stage_data[:, split_column],
                                   stage_data[:, self.w_column])
            step = max(step, self.MINIMUM_STEP[param])
            xbins = np.arange(stage_data[:, split_column].min() - step * 0.5,
                              stage_data[:, split_column].max() + step * 1.5, step)
            max_order = 4
        histogram = np.histogram(stage_data[:, split_column],
                                 bins=xbins,
                                 weights=stage_data[:, self.w_column])
        for split in histogram_splitter(histogram[0], histogram[1],
                                        use_spikes=False, dip_depth=dip,
                                        max_order=max_order):
            yield stage_data[stage_data[:, split_column] <= split]
            stage_data = stage_data[stage_data[:, split_column] > split]
        # Do not forget to yield last item
        yield stage_data

    def split_multimodal(self, stage_data):
        """
        Simplest approach to splitting histogram into peaks,
        separated by valleys.
        """
        for part in self.split_mass(stage_data):
            for part3 in self.split_other(part, 'age'):
                yield part3

    def get_bin_count(self, name, mode_data, weights):
        """
        Estimate the number of bins needed to represent the data.
        """
        m_min = mode_data.min()
        m_max = mode_data.max()
        # We use different binning for different parameters.
        if name == 'mass':
            # Fixed step in mass, as masses are discrete
            # for some isochrones.
            bins = np.arange(m_min * 0.95, m_max * 1.1, 0.06)
        elif name == 'distance':
            # Fixed number of bins for distances
            bins = np.linspace(m_min * 0.95, m_max, 50)
        elif name == 'age':
            bins = to_bins(self.age_grid)
        else:
            if name == 'extinction':
                m_min = mode_data[mode_data > 0].min() / 2.
            h, _ = bin_estimate(mode_data, weights)
            if name in self.MINIMUM_STEP:
                h = max(h, self.MINIMUM_STEP[name])
            if h < np.finfo(np.float32).eps:
                # In some (very ugly) cases h cannot be properly
                # determined...
                bin_count = 3
            else:
                bin_count = int((m_max - m_min) / h) + 1
            bins = np.linspace(m_min, m_max, bin_count)
        return bins

    def _get_histogram(self, name, mode_data, weights, bins,
                       smooth):
        if name == 'age':
            # For ages we always use a fixed grid.
            bin_centers = self.age_grid
        else:
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # We need mode_data >= bins[0], otherwise numpy behaves
        # strangely sometimes.
        hist = np.histogram(mode_data[mode_data >= bins[0]]
                            , bins,
                            weights=weights[mode_data >= bins[0]]
                            )[0]
        hist = hist / (hist.sum() * (bins[1:] - bins[:-1]))
        if smooth is not None:
            if name in ['distance_modulus', 'extinction']:
                hist = gaussian_filter1d(
                    hist,
                    smooth / (bins[1] - bins[0]),
                    mode='constant')
            else:
                hist = vargauss_filter1d(bin_centers, hist, smooth)
        return hist, bin_centers

    def _get_histogram_parts(self, name, mode_data, weights, bins,
                             extinction_data, smooth):
        part1 = mode_data[extinction_data < mf.extinction]
        weight1 = weights[extinction_data < mf.extinction]
        bin_centers = from_bins(bins)
        hist = np.zeros_like(bin_centers)
        if len(part1) > 0:
            bin_left = bins.searchsorted(part1.min()) - 1
            if bin_left < 0:
                bin_left = 0
            bin_right = bins.searchsorted(part1.max()) + 1
            if bin_right - bin_left == 1:
                hist[bin_left] = weight1.sum()
            else:
                xbins = bins[bin_left:bin_right]
                hist1, _ = \
                    self._get_histogram(name, part1,
                                        weight1, xbins,
                                        smooth[0])
                hist[bin_left:bin_right - 1] = hist1
        part2 = mode_data[extinction_data >= mf.extinction]
        weight2 = weights[extinction_data >= mf.extinction]
        if len(part2) > 0:
            hist2, _ = \
                self._get_histogram(name, part2,
                                    weight2, bins, smooth[1])
            hist += hist2
        return hist, bin_centers

    def process_mode(self, name, mode_data, weights, smooth=None,
                     extinction_data=None):
        """
        Produce PDF representation for a given value of the mode.
        """
        m_min = mode_data.min()
        m_max = mode_data.max()
        bin_centers = None
        if len(mode_data) == 1 or \
                np.abs(m_min - m_max) < np.finfo(np.float32).eps:
            mode = mode_data[0]
            avg = mode_data[0]
            median = mode_data[0]
            err = 0.
            fit, par, kl_div = 'N', [], 1e10
        else:
            median = quantile(mode_data, weights)
            avg, err = wstatistics(mode_data, weights, 2)
            if smooth is not None:
                if name in ['distance_modulus', 'extinction']:
                    err = np.sqrt(err**2 + np.sum(smooth**2))
                else:
                    err = np.sqrt(err**2 + np.sum((avg * smooth)**2))
            if err == 0.:
                # This is done for the case of very low weights...
                # I guess it should be done otherwise, but...
                err = np.std(mode_data)
            elif err > 0.5 * (m_max - m_min):
                # This is for the case of very high smoothing values.
                err = 0.5 * (m_max - m_min)
            # Get a first guess on the number of bins needed
            bins = self.get_bin_count(name, mode_data, weights)
            if len(bins) <= 4 and name != 'age':
                # This happens sometimes, huh.
                mode = avg
                fit, par, kl_div = 'N', [], 1e10
            else:
                if name in ['distance_modulus', 'extinction'] and \
                    len(smooth) == 2:
                    hist, bin_centers = self._get_histogram_parts(
                        name, mode_data, weights, bins,
                        extinction_data, smooth)
                    avg, err = wstatistics(bin_centers, hist, 2)
                else:
                    hist, bin_centers = \
                        self._get_histogram(name, mode_data,
                                            weights, bins, smooth)
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
        if name == 'extinction':
            # Todo: fix!
            result['_zero'] = 1. - mode_data[mode_data > 0.].shape[0] \
                / float(mode_data.shape[0])
        if self.dump and bin_centers is not None:
            result.update({'_bins_debug': bin_centers,
                           '_hist_debug': hist})
        result_par = np.array(list(par) + [0] * 5)[:5]
        if fit in 'TL':
            result_par[2] = mode_data.min()
            result_par[3] = mode_data.max()
        elif fit == 'P':
            result_par[3] = mode_data.min()
            result_par[4] = mode_data.max()
        if fit in 'GSTPLF':
            if fit != 'F':
                result_par[1] = abs(result_par[1])
            # Find confidence intervals.
            sigma1 = to_borders(find_confidence(bin_centers, hist,
                                                ONE_SIGMA),
                                mode_data.min(), mode_data.max())
            sigma3 = to_borders(find_confidence(bin_centers, hist,
                                                THREE_SIGMA),
                                mode_data.min(), mode_data.max())
            result.update({'_low_1sigma': sigma1[0],
                           '_up_1sigma': sigma1[1],
                           '_low_3sigma': sigma3[0],
                           '_up_3sigma': sigma3[1]})
        else:
            # If we have no fit, than report just mean and err.
            result_par = np.array([avg, err, 0., 0., 0.])
            result.update({'_low_1sigma': result_par[0] - result_par[1],
                           '_up_1sigma': result_par[0] + result_par[1],
                           '_low_3sigma': result_par[0] - result_par[1] * 3,
                           '_up_3sigma': result_par[0] + result_par[1] * 3})
        result['_par'] = result_par
        if name in ['extinction', 'distance', 'parallax']:
            # These values cannot be negative...
            if result['_low_1sigma'] < 0.:
                result['_low_1sigma'] = 0.
            if result['_low_3sigma'] < 0.:
                result['_low_3sigma'] = 0.
        return {'%s%s' % (name, key): value
                for (key, value) in result.iteritems()}

    def get_correlations(self, first_parameter, second_parameter, adata):
        """
        Calculate correlations between parameter :name:
        and distance modulus.
        Use a linear fit, report slope, intercept and scatter.
        """
        dm_column = self.fitted_columns.keys().index(first_parameter)
        second_column = self.fitted_columns.keys().index(second_parameter)
        xdata = adata[:, dm_column]
        if second_parameter == 'mass':
            # Log-mass used, as it has linear relation to mu
            ydata = np.log10(adata[:, second_column])
        else:
            ydata = adata[:, second_column]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.RankWarning)
            fit = np.polyfit(xdata, ydata, 1, w=adata[:, self.w_column])
        residuals = xdata * fit[0] + fit[1] - ydata
        if first_parameter == 'distance_modulus':
            name1 = 'dm'
        else:
            name1 = first_parameter
        if second_parameter == 'distance_modulus':
            name2 = 'dm'
        else:
            name2 = second_parameter
        name = '%s_%s' % (name1, name2)
        return {'%s_slope' % name: fit[0],
                '%s_intercept' % name: fit[1],
                '%s_mad' % name: quantile(np.abs(residuals),
                                          adata[:, self.w_column]),
                '%s_scatter' % name: wstatistics(residuals,
                                                 adata[:, self.w_column],
                                                 moments=2)[1]}

    def add_to_pdf(self, xdata, mode_weight):
        """
        Collecting total PDFs in age and age-distance modulus.
        """
        age_col = self.fitted_columns.keys().index('age')
        age_histogram = np.histogram(xdata[:, age_col],
                                     to_bins(constants.AGE_RANGE),
                                     weights=xdata[:, self.w_column],
                                     normed=True)[0]
        age_histogram = age_histogram * mode_weight / age_histogram.sum()
        age_histogram[np.isnan(age_histogram)] = 0.
        self.total_age_pdf += age_histogram
        if 'distance_modulus' in self.fitted_columns:
            dm_col = self.fitted_columns.keys().index('distance_modulus')
            two_histogram = np.histogram2d(
                xdata[:, dm_col], xdata[:, age_col],
                (to_bins(constants.DM_RANGE), to_bins(constants.AGE_RANGE)),
                weights=xdata[:, self.w_column], normed=True)[0]
            two_histogram = two_histogram * mode_weight / two_histogram.sum()
            two_histogram[np.isnan(two_histogram)] = 0.
            self.total_2d_pdf += two_histogram

    def dump_sed(self, adata):
        """
        Store Spectral Energy Distribution (SED) to result dictionary.
        SED is calculated as a weighted mean (with scatter) for predicted
        visible magnitudes.
        """
        mdata = mf.models[np.asarray(adata[:, self.w_column + 1] - 1, dtype=int)]
        dm = adata[:, self.fitted_columns.keys().index('distance_modulus')]
        ext = adata[:, self.fitted_columns.keys().index('extinction')]
        weight = adata[:, self.w_column]
        sed_dict = {'Predicted': {}, 'PredErr': {},
                    'Observed': {}, 'ObsErr': {}}
        for iband, band in enumerate(self.mag_names):
            sed_dict['Predicted'][band], sed_dict['PredErr'][band] = \
                wstatistics(mdata[:, self.default_bands[band]] +
                            dm + self.RK[band] * ext,
                            weight, 2)
            sed_dict['Observed'][band] = self.mag[iband]
            sed_dict['ObsErr'][band] = 1./np.sqrt(self.mag_err[iband])
        return sed_dict

    def get_row(self, xdata, wtotal):
        """
        Prepare output row for the selection of models.
        """
        dof = len(self.mag)
        if self.mag_err.size > 1:
            # Calculating smoothing parameters from the inverse
            # Hessian matrix
            covariance = np.linalg.inv(self.mag_matrix)
            smooth_distance = np.sqrt(covariance[0, 0])
            smooth_extinction = np.sqrt(covariance[1, 1])
        else:
            # Only one magnitude - calculate smoothing
            # from magnitude uncertainties directly
            smooth_distance = np.sqrt(1. / self.mag_err[0])
            smooth_extinction = np.sqrt(1. / self.mag_err[0]) / self.Rk[0]
        if mf.parallax_known:
            hess_matrix = np.copy(self.mag_matrix)
            hess_matrix[0, 0] += 0.212 * mf.parallax**2 / mf.parallax_error**2
            #hess_matrix[1, 1] += 1./mf.extinction_error ** 2
            # Magic constant 0.212 is (0.2 log(10))**2
            covariance = np.linalg.inv(hess_matrix)
            hess_matrix[1, 1] += 1./mf.extinction_error ** 2
            covariance2 = np.linalg.inv(hess_matrix)
            smooth_distance = [np.sqrt(covariance[0, 0]),
                               np.sqrt(covariance2[0, 0])]
            smooth_extinction = [np.sqrt(covariance[1, 1]),
                                 np.sqrt(covariance2[1, 1])]
            dof += 2
        smooth_distance = np.atleast_1d(smooth_distance)
        smooth_extinction = np.atleast_1d(smooth_extinction)
        best_model = np.argmin(xdata[:, self.w_column - 1] +
                               xdata[:, self.w_column - 2])
        l_sed = xdata[best_model, self.w_column - 1]
        l_best = l_sed + xdata[best_model, self.w_column - 2]
        new_result = {'stage': xdata[0, 0],
                      'uspdf_points': xdata.shape[0],
                      'uspdf_weight': np.sum(xdata[:, self.w_column]) / wtotal,
                      'p_best': 1. - chi2.cdf(2. * l_best, dof + 3),
                      'p_sed': 1. - chi2.cdf(2. * l_sed, dof)
                     }
        if self.dump:
            new_result['sed_debug'] = self.dump_sed(xdata)
        for ikey, key in enumerate(self.fitted_columns.keys()):
            extinction_if_needed = None
            if key == 'stage':
                continue
            elif key == 'distance_modulus':
                new_result['distance_modulus_smooth'] = smooth_distance[0]
                smooth = smooth_distance
                if len(smooth) == 2:
                    extinction_if_needed = xdata[:, self.fitted_columns.keys().index('extinction')]
            elif key == 'extinction':
                new_result['extinction_smooth'] = smooth_extinction[0]
                smooth = smooth_extinction
                if len(smooth) == 2:
                    extinction_if_needed = xdata[:, ikey]
            elif key == 'distance' or key == 'parallax':
                smooth = 0.2 * np.log(10.) * smooth_distance[0]
            else:
                smooth = None
            new_result.update(self.process_mode(key,
                                                xdata[:, ikey],
                                                xdata[:, self.w_column],
                                                smooth,
                                                extinction_if_needed))
        if self.dump_pdf and 'age' in self.fitted_columns:
            self.add_to_pdf(xdata, new_result['uspdf_weight'])
        if len(xdata) > 3 and 'distance_modulus' in self.fitted_columns:
            if 'age' in self.fitted_columns:
                new_result.update(self.get_correlations('distance_modulus',
                                                        'age', xdata))
                new_result.update(self.get_correlations('age',
                                                        'distance_modulus',
                                                        xdata))
            if 'mass' in self.fitted_columns:
                new_result.update(self.get_correlations('distance_modulus',
                                                        'mass', xdata))
        return new_result

    @classmethod
    def assign_quality(cls, results):
        """
        Qulity flag assignment:
        1 - Single mode;
        A - best mode has power of 0.9 or more;
        B - 1st and 2nd best modes together have power of 0.9 or more;
        C - 1st, 2nd and 3rd best modes together have power of 0.9 or more;
        D - 1st, 2nd and 3rd best modes together have power less than 0.9;
        L - low power mode (below 0.1);
        E - Mode has p_sed < 0.1 (possibly bad photometry);
        N - Mode has less than 10 model points (unreliable result);
        X - Best mode has p_best < 0.1 (likely off the model grid);
        """
        if len(results) == 1:
            results[0]['quality'] = '1'
        else:
            for result in results:
                # Setting default value
                if result['uspdf_weight'] > 0.1:
                    result['quality'] = 'D'
                else:
                    result['quality'] = 'L'
            wcum = np.cumsum([res['uspdf_weight'] for res in results])
            if wcum[0] > 0.9:
                results[0]['quality'] = 'A'
            elif len(results) >= 2 and wcum[1] > 0.9:
                results[0]['quality'] = 'B'
                results[1]['quality'] = 'B'
            elif len(results) >= 3 and wcum[2] > 0.9:
                results[0]['quality'] = 'C'
                results[1]['quality'] = 'C'
                results[2]['quality'] = 'C'
        for result in results:
            if result['p_best'] < 0.1:
                result['quality'] = 'X'
            elif result['p_sed'] < 0.1:
                result['quality'] = 'E'
            elif result['uspdf_points'] < 10:
                result['quality'] = 'N'
        return results

    def dump_results(self, model_params, row, result):
        """
        Save results into a dat/json files.
        """
        idstr = str(row[self.id_column]).strip()
        dump_array = range(self.w_column + 1)
        header = '%s L_iso L_sed p_w' % ' '.join(self.fitted_columns.keys())
        np.savetxt('dump/dump_%s.dat' % idstr,
                   model_params[:, dump_array],
                   header=header)
        json.dump(result,
                  open('dump/dump_%s.json' % idstr, 'w'),
                  indent=2, cls=NumpyAwareJSONEncoder)

    def get_table(self, data, idtype=str):
        """
        Prepare output table object.
        """
        ucd = {'age': 'time.age',
               'mass': 'phys.mass',
               'distance': 'pos.distance',
               'distance_modulus': 'phot.mag.distMod',
               'T': 'phys.temperature',
               'parallax': 'pos.parallax',
               'extinction': 'phys.absorption.gal'}

        def float_column(unit=None):
            if unit is None:
                return Column(dtype=float)
            return Column(dtype=float, unit=unit)

        columns = OrderedDict([
            ('id', Column(dtype=idtype)),
            ('uspdf_priority', Column(dtype=int)),
            ('uspdf_points', Column(dtype=int)),
            ('stage', Column(dtype=int)),
            ('uspdf_weight', float_column()),
            ('total_uspdfs', Column(dtype=int)),
            ('quality', Column(dtype='S1')),
            ('p_best', float_column()),
            ('p_sed', float_column())])
        final = Table()
        for key, value in columns.iteritems():
            final[key] = value
        units = {'age': 'Gyr', 'mass': 'MSun',
                 'distance_modulus': 'mag',
                 'extinction': 'mag',
                 'distance': 'kpc',
                 'T': 'K',
                 'parallax': 'mas'}
        for key in self.fitted_columns.iterkeys():
            if key in ['stage']:
                continue
            if key in units:
                unit = units[key]
                meta = ucd[key]
            else:
                unit, meta = '', ''
            for suffix in ['_mean', '_err', '_mode', '_median',
                           '_low_1sigma', '_up_1sigma',
                           '_low_3sigma', '_up_3sigma']:
                final['%s%s' % (key, suffix)] = float_column(unit=unit)
            final['%s_mean' % key].meta['ucd'] = meta
            final['%s_fit' % key] = Column(dtype='S1')
            final['%s_par' % key] = Column(dtype=float, shape=(5))
        if 'distance_modulus' in self.fitted_columns:
            for key in ['dm_age', 'age_dm', 'dm_mass']:
                final['%s_slope' % key] = float_column()
                final['%s_intercept' % key] = float_column()
                final['%s_scatter' % key] = float_column()
                final['%s_mad' % key] = float_column()
            final['distance_modulus_smooth'] = float_column(unit='mag')
        if 'extinction' in self.fitted_columns:
            final['extinction_smooth'] = float_column(unit='mag')
            final['extinction_zero'] = float_column(unit='fraction')
        for keep in self.keep_columns:
            final[keep] = Column(dtype=data[keep].dtype)
        return final

# DO NOT MAKE main() function here: will cause problems for parallel...
