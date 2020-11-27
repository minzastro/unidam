#!/usr/bin/env python
import os
import warnings
from collections import OrderedDict
from configparser import ConfigParser
import numpy as np
import simplejson as json
from astropy.table import Table, Column
from astropy.io import fits
from scipy.stats import chi2, norm, truncnorm
from unidam.core.histogram_analyzer import HistogramAnalyzer
from unidam.core.model_fitter import model_fitter as mf  # pylint: disable=no-member
from unidam.core.histogram_splitter import histogram_splitter

from unidam.utils.mathematics import wstatistics, quantile, bin_estimate, \
    to_borders, move_to_end, move_to_beginning
from unidam.utils.stats import to_bins, from_bins
from unidam.utils import constants


def ensure_dir(directory):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


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


def get_modified_chi2(offset, dof, sum_of_squares):
    """
    Chi-squared analog for truncated multivariate
    Gaussian.
    """
    random = [truncnorm.rvs(a=offset, b=np.inf, size=10000)]
    for _ in range(dof - 1):
        random.append(norm.rvs(size=10000))
    chi2_values = np.sum(np.array(random) ** 2, axis=0)
    return np.sum(chi2_values < sum_of_squares) * 1e-4


class UniDAMTool():
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
        'dump_pdf': False,
        'dump_prefix': 'dump',
        'save_sed': False
    }

    MIN_USPDF_WEIGHT = 0.03

    # These special distance-related columns are treated differently
    # by the FORTRAN module.
    SPECIAL = ['distance_modulus', 'extinction', 'distance', 'parallax']

    RK = {band: constants.R_FACTORS[band] / constants.R_FACTORS['K']
          for band in constants.R_FACTORS}
    RV = {band: constants.R_FACTORS[band] / constants.R_FACTORS['V']
          for band in constants.R_FACTORS}

    def __init__(self, config_filename=None, config_override=None):
        self.mag = None
        self.mag_matrix = None
        self.mag_err = None
        self.mag_names = None
        self.abs_mag = None
        self.Rk = None
        self.param = None
        self.param_err = None
        self.id_column = None
        self.model_column_names = None
        self.config = {'dump': False}
        config = ConfigParser()
        config.optionxform = str
        if config_filename is None:
            config_filename = os.path.join(os.path.dirname(__file__),
                                           'unidam.conf')
        if not os.path.exists(config_filename):
            raise Exception('Config file %s not found' % config_filename)
        config.read(config_filename)
        if config_override is not None:
            for key, value in config_override.items():
                if '.' in key:
                    group, key = key.split('.')
                else:
                    group = 'general'
                config.set(group, key, str(value))
        for key, value in self.DEFAULTS.items():
            if not config.has_option('general', key):
                config.set('general', key, str(value))
        self.keep_columns = get_splitted(config, 'keep_columns')
        self.fitted_columns = get_splitted(config, 'fitted_columns')
        # This array contains indices in the model table for input data
        self.model_columns = get_splitted(config, 'model_columns')
        self.default_bands = get_splitted(config, 'band_columns')
        if len(self.default_bands) == 0:
            mf.use_photometry = False
        mf.parallax_known = config.getboolean('general', 'parallax_known')
        if mf.parallax_known:
            if not mf.use_photometry:
                raise ValueError('Cannot use parallaxes without'
                                 'photometry. Please set band_columns'
                                 'parameter in config')
            self._update_config('parallax', config)
            self._update_config('extinction', config)
            HistogramAnalyzer.MINIMUM_STEP['distance_modulus'] = 1e-5
        for item in self.SPECIAL:
            # Special columns should appear at the end
            # of fitted_columns list, and in the prescribed order.
            move_to_end(self.fitted_columns, item)
            if item in self.fitted_columns and not mf.use_photometry:
                raise ValueError('Distance-related output is requested, but'
                                 ' no photometry provided.')
        if 'stage' not in self.fitted_columns:
            self.fitted_columns.insert(0, 'stage')
        else:
            move_to_beginning(self.fitted_columns, 'stage')
            # raise ValueError('stage column has to be in the list of fitted columns')
        # This is the index of column with output model weights.
        # In the output table (mf.model_data) there are columns for
        # fitted columns + columns for L_iso, L_sed and
        # at the very end - weight column
        self.w_column = len(self.fitted_columns) + 2
        mf.max_param_err = config.getint('general', 'max_param_err')
        mf.use_model_weight = True
        mf.use_magnitude_probability = \
            config.getboolean('general', 'use_magnitudes')
        mf.distance_prior = config.getint('general', 'distance_prior')
        mf.allow_negative_extinction = config.getboolean(
            'general', 'allow_negative_extinction')
        for icolumn, column in enumerate(self.SPECIAL):
            mf.special_columns[icolumn] = column in self.fitted_columns
        self.config['dump_pdf'] = config.getboolean('general', 'dump_pdf')
        self.config['dump_prefix'] = config.get('general', 'dump_prefix')
        self.config['save_sed'] = config.getboolean('general', 'save_sed')
        if self.config['dump_pdf']:
            self.total_age_pdf = np.zeros(constants.AGE_RANGE.shape[0])
            self.total_2d_pdf = np.zeros((constants.DM_RANGE.shape[0],
                                          constants.AGE_RANGE.shape[0]))
        model_file = config.get('general', 'model_file')
        if not (os.path.isabs(model_file) or
                model_file.startswith('./')):
            model_file = os.path.join(os.path.dirname(__file__),
                                      model_file)
        self._load_models(model_file)

    def _names_to_indices(self, columns, validate=False):
        """
        Convert a list of column names to name-index dictionary.
        """
        if self.model_column_names is None:
            raise Exception('Cannot convert names to indices - model file not loaded yet')
        result = []
        for name in columns:
            if name in self.model_column_names:
                result.append((name, self.model_column_names.index(name)))
            elif validate and name not in self.SPECIAL:
                raise ValueError("%s column is not in the model, cannot fit" % name)
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
        print('Opening FITS')
        table = fits.open(filename)
        self.model_data = None
        if os.path.exists(filename + '.npy') and os.path.getmtime(filename + '.npy') > os.path.getmtime(filename):
                print('Opening npy')
                self.model_data = np.load(filename + '.npy')
        if self.model_data is None:
            print('Converting data to nparray')
            v = table[1].data
            self.model_data = np.empty((v.shape[0], len(v[0])))
            for i in range(len(v[0])):
                name = v.columns[i].name
                self.model_data[:, i] = v[name]
            print('Saving...')
            np.save(filename + '.npy', self.model_data)
        self.model_data = np.hstack(
            (self.model_data[:, :-1],
             np.arange(len(self.model_data))[:, np.newaxis],
             self.model_data[:, -1][:, np.newaxis]))
        self.age_grid = np.asarray(table[2].data, dtype=float)
        self.model_column_names = [column.name for column in table[1].columns]
        self.fitted_columns = self._names_to_indices(self.fitted_columns, validate=True)
        self.model_columns = self._names_to_indices(self.model_columns, validate=True)
        self.default_bands = self._names_to_indices(self.default_bands)
        self.mag_names = np.array(list(self.default_bands.keys()), dtype=str)
        print('Pass to F90')
        mf.alloc_models(self.model_data)
        print('Ready')

    def _apply_mask(self, mask):
        """
        Mask out missing magnitudes.
        """
        self.mag_names = self.mag_names[mask]
        self.mag = self.mag[mask]
        self.mag_err = self.mag_err[mask]
        self.abs_mag = self.abs_mag[mask]
        self.Rk = self.Rk[mask]

    def _validate_star(self, row_id):
        """
        Check if the row is good for processing.
        """
        if np.isnan(self.param).any():
            print(('No spectral params for %s' % row_id))
            return {'id': row_id,
                    'error': 'No spectral params'}
        if np.any(self.param < -100.) or np.any(self.param_err <= 0):
            print(('No spectral params or invalid params for %s' % row_id))
            return {'id': row_id,
                    'error': 'No spectral params or invalid params'}
        if self.mag.size == 0 and mf.use_photometry:
            print(('No photometry for a %s' % row_id))
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
        if self.mag.size > 0:
            mf.alloc_mag(self.mag, self.mag_err, self.Rk)
        if len(self.param) > 0:
            mf.alloc_param(self.param, self.param_err)
        # Collect model-file column indices of fitted columns
        fitted = [item for item in list(self.fitted_columns.values()) if item >= 0]
        mf.alloc_settings(self.abs_mag, list(self.model_columns.values()),
                          fitted)
        if mf.parallax_known:
            mf.parallax = row[self.config['parallax']]
            mf.parallax_error = row[self.config['parallax_err']]
            mf.parallax_l_correction = np.log(norm.cdf(mf.parallax / mf.parallax_error))
            mf.extinction = row[self.config['extinction']]
            mf.extinction_error = row[self.config['extinction_err']]

    def get_fitting_models(self, row):
        mask = np.ones(len(self.model_data), dtype=bool)
        for param, param_err, model in zip(self.param, self.param_err,
                                           self.model_columns.values()):
            mask *= np.abs(self.model_data[:, model] - param) \
                    <= (mf.max_param_err * param_err)
        if mask.sum() == 0:
            print('No model fitting for %s' % row[self.id_column])
            return None, None
        xsize = len(self.fitted_columns) + 4
        model_params = np.zeros((mask.sum(), xsize))
        special_params = np.zeros((mask.sum(), 5))
        for i, model in enumerate(self.model_data[mask]):
            success, model_params[i], special_params[i] = \
                mf.process_model(i, model, xsize)
            if not success:
                model_params[i, -2] = 0.
            else:
                model_params[i, -1] = i
        if 0 < (model_params[:, -2] > 0).sum() < 50:
            print('Adding more models for %s' % row[self.id_column])
            max_id = model_params.max() + 1
            # Add intermediate models
            ind = np.arange(len(self.model_data), dtype=int)[mask][model_params[:, -2] > 0]
            new_models = []
            new_special = []
            for ii in ind:
                m1 = self.model_data[ii]
                for offset in [1, -1]:
                    m2 = self.model_data[ii + offset]
                    if ii + offset in ind:
                        t_current = 0.5
                    else:
                        t_current = 1.
                        for param, param_err, model in zip(self.param, self.param_err,
                                                           self.model_columns.values()):
                            v1 = m1[model]
                            v2 = m2[model]
                            if v2 > v1:
                                t_current = min(t_current,
                                                np.abs(param + mf.max_param_err * param_err - v1) / (v2 - v1))
                            elif v1 > v2:  # If v1 == v2 then t is not updated.
                                t_current = min(t_current,
                                                np.abs(param - mf.max_param_err * param_err - v1) / (v1 - v2))
                    extra_models = m1 + np.linspace(0, t_current, 50)[:, np.newaxis] * (m2 - m1)
                    # This is an extra fix for the stage column.
                    extra_models[:, 0] = m1[0]
                    for model in extra_models:
                        res = mf.process_model(-1, model, xsize)
                        if res[0] > 0:
                            new_models.append(res[1])
                            new_special.append(res[2])
            model_params = np.atleast_2d(new_models)
            special_params = np.atleast_2d(new_special)
            model_params[:, -1] = np.arange(len(model_params))
            special_params[:, -1] = np.arange(len(model_params))
        return model_params[model_params[:, -2] > 0], \
               special_params[model_params[:, -2] > 0]

    def get_mode_weights(self, row, model_params):
        stages = np.asarray(model_params[:, 0], dtype=int)
        uniq_stages = np.unique(stages)
        mode_weight = np.zeros(len(uniq_stages))
        for istage, stage in enumerate(uniq_stages):
            mode_weight[istage] = np.sum(model_params[stages == stage,
                                                      self.w_column])
        total_mode_weight = np.sum(mode_weight)
        if total_mode_weight == 0.:
            # Does this ever work?
            print(('No model fitting (test) for %s' % row[self.id_column]))
            return {'id': row[self.id_column],
                    'error': 'No model fitting'}
        try:
            mode_weight = mode_weight / total_mode_weight
        except ZeroDivisionError:
            print(('Zero weight for %s' % row[self.id_column]))
            return {'id': row[self.id_column],
                    'error': 'Zero weight'}
        result = {}
        for istage, stage in enumerate(uniq_stages):
            result[stage] = mode_weight[istage]
        return result

    def process_star(self, row, dump=False):
        """
        Estimate distance and other parameters set in self.fitted_columns.
        """
        # Set maximum differnce between model and observation in units
        # of the observational error.
        self.config['dump'] = dump
        self.prepare_star(row)
        validate = self._validate_star(row[self.id_column])
        if validate is not None:
            return validate
        self._push_to_fortran(row)
        if np.isinf(mf.parallax_l_correction):
            return {'id': row[self.id_column],
                    'error': 'Parallax is too negative'}
        # HERE THINGS HAPPEN!
        model_params, model_special = self.get_fitting_models(row)
        if model_params is None or len(model_params) == 0:
            return {'id': row[self.id_column],
                    'error': 'No model fitting'}
        stage_weights = self.get_mode_weights(row, model_params)
        # Setting best stage
        result = []
        for part_weight, part_data, part_special in self.data_splitter(
                stage_weights, model_params, model_special):
            result.append(self.get_row(part_data, part_special,
                                       part_weight))
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
        if self.config['dump']:
            # Store results into a json-file.
            result = self.dump_results(model_params, row, result)
        return result

    def data_splitter(self, stage_weights, model_params, model_special):
        stages = np.asarray(model_params[:, 0], dtype=int)
        for stage, mode_weight in stage_weights.items():
            if mode_weight < self.MIN_USPDF_WEIGHT:
                # Ignore stages with a small weight
                continue
            stage_data = model_params[stages == stage]
            stage_data = stage_data[stage_data[:, self.w_column] > 0]
            current_stage_weight = np.sum(stage_data[:, self.w_column])
            # Split stage data into USPDFs
            for part_data in self.split_multimodal(stage_data):
                part_weight = mode_weight * \
                              (np.sum(part_data[:, self.w_column]) / current_stage_weight)
                if part_weight < self.MIN_USPDF_WEIGHT:
                    # ignore USPDF with small weight
                    continue
                part_model_ids = part_data[:, -1]
                part_special = np.in1d(model_special[:, -1], part_model_ids)
                yield part_weight, part_data, model_special[part_special]

    def prepare_star(self, row):
        """
        Prepare magnitudes and spectral parameters
        for a given row.
        """
        self.mag_names = np.array(list(self.default_bands), dtype=str)
        self.mag = np.zeros(len(self.default_bands))
        self.mag_err = np.zeros_like(self.mag)
        for iband, band in enumerate(self.default_bands):
            self.mag[iband] = row['%smag' % band]
            # Storing the inverse uncertainty squared
            # for computational efficiency.
            if np.abs(self.mag[iband]) > 50 or row['e_%smag' % band] > 50  or row['e_%smag' % band] < 1e-4:
                self.mag[iband] = np.nan
                self.mag_err[iband] = np.nan
            else:
                self.mag_err[iband] = 1. / (row['e_%smag' % band]) ** 2
        self.Rk = np.array([self.RK[band] for band in self.default_bands.keys()])
        self.abs_mag = np.array(list(self.default_bands.values()), dtype=int)
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
        mass_column = list(self.fitted_columns.keys()).index('mass')
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
        split_column = list(self.fitted_columns.keys()).index(param)
        if param == 'age':
            xbins = to_bins(self.age_grid)
            max_order = 10
        else:
            step, _ = bin_estimate(stage_data[:, split_column],
                                   stage_data[:, self.w_column])
            step = max(step, HistogramAnalyzer.MINIMUM_STEP[param])
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

    def get_correlations(self, first_parameter, second_parameter, adata):
        """
        Calculate correlations between parameter :name:
        and distance modulus.
        Use a linear fit, report slope, intercept and scatter.
        """
        dm_column = list(self.fitted_columns.keys()).index(first_parameter)
        second_column = list(self.fitted_columns.keys()).index(second_parameter)
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
        age_col = list(self.fitted_columns.keys()).index('age')
        age_histogram = np.histogram(xdata[:, age_col],
                                     to_bins(constants.AGE_RANGE),
                                     weights=xdata[:, self.w_column],
                                     density=True)[0]
        age_histogram = age_histogram * mode_weight / age_histogram.sum()
        age_histogram[np.isnan(age_histogram)] = 0.
        self.total_age_pdf += age_histogram
        if 'distance_modulus' in self.fitted_columns:
            dm_col = list(self.fitted_columns.keys()).index('distance_modulus')
            two_histogram = np.histogram2d(
                xdata[:, dm_col], xdata[:, age_col],
                (to_bins(constants.DM_RANGE), to_bins(constants.AGE_RANGE)),
                weights=xdata[:, self.w_column], density=True)[0]
            two_histogram = two_histogram * mode_weight / two_histogram.sum()
            two_histogram[np.isnan(two_histogram)] = 0.
            self.total_2d_pdf += two_histogram

    def dump_sed(self, adata):
        """
        Store Spectral Energy Distribution (SED) to result dictionary.
        SED is calculated as a weighted mean (with scatter) for predicted
        visible magnitudes.
        """
        if 'distance_modulus' not in self.fitted_columns:
            raise ValueError('Cannot produce SED -- no distance estimate')
        if 'extinction' not in self.fitted_columns:
            raise ValueError('Cannot produce SED -- no extinction estimate')
        sed_dict = {'Predicted': {}, 'PredErr': {},
                    'Observed': {}, 'ObsErr': {}}
        if np.any(adata[:, self.w_column + 1] >= len(mf.models)) or \
                np.any(adata[:, self.w_column + 1] < 0):
            print("Cannot do anything for added models...so far")
            return sed_dict
        mdata = mf.models[np.asarray(adata[:, self.w_column + 1] - 1, dtype=int)]
        dm = adata[:, list(self.fitted_columns.keys()).index('distance_modulus')]
        ext = adata[:, list(self.fitted_columns.keys()).index('extinction')]
        weight = adata[:, self.w_column]
        for iband, band in enumerate(self.mag_names):
            sed_dict['Predicted'][band], sed_dict['PredErr'][band] = \
                wstatistics(mdata[:, self.default_bands[band]] +
                            dm + self.RK[band] * ext,
                            weight, 2)
            sed_dict['Observed'][band] = self.mag[iband]
            sed_dict['ObsErr'][band] = 1. / np.sqrt(self.mag_err[iband])
        return sed_dict

    def get_psed_pbest(self, xdata, dof):
        """
        Calculate values related to best-fitting model
        (best fitting = model with highest weight).
        Note that the best-fitting model is not neccesserily the one
        with the lowest log-likelihood.
        """
        best_model = np.argmin(xdata[:, self.w_column - 1] +
                               xdata[:, self.w_column - 2])
        l_sed = xdata[best_model, self.w_column - 1]
        l_best = l_sed + xdata[best_model, self.w_column - 2]
        if mf.parallax_known:
            fracpar = -mf.parallax / mf.parallax_error
            if fracpar > -10:
                return {'p_sed': 1. - get_modified_chi2(fracpar, dof,
                                                        2. * (l_sed - mf.parallax_l_correction)),
                        'p_best': 1. - get_modified_chi2(fracpar, dof + len(self.model_columns),
                                                         2. * (l_best - mf.parallax_l_correction)),
                        }
            else:
                return {'p_sed': 1. - chi2.cdf(2. * l_sed, dof),
                        'p_best': 1. - chi2.cdf(2. * l_best, dof + len(self.model_columns))}
        else:
            return {'p_sed': 1. - chi2.cdf(2. * l_sed, dof),
                    'p_best': 1. - chi2.cdf(2. * l_best, dof + len(self.model_columns))}

    def get_row(self, xdata, xspecial, xweight):
        """
        Prepare output row for the selection of models.
        """
        dof = len(self.mag)
        if mf.parallax_known and self.mag.size > 0:
            # If the parallax is known, we have to modify the Hessian,
            # because L_sed now includes new term for parallax prior
            hess_matrix = np.copy(self.mag_matrix)
            if self.mag_err.size > 1 or abs(mf.parallax) > 0:
                hess_matrix[0, 0] += 0.212 * mf.parallax ** 2 / mf.parallax_error ** 2
                # Magic constant 0.212 is (0.2 log(10))**2
                covariance = np.linalg.inv(hess_matrix)
            else:
                covariance = np.zeros((2, 2))
                covariance[0][0] = np.sqrt(1. / self.mag_err[0])
                covariance[1][1] = np.sqrt(1. / self.mag_err[0]) / self.Rk[0]
            # For models with extinction > A_0 we need also to modify
            # H_1,1 to account for extinction term in L_sed
            hess_matrix[1, 1] += 1. / mf.extinction_error ** 2
            covariance2 = np.linalg.inv(hess_matrix)
            smooth_distance = [np.sqrt(covariance[0, 0]),
                               np.sqrt(covariance2[0, 0])]
            smooth_extinction = [np.sqrt(covariance[1, 1]),
                                 np.sqrt(covariance2[1, 1])]
            # ...and we have two more degrees of freedom
            # (extinction and parallax)
            dof += 2
        elif self.mag_err.size == 0:
            smooth_distance = 0
            smooth_extinction = 0
        elif self.mag_err.size > 1:
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

        smooth_distance = np.atleast_1d(smooth_distance)
        smooth_extinction = np.atleast_1d(smooth_extinction)
        new_result = {'stage': xdata[0, 0],
                      'uspdf_points': xdata.shape[0],
                      'uspdf_weight': xweight,
                      }
        new_result.update(self.get_psed_pbest(xdata, dof))
        if (self.config['dump'] or self.config['save_sed']) and \
                self.mag.size > 0:
            sed = self.dump_sed(xdata)
            if self.config['dump']:
                new_result['sed_debug'] = sed
            if self.config['save_sed']:
                for band in self.mag_names:
                    new_result['sed_%s' % band] = \
                        sed['Observed'][band] - sed['Predicted'][band]
                    new_result['sed_%s_relative' % band] = \
                        new_result['sed_%s' % band] / sed['ObsErr'][band]
        for ikey, key in enumerate(self.fitted_columns.keys()):
            extinction_if_needed = xspecial[:, 1]
            if key == 'stage':
                continue
            elif key == 'distance_modulus':
                new_result['distance_modulus_smooth'] = smooth_distance[0]
                smooth = smooth_distance
                if len(smooth) == 2 and ('extinction' in self.fitted_columns):
                    extinction_if_needed = xdata[:, list(self.fitted_columns.keys()).index('extinction')]
            elif key == 'extinction':
                new_result['extinction_smooth'] = smooth_extinction[0]
                smooth = smooth_extinction
                if len(smooth) == 2:
                    extinction_if_needed = xdata[:, ikey]
            elif key in ('distance', 'parallax'):
                smooth = 0.2 * np.log(10.) * smooth_distance[0]
            else:
                smooth = None

            histogram = HistogramAnalyzer(key,
                                          xdata[:, ikey],
                                          xdata[:, self.w_column],
                                          smooth,
                                          extinction_if_needed,
                                          self.config['dump'])
            histogram.age_grid = self.age_grid
            new_result.update(histogram.process_mode())
        if self.config['dump_pdf'] and 'age' in self.fitted_columns:
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
        Quality flag assignment:
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
        dump_array = list(range(self.w_column + 1))
        header = '%s L_iso L_sed p_w' % ' '.join(list(self.fitted_columns.keys()))
        ensure_dir(self.config['dump_prefix'])
        np.savetxt('%s/dump_%s.dat' % (self.config['dump_prefix'], idstr),
                   model_params[:, dump_array],
                   header=header)
        json.dump(result,
                  open('%s/dump_%s.json' % (self.config['dump_prefix'],
                                            idstr), 'w'),
                  indent=2, cls=NumpyAwareJSONEncoder)
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
        for key, value in columns.items():
            final[key] = value
        units = {'age': 'Gyr', 'mass': 'MSun',
                 'distance_modulus': 'mag',
                 'extinction': 'mag',
                 'distance': 'kpc',
                 'T': 'K',
                 'parallax': 'mas'}
        for key in self.fitted_columns.keys():
            if key in ['stage']:
                continue
            elif key in units:
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
            final['%s_par' % key] = Column(dtype=float, shape=5)
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
        if self.config['save_sed']:
            for band in self.mag_names:
                final['sed_%s' % band] = float_column(unit='mag')
                final['sed_%s_relative' % band] = float_column(unit='fraction')
        for keep in self.keep_columns:
            final[keep] = Column(dtype=data[keep].dtype)
        return final

# DO NOT MAKE main() function here: will cause problems for parallel...
