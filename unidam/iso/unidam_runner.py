#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:09:00 2016

@author: mints

Script to run distance estimator for various regimes from
the command-line.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import zip
from builtins import int
from future import standard_library
standard_library.install_aliases()
import numpy as np
import argparse
import os
import sys
import traceback
import warnings
from astropy.table import Table, Column
from unidam.iso.model_fitter import model_fitter as mf
from unidam.iso.unidam_main import UniDAMTool
from unidam.utils.constants import AGE_RANGE

np.set_printoptions(linewidth=200)
parser = argparse.ArgumentParser(description="""
Tool to estimate distances to stars.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input', type=str, default=None,
                    required=True, help='Input file name')
parser.add_argument('-o', '--output', type=str, default='result.fits',
                    help='Output file name')
parser.add_argument('-c', '--config', type=str,
                    required=True, default=None,
                    help='Config file name')
parser.add_argument('--id', type=str, default=None,
                    help='Run for just a single ID or a comma-separated list of IDs')
parser.add_argument('-d', '--dump-results', action="store_true",
                    default=False,
                    help='Dump model data for each star')
parser.add_argument('-p', '--parallel', action="store_true",
                    default=False,
                    help='Run in parallel (uses OMP_NUM_THREADS if given, otherwise 2 threads)')
args = parser.parse_args()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data = Table.read(args.input)
de = UniDAMTool(config_filename=args.config)

de.id_column = 'id'
idtype = data.columns[de.id_column].dtype
final = de.get_table(data, idtype)
unfitted = Table()
unfitted['id'] = Column(dtype=idtype)
unfitted['error'] = Column(dtype='S100')

mf.use_model_weight = True

if args.id is not None:
    ids = (args.id).split(',')
    if idtype.kind == 'S':
        len_id = idtype.itemsize
        mask = [j.strip() in np.asarray(ids, dtype=idtype) for j in data[de.id_column]]
    else:
        mask = [j in np.asarray(ids, dtype=idtype) for j in data[de.id_column]]
    data = data[np.where(mask)]
    print(data)

if args.parallel:
    mf.debug = False
    import multiprocessing as mp
    from copy import deepcopy
    from astropy.table import vstack
    if 'OMP_NUM_THREADS' in os.environ:
        pool_size = int(os.environ['OMP_NUM_THREADS'])
    else:
        pool_size = 2

    def run_single(patch):
        try:
            des = deepcopy(de)
            tbl = deepcopy(final)
            bad = deepcopy(unfitted)
            for xrow in patch:
                result = des.get_estimates(xrow, dump=False)
                if result is None:
                    continue
                elif isinstance(result, dict):
                    bad.add_row(result)
                    continue
                for new_row in result:
                    tbl.add_row(new_row)
            return tbl, des, bad
        except:
            # Put all exception text into an exception and raise that
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    pool = mp.Pool(pool_size)
    pool_result = pool.map(run_single, np.array_split(data, pool_size))
    pool_result, des, bads = list(zip(*pool_result))
    final = vstack(pool_result)
    unfitted = vstack(bads)
    if de.dump_pdf:
        out_age_pdf = np.zeros_like(des[0].total_age_pdf)
        out_2d_pdf = np.zeros_like(des[0].total_2d_pdf)
        for de_ in des:
            out_age_pdf += de_.total_age_pdf
            out_2d_pdf += de_.total_2d_pdf
        output_prefix = '%s_stacked' % args.output[:-5]
        np.savetxt('%s_age_pdf.dat' % output_prefix,
                   np.vstack((AGE_RANGE, out_age_pdf)).T)
        np.savetxt('%s_2d_pdf.dat' % output_prefix, out_2d_pdf)
else:
    if args.id is not None:
        # Debug mode is allowed only when a list of IDs is given.
        # This is done to prevent dumping HUGE amounts of data
        # if a debug-mode is switched on for a complete survey.
        mf.debug = args.dump_results
    else:
        mf.debug = False
    i = 0
    for xrow in data:
        print(xrow[de.id_column])
        #with warnings.catch_warnings():
        #    warnings.filterwarnings("error")
        result = de.get_estimates(xrow, dump=args.dump_results)
        if result is None:
            continue
        elif isinstance(result, dict):
            unfitted.add_row(result)
            continue
        for new_row in result:
            for k in list(new_row.keys()):
                if k not in final.colnames:
                    print('%s not in columns' % k)
            final.add_row(new_row)
        i += 1
    if de.dump_pdf:
        output_prefix = '%s_stacked' % args.output[:-5]
        np.savetxt('%s_age_pdf.dat' % output_prefix,
                   np.vstack((AGE_RANGE, de.total_age_pdf)).T)
        np.savetxt('%s_2d_pdf.dat' % output_prefix, de.total_2d_pdf)
if os.path.exists(args.output):
    os.remove(args.output)
final.meta = vars(args)
final.write(args.output, format='fits')
if os.path.exists('%s_unfitted.fits' % args.output):
    os.remove('%s_unfitted.fits' % args.output)
unfitted.write('%s_unfitted.fits' % args.output)
