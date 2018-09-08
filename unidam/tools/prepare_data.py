#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:43:39 2018

@author: mints
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import argparse
import numpy as np
from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astroquery.xmatch import XMatch
from configobj import ConfigObj
from unidam.iso import extinction
import warnings

warnings.filterwarnings('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

parser = argparse.ArgumentParser(description="""
Tool prepare files for UniDAM.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Input filename')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Output filename')
parser.add_argument('-c', '--config', type=str, default='default.conf',
                    help='Config filename')
parser.add_argument('--ra', type=str, default='RAJ2000',
                    help='RA column name')
parser.add_argument('--dec', type=str, default='DEJ2000',
                    help='DEC column name')
parser.add_argument('-f', '--force', action="store_true",
                    default=False,
                    help='Overwrite file if exists')
args = parser.parse_args()

extinction.init()

data = Table.read(args.input)
config = ConfigObj(args.config)
if 'keep' in config:
    data.keep_columns(config['keep'])
for key, value in list(config['mapping'].items()):
    if value.startswith('!'):
        data.add_column(Column(name=key,
                               data=np.ones(len(data)) * float(value[1:])))
    elif value in data.colnames:
        if value != key:
            data.rename_column(value, key)
    else:
        raise ValueError('%s not in table' % key)

if 'galactic' in config:
    data.rename_column([config['galactic']['longitude']], 'l')
    data.rename_column([config['galactic']['lattitude']], 'b')
else:
    data[args.ra].unit = u.degree
    data[args.dec].unit = u.degree
    c = SkyCoord(ra=data[args.ra],
                 dec=data[args.dec],
                 frame='icrs')
    data['l'] = c.galactic.l.degree
    data['b'] = c.galactic.b.degree

def clean(table):
    for column in ['RAJ2000_2', 'DEJ2000_2', 'l_2', 'b_2']:
        if column in table.colnames:
            table.remove_column(column)

print('XMatching with 2MASS')
data = XMatch.query(cat1=data,
                    cat2='vizier:II/246/out',
                    max_distance=3 * u.arcsec,
                    colRA1=args.ra, colDec1=args.dec,
                    responseformat='votable',
                    selection='best')
data.remove_columns(['angDist',
                     'errHalfMaj', 'errHalfMin', 'errPosAng',
                     'X', 'MeasureJD'])
clean(data)
print('XMatching with AllWISE')
data = XMatch.query(cat1=data,
                    cat2='vizier:II/328/allwise',
                    max_distance=3 * u.arcsec,
                    colRA1=args.ra, colDec1=args.dec,
                    responseformat='votable',
                    selection='best')
bad_col = ['%smag_2' % b for b in 'JHK'] + ['e_%smag_2' % b for b in 'JHK']
clean(data)
data.remove_columns(['angDist',
                     'eeMaj', 'eeMin', 'eePA',
                     'W3mag', 'W4mag',
                     'e_W3mag', 'e_W4mag',
                     'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'ID', 'd2M'] +
                    bad_col)
print('XMatching with Gaia')
data = XMatch.query(cat1=data,
                    cat2='vizier:I/337/gaia',
                    max_distance=3 * u.arcsec,
                    colRA1=args.ra, colDec1=args.dec,
                    responseformat='votable',
                    selection='best')
data.remove_columns(['ra_ep2000', 'dec_ep2000',
                     'errHalfMaj', 'errHalfMin', 'errPosAng',
                     'ra', 'dec', 'source_id', 'ref_epoch',
                     'ra_error', 'dec_error',
                     'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                     'ra_dec_corr', 'duplicated_source',
                     'phot_g_mean_mag', 'phot_variable_flag'])
clean(data)
data['parallax'] *= 1e-3
data['parallax_error'] *= 1e-3
print('Adding extinction data')
extinction_data = extinction.get_schlegel_Av(data['l'],
                                             data['b'],
                                             uncertainty=True)
data['extinction'] = extinction_data[:, 0]
data['extinction_error'] = extinction_data[:, 1]
if args.force and os.path.exists(args.output):
    os.remove(args.output)
data.write(args.output)
