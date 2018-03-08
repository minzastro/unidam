#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:34:12 2016

@author: mints
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import open
from future import standard_library
standard_library.install_aliases()
import healpy as hp
import numpy as np
from os import path

MAP = None
EXTINCTION_FILENAME = 'extinction_data/lambda_sfd_ebv.fits'


def _download(url, filename):
    from urllib.request import urlopen  # Python 2
    #from urllib.request import urlopen # Python 3
    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(filename, 'wb') as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)


def init():
    global MAP
    global EXTINCTION_FILENAME
    if not path.exists(EXTINCTION_FILENAME):
        _download('http://lambda.gsfc.nasa.gov/data/foregrounds/SFD/lambda_sfd_ebv.fits',
                  EXTINCTION_FILENAME)
    MAP = hp.read_map(EXTINCTION_FILENAME, verbose=False)


def get_schlegel_Av(l, b):
    """
    Get A_v from Schlegel healpix-based map.
    """
    posr = np.radians((l, b))
    return 3.1 * MAP[hp.ang2pix(512, np.pi * 0.5 - posr[1], posr[0])]
