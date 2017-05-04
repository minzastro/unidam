#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:34:12 2016

@author: mints
"""
import healpy as hp
import numpy as np


MAP = None


def init():
    global MAP
    MAP = hp.read_map('extinction_data/lambda_sfd_ebv.fits', verbose=False)


def get_schlegel_Av(l, b):
    """
    Get A_v from Schlegel healpix-based map.
    """
    posr = np.radians((l, b))
    return 3.1 * MAP[hp.ang2pix(512, np.pi * 0.5 - posr[1], posr[0])]
