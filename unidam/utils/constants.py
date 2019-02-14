#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:37:34 2017

@author: mints

Some useful constants.
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from numpy import arange
# Extinctions references:
# JHK (2MASS), W1-W2 (AllWISE), ugriz (SDSS): Yuan et al. 2013
# GaiaDR2 from PARSEC
# UBVRI: Cardelli, Clayton and Mathis 1989
R_FACTORS = {'J': 0.72, 'H': 0.46, 'K': 0.306, # 2MASS
             'UKIDSSJ': 0.28887 * 3.1, # UKIDSS
             'W1': 0.18, 'W2': 0.16, # AllWISE
             'u': 4.39, 'g': 3.30, 'r': 2.31, 'i': 1.71, 'z': 1.29, # SDSS
             'G': 0.86209 * 3.1, 'G_BP': 1.072 * 3.1, 'G_RP': 0.64648 * 3.1, # Gaia2 Weiler 2018
             'B_T': 1.35552, 'V_T': 1.05047,  # Tycho2
             'U': 4.7471, 'B': 4.1075, 'V': 3.1, 'R': 2.32, 'I': 1.49, # Johnson
             'DECAMu': 4.495, 'DECAMg': 3.636, 'DECAMr': 2.585, 'DECAMi': 1.975, 'DECAMz': 1.457, 'DECAMY': 1.286,
             'VISTAZ': 1.578, 'VISTAY': 1.209, 'VISTAJ': 0.893, 'VISTAH': 0.564, 'VISTAKs': 0.372 }

AGE_RANGE = arange(6.13, 11.13, 0.02)
DM_RANGE = arange(0., 20., 0.25)
