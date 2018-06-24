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
R_FACTORS = {'J': 0.72, 'H': 0.46, 'K': 0.306,
             'W1': 0.18, 'W2': 0.16,
             'u': 4.39, 'g': 3.30, 'r': 2.31, 'i': 1.71, 'z': 1.29,
             'G': 0.85926 * 3.1, 'G_BP': 1.06794 * 3.1, 'G_RP': 0.65199 * 3.1,
             'U': 4.7471, 'B': 4.1075, 'V': 3.1, 'R': 2.32, 'I': 1.49}

AGE_RANGE = arange(6.13, 11.13, 0.02)
DM_RANGE = arange(0., 20., 0.25)
