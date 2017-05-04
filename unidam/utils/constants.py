#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:37:34 2017

@author: mints

Some useful constants.
"""
from numpy import arange
# Extinctions references:
# JHK (2MASS), W1-W2 (AllWISE), ugriz (SDSS): Yuan et al. 2013
# UBVRI: Cardelli, Clayton and Mathis 1989
R_FACTORS = {'J': 0.72, 'H': 0.46, 'K': 0.306,
             'W1': 0.18, 'W2': 0.16,
             'u': 4.39, 'g': 3.30, 'r': 2.31, 'i': 1.71, 'z': 1.29,
             'U': 4.7471, 'B': 4.1075, 'V': 3.1, 'R': 2.32, 'I': 1.49}

AGE_RANGE = arange(6.13, 11.13, 0.02)
DM_RANGE = arange(0., 20., 0.25)