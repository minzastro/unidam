#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:30:31 2017

@author: mints
"""

from astropy.io import fits
f1 = fits.open('PAR.fits')
f2 = fits.open('PARSEC.fits')
f2[1].data['Weight'] = f2[1].data['Weight'] / f2[1].data['Weight'].mean()
f3 = fits.HDUList([f1[0], f2[1], f1[2]])
f3.writeto('P.fits')