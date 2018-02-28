#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for truncated line distribution.
Created on Thu Aug 31 17:55:10 2017

@author: mints
"""
from unidam.utils.fit import trunc_line
import numpy as np


class TruncLine(object):
    @classmethod
    def pdf(cls, *par):
        return trunc_line(*par)

    @classmethod
    def cdf(cls, xin, val, val2, lower, upper):
        result = np.zeros_like(xin)
        result[xin < lower] = 0.
        result[xin > upper] = 1.
        norm = val * (upper**2 - lower**2) + val2 * (upper - lower)
        mask = (xin >= lower) * (xin <= upper)
        result[mask] = \
            (val * (xin[mask]**2 - lower**2) + val2 * (xin[mask] - lower)) / norm
        return result
