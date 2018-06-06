#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:19:43 2018

@author: mints
"""
import argparse
import simplejson as json
import pylab as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(description="""
Tool to plot PDFs from debug data.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input', type=str, default=None,
                    help='Comma-separated list of IDs')
parser.add_argument('--log', action="store_true",
                    default=False,
                    help='Log scale X axis')
parser.add_argument('-d', '--dump', type=str, default='dump',
                    help='Dump folder name')

parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

BAND_POS = {'U': 3508.2,
            'B': 4329,
            'V': 5421.7,
            'R': 6427.8,
            'I': 8048.4,
            'J': 12329.79,
            'H': 16395.59,
            'K': 21522.05,
            'G': 6437.70,
            'G_BP': 5309.57,
            'G_RP': 7709.85,
            'W1': 33159.26,
            'W2': 45611.97
            # TODO: SDSS
            }

print(args)
if args.input is None:
    ids = []
    for arg in args.rest:
        ids += arg.split(',')
else:
    ids = args.input.split(',')
for id in ids:
    print(id)
    data = json.load(open('../iso/%s/dump_%s.json' % (args.dump, id)))
    bands = []
    xbands = []
    for b, freq in BAND_POS.items():
        if b in data[0]['sed_debug']['Observed']:
            bands.append(b)
            xbands.append(freq)
    bands = np.array(bands, dtype=str)
    xbands = np.array(xbands)
    bands = bands[np.argsort(xbands)]
    xbands = xbands[np.argsort(xbands)]
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
    colors = list('brm')
    for irow, row in enumerate(data):
        obs = [row['sed_debug']['Observed'][b] for b in bands]
        err = np.array([row['sed_debug']['ObsErr'][b] for b in bands])
        pred = [row['sed_debug']['Predicted'][b] for b in bands]
        perr = np.array([row['sed_debug']['PredErr'][b] for b in bands])
        err = np.sqrt(err**2 + perr**2)
        ax1.scatter(xbands, obs,
                    s=(plt.rcParams['lines.markersize']*1.5)**2,
                    edgecolor='black', facecolors='none',
                      )
        ax1.plot(xbands, pred, '-+',
                 color=colors[irow])
        ax2.errorbar(xbands, np.array(obs)-np.array(pred), yerr=err,
                      label='Stage=%s' % int(row['stage']), color=colors[irow])
        ax2.axhline(0, color='grey')
    ax1.set_ylabel('Visible magnitudes')
    ax2.set_ylabel('Offset')
    ax2.legend()
    ax2.set_xticks(xbands)
    ax2.set_xticklabels(bands)
    if args.log:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.savefig('../iso/%s/dump_%s.sed.png' % (args.dump, id))