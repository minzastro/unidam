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
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

print(args)
if args.input is None:
    ids = []
    for arg in args.rest:
        ids += arg.split(',')
else:    
    ids = args.input.split(',')
for id in ids:
    print(id)
    data = json.load(open('dump/dump_%s.json' % id))
    bands = list('UBVI') # TODO: generalize
    xbands = np.arange(4)
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
        ax1.plot(xbands + 0.01 * irow, pred, '-+',
                 color=colors[irow])
        ax2.errorbar(xbands + 0.01*irow, np.array(obs)-np.array(pred), yerr=err,
                      label='Stage=%s' % int(row['stage']), color=colors[irow])
        ax2.axhline(0, color='grey')
    ax1.set_ylabel('Visible magnitudes')
    ax2.set_ylabel('Offset')
    ax2.legend()
    ax2.set_xticks(xbands)
    ax2.set_xticklabels(bands)
    fig.subplots_adjust(hspace=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.savefig('dump/dump_%s.sed.png' % id)