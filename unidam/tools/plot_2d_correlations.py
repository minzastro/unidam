# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:07:35 2016

@author: mints
"""

import numpy as np
import pylab as plt
import simplejson as json
import argparse
from unidam.utils.local import get_param
from matplotlib.colors import LogNorm
from matplotlib.colorbar import make_axes


UNITS = {'mass': 'Mass ($M_{sun}$)', 'age': 'log(age) (log years)',
         'distance_modulus': 'Distance modulus (mag)',
         'distance': 'Distance (pc)'}

STAGE_CMAPS = {1: 'Reds', 2: 'Blues', 3: 'Oranges'}

def plot_pdf(xid, name, data, column, axes, correlations=True,
             stage_list=[1, 2, 3], labels=False):
    stages = np.asarray(data[:, 0], dtype=int)
    adata = data[:, column]
    xdata = data[:, 3]
    wdata = data[:, -1]
    wmax = 0
    bins = (np.linspace(xdata.min()*0.9, xdata.max()*1.1, 20),
            np.linspace(adata.min()*0.9, adata.max()*1.1, 20))
    for stage in stage_list:
        hist = np.histogram2d(xdata[stages == stage], adata[stages == stage],
                              bins=bins,
                              weights=wdata[stages == stage])[0]
        if hist.max() > wmax:
            wmax = hist.max()
    for stage in stage_list:
        if isinstance(axes, list) or isinstance(axes, np.ndarray):
            ax = axes[column - 1][stage - 1]
        else:
            ax = axes
        if (stages == stage).sum() > 0:
            cm = plt.cm.ScalarMappable(cmap=STAGE_CMAPS[stage])
            cm.set_clim(vmin=0, vmax=wmax)
            _, _, _, im = ax.hist2d(xdata[stages == stage],
                                    adata[stages == stage],
                                    bins=bins, cmap=cm.cmap,
                                    vmax=wmax,
                                    norm=LogNorm(vmax=wmax),
                                    weights=wdata[stages == stage])
        xvalues = np.linspace(xdata.min(), xdata.max(), 10)
        labels_printed = False
        if correlations:
            for fit in fits:
                if fit is None:
                    continue
                if int(fit['stage']) != stage:
                    continue
                mu_func, mu_par = get_param(fit['distance_modulus_fit'],
                                            fit['distance_modulus_par'])
                mu_pdf = mu_func.pdf(xvalues, *mu_par)
                mu_pdf = mu_pdf / mu_pdf.max()
                if name == 'age':
                    ax.plot(xvalues,
                            xvalues * fit['dm_age_slope'] +
                            fit['dm_age_intercept'],
                            color='red', lw=2)
                    ax.plot(xvalues,
                            (xvalues - fit['age_dm_intercept']) / \
                            fit['age_dm_slope'],
                            color='blue', lw=2)
                else:
                    ax.plot(xvalues,
                            np.power(10, xvalues*fit['dm_%s_slope' % name] +
                                     fit['dm_%s_intercept' % name]),
                            color='red', lw=2)
                if labels and not labels_printed:
                    ax.text(0.05, 0.9, '%.2f' % fit['dm_age_scatter'],
                    transform=ax.transAxes)
                    ax.text(0.05, 0.83, '%.2f' % fit['age_dm_scatter'],
                    transform=ax.transAxes)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        if stage == 1 or len(stage_list) == 1:
            ax.set_ylabel(UNITS[name])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        r_from = int(xdata.min()) - 1
        r_to = int(xdata.max()) + 1
        if r_to - r_from > 4:
            step = ((r_to - r_from) / 6) + 1
        elif r_to - r_from > 1.5:
            step = 0.5
        else:
            step = 0.2
        ticks = np.arange(r_from, r_to-step * .1, step)
        print(ticks)
        ax.set_xticks(ticks)
        if column == 2 or len(stage_list) == 1:
            ax.set_xticklabels(['%.0f' % t for t in ticks])
            ax.set_xlabel('Distance modulus')
        else:
            ax.set_xticklabels([])
            ax.set_title('Stage %s' % ('I'*stage))
    return im


parser = argparse.ArgumentParser(description="""
Tool to plot PDFs from debug data.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input', type=str, default=None,
                    help='Comma-separated list of IDs')
parser.add_argument('-s', '--stage', type=int, default=None,
                    help='Stage to plot (in this case only distance-age will be plotted)')
parser.add_argument('-c', '--correlations', action="store_true",
                    default=False,
                    help='Add fits for correlations')
parser.add_argument('-l', '--labels', action="store_true",
                    default=False,
                    help='Add labels with scatter values')
parser.add_argument('-d', '--dump', type=str, default='dump',
                    help='Dump folder name')
parser.add_argument('--dev', action="store_true",
                    default=False,
                    help='Use development file path')
args = parser.parse_args()

folder = args.dump
if args.dev:
    folder = f'../iso/{folder}'

#fig = plt.figure(figsize=(12, 8))
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 18})
for xid in args.input.split(','):
    data = np.loadtxt('%s/dump_%s.dat' % (folder, xid))
    fits = json.load(open('%s/dump_%s.json' % (folder, xid), 'r'))
    fits = [xfits if str(xfits['id']).strip() == xid
            else None for xfits in fits]
    if args.stage is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ii, item in enumerate([('age', 1), ('mass', 2)]):
            im = plot_pdf(xid, item[0], data, item[1], axes,
                          args.correlations, labels=args.labels)
        fig.subplots_adjust(hspace=0, wspace=0)
        output_name = '%s/dump_%s_correlations.png' % (folder, xid)
        cax, kw = make_axes([ax for ax in axes.flat])
        cbar = plt.colorbar(im, cax=cax, use_gridspec=True, **kw)
        cbar.set_label('PDF')
        fig.savefig(output_name)
        plt.clf()
    else:
        for column, col_index in [('age', 1), ('mass', 2)]:
            fig, axes = plt.subplots()
            im = plot_pdf(xid, column, data, col_index, axes,
                          args.correlations, [args.stage],
                          labels=args.labels)
            output_name = '%s/dump_%s_correlations_%d_%s.png' % (
                folder, xid, args.stage, column)
            cbar = plt.colorbar(im)
            cbar.set_label('PDF')
            plt.tight_layout()
            fig.savefig(output_name)
            plt.clf()
