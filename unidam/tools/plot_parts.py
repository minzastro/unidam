#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:07:35 2016

@author: mints
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import open
from builtins import int
from future import standard_library
standard_library.install_aliases()
import numpy as np
import pylab as plt
import simplejson as json
import argparse
from unidam.utils.local import get_param, get_ydata
from matplotlib.ticker import FormatStrFormatter

#plt.rc('text', usetex=True)  # Commented out for performance
plt.rc('font', family='serif')
UNITS = {'mass': 'Mass ($M_{sun}$)', 'age': 'log(age) (log years)',
         'distance_modulus': 'Distance modulus (mag)',
         'distance': 'Distance (pc)',
         'parallax': 'Parallax (mas)',
         'extinction': 'Extinction (mag)'}


def plot_pdf(xid, fits, name, data, column, ax, each=False,
             total=False, correlations=False, legend=True,
             plot_fits=True, plot_debug=True, dump='dump'):

    lcolors = {0: 'c', 1: 'r', 2: 'b', 3: 'orange'}
    label = {0: '0', 1: 'I', 2: 'II', 3: 'III'}
    stages = np.asarray(data[:, 0], dtype=int)
    adata = data[:, column]
    wdata = data[:, -1]
    total_weight = np.sum(wdata)
    bins = np.linspace(adata.min()*0.9, adata.max(), 50)
    binx = 0.5*(bins[1:] + bins[:-1])
    ns = []
    w = {}
    lines = []
    labels = []
    n_total = np.zeros(len(binx))
    for stage in np.sort(np.unique(stages)):
        n, _ = np.histogram(adata[stages == stage], bins,
                            weights=wdata[stages == stage], density=True)
        n = n * np.sum(wdata[stages == stage]) / total_weight
        n_total = n_total + n
        w[stage] = np.sum(n)
        if w[stage] < 0.03:
            continue
        if not plot_debug and not plot_fits:
            lw = 2.5
        else:
            lw = 1
        if total < 2:
            l = ax.step(binx, n, label='Stage %s' % label[stage], where='mid',
                        linewidth=lw, color=lcolors[stage])
            lines.append(l[0])
            labels.append('Stage %s' % label[stage])
        ns.append(n)
    fit_norm = np.sum(ns)
    if plot_debug:
        for row in fits:
            if row['%s_fit' % name] == 'N':
                continue
            stage = row['stage']
            l = ax.plot(row['%s_bins_debug' % name],
                        np.array(row['%s_hist_debug' % name]) * row['uspdf_weight'],
                        label='Stage %s' % label[stage],
                        linewidth=1.5, color=lcolors[stage])
            lines.append(l[0])
            labels.append('Stage %s' % label[stage])
            ns.append(n)
    ydata_total = np.zeros(len(binx))
    if plot_fits:
        for row in fits:
            ydata = get_ydata(name, row, binx)
            if ydata is None:
                continue
            ydata = ydata * row['uspdf_weight'] * fit_norm / np.sum(ydata)
            ydata_total = ydata_total + ydata
            ax.plot(binx, ydata, color=lcolors[row['stage']], linewidth=2.5)
            if correlations and (name == 'mass' or name == 'age') \
                and 'dm_age_intercept' in row:
                if name == 'mass':
                    binx2 = (np.log10(binx) - row['dm_mass_intercept']) / \
                        row['dm_mass_slope']
                else:
                    binx2 = (binx - row['dm_age_intercept'])/row['dm_age_slope']
                ydata2 = get_ydata('distance_modulus', row, binx2)
                if ydata2 is None:
                    continue
                ydata2 = ydata2 * row['uspdf_weight'] * ns[0].sum() / np.sum(ydata2)
                ax.plot(binx, ydata2, color=lcolors[row['stage']],
                        linestyle='--', linewidth=1.5,
                        label='Fit %s' % label[row['stage']])
    if total > 0:
        if total == 1:
            linestyle = '--'
        else:
            linestyle = '-'
        line = plt.step(binx, n_total, label='Total', where='mid',
                        color='black', linestyle=linestyle)
        lines.append(line[0])
        if plot_fits:
            line = plt.plot(binx, ydata_total, color='black',
                            linestyle=linestyle)
            lines.append(line[0])
            labels.append('Total')
        ns.append(n_total)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    if name in UNITS:
        ax.set_xlabel(UNITS[name])
    else:
        ax.set_xlabel(name)
    ax.set_ylabel('PDF')
    step = (binx[-1] - binx[0])/5
    min_x = binx[0]
    max_x = binx[-1]
    if step > 100:
        step = np.power(10, int(np.log10(max_x)))
        min_x = 0.
    if step > 1.:
        step = int(step)
        min_x = int(min_x)
        max_x = int(max_x) + 1
    elif step > 0.5:
        step = 1.
        min_x = int(min_x)
        max_x = int(max_x) + 1
    elif step > 0.25:
        step = 0.5
        min_x = int(2. * min_x) * 0.5
        max_x = int(2. * max_x) * 0.5
    else:
        step = 0.25 * (binx[-1] - binx[0])
    ax.xaxis.set_ticks(np.arange(min_x, max_x+step, step))
    ax.set_ylim(0., np.max(ns)*1.1)
    ax.yaxis.set_ticks(np.linspace(0., np.max(ns)*1.1, 6))
    ax.yaxis.set_ticklabels(np.linspace(0., 1., 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    if each:
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig('%s/dump_%s%s.png' % (dump, xid, name.replace('/', '_')))
        plt.clf()
    return lines, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Tool to plot PDFs from debug data.
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Comma-separated list of IDs')
    parser.add_argument('--title', type=str, default=None,
                        help='Title to add at the top')
    parser.add_argument('-w', '--what', type=str, default='AMD',
                        help="""What to plot:
                            A: age,
                            M: mass,
                            D: distance_modulus,
                            E: extinction,
                            d: distance,
                            P: parallax""")
    parser.add_argument('-d', '--dump', type=str, default='dump',
                        help='Dump folder name')
    parser.add_argument('-g', '--grid', action="store_true",
                        default=False,
                        help='Place all plots into one file')
    parser.add_argument('-t', '--total', type=int, default=0,
                        help="""Total histogram and fit mode:
                            0 - no totals,
                            1 - add (with black dashed line),
                            2 - plot only totals""")
    parser.add_argument('-l', '--legend', action="store_true",
                        default=False,
                        help='Add a legend on each plot')
    parser.add_argument('--nofit', action="store_true",
                        default=False,
                        help='Do not plot fits')
    parser.add_argument('--nodebug', action="store_true",
                        default=False,
                        help='Do not plot debug histogram')
    parser.add_argument('-c', '--correlations', action="store_true",
                        default=False,
                        help='Add fits for ages and masses derived '
                             'from their correlation with distance')
    parser.add_argument('--dev', action="store_true",
                        default=False,
                        help='Use development file path')
    args = parser.parse_args()

    folder = args.dump
    if args.dev:
        folder = f'../iso/{folder}'

    PLOTS = {'A': ('age', 1),
             'M': ('mass', 2),
             'D': ('distance_modulus', 3),
             'E': ('extinction', 4),
             'd': ('distance', 5),
             'P': ('parallax', 6)}
    if args.grid:
        if len(args.what) <= 2:
            cols = len(args.what)
            rows = 1
        else:
            cols = 2
            rows = len(args.what) // 2 + 1
        figsize = plt.rcParams['figure.figsize']
        fig = plt.figure(figsize=(figsize[0] * cols, figsize[1] * rows))
    plt.rcParams.update({'font.size': 18})
    if args.title is not None:
        plt.title(args.title)

    for xid in args.input.split(','):
        data_name = '%s/dump_%s.dat' % (folder, xid)
        columns = open(data_name, 'r').readline()[2:].split()
        data = np.loadtxt(data_name)
        fits = json.load(open('%s/dump_%s.json' % (folder, xid), 'r'))
        plot_params = []
        for name in args.what:
            if name in PLOTS:
                plot_params.append(PLOTS[name])
            else:
                for item in args.what.split(','):
                    plot_params.append([item, 0])
                break

        for ii, item in enumerate(plot_params):
            if args.grid:
                ax = plt.subplot(rows, cols, ii + 1)
            else:
                ax = plt.subplot(111)
            if item[0] not in columns:
                raise ValueError('%s not in data columns: %s', item[0], columns)
            col_index = columns.index(item[0])
            lines, labels = plot_pdf(xid, fits, item[0], data, col_index, ax,
                                     not args.grid, args.total,
                                     args.correlations,
                                     args.legend, not args.nofit,
                                     not args.nodebug,
                                     dump=folder)
        if args.grid:
            if args.legend:
                leg = plt.figlegend(lines, labels, loc=(0.05, 0.01),
                                    ncol=4, frameon=False)
                fig.subplots_adjust(bottom=0.25)
                fig.savefig('%s/dump_%s.png' % (folder, xid),
                            bbox_extra_artists=(leg,), bbox_inches='tight')
            else:
                fig.savefig('%s/dump_%s.png' % (folder, xid),
                            bbox_inches='tight')
            plt.clf()
