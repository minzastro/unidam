#!/usr/bin/env python3
import numpy as np
import pylab as plt
from itertools import cycle
from unidam.utils import mathematics
from astropy.io import fits

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (5, 3.5)
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.size'] = 10.
np.set_printoptions(linewidth=200)

def plot_multiple(files, labels, filename, title=None):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    colors = cycle(plt.cm.tab10(np.linspace(0, 1, len(files))))
    if title is not None:
        fig.suptitle(title)
    hists = []
    for name, label in zip(files, labels):
        print(name)
        f = fits.open(name)[1].data
        f = f[f['quality'] != 'X']
        f = f[f['quality'] != 'N']
        f = f[f['quality'] != 'E']
        ind = np.digitize(f['distance_modulus_mean'], bins=np.linspace(0, 20, 21))
        y1 = np.ones(22) * np.nan
        y2 = np.ones(22) * np.nan
        for i in np.unique(ind):
            d = f[ind == i]
            if d['uspdf_weight'].sum() > 30:
                y1[i] = mathematics.quantile(d['distance_modulus_err'], d['uspdf_weight'], 0.5)
                y2[i] = mathematics.quantile(d['age_err'], d['uspdf_weight'], 0.5)
        if 'DR2' in label:
            color = 'blue'
        elif 'eDR3' in label:
            color = 'red'
            if 'MRS1' in label:
                color = 'black'
        elif 'Without' in label:
            color = 'grey'
        else:
            color = colors.__next__()
        ax[0].plot(y1, label=label, color=color)
        ax[1].plot(y2, label=label, color=color)
        if 'all' in filename:
            hists.append(np.histogram(d['p_best'], bins=40, range=(0, 1), density=True,
                                      weights=d['uspdf_weight'])[0])
    ax[0].legend()
    ax[0].set_xlabel('Distance modulus (mag)')
    ax[1].set_xlabel('Distance modulus (mag)')
    ax[0].set_ylabel('Distance modulus err (mag)')
    ax[1].set_ylabel('log(age) err')
    ax[0].set_xlim(0, 20)
    ax[1].set_xlim(0, 20)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    if 'all' in filename:
        for label, hist in zip(labels, hists):
            plt.bar(np.linspace(0, 1, 40), hist, width=1, label=label,
                    color=None)
        plt.legend()
        plt.savefig('pbest_all.png')
        plt.clf()


TITLES = {kk: vv for kk, vv in zip([
    'APOGEE_dr16', 'Bensby', 'GAIA_ESO4', 'GALAH_DR3', 'GCS',
    'LAMOST_DR6',
    'LAMOST_MRS1', 'RAVE_DR6', 'SEGUE'],
    ['APOGEE (DR16)', 'Bensby (2014)', 'Gaia-ESO (DR3)', 'GALAH (DR3)',
     'GCS', 'LAMOST (DR6)',
     'LAMOST MRS (DR1)', 'RAVE (DR6)', 'SEGUE'])}
for name in ['GCS', 'Bensby', 'GAIA_ESO4', 'SEGUE']:
    files = ['/home/mints/data/MPS/results/%s.fits' % name,
             '/home/mints/data/MPS/results_gaia_dr2/%s.fits' % name,
             '/home/mints/data/MPS/results_gaia_dr3/%s.fits' % name]
    labels = ['Without parallax', 'Gaia DR2', 'Gaia eDR3']
    filename = 'compare_%s.png' % name
    #plot_multiple(files, labels, filename, TITLES[name])

for name in ['APOGEE_dr16']:
    files = ['/home/mints/data/MPS/results_gaia_dr2/%s.fits' % name,
             '/home/mints/data/MPS/results_gaia_dr3/%s.fits' % name]
    labels = ['Gaia DR2', 'Gaia eDR3']
    filename = 'compare_%s.png' % name
    #plot_multiple(files, labels, filename, TITLES[name])

for name in ['LAMOST_DR6']:
    files = ['/home/mints/data/MPS/results_gaia_dr2/LAMOST_5.fits',
             '/home/mints/data/MPS/results_gaia_dr3/LAMOST_MRS1.fits',
             '/home/mints/data/MPS/results_gaia_dr3/%s.fits' % name]
    labels = ['Gaia DR2 (DR5)', 'Gaia eDR3 (MRS1)', 'Gaia eDR3 (DR6)']
    filename = 'compare_%s.png' % name
    plot_multiple(files, labels, filename, TITLES[name])

for name in ['RAVE_DR6']:
    files = ['/home/mints/data/MPS/results/RAVE.fits',
             '/home/mints/data/MPS/results_gaia_dr2/%s.fits' % name,
             '/home/mints/data/MPS/results_gaia_dr3/%s.fits' % name]
    labels = ['Without parallax (DR4)', 'Gaia DR2', 'Gaia eDR3']
    filename = 'compare_%s.png' % name
    #plot_multiple(files, labels, filename, TITLES[name])

files = ['/home/mints/data/MPS/results_gaia_dr3/%s.fits' % name
            for name in ['APOGEE_dr16', 'Bensby', 'GAIA_ESO4', 'GALAH_DR3', 'GCS',
                         'LAMOST_DR6',
                         'LAMOST_MRS1',
                         'RAVE_DR6', 'SEGUE']]
labels = ['APOGEE (DR16)', 'Bensby (2014)', 'Gaia-ESO (DR3)', 'GALAH (DR3)',
          'GCS', 'LAMOST (DR6)', 'LAMOST MRS (DR1)',
          'RAVE (DR6)', 'SEGUE']
#plot_multiple(files, labels, 'compare_all.png')
