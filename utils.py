"""Module contains commonly used functions analysis."""
import re
import os
import numpy as np

import requests
import wget
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.decomposition import PCA


def read_file(filename):
    """Parse file."""
    data = []
    parser = re.compile('  *')
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                parsed = parser.sub(';', line).replace('\n', '').split(';')[1:]
                data.append(np.array(parsed, dtype=float))
    data = np.array(data)
    f.close()
    return data

def download_files():
    possible_ids = np.arange(1, 21000, 25)
    possible_ids = np.array(possible_ids, dtype=int)
    possible_IDS = [str(i).zfill(6) for i in possible_ids]

    datafiles = os.listdir('Data/')
    url = 'https://arrows.emsl.pnnl.gov/api/eric_view/download=we31869:/home/bylaska/' + \
          'Projects/BES/Hematite-Zn/PFe2-ZnH-H3-corner54a/exafs_Zn_ih_1_de_-2.0_s02_1.0_rc_8.5/'

    for ID in possible_IDS:
        filename = f'chi_run{ID}_70.dat'
        req = requests.get(f'{url}{filename}')
        if req.status_code == 200:
            if filename not in datafiles:
                print(f"Dowloading {filename}")
                fname = wget.download(url + filename)
                with zipfile.ZipFile(fname, 'r') as zip_ref:
                    zip_ref.extractall('Data/')
                os.remove(fname)
            else:
                print(f'Already downloaded {filename}')

def make_scree_plot(data, n=5, threshold=0.95, show_first_PC=True, mod=0):
    """
    Make PCA scree plot.
    
    Attributes:
        data - Data on which to perform PCA analysis.
            type = ndarray
    kwargs:
        n - The number of PC's to display/ keep on the x-axis.
            default = 5
            type = int
        threshold - The variance cutoff to display.
            default = 0.95
            type = float
        show_first_PC - Opt to display the first PC.
            default = True
            type = boolean
    """
    fig, ax = plt.subplots(figsize=(8,6))
    pca = PCA()
    pca_components = pca.fit_transform(data)

    x = np.arange(n) + 1
    cdf = [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(n)]

    ax.plot(x, cdf, 's-', markersize=10, fillstyle='none',
            color=plt.cm.tab10(.15))
    ax.plot(x, np.ones(len(x)) * threshold, 'k--', linewidth=3)

    if show_first_PC:
        PC1 = pca.components_[0]
        plt.plot(np.linspace(1, n, len(PC1)), -PC1*0.3 + min(cdf) + 0.1, 'k', linewidth=2)
        text = ax.text(n - 1, min(cdf) + 0.13, '$PC_1$', ha="right", va="bottom", size=20)

    if mod == 0:
        xticks = np.arange(n) + 1
    else:
        xticks = np.arange(0, n + 1, mod)
    plt.xticks(xticks, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(min(cdf) - 0.05, 1.02)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Number of Parameters', fontsize=22)
    plt.ylabel(f'Cumultative\nExplained Variance', fontsize=22)
    ax.tick_params(direction='in', width=2, length=8)

def set_axes(ax, minval, maxval, nticks=4, axis='x'):
    spacing = 1 / (nticks + 2)
    ticks = np.linspace(spacing, 1 - spacing, nticks)
    if axis == 'x':
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)
    tick_labels = [v * (maxval - minval) + minval for v in ticks]
    if axis == 'x':
        ax.set_xticklabels([f'{l:.2f}' for l in tick_labels])
    else:
        ax.set_yticklabels([f'{l:.2f}' for l in tick_labels])
