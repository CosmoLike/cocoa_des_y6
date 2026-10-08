"""This script plots the n(z) columns of one or more .nz files.

Each .nz file holds z in column 0 and the redshift distribution n_i(z) of
one tomographic bin in each further column. Every file gets one color,
and its name labels its first curve in the legend. At most ten files are
drawn: the colors come from matplotlib's ten Tableau colors, and zip stops
at the shorter of the two sequences.

Run: python plot_nz.py DESY6_lens.nz [DESY6_source.nz ...]
The figure opens in a window (plt.show); nothing is saved.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# True only when this file runs as a script, not when it is imported.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: .nz file not provided")
        print(f"Usage: python3 {sys.argv[0]} nzfile1 [nzfile2 ...]")
        exit(1)

    nzfiles = sys.argv[1:]
    colors = mcolors.TABLEAU_COLORS

    for nzfile, color in zip(nzfiles, colors):
        if not os.path.exists(nzfile):
            print(f"ERROR: could not find .nz file {nzfile}")
            print(f"Usage: python3 {sys.argv[0]} nzfile1 [nzfile2 ...]")
            exit(1)

        # unpack=True returns the columns as rows: data[0] is z and
        # data[1:] holds one n_i(z) per bin.
        data = np.loadtxt(nzfile, unpack=True)
        z = data[0]
        nz = data[1:]
        num_bins = len(nz)

        # Only the first curve of a file gets the file name as label, so
        # the legend lists each file once (it skips empty labels).
        for i, nz_i in enumerate(nz):
            plt.plot(z, nz_i, color=color, label=nzfile if i == 0 else "")
    plt.legend()
    plt.show()