"""This script writes the DES Y6 n(z) files DESY6_lens.nz and DESY6_source.nz.

An n(z) file holds the redshift distribution of each tomographic bin (a
slice of the galaxy sample in photometric redshift): column 0 is z and
column i is n_i(z) of bin i. The inputs are the tables of CosmoSIS (another
cosmology analysis framework) in the DES Y6 code-comparison repository,
which the user must clone from
https://github.com/joaoreboucas1/y6_code_comparison
(cosmosis/lcdm_datavector_run/nz_source and nz_lens: z.txt and one
bin_<i>.txt per bin).

Run: set Y6_COMPARISON_PATH below to the local clone, then
    cd projects/des_y6/data; python generate_nz.py
The two files are written into the working directory.
"""
import os
import numpy as np

# Local clone of the code-comparison repository named in the module
# docstring; the script stops until it is set.
Y6_COMPARISON_PATH = None
if Y6_COMPARISON_PATH is None:
    raise Exception("If you want to regenerate the *.nz files, you need to change the Y6_COMPARISON_PATH variable in this script to point to the Y6 code comparison repository!")

cosmosis_path  = f"{Y6_COMPARISON_PATH}/cosmosis/lcdm_datavector_run/"
nz_source_path = f"{cosmosis_path}/nz_source/"
nz_lens_path   = f"{cosmosis_path}/nz_lens/"
# Tomographic bin counts of the DES Y6 source and lens samples
NUM_SOURCE_BINS = 4
NUM_LENS_BINS = 6

# One array per bin, read from bin_1.txt ... bin_<N>.txt; both samples
# share the redshift column z.txt.
redshifts  = np.loadtxt(f"{nz_source_path}/z.txt")
nzs_source = [np.loadtxt(f"{nz_source_path}/bin_{i}.txt") for i in range(1, NUM_SOURCE_BINS+1)]
nzs_lens   = [np.loadtxt(f"{nz_lens_path}/bin_{i}.txt")   for i in range(1, NUM_LENS_BINS+1)]

# CosmoSIS and cosmolike read the z column differently. The CosmoSIS tables
# list (z_i, n(z_i)) at the bin centers z_i. Cosmolike, with
# photoz_zmid_convention = 0, reads each row as a histogram bin that starts
# at the listed z: (z_lo, n(z_lo + dz/2)), with dz the uniform spacing
# (cosmolike/redshift_spline.c). Shifting every z by -dz/2 turns the
# centers into lower edges.
dz = redshifts[1] - redshifts[0] # Assuming uniform spacing
redshifts = redshifts - dz/2

# The first center is z = 0, which the shift makes negative (-dz/2); the
# [1:] below drops that row. [redshifts] + nzs_lens is the list (z, n_1,
# ..., n_6); np.vstack stacks it as rows, and .T turns those rows into the
# columns of the file.
nzhisto_lens = np.vstack([redshifts] + nzs_lens).T[1:]
nzhisto_source = np.vstack([redshifts] + nzs_source).T[1:]
np.savetxt("DESY6_lens.nz", nzhisto_lens)
np.savetxt("DESY6_source.nz", nzhisto_source)