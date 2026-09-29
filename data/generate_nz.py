"""
    This is the script originally used to create the DES-Y6 *.nz files
    The user must have access to the data at https://github.com/joaoreboucas1/y6_code_comparison
"""
import os
import numpy as np

# NOTE: user must input the path for the directory mentioned above
Y6_COMPARISON_PATH = None
if Y6_COMPARISON_PATH is None:
    raise Exception("If you want to regenerate the *.nz files, you need to change the Y6_COMPARISON_PATH variable in this script to point to the Y6 code comparison repository!")

cosmosis_path  = f"{Y6_COMPARISON_PATH}/cosmosis/lcdm_datavector_run/"
nz_source_path = f"{cosmosis_path}/nz_source/"
nz_lens_path   = f"{cosmosis_path}/nz_lens/"
NUM_SOURCE_BINS = 4
NUM_LENS_BINS = 6

redshifts  = np.loadtxt(f"{nz_source_path}/z.txt")
nzs_source = [np.loadtxt(f"{nz_source_path}/bin_{i}.txt") for i in range(1, NUM_SOURCE_BINS+1)]
nzs_lens   = [np.loadtxt(f"{nz_lens_path}/bin_{i}.txt")   for i in range(1, NUM_LENS_BINS+1)]

# NOTE: there is a difference in conventions between Cosmosis and Cosmolike.
# The output Cosmosis z-n(z) tables assume that the i-th entry is (z_i, n(z_i))
# Whereas Cosmolike assumes that the input z-n(z) is in a "bin edge" convention, i.e.
# each entry is (z_i, n(z_i + dz/2)), where dz is the bin spacing, see cosmolike/redshift_spline.c
dz = redshifts[1] - redshifts[0] # Assuming uniform spacing
redshifts = redshifts - dz/2

# In the original `redshifts`, the first entry is zero, so it becomes negative.
# I remove the first entry.

nzhisto_lens = np.vstack([redshifts] + nzs_lens).T[1:]
nzhisto_source = np.vstack([redshifts] + nzs_source).T[1:]
np.savetxt("DESY6_lens.nz", nzhisto_lens)
np.savetxt("DESY6_source.nz", nzhisto_source)