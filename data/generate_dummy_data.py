"""This script writes the placeholder data files of the DES Y6 examples.

It writes three text files into the working directory (run it from
data/):
  ones.mask         = index and 1 on every row: the mask keeps every entry;
  dummy.modelvector = index and the value 1.0: a placeholder data vector;
  dummy.cov         = rows (i, i, 1): the identity covariance, diagonal
                      entries only.

The length is that of the full real-space 3x2pt vector of DESY6.dataset:
xi_+ and xi_- (NUM_SRC_BINS (NUM_SRC_BINS + 1)/2 source pairs each),
gamma_t (NUM_SRC_BINS x NUM_LENS_BINS pairs) and w (NUM_LENS_BINS
auto-correlations), each at NUM_ANG_BINS angles: 1300 entries. Likelihood
values computed with these files carry no information about DES data.

Run: cd projects/des_y6/data; python generate_dummy_data.py
"""
import numpy as np

# Bin counts of data/DESY6.dataset: 26 angular bins, 4 source and 6 lens
# tomographic bins.
NUM_ANG_BINS = 26
NUM_SRC_BINS = 4
NUM_LENS_BINS = 6

# xi_+ and xi_- (2 x 10 pairs) + gamma_t (24 pairs) + w (6), times 26 angles
DV_LEN = NUM_SRC_BINS*(NUM_SRC_BINS+1)*NUM_ANG_BINS + NUM_SRC_BINS*NUM_LENS_BINS*NUM_ANG_BINS + NUM_LENS_BINS*NUM_ANG_BINS

mask = np.ones(DV_LEN)
indices = np.arange(DV_LEN)
np.savetxt("ones.mask", np.column_stack((indices, mask)), fmt="%d")
np.savetxt("dummy.modelvector", np.column_stack((indices, mask)), fmt="%d %.8e")
np.savetxt("dummy.cov", np.column_stack((indices, indices, mask)), fmt="%d")
