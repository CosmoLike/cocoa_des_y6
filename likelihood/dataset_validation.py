"""Check the DES Y6 file layouts before the compiled code reads them.

The data-vector, mask and covariance files index the entries of the full
data vector (every bin, before probe selection or scale cuts) with
integers: the data and mask files have two columns (index, value), and
each covariance row starts with two indices (i, j). validate_layout
refuses files whose indices do not match that layout, before
_cosmolike_prototype_base hands them to the compiled code.
"""

import numpy as np


def validate_layout(data_file, mask_file, cov_file, size):
    """Require data, mask and covariance indices in the same full layout.

    The checks: the data and mask files list the indices 0, 1, ...,
    size-1 in this order; every covariance index is an integer in
    [0, size); each index has exactly one diagonal row (i, i). They test
    ordering and bounds, not covariance accuracy or positivity.

    Arguments:
        data_file, mask_file = two-column index/value text files
        cov_file = covariance text file, indices i, j in its first two
                   columns
        size = expected full vector length before any probe or scale
               selection (1300 for data/DESY6.dataset)

    Returns:
        None.

    Raises:
        ValueError naming the file and the violated condition, before
        the C++ initialization.
    """
    expected = np.arange(size)
    for filename in (data_file, mask_file):
        indices = np.loadtxt(fname=filename, usecols=0, ndmin=1)
        if not np.array_equal(indices, expected):
            raise ValueError(f"{filename}: expected consecutive indices 0 to {size-1}")

    # The inactive DESY6.cov has 1690 indices, while this project's measured
    # layout has 1300. Reject a mismatch rather than cropping a covariance
    # whose tomographic ordering has not been established.
    pairs = np.loadtxt(fname=cov_file, usecols=(0, 1), ndmin=2)
    if (not np.all(np.isfinite(pairs)) or np.any(pairs != np.floor(pairs))
            or np.any(pairs < 0) or np.any(pairs >= size)):
        raise ValueError(f"{cov_file}: covariance indices must be integers in 0 to {size-1}")
    # pairs[:, 0] == pairs[:, 1] marks the diagonal rows (i, i); indexing
    # with that boolean array keeps those rows, and column 0 their index i.
    diagonal = pairs[pairs[:, 0] == pairs[:, 1], 0]
    if not np.array_equal(np.sort(diagonal), expected):
        raise ValueError(f"{cov_file}: expected exactly one diagonal entry per measurement")
