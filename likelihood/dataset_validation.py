"""Check the DES Y6 file ordering before passing indices to compiled code."""

import numpy as np


def validate_layout(data_file, mask_file, cov_file, size):
    """Require data, mask and covariance indices in the same full layout.

    Arguments:
        data_file, mask_file = two-column index/value files.
        cov_file = indexed covariance, with indices in its first two columns.
        size = expected full vector length before any probe or scale selection.
    Returns:
        Nothing. Incompatible files raise ValueError before C++ initialization.
    This checks ordering and bounds, not covariance accuracy or positivity.
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
    diagonal = pairs[pairs[:, 0] == pairs[:, 1], 0]
    if not np.array_equal(np.sort(diagonal), expected):
        raise ValueError(f"{cov_file}: expected exactly one diagonal entry per measurement")
