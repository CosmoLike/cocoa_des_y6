"""Changing n(z) conventions must invalidate the corresponding tables.

Cosmolike caches its n(z) tables. Changing the interpolation type (linear,
Steffen) or the z-column convention (midpoints instead of lower edges)
must rebuild them, and restoring the defaults must give back the default
vector bit for bit. The midpoint run reads the lower-edge files with the
other convention on purpose: only the switch is under test.
"""

import numpy as np

import cocoa_test_utils as u


def test_photoz_flags_and_round_trip(tmp_path):
    """All alternative conventions are active and the original is restored.

    Arguments:
      tmp_path = a temporary folder pytest creates for this test
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    output = tmp_path/"photoz.npz"
    u.run_vector_worker(mode="photoz", example="example2", tatt=False, output=output)
    with np.load(file=output, allow_pickle=False) as result:
        default = result["default"]
        for label in ("linear", "steffen", "midpoint"):
            alternative = result[label]
            assert np.all(np.isfinite(alternative))
            assert not np.array_equal(alternative, default), label
        np.testing.assert_array_equal(result["restored"], default)
