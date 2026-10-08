"""Check that cosmolike's caches follow every parameter change.

Cosmolike keeps intermediate tables between evaluations (caches) and
rebuilds a table only when a tag of its inputs changed; a tag is a random
number redrawn whenever those inputs receive new values. The cache worker
changes one parameter sector at a time (cosmology, source and lens photo-z
shifts, shear calibration, IA, galaxy bias), returns to the fiducial point
and evaluates it again; a fresh process evaluates the fiducial once. These
full-vector checks supplement the dummy-covariance likelihood.
"""

import numpy as np
import pytest

import cocoa_test_utils as u


@pytest.mark.parametrize("tatt", (False, True), ids=("NLA", "TATT"))
def test_cache_round_trips_and_fresh_process(tatt, tmp_path):
    """Each sampled sector changes the result; returning restores every bit.

    The checks: each changed vector is finite and differs from the initial
    one; the repeated fiducial evaluation equals the initial vector bit for
    bit; the initial vector and chi2 equal those of a fresh process.

    Arguments:
      tatt = False for the NLA model, True for TATT (one test run each)
      tmp_path = a temporary folder pytest creates for this test
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    cache_file = tmp_path/"cache.npz"
    fresh_file = tmp_path/"fresh.npz"
    u.run_vector_worker(mode="cache", example="example1", tatt=tatt, output=cache_file)
    u.run_vector_worker(mode="snapshot", example="example1", tatt=tatt, output=fresh_file)
    with np.load(file=cache_file, allow_pickle=False) as cached:
        initial = cached["initial"]
        for name in cached.files:
            if name.startswith("changed_"):
                assert np.all(np.isfinite(cached[name]))
                assert not np.array_equal(cached[name], initial), name
        np.testing.assert_array_equal(cached["repeated"], initial)
        with np.load(file=fresh_file, allow_pickle=False) as fresh:
            np.testing.assert_array_equal(initial, fresh["vector"])
            assert cached["initial_chi2"] == fresh["chi2"]
