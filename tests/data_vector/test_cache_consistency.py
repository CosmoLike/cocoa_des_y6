"""Full-vector cache checks supplement the dummy-covariance likelihood."""

import numpy as np
import pytest

import cocoa_test_utils as u


@pytest.mark.parametrize("tatt", (False, True), ids=("NLA", "TATT"))
def test_cache_round_trips_and_fresh_process(tatt, tmp_path):
    """Each sampled sector changes the result; returning restores every bit."""
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
