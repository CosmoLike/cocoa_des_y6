"""Check non-Limber switches, affected blocks and cache restoration.

The switches are physics choices, not convergence settings. No likelihood
accuracy threshold is inferred from the shipped identity covariance.
"""

import numpy as np
import pytest

import cocoa_test_utils as u


@pytest.mark.parametrize("tatt", (False, True), ids=("NLA", "TATT"))
@pytest.mark.parametrize(
    "setting,start,stop",
    (("adopt_limber_gg", 1144, 1300), ("adopt_limber_gs", 520, 1144)),
)
def test_nonlimber_round_trip(setting, start, stop, tatt, tmp_path):
    """Only the selected probe changes, and returning restores every entry."""
    output = tmp_path/"nonlimber.npz"
    u.run_vector_worker(
        mode="nonlimber", example="example1", tatt=tatt, output=output,
        setting=setting)
    with np.load(output, allow_pickle=False) as result:
        limber = result["limber"]
        exact = result["nonlimber"]
        assert np.all(np.isfinite(exact))
        np.testing.assert_array_equal(result["restored"], limber)
    np.testing.assert_array_equal(exact[:start], limber[:start])
    np.testing.assert_array_equal(exact[stop:], limber[stop:])
    assert np.any(exact[start:stop] != limber[start:stop]), setting
    print(f"{setting}: max block change "
          f"{np.max(np.abs(exact[start:stop]-limber[start:stop])):.6e}")
