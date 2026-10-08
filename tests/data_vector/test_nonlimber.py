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
    """Only the selected probe changes, and returning restores every entry.

    The worker evaluates the switch at 1 (Limber), 0 (exact projection) and
    1 again.

    Arguments:
      setting, start, stop = the Limber switch and the entry range of the
          probe it controls: adopt_limber_gg and w (1144 to 1300), or
          adopt_limber_gs and gamma_t (520 to 1144)
      tatt = False for the NLA model, True for TATT; the two parametrize
             decorators run every combination of the listed values
      tmp_path = a temporary folder pytest creates for this test
    """
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
