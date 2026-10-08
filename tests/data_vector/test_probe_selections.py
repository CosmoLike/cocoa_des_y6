"""Check every shipped probe selection against the same full-layout prediction."""

import numpy as np
import pytest

import cocoa_test_utils as u


@pytest.mark.parametrize(
    "combination,excluded",
    (("combo_2x2pt", slice(0, 520)),
     ("combo_xi_gg", slice(520, 1144)),
     ("combo_xi_ggl", slice(1144, 1300))),
)
def test_probe_selection(combination, excluded):
    """Keep each retained prediction and zero the unselected probe entries.

    Arguments:
      combination, excluded = a combination module and the entry range of
          the probe it leaves out: combo_2x2pt and xi (0 to 520),
          combo_xi_gg and gamma_t (520 to 1144), or combo_xi_ggl and w
          (1144 to 1300); one test run each
    """
    import cosmolike_des_y6_interface as interface

    u.require_cocoa_environment()
    u.verify_frozen()
    info = u.load_frozen_info(example="example1", tatt=False)
    # Reuse the frozen 3x2pt options under another combination's name, so
    # only the probe selection differs from the reference.
    block = info["likelihood"].pop("des_y6.combo_3x2pt")
    info["likelihood"][f"des_y6.{combination}"] = block
    model = u.make_model(info=info)
    point = u.build_point(model=model, example="example1", tatt=False)
    u.evaluate_chi2(model=model, point=point)
    actual = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    expected = np.load(u.FROZEN_DIR/"example1_nla.npy", allow_pickle=False)
    expected[excluded] = 0.0
    np.testing.assert_allclose(actual, expected, rtol=1.e-8, atol=1.e-14)
    model.close()


def test_linear_power_option():
    """Check non_linear_emul = 0, the linear P(k) in both power slots.

    The shear entries must differ from the nonlinear reference, and the
    galaxy entries of this shear-only example stay zero.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    # evaluate_vector returns (chi2, vector); the name _ receives the chi2,
    # which this test does not use.
    _, vector = u.evaluate_vector(
        example="example2", tatt=False, overrides={"non_linear_emul": 0})
    nonlinear = np.load(u.FROZEN_DIR/"example2_nla.npy", allow_pickle=False)
    assert vector.shape == nonlinear.shape
    assert np.all(np.isfinite(vector))
    assert np.any(vector[:520] != nonlinear[:520])
    np.testing.assert_array_equal(vector[520:], np.zeros(780))
