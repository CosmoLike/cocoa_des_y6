"""NLA/TATT example regressions use every full-precision vector entry."""

import numpy as np
import pytest

import cocoa_test_utils as u


@pytest.mark.parametrize("example", ("example1", "example2"))
@pytest.mark.parametrize("tatt", (False, True), ids=("NLA", "TATT"))
def test_frozen_example(example, tatt, tmp_path):
    """Protect vector shape, probe selection and numerical reproducibility.

    rtol=1e-8 and atol=1e-14 are regression guards, not a claim of survey
    accuracy. The identity covariance gives no meaningful DES Y6 error bar.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    label = "tatt" if tatt else "nla"
    key = f"{example}_{label}"
    output = tmp_path/f"{key}.npz"
    u.run_vector_worker(mode="snapshot", example=example, tatt=tatt, output=output)
    reference = np.load(file=u.FROZEN_DIR/f"{key}.npy", allow_pickle=False)
    with np.load(file=output, allow_pickle=False) as result:
        vector = result["vector"]
        assert vector.shape == (1300,)
        assert np.all(np.isfinite(vector))
        np.testing.assert_allclose(vector, reference, rtol=1.e-8, atol=1.e-14)
        assert np.isfinite(result["chi2"])
    # xi+ and xi- each contain 10 source pairs x 26 angular bins.
    assert np.any(vector[:260] != 0.0)
    assert np.any(vector[260:520] != 0.0)
    if example == "example2":
        np.testing.assert_array_equal(vector[520:], np.zeros(780))
    else:
        assert np.any(vector[520:1144] != 0.0)
        assert np.any(vector[1144:] != 0.0)
