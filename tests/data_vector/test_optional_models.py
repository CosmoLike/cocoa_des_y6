"""Exercise the public CPU hybrid and external baryonic-feedback examples.

These checks need the shared emulator assets installed by Cocoa. They are
workflow and cache checks, not calibration or measured-likelihood tests.
"""

import numpy as np
import pytest

import cocoa_test_utils as u


@pytest.mark.parametrize("example", ("example1", "example2"))
def test_hybrid_example(example, tmp_path):
    """Both public hybrid configurations evaluate a finite full-layout vector."""
    output = tmp_path/"hybrid.npz"
    u.run_vector_worker(mode="hybrid", example=example, tatt=False, output=output)
    with np.load(output, allow_pickle=False) as result:
        assert np.isfinite(result["prior"])
        assert np.isfinite(result["likelihood"])
        assert result["vector"].shape == (1300,)
        assert np.all(np.isfinite(result["vector"]))
        assert np.any(result["vector"] != 0)
        if example == "example2":
            np.testing.assert_array_equal(result["vector"][520:], 0)


def test_external_baryon_round_trip(tmp_path):
    """External suppression changes every probe and switching off restores it."""
    output = tmp_path/"baryons.npz"
    u.run_vector_worker(mode="baryons", example="example1", tatt=False, output=output)
    with np.load(output, allow_pickle=False) as result:
        off, on = result["off"], result["on"]
        assert on.shape == off.shape == (1300,)
        assert np.all(np.isfinite(on))
        np.testing.assert_array_equal(result["restored"], off)
        for start, stop in ((0, 260), (260, 520), (520, 1144), (1144, 1300)):
            assert np.any(on[start:stop] != off[start:stop])
            print(f"rows {start}:{stop}: max feedback change "
                  f"{np.max(np.abs(on[start:stop]-off[start:stop])):.6e}")
