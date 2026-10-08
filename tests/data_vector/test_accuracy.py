"""Report independent interpolation and quadrature refinements on DES Y6.

The identity covariance is not a survey error model. These diagnostics
report the largest absolute change and its fraction of each probe's peak
signal, rather than treating a tiny dummy chi2 as evidence of accuracy.
Each resolution runs in a fresh process so persistent tables cannot hide
the effect of a changed grid. Finite arrays and the frozen default are
asserted; the refinement differences are advisory.
"""

import numpy as np
import pytest

import cocoa_test_utils as u
from vector_worker import ACCURACY_SETTINGS


@pytest.mark.parametrize("tatt", (False, True), ids=("NLA", "TATT"))
def test_accuracy_refinements(tatt, tmp_path):
    """Measure one changed control at a time over all 1,300 entries."""
    u.require_cocoa_environment()
    u.verify_frozen()
    label = "tatt" if tatt else "nla"
    reference = np.load(u.FROZEN_DIR/f"example1_{label}.npy", allow_pickle=False)
    blocks = {
        "xi+": slice(0, 260),
        "xi-": slice(260, 520),
        "gamma_t": slice(520, 1144),
        "w": slice(1144, 1300),
    }
    for setting in ACCURACY_SETTINGS:
        if setting == "fastpt" and not tatt:
            continue
        output = tmp_path/f"{setting}.npz"
        u.run_vector_worker(
            mode="accuracy", example="example1", tatt=tatt, output=output,
            setting=setting)
        with np.load(output, allow_pickle=False) as result:
            vector = result["vector"]
        assert vector.shape == reference.shape
        assert np.all(np.isfinite(vector))
        if setting == "default":
            np.testing.assert_allclose(vector, reference, rtol=1.e-8, atol=1.e-14)
        for probe, indices in blocks.items():
            absolute = np.max(np.abs(vector[indices]-reference[indices]))
            scale = np.max(np.abs(reference[indices]))
            assert scale > 0.0
            print(f"{label} {setting:20s} {probe:7s}: max |delta|={absolute:.6e}; "
                  f"relative to probe peak={absolute/scale:.6e}", flush=True)
