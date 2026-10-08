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
    """Measure one changed control at a time over all 1,300 entries.

    Arguments:
      tatt = False for the NLA model, True for TATT; pytest.mark.parametrize
             runs the test once per listed value
      tmp_path = a temporary folder pytest creates for this test
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    label = "tatt" if tatt else "nla"
    reference = np.load(u.FROZEN_DIR/f"example1_{label}.npy", allow_pickle=False)
    # Entry ranges of the four probes in the full layout: 10 source pairs x
    # 26 angles for xi+ and for xi-, 24 lens-source pairs x 26 for gamma_t,
    # and 6 lens bins x 26 for w.
    blocks = {
        "xi+": slice(0, 260),
        "xi-": slice(260, 520),
        "gamma_t": slice(520, 1144),
        "w": slice(1144, 1300),
    }
    for setting in ACCURACY_SETTINGS:
        # internal_accuracyboost refines only the C-FAST-PT grid. Its spectra
        # enter through the TATT terms, or through one-loop bias terms when
        # some b2 is nonzero; the frozen point fixes b2 = 0, so under NLA
        # the setting cannot change the vector.
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
        # The default run repeats the frozen settings in a fresh process, so
        # it must reproduce the frozen vector up to floating-point rounding.
        if setting == "default":
            np.testing.assert_allclose(vector, reference, rtol=1.e-8, atol=1.e-14)
        for probe, indices in blocks.items():
            absolute = np.max(np.abs(vector[indices]-reference[indices]))
            scale = np.max(np.abs(reference[indices]))
            assert scale > 0.0
            print(f"{label} {setting:20s} {probe:7s}: max |delta|={absolute:.6e}; "
                  f"relative to probe peak={absolute/scale:.6e}", flush=True)
