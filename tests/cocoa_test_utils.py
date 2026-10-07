"""DES Y6 inputs bound to Cocoa's shared frozen-state test harness.

The shipped descriptor selects dummy data and an identity covariance.
These tests protect software behavior; they do not establish DES Y6
likelihood accuracy. Full-precision vectors supplement the chi2 checks.
"""

from pathlib import Path
import os
import subprocess
import sys

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
FROZEN_DIR = TESTS_DIR/"frozen"
MANIFEST_FILE = TESTS_DIR/"manifest_sha256.json"
REFERENCE_FILE = FROZEN_DIR/"reference_chi2.json"
CORE_DIR = PROJECT_DIR.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(PROJECT_DIR/"interface"))

import cocoa_testing as _cct

EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "des_y6.combo_3x2pt",
        "nla_dataset": "synthetic_des_y6.dataset",
        "tatt_dataset": "tatt_des_y6.dataset",
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "des_y6.cosmic_shear",
        "nla_dataset": "synthetic_des_y6.dataset",
        "tatt_dataset": "tatt_des_y6.dataset",
    },
}

# Nonzero tidal-torquing terms exercise TATT, rather than its NLA limit.
TATT_POINT = {
    "DES_A2_1": 0.5,
    "DES_A2_2": -1.0,
    "DES_BTA_1": 0.5,
}

_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_des_y6_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
    accuracy_knobs=[],
    high_accuracy_likelihood={},
    fastpt_low_settings={},
    fastpt_high_settings={},
    fastpt_points=[],
)

require_cocoa_environment = _cct.require_cocoa_environment
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
sha256_of = _cct.sha256_of
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
load_frozen_info = _H.load_frozen_info
build_point = _H.build_point
_frozen_module = _H._frozen_module
_worker = _H._worker


def evaluate_vector(example, tatt, overrides=None):
    """Build one frozen model and return its chi2 and full-layout vector.

    The vector has 1300 entries, including zeros for unselected probes.
    The caller verifies the frozen manifest before using this helper.
    """
    import numpy as np
    import cosmolike_des_y6_interface as interface

    info = load_frozen_info(example=example, tatt=tatt)
    if overrides is not None:
        block = info["likelihood"][EXAMPLES[example]["likelihood"]]
        block.update(overrides)
    model = make_model(info=info)
    point = build_point(model=model, example=example, tatt=tatt)
    chi2 = evaluate_chi2(model=model, point=point)
    vector = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    return chi2, vector


def run_vector_worker(mode, example, tatt, output):
    """Run one numerical check in a fresh process and save an NPZ result.

    Every worker has four OpenMP threads and serial BLAS. A failed worker
    raises immediately; tests never accept a missing result or fallback.
    """
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "4"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    environment["COBAYA_NOMPI"] = "1"
    command = [
        sys.executable,
        str(TESTS_DIR/"vector_worker.py"),
        "--mode", mode,
        "--example", example,
        "--output", str(output),
    ]
    if tatt:
        command.append("--tatt")
    subprocess.run(args=command, env=environment, check=True)
