"""This module adapts Cocoa's shared test harness to the DES Y6 project.

cocoa_testing.py in cosmolike_core holds the test machinery every Cocoa
project shares: frozen configurations pinned by SHA-256 hashes in
manifest_sha256.json, reference chi2 values, worker subprocesses. This
"shim" supplies the DES Y6 facts (the two examples and the TATT point),
builds one CocoaTestHarness from them, and re-exports its functions as
module-level names; the test modules and generate_frozen_reference.py
import everything from here. Importing this module also puts
cosmolike_core and the project's interface folder at the front of
sys.path, the list of folders Python searches on import.

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

# The two frozen examples: the full 3x2pt likelihood (example1) and cosmic
# shear (example2). "provenance" names the project yaml each frozen
# configuration was made from; the NLA and TATT datasets are synthetic
# descriptors whose data vector is the model at the fiducial point
# (generate_frozen_reference.py).
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
# With IA_redshift_evolution = 3, DES_A2_1 and DES_A2_2 are the amplitude
# and the redshift exponent of tidal torquing, and DES_BTA_1 is b_TA.
TATT_POINT = {
    "DES_A2_1": 0.5,
    "DES_A2_2": -1.0,
    "DES_BTA_1": 0.5,
}

# The shared accuracy scans and the CFASTPT-vs-FASTPT comparison receive
# no settings here: test_accuracy.py runs this project's own refinements
# through vector_worker.py.
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

# Module-level names for the harness functions, so a test calls, for
# example, u.verify_frozen().
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

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES)
      tatt = True for the TATT model and point, False for NLA
      overrides = optional dict of likelihood options that replace the
                  frozen ones, for example {"accuracyboost": 2}

    Returns:
      (chi2, vector): a float and a float numpy array [1300].
    """
    # Imported here, not at the top, so that importing this module does not
    # load the compiled interface.
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


def run_vector_worker(mode, example, tatt, output, setting=None):
    """Run one numerical check in a fresh process and save an NPZ result.

    Every worker has four OpenMP threads, the count the shared harness
    requires (cocoa_testing.REQUIRED_OMP_THREADS), and serial BLAS. A
    failed worker raises immediately; tests never accept a missing result
    or fallback.

    Arguments:
      mode = vector_worker.py mode: snapshot, cache, photoz, accuracy,
             nonlimber, hybrid or baryons
      example = "example1" or "example2"
      tatt = True adds --tatt (TATT model and point)
      output = path of the .npz archive the worker writes
      setting = optional --setting value: an ACCURACY_SETTINGS name or a
                Limber switch

    Raises:
      subprocess.CalledProcessError when the worker exits with an error.
    """
    # A copy of this process's environment: only the worker sees the changes.
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
    if setting is not None:
        command.extend(["--setting", setting])
    subprocess.run(args=command, env=environment, check=True)
