"""Compute one DES Y6 test diagnostic in a fresh Python process.

The tests run this script through cocoa_test_utils.run_vector_worker. A
fresh process starts with no cosmolike state: the compiled interface keeps
its tables in process-wide C globals, so a diagnostic that compares
settings or checks cache restoration must not inherit tables from another
test. Each mode evaluates frozen models, collects full 1300-entry model
vectors (and chi2 values) in a dict, and saves them with numpy.savez to the
.npz archive named by --output. Modes: snapshot (one vector and chi2),
cache (parameter round trips), photoz (n(z) conventions), accuracy (one
ACCURACY_SETTINGS entry), nonlimber (one Limber switch), hybrid (an
EXAMPLE_EMUL2 configuration), baryons (external suppression off and on).
"""

import argparse
import os

# Thread counts and MPI must be fixed before numpy, cobaya or cosmolike
# load, since those libraries read them once; the values match
# tests/conftest.py.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

import numpy as np
import cocoa_test_utils as u

# One refined control per entry, doubled (or raised one level) from the
# frozen default, which "default" reproduces: interpolation tables
# (accuracyboost), quadrature level (integration_accuracy), power-table
# redshifts (pk_z_refinement), non-Limber distance samples
# (nonlimber_accuracyboost) and the C-FAST-PT convolution grid
# (internal_accuracyboost).
ACCURACY_SETTINGS = {
    "default": {},
    "interpolation": {"accuracyboost": 2},
    "quadrature": {"integration_accuracy": 1},
    "power_redshift": {"pk_z_refinement": 2},
    "nonlimber_distance": {"nonlimber_accuracyboost": 2},
    "fastpt": {"internal_accuracyboost": 2},
}


def hybrid_vector(example):
    """Evaluate the public hybrid YAML without writing its example outputs.

    Arguments:
      example = "example1" or "example2", for EXAMPLE_EMUL2_EVALUATE1.yaml
                or EXAMPLE_EMUL2_EVALUATE2.yaml

    Returns:
      dict with prior (ln prior), likelihood (ln L) and vector, the
      1300-entry model vector at the yaml's fiducial point.
    """
    import cocoa_hybrid_sampling as hybrid
    import cosmolike_des_y6_interface as interface

    number = 1 if example == "example1" else 2
    filename = u.PROJECT_DIR/f"EXAMPLE_EMUL2_EVALUATE{number}.yaml"
    # load_model returns (model, parameter names, start point, yaml hash);
    # the names _ receive the two values this check does not use.
    model, _, start, _ = hybrid.load_model(filename=filename)
    prior, likelihood = hybrid.evaluate(model=model, values=start)
    vector = np.array(interface.compute_data_vector_masked(), dtype=float)
    model.close()
    return {"prior": prior, "likelihood": likelihood, "vector": vector}


def baryon_vectors():
    """Compare BCEmu on/off on the same gravity-only power prescription.

    BCEmu, provided by the bfmt theory of EXAMPLE_EVALUATE1_BARYONS.yaml,
    emulates the baryonic suppression S(k, z) of the matter power
    spectrum. The off runs remove that theory and its *_bcemu parameters.

    Returns:
      dict with the 1300-entry model vectors off, on and restored (off
      again, evaluated last).
    """
    from copy import deepcopy
    from cobaya.yaml import yaml_load_file
    import cosmolike_des_y6_interface as interface

    original = yaml_load_file(str(u.PROJECT_DIR/"EXAMPLE_EVALUATE1_BARYONS.yaml"))
    point = original.pop("sampler")["evaluate"]["override"]
    original.pop("output", None)
    original["likelihood"]["des_y6.combo_3x2pt"]["print_datavector"] = False
    output = {}
    for name, enabled in (("off", False), ("on", True), ("restored", False)):
        info = deepcopy(original)
        info["likelihood"]["des_y6.combo_3x2pt"]["external_baryon_suppression"] = enabled
        if not enabled:
            info["theory"].pop("bfmt")
            for parameter in list(info["params"]):
                if parameter.endswith("_bcemu"):
                    info["params"].pop(parameter)
        model = u.make_model(info=info)
        # A dict comprehension: the yaml fiducial value of every parameter
        # this model samples.
        sampled = {key: point[key] for key in model.parameterization.sampled_params()}
        u.evaluate_chi2(model=model, point=sampled)
        output[name] = np.array(interface.compute_data_vector_masked(), dtype=float)
        model.close()
    return output


def nonlimber_vectors(setting, tatt):
    """Toggle one projection choice without changing the other probes.

    Return the Limber, non-Limber and restored full vectors. All three
    evaluations use identical physics except for the named projection.

    Arguments:
      setting = "adopt_limber_gg" or "adopt_limber_gs"
      tatt = True for the TATT model and point, False for NLA

    Returns:
      dict with the 1300-entry vectors limber (setting = 1), nonlimber
      (setting = 0) and restored (setting = 1 again).

    Raises:
      ValueError for any other setting.
    """
    if setting not in ("adopt_limber_gg", "adopt_limber_gs"):
        raise ValueError("nonlimber mode requires adopt_limber_gg or adopt_limber_gs")
    output = {}
    for label, value in (("limber", 1), ("nonlimber", 0), ("restored", 1)):
        _, vector = u.evaluate_vector(
            example="example1", tatt=tatt, overrides={setting: value})
        output[label] = vector
    return output


def cache_vectors(example, tatt):
    """Perturb each parameter sector and return to the same fiducial.

    Each changed point starts from the fiducial, so a missing cache
    invalidation is localized to one cosmology or nuisance parameter.

    Arguments:
      example = "example1" or "example2"
      tatt = True for the TATT model and point, False for NLA

    Returns:
      dict with initial and initial_chi2, one changed_<parameter> vector
      per entry of `changes`, and repeated, the fiducial evaluated again
      at the end.

    Raises:
      ValueError when a parameter of `changes` is not sampled;
      AssertionError when returning to the fiducial does not restore the
      vector and chi2 exactly.
    """
    import cosmolike_des_y6_interface as interface

    info = u.load_frozen_info(example=example, tatt=tatt)
    model = u.make_model(info=info)
    point = u.build_point(model=model, example=example, tatt=tatt)
    initial_chi2 = u.evaluate_chi2(model=model, point=point)
    initial = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    output = {"initial": initial, "initial_chi2": initial_chi2}
    # One parameter per sector: cosmology, source photo-z, lens photo-z,
    # shear calibration, IA amplitude and linear galaxy bias. Each step is
    # far above double-precision rounding, so the vector must change when
    # the sector's tables are rebuilt.
    changes = {
        "omegam": 0.005,
        "DES_DZ_S1": 0.001,
        "DES_DZ_L1": 0.001,
        "DES_M1": 0.001,
        "DES_A1_1": 0.1,
        "DES_B1_1": 0.1,
    }
    for parameter, delta in changes.items():
        if parameter not in point:
            raise ValueError(f"cache test requires sampled parameter {parameter}")
        changed = dict(point)
        changed[parameter] += delta
        u.evaluate_chi2(model=model, point=changed)
        output[f"changed_{parameter}"] = np.asarray(
            interface.compute_data_vector_masked(), dtype=float)
        restored_chi2 = u.evaluate_chi2(model=model, point=point)
        restored = np.asarray(interface.compute_data_vector_masked(), dtype=float)
        # Back at the fiducial, the rebuilt tables must give the initial
        # vector and chi2 bit for bit.
        np.testing.assert_array_equal(restored, initial)
        assert restored_chi2 == initial_chi2
    u.evaluate_chi2(model=model, point=point)
    output["repeated"] = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    return output


def photoz_vectors():
    """Switch both n(z) conventions in one process and restore them.

    Returns:
      dict with the 1300-entry example2 vectors default, linear, steffen,
      midpoint and restored.
    """
    # (label, photoz_interpolation_type, photoz_zmid_convention):
    # interpolation 0 cubic spline, 1 linear, 2 Steffen; z column 0 lower
    # bin edges, 1 midpoints.
    settings = (
        ("default", 0, 0),
        ("linear", 1, 0),
        ("steffen", 2, 0),
        ("midpoint", 0, 1),
        ("restored", 0, 0),
    )
    output = {}
    for label, interpolation, midpoint in settings:
        overrides = {
            "photoz_interpolation_type": interpolation,
            "photoz_zmid_convention": midpoint,
        }
        _, vector = u.evaluate_vector(
            example="example2", tatt=False, overrides=overrides)
        output[label] = vector
    return output


def main():
    """Select a fresh-process diagnostic and save every computed vector.

    argparse reads the command options: --mode, --example, --tatt,
    --output and --setting (an ACCURACY_SETTINGS name or a Limber switch).

    Side effects:
      writes the .npz archive named by --output; parser.error exits with
      status 2 for an accuracy mode without a valid setting.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("snapshot", "cache", "photoz", "accuracy", "nonlimber", "hybrid", "baryons"),
        required=True)
    parser.add_argument("--example", choices=tuple(u.EXAMPLES), required=True)
    parser.add_argument("--tatt", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--setting")
    args = parser.parse_args()
    u.require_cocoa_environment()
    u.verify_frozen()
    if args.mode == "snapshot":
        chi2, vector = u.evaluate_vector(example=args.example, tatt=args.tatt)
        output = {"chi2": chi2, "vector": vector}
    elif args.mode == "cache":
        output = cache_vectors(example=args.example, tatt=args.tatt)
    elif args.mode == "photoz":
        output = photoz_vectors()
    elif args.mode == "accuracy":
        if args.setting not in ACCURACY_SETTINGS:
            parser.error("accuracy mode requires a name from ACCURACY_SETTINGS")
        chi2, vector = u.evaluate_vector(
            example=args.example, tatt=args.tatt,
            overrides=ACCURACY_SETTINGS[args.setting])
        output = {"chi2": chi2, "vector": vector}
    elif args.mode == "hybrid":
        output = hybrid_vector(example=args.example)
    elif args.mode == "baryons":
        output = baryon_vectors()
    else:
        output = nonlimber_vectors(setting=args.setting, tatt=args.tatt)
    # **output passes each dict entry as a named array of the archive.
    np.savez(file=args.output, **output)


# True only when this file runs as a script, not when it is imported.
if __name__ == "__main__":
    main()
