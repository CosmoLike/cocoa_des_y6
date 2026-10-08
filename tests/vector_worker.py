"""Fresh-process vector diagnostics used by the DES Y6 tests."""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

import numpy as np
import cocoa_test_utils as u

ACCURACY_SETTINGS = {
    "default": {},
    "interpolation": {"accuracyboost": 2},
    "quadrature": {"integration_accuracy": 1},
    "power_redshift": {"pk_z_refinement": 2},
    "nonlimber_distance": {"nonlimber_accuracyboost": 2},
    "fastpt": {"internal_accuracyboost": 2},
}


def hybrid_vector(example):
    """Evaluate the public hybrid YAML without writing its example outputs."""
    import cocoa_hybrid_sampling as hybrid
    import cosmolike_des_y6_interface as interface

    number = 1 if example == "example1" else 2
    filename = u.PROJECT_DIR/f"EXAMPLE_EMUL2_EVALUATE{number}.yaml"
    model, _, start, _ = hybrid.load_model(filename=filename)
    prior, likelihood = hybrid.evaluate(model=model, values=start)
    vector = np.array(interface.compute_data_vector_masked(), dtype=float)
    model.close()
    return {"prior": prior, "likelihood": likelihood, "vector": vector}


def baryon_vectors():
    """Compare BCEmu on/off on the same gravity-only power prescription."""
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
        sampled = {key: point[key] for key in model.parameterization.sampled_params()}
        u.evaluate_chi2(model=model, point=sampled)
        output[name] = np.array(interface.compute_data_vector_masked(), dtype=float)
        model.close()
    return output


def nonlimber_vectors(setting, tatt):
    """Toggle one projection choice without changing the other probes.

    Return the Limber, non-Limber and restored full vectors. All three
    evaluations use identical physics except for the named projection.
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
    """
    import cosmolike_des_y6_interface as interface

    info = u.load_frozen_info(example=example, tatt=tatt)
    model = u.make_model(info=info)
    point = u.build_point(model=model, example=example, tatt=tatt)
    initial_chi2 = u.evaluate_chi2(model=model, point=point)
    initial = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    output = {"initial": initial, "initial_chi2": initial_chi2}
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
        np.testing.assert_array_equal(restored, initial)
        assert restored_chi2 == initial_chi2
    u.evaluate_chi2(model=model, point=point)
    output["repeated"] = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    return output


def photoz_vectors():
    """Switch both n(z) conventions in one process and restore them."""
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
    """Select a fresh-process diagnostic and save every computed vector."""
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
    np.savez(file=args.output, **output)


if __name__ == "__main__":
    main()
