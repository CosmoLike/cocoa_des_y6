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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("snapshot", "cache", "photoz"), required=True)
    parser.add_argument("--example", choices=tuple(u.EXAMPLES), required=True)
    parser.add_argument("--tatt", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    u.require_cocoa_environment()
    u.verify_frozen()
    if args.mode == "snapshot":
        chi2, vector = u.evaluate_vector(example=args.example, tatt=args.tatt)
        output = {"chi2": chi2, "vector": vector}
    elif args.mode == "cache":
        output = cache_vectors(example=args.example, tatt=args.tatt)
    else:
        output = photoz_vectors()
    np.savez(file=args.output, **output)


if __name__ == "__main__":
    main()
