"""Create the DES Y6 regression baseline from the current compiled code.

The baseline, the "frozen state", is everything the tests compare against,
stored under tests/frozen/ and pinned by SHA-256 hashes in
tests/manifest_sha256.json: copies of the referenced input files, one
generated configuration module per example, the synthetic NLA and TATT
data vectors, the reference vectors and chi2 values, and a provenance
record.

Run from Cocoa/ after starting its environment:
    python ./projects/des_y6/tests/generate_frozen_reference.py --overwrite

The first freeze creates references; a later freeze changes what is tested.
Review every intended numerical change before replacing a baseline. Previous
state is preserved in tests/.reference_backups/ when --overwrite is used.
The options --freeze-one, --synthetic-one and --snapshot-one run single
steps; main() passes them to fresh copies of this script.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Thread counts and MPI must be fixed before numpy, cobaya or cosmolike
# load, since those libraries read them once; the values match
# tests/conftest.py.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

import cocoa_test_utils as u

# Frozen data folder relative to Cocoa/, the working directory that
# require_cocoa_environment sets. It is written into the frozen
# configurations as the likelihood path; the tests replace it by the
# absolute path when they load them.
FROZEN_DATA_RELPATH = "./projects/des_y6/tests/frozen/data"


def write_json(path, value):
    """Write a readable, deterministic JSON record.

    Arguments:
      path = pathlib.Path of the file to write
      value = data JSON can represent (dict, list, number, string)

    Side effects:
      writes the file with two-space indentation, sorted keys and a final
      newline.
    """
    with path.open(mode="w") as output:
        json.dump(obj=value, fp=output, indent=2, sort_keys=True)
        output.write("\n")


def copy_referenced_inputs():
    """Freeze only files selected by the examples and their descriptors.

    In particular, the unselected 1690-entry DESY6.cov and DESY6.mask
    are not substituted for the active 1300-entry dummy-data contract.

    Returns:
      (copied, absent_optional): a dict from each copied file name to its
      SHA-256 hash, and the list of optional descriptor files that do not
      exist.

    Raises:
      FileNotFoundError for a missing required file; ValueError for an
      input outside data/.

    Side effects:
      creates tests/frozen/data/ and copies the selected files into it, and
      the example yaml files into tests/frozen/.
    """
    from cobaya.yaml import yaml_load_file

    data_source = u.PROJECT_DIR/"data"
    destination = u.FROZEN_DIR/"data"
    destination.mkdir()
    copied = {}
    absent_optional = []
    required = (
        "data_file",
        "cov_file",
        "mask_file",
        "nz_lens_file",
        "nz_source_file",
    )
    optional = ("all_sims_hdf5_file", "baryon_pca_file")
    for configuration in u.EXAMPLES.values():
        yaml_path = u.PROJECT_DIR/configuration["provenance"]
        info = yaml_load_file(str(yaml_path))
        block = info["likelihood"][configuration["likelihood"]]
        descriptor_name = block["data_file"]
        descriptor_path = data_source/descriptor_name
        entries = {}
        # Descriptor lines read key = value; text after # is a comment.
        for line in descriptor_path.read_text().splitlines():
            text = line.split("#", 1)[0].strip()
            if "=" in text:
                key, value = text.split("=", 1)
                entries[key.strip()] = value.strip()
        filenames = [descriptor_name]
        for key in required:
            filename = entries[key]
            path = data_source/filename
            if not path.is_file():
                raise FileNotFoundError(f"{descriptor_name}: required {key} file {path} is absent")
            filenames.append(filename)
        for key in optional:
            if key not in entries:
                continue
            filename = entries[key]
            if (data_source/filename).is_file():
                filenames.append(filename)
            elif filename not in absent_optional:
                absent_optional.append(filename)
        for filename in filenames:
            source = data_source/filename
            if source.parent != data_source:
                raise ValueError(f"frozen input {filename} must be a file directly under data/")
            shutil.copy2(src=source, dst=destination/filename)
            copied[filename] = u.sha256_of(path=source)
        shutil.copy2(src=yaml_path, dst=u.FROZEN_DIR/yaml_path.name)
    return copied, absent_optional


def freeze_configuration(example, timestamp):
    """Resolve one example's defaults and save its complete parameter point.

    Arguments:
      example = "example1" or "example2"
      timestamp = UTC time stamp written into the generated module's
                  docstring

    Raises:
      ValueError when a sampled parameter has neither an evaluate override
      nor a ref center.

    Side effects:
      writes tests/frozen/frozen_config_<example>.py, which defines point
      (a dict) and yaml_string (the configuration with every default
      filled in, as yaml text).
    """
    from cobaya.yaml import yaml_dump, yaml_load_file

    configuration = u.EXAMPLES[example]
    info = yaml_load_file(str(u.PROJECT_DIR/configuration["provenance"]))
    override = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler", None)
    info.pop("output", None)
    # Log level 30 (WARNING, in Python's logging levels) and no timing
    # report. The frozen configuration stores NLA (IA_model = 0), reads the
    # frozen data and writes no model-vector file; load_frozen_info
    # switches to TATT when a test asks for it.
    info["debug"] = 30
    info["timing"] = False
    block = info["likelihood"][configuration["likelihood"]]
    block["path"] = FROZEN_DATA_RELPATH
    block["IA_model"] = 0
    block["print_datavector"] = False
    model = u.make_model(info=info)
    # model.info() returns the configuration with every default filled in;
    # packages_path, which depends on the machine, is dropped.
    resolved = model.info()
    resolved.pop("packages_path", None)
    point = {}
    for name in model.parameterization.sampled_params():
        if name in override:
            point[name] = override[name]
        else:
            reference = resolved["params"][name].get("ref")
            if isinstance(reference, dict):
                reference = reference.get("loc")
            if reference is None:
                raise ValueError(f"{example}: sampled {name} has no evaluate override or ref center")
            point[name] = reference
            print(f"{example}: {name} uses explicit ref center {reference}", flush=True)
    # repr writes point and the yaml text as Python literals, so the
    # generated module recreates them exactly.
    content = (
        f'"""Generated DES Y6 frozen configuration ({timestamp}); do not edit."""\n\n'
        + "point = " + repr(point) + "\n\n"
        + "yaml_string = " + repr(yaml_dump(resolved)) + "\n"
    )
    (u.FROZEN_DIR/configuration["frozen_module"]).write_text(content)


def generate_synthetic_vector(tatt):
    """Generate a full 3x2pt fiducial with lossless text precision.

    The two IA models receive separate synthetic vectors. Both therefore
    evaluate at their own likelihood minimum instead of measuring distance
    from the arbitrary shipped dummy vector.

    Arguments:
      tatt = True for the TATT model and point, False for NLA

    Raises:
      ValueError when the vector is not 1300 finite entries, or the
      descriptor does not have exactly one data_file line.

    Side effects:
      writes frozen/data/<synthetic or tatt>_des_y6.modelvector (index and
      value; %.17e prints 18 significant digits, enough to read every
      double back exactly) and the matching .dataset descriptor, a copy of
      the original whose data_file names the new vector.
    """
    import numpy as np
    from cobaya.yaml import yaml_load
    import cosmolike_des_y6_interface as interface

    example = "example1"
    configuration = u.EXAMPLES[example]
    # load_frozen_info points data_file at the synthetic descriptor this
    # function is about to write; the model is evaluated with the frozen
    # configuration's own descriptor instead (only the model vector is used).
    raw = yaml_load(u._frozen_module(example=example).yaml_string)
    original = raw["likelihood"][configuration["likelihood"]]["data_file"]
    info = u.load_frozen_info(example=example, tatt=tatt)
    info["likelihood"][configuration["likelihood"]]["data_file"] = original
    model = u.make_model(info=info)
    point = u.build_point(model=model, example=example, tatt=tatt)
    u.evaluate_chi2(model=model, point=point)
    vector = np.asarray(interface.compute_data_vector_masked(), dtype=float)
    if vector.shape != (1300,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"fiducial must contain 1300 finite entries; got {vector.shape}")
    key = "tatt_dataset" if tatt else "nla_dataset"
    descriptor_name = configuration[key]
    vector_name = descriptor_name.replace(".dataset", ".modelvector")
    data_dir = u.FROZEN_DIR/"data"
    table = np.column_stack((np.arange(vector.size), vector))
    np.savetxt(fname=data_dir/vector_name, X=table, fmt=("%d", "%.17e"))
    replacement = []
    count = 0
    for line in (data_dir/original).read_text().splitlines():
        if line.split("=", 1)[0].strip() == "data_file":
            replacement.append(f"data_file = {vector_name}")
            count += 1
        else:
            replacement.append(line)
    if count != 1:
        raise ValueError(f"{original} has {count} data_file entries; expected one")
    (data_dir/descriptor_name).write_text("\n".join(replacement)+"\n")


def generate_snapshot(example, tatt):
    """Save one full-precision reference vector and its chi2.

    Arguments:
      example = "example1" or "example2"
      tatt = True for the TATT model and point, False for NLA

    Raises:
      ValueError for a non-finite chi2 or vector entry.

    Side effects:
      writes frozen/<example>_<nla|tatt>.npy and a temporary
      <example>_<nla|tatt>_chi2.json, which main() moves into
      reference_chi2.json.
    """
    import numpy as np

    label = "tatt" if tatt else "nla"
    key = f"{example}_{label}"
    chi2, vector = u.evaluate_vector(example=example, tatt=tatt)
    if not np.isfinite(chi2) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{key}: non-finite reference result")
    np.save(file=u.FROZEN_DIR/f"{key}.npy", arr=vector, allow_pickle=False)
    write_json(path=u.FROZEN_DIR/f"{key}_chi2.json", value=chi2)
    print(f"{key}: chi2={chi2:.17g}, vector entries={vector.size}", flush=True)


def main():
    """Rebuild the frozen state, or run one step of it in this process.

    With a worker option (--freeze-one, --synthetic-one or --snapshot-one)
    the script runs that single step and returns. Otherwise --overwrite is
    required: the old frozen state moves to
    tests/.reference_backups/<timestamp>/, the referenced inputs are
    copied, and every step runs in a fresh copy of this script, so no C
    global state of one step reaches the next. reference_chi2.json,
    provenance.json and the manifest are written last.

    Side effects:
      replaces tests/frozen/ and tests/manifest_sha256.json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--freeze-one", choices=tuple(u.EXAMPLES))
    parser.add_argument("--synthetic-one", choices=("nla", "tatt"))
    parser.add_argument("--snapshot-one", choices=tuple(u.EXAMPLES))
    parser.add_argument("--tatt", action="store_true")
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    # `a or b or c` gives the first value that is set (not None), so worker
    # names the single step requested, or is None.
    worker = args.freeze_one or args.synthetic_one or args.snapshot_one
    if not worker and not args.overwrite:
        parser.error("reference generation requires --overwrite; it changes the regression baseline")
    u.require_cocoa_environment()
    if args.freeze_one:
        freeze_configuration(example=args.freeze_one, timestamp=args.timestamp)
        return
    if args.synthetic_one:
        generate_synthetic_vector(tatt=args.synthetic_one == "tatt")
        return
    if args.snapshot_one:
        generate_snapshot(example=args.snapshot_one, tatt=args.tatt)
        return

    # UTC time as YYYYMMDDTHHMMSS, then microseconds and Z: it names the
    # backup folder and stamps the generated files.
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if u.FROZEN_DIR.exists() or u.MANIFEST_FILE.exists():
        backup = u.TESTS_DIR/".reference_backups"/timestamp
        backup.mkdir(parents=True)
        if u.FROZEN_DIR.exists():
            shutil.move(src=str(u.FROZEN_DIR), dst=str(backup/"frozen"))
        if u.MANIFEST_FILE.exists():
            shutil.move(src=str(u.MANIFEST_FILE), dst=str(backup/u.MANIFEST_FILE.name))
        print(f"Previous frozen state preserved in {backup}", flush=True)
    u.FROZEN_DIR.mkdir()
    copied, absent_optional = copy_referenced_inputs()
    command = [sys.executable, str(Path(__file__).resolve())]
    for example in u.EXAMPLES:
        subprocess.run(
            args=command+["--freeze-one", example, "--timestamp", timestamp],
            check=True,
        )
    for label in ("nla", "tatt"):
        subprocess.run(args=command+["--synthetic-one", label], check=True)
    reference = {
        "_meta": {
            "generated_utc": timestamp,
            "omp_num_threads": 4,
            "data_contract": "synthetic fiducials; shipped dummy identity covariance and ones mask",
            "vector_rtol": 1.e-8,
            "vector_atol": 1.e-14,
            "tolerance_scope": "software regression guards, not survey precision",
        },
    }
    for example in u.EXAMPLES:
        for tatt in (False, True):
            arguments = command+["--snapshot-one", example]
            if tatt:
                arguments.append("--tatt")
            subprocess.run(args=arguments, check=True)
            label = "tatt" if tatt else "nla"
            key = f"{example}_{label}"
            snapshot = u.FROZEN_DIR/f"{key}_chi2.json"
            reference[key] = json.loads(snapshot.read_text())
            snapshot.unlink()
    write_json(path=u.REFERENCE_FILE, value=reference)
    # Hashes of the likelihood and interface sources at freeze time: a
    # record for the reader; no test compares them with the live files.
    sources = {}
    for path in sorted((u.PROJECT_DIR/"likelihood").glob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml"):
            sources[str(path.relative_to(u.PROJECT_DIR))] = u.sha256_of(path=path)
    for filename in ("interface.cpp", "cosmolike_des_y6_interface.py"):
        path = u.PROJECT_DIR/"interface"/filename
        sources[str(path.relative_to(u.PROJECT_DIR))] = u.sha256_of(path=path)
    provenance = {
        "input_sha256": copied,
        "source_sha256": sources,
        "absent_optional_descriptor_files": absent_optional,
        "excluded_unselected_files": ["DESY6.cov", "DESY6.mask"],
    }
    write_json(path=u.FROZEN_DIR/"provenance.json", value=provenance)
    manifest = {
        "_comment": "Generated SHA-256 pins; regenerate deliberately, never edit references by hand.",
        "files": u.compute_manifest(),
    }
    write_json(path=u.MANIFEST_FILE, value=manifest)
    print(f"Frozen {len(manifest['files'])} files. Review references, then run both test sectors.", flush=True)


if __name__ == "__main__":
    main()
