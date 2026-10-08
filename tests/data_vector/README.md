# Data-vector checks <a name="data-vector"></a>

These checks compare every entry of the shipped 1,300-entry prediction and
exercise cache invalidation, input validation and probe selection. The active
likelihood has a dummy vector and identity covariance; the checks do not
validate a measured DES Y6 likelihood.

| Check | Configuration |
|---|---|
| Full-vector reference comparison | 3×2pt, IA modeling: NLA |
| Full-vector reference comparison | 3×2pt, IA modeling: TATT |
| Full-vector reference comparison | Cosmic shear, IA modeling: NLA |
| Full-vector reference comparison | Cosmic shear, IA modeling: TATT |
| Return to the original prediction after parameter changes | IA modeling: NLA |
| Return to the original prediction after parameter changes | IA modeling: TATT |
| Agreement with a fresh process | IA modeling: NLA |
| Agreement with a fresh process | IA modeling: TATT |
| Retained entries agree with the full prediction | Clustering and galaxy–galaxy lensing |
| Retained entries agree with the full prediction | Cosmic shear and clustering |
| Retained entries agree with the full prediction | Cosmic shear and galaxy–galaxy lensing |
| Linear-power option returns a finite, distinct prediction | Cosmic shear |
| n(z) setting changes and restoration | Photo-z interpolation and coordinate convention |
| Invalid data, mask or covariance indices are rejected | Temporary input files |
| YAML thread keys are rejected | OpenMP threads belong to the environment |
| Stored input hashes and descriptor layout are checked | `manifest_sha256.json` |
| Separate accuracy-control refinements | NLA and TATT, every vector entry |
| Non-Limber gg/gs switches and restoration | NLA and TATT, selected block only |
| Public hybrid YAMLs on CPU | 3×2pt NLA and cosmic-shear TATT |
| External BCEmu suppression and restoration | Same Takahashi baseline, all probes |

We assume the project is enabled in `set_installation_options.sh` and
compiled, the Cocoa Conda environment is active, the shell is Bash, and the
current folder is `cocoa/Cocoa/`. Test workers
use four OpenMP threads and one BLAS thread. The optional-model checks need
the shared hybrid emulator assets and BCEmu installed through the
[main Cocoa recipe](https://github.com/CosmoLike/cocoa#cobaya_base_code_examples_emul2).
A missing dependency is a failure with an explicit error, not a silent skip.

## Running the data-vector checks <a name="run-data-vector"></a>

**Step :one:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :two:**: run the data-vector checks.

```bash
python -m pytest ./projects/des_y6/tests/data_vector
```

## Accuracy and model-choice checks

`test_accuracy.py` changes each control separately in a fresh process:
`accuracyboost`, `integration_accuracy`, `pk_z_refinement`,
`nonlimber_accuracyboost` and, for TATT, `internal_accuracyboost`. The default
must reproduce its frozen vector. Refined vectors must be finite; per-probe
maximum absolute and peak-normalized changes are printed as diagnostics.
The dummy identity covariance cannot supply a meaningful survey chi-squared
accuracy threshold, so the test does not invent one.

`test_nonlimber.py` switches gg or galaxy–shear projection individually,
checks that only the selected block changes, then requires exact restoration.
This tests the switch and cache behavior, not non-Limber convergence.

`test_optional_models.py` evaluates both public CPU hybrid configurations.
It also applies BCEmu off/on/off restoration against the same
gravity-only Takahashi spectrum, checking every probe. No frozen reference
is regenerated, and no HMcode feedback is mixed with the external ratio.

# Appendix <a name="appendix"></a>

## FAQ: Which inputs are tested? <a name="inputs"></a>

The tests store their own configuration and input snapshots in `frozen/`.
The generator copies only each descriptor and its referenced inputs, then
records hashes in `manifest_sha256.json`. Every data-vector evaluation
verifies this manifest before using the snapshot.

The active layout has four source bins, six lens bins and 26 angular bins.
Its 1,300-entry identity covariance supplies no survey weighting. The separate
1,690-entry `data/DESY6.cov` and optional `DESY6.mask` are not selected by the
active descriptor and are excluded from the snapshot. No verified mapping
allows that covariance to be cropped into the active layout.

Input-validation tests create temporary malformed files. They check short or
reordered vectors/masks, fractional or out-of-range covariance indices, and
missing or duplicated diagonals before the C++ reader is called.

## FAQ: What do the reference comparisons mean? <a name="references"></a>

NLA and TATT each have a stored fiducial vector and resolved configuration.
The generator uses the corresponding prediction as that configuration's
test data vector. This checks reproducibility at a known point; its raw
$`\chi^2`$ has no measured-survey interpretation.

| Comparison | Requirement |
|---|---|
| Every stored data-vector entry | Relative tolerance `1e-8`, absolute tolerance `1e-14` |
| Same-process cache return | Bitwise agreement |
| Fresh-process data vector | Bitwise agreement |

These are software regression guards, not DES Y6 accuracy requirements.
Assessing physical accuracy needs an appropriate covariance and independent
interpolation, quadrature and model checks. Passing the reduced covariance
assembly test does not establish those conditions.

The [initial port validation record](../validation/20261007.json) preserves the
build checks, full-matrix comparisons, notebook fingerprints and their scope.

The [expanded example/accuracy validation](../validation/hybrid_accuracy_20261007.json)
records 35 data-vector checks and the separate covariance adapter check.

## FAQ: How can users replace a reference? <a name="replace-reference"></a>

Only the documented generator writes the reference state and its manifest.
Replace them after reviewing an intended physics or input change, never to
hide an unexplained failure. If that is the case, follow the steps below.

We assume the project is enabled in `set_installation_options.sh` and
compiled, the Cocoa Conda environment is active, the shell is Bash, and the
current folder is `cocoa/Cocoa/`.

**Step :one:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :two:**: regenerate the reference vectors and manifest.

```bash
python ./projects/des_y6/tests/generate_frozen_reference.py --overwrite
```

**Step :three:**: inspect the reference changes.

```bash
git -C ./projects/des_y6 diff -- tests/frozen tests/manifest_sha256.json
```

**Step :four:**: rerun the data-vector checks.

```bash
python -m pytest ./projects/des_y6/tests/data_vector
```

The generator preserves previous references in the ignored
`tests/.reference_backups/` folder. Use the [separate covariance flow](../covariance/README.md)
when the change also affects covariance calculations.
