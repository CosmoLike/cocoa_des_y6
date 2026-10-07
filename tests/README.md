# Table of contents

1. [Data-vector checks](#data-vector)
   1. [Running the data-vector checks](#run-data-vector)
2. [Covariance checks](#covariance)
   1. [Running the covariance checks](#run-covariance)
3. [Appendix](#appendix)
   1. [FAQ: Which inputs are tested?](#inputs)
   2. [FAQ: What do the reference comparisons mean?](#references)
   3. [FAQ: How can users replace a reference?](#replace-reference)

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

We assume the project is enabled in `set_installation_options.sh` and
compiled, the Cocoa Conda environment is active, the shell is Bash, and the
current folder is `cocoa/Cocoa/`. Test workers
use four OpenMP threads and one BLAS thread.

## Running the data-vector checks <a name="run-data-vector"></a>

**Step :one:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :two:**: run the data-vector checks.

```bash
python -m pytest ./projects/des_y6/tests/data_vector
```

# Covariance checks <a name="covariance"></a>

The covariance test uses the project's n(z), densities and shape dispersions
with a six-entry forecast and reduced numerical settings. It checks the
interface and matrix assembly, not full-survey convergence.

| Check | Configuration |
|---|---|
| Full layout has 1,300 entries | Real space |
| Full layout has 600 entries | Fourier space |
| Scientific bins stay fixed under accuracy refinement | Shared forecast settings |
| G, SSC, cNG and total are finite and symmetric | Six-entry forecast |
| Total equals the sum of components | Six-entry forecast |
| Total is positive definite | Six-entry forecast |
| CLI and notebook outputs agree bitwise | Production and wrapper interfaces |
| OpenMP outputs agree bitwise | One versus eight threads, one BLAS thread |
| Saved matrices, row indices and settings agree with memory | NPZ output |

We assume the project is installed, the Cocoa Conda environment is active,
the shell is Bash, and the current folder is `cocoa/Cocoa/`. Run these tests
separately from the data-vector tests because the calculations use different
compiled-library state.

## Running the covariance checks <a name="run-covariance"></a>

**Step :one:**: comment out `export IGNORE_COSMOLIKE_DES_Y6_CODE=1` in
`set_installation_options.sh` before starting Cocoa.

**Step :two:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :three:**: enable covariance generation.

```bash
unset IGNORE_COSMOLIKE_DES_Y6_COVARIANCE
```

**Step :four:**: compile the project interface.

```bash
source ./projects/des_y6/scripts/compile_des_y6.sh
```

**Step :five:**: run the covariance checks.

```bash
python -m pytest ./projects/des_y6/tests/covariance
```

> [!NOTE]
> If covariance bindings were deliberately omitted, the covariance test
> skips. The commands above build those bindings before testing.

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

The [initial port validation record](validation/20261007.json) preserves the
build checks, full-matrix comparisons, notebook fingerprints and their scope.

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
`tests/.reference_backups/` folder. Use the separate covariance flow above
when the change also affects covariance calculations.
