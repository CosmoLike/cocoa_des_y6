# Table of contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Computing data vectors](#data-vectors)
4. [Computing covariances](#computing_covariances)
5. [Exploring notebooks](#notebooks)
6. [Running the tests](#tests)
7. [Appendix](#appendix)
   1. [FAQ: Which inputs do the examples use?](#inputs)
   2. [FAQ: Which accuracy settings are available?](#accuracy)
   3. [FAQ: What does the model include?](#model)

# Overview <a name="overview"></a>

This project evaluates DES Y6 galaxy clustering, galaxy–galaxy lensing and
cosmic shear in [CoCoA](https://github.com/CosmoLike/cocoa). It provides Cobaya
data-vector likelihoods, a covariance forecast CLI, notebooks and tests.

> [!WARNING]
> **CLI for production; notebook wrappers for exploration.** Run production
> and HPC calculations from YAML through the `_interface` bindings.
> Notebook `_wrapper` APIs expose intermediate quantities through Armadillo,
> pybind11 and CARMA. Both routes call the same C kernels; copying and
> rearranging notebook arrays adds overhead.

> [!IMPORTANT]
> **The shipped likelihood uses dummy data.** `DESY6.dataset` selects a
> 1,300-entry dummy vector, identity covariance and all-ones mask. Its
> $`\chi^2`$ is not a fit to DES observations. The separate `DESY6.cov` has
> 1,690 entries and no verified mapping to this layout; it is neither
> activated nor cropped. See [the input inventory](data/README.md).

# Installation <a name="installation"></a>

Use the [main CoCoA installation recipe](https://github.com/CosmoLike/cocoa#required_packages_conda)
for dependencies. We assume the Cocoa Conda environment is active, the
shell is Bash, and the current folder is `cocoa/Cocoa/`. This project requires
the current shared CosmoLike core.

**Step :one:**: clone the project if it is not already installed.

```bash
git clone https://github.com/CosmoLike/cocoa_des_y6.git projects/des_y6
```

**Step :two:**: comment out `export IGNORE_COSMOLIKE_DES_Y6_CODE=1` in
`set_installation_options.sh` before starting Cocoa.

**Step :three:**: activate Cocoa's private environment and likelihood links.

```bash
source start_cocoa.sh
```

**Step :four:**: compile the project interface.

```bash
source ./projects/des_y6/scripts/compile_des_y6.sh
```

> [!WARNING]
> CosmoLike supports the optimized strict-IEEE default build and
> `COSMOLIKE_DEBUG_MODE`. `COSMOLIKE_AGGRESSIVE_MODE` is retired because its
> fast-math configuration produced incorrect covariance inverses. Unset it
> before compiling. Do not enable `-ffast-math`, `-Ofast`,
> `-funsafe-math-optimizations`, `-fassociative-math`, `-ffinite-math-only`,
> `-freciprocal-math`, `-fno-signed-zeros` or `-fno-trapping-math`.

# Computing data vectors <a name="data-vectors"></a>

We assume the project is enabled in `set_installation_options.sh` and
compiled, the Cocoa Conda environment is active, the shell is Bash, and the
current folder is `cocoa/Cocoa/`.

**Step :one:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :two:**: select the OpenMP team size.

```bash
export OMP_NUM_THREADS=8
```

**Step :three:**: evaluate the 3×2pt fiducial.

```bash
cobaya-run ./projects/des_y6/EXAMPLE_EVALUATE1.yaml --force
```

**Step :four:**: evaluate the cosmic-shear fiducial.

```bash
cobaya-run ./projects/des_y6/EXAMPLE_EVALUATE2.yaml --force
```

| Calculation | Written vector |
|---|---|
| 3×2pt | `EXAMPLE.modelvector` |
| Cosmic shear | `EXAMPLE_SHEAR.modelvector` |

> [!TIP]
> For the row ordering and dummy-data limits, see [which inputs the examples use](#inputs).

# Computing covariances <a name="computing_covariances"></a>

The forecast saves Gaussian (G), super-sample (SSC), connected non-Gaussian
(cNG) and total matrices in the full 1,300-entry real-space layout.
The [covariance guide](covariance/README.md) records its survey assumptions,
model limits, output files and numerical controls.

We assume the project is installed, the Cocoa Conda environment is active,
the shell is Bash, and the current folder is `cocoa/Cocoa/`.

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

**Step :five:**: select the OpenMP team size.

```bash
export OMP_NUM_THREADS=8
```

**Step :six:**: generate the covariance at the YAML's fixed cosmology.

```bash
python ./projects/des_y6/covariance/compute_covariance.py ./projects/des_y6/EXAMPLE_EVALUATE_COVARIANCE.yaml
```

- `--output PATH`: save to another NPZ file.
- `--overwrite`: replace an existing output explicitly.

No thread count belongs in the YAML; use `OMP_NUM_THREADS`. Generated NPZ
files under `covariance/` are ignored by Git. Ordinary likelihood evaluation
uses its supplied covariance and never generates a forecast.

# Exploring notebooks <a name="notebooks"></a>

**Armadillo** was chosen to make a convenient Python API for notebook
exploration. It is a C++ library for vectors, matrices and three-dimensional
arrays called cubes. This small interface layer connects them to NumPy
through **pybind11**, with **CARMA** handling array conversion.

We assume the project is enabled in `set_installation_options.sh`, its
required interfaces are compiled, the Cocoa Conda environment is active,
the shell is Bash, and the current folder is `cocoa/Cocoa/`. The covariance
notebook requires the covariance interface to be compiled.

**Step :one:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :two:**: select the OpenMP team size.

```bash
export OMP_NUM_THREADS=8
```

**Step :three:**: start Jupyter.

```bash
jupyter notebook --no-browser --port=8888
```

**Step :four:**: open the printed URL and select a notebook below.

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

| Notebook | Contents |
|---|---|
| [3×2pt data vectors](EXAMPLE_EVALUATE1.ipynb) | Spectra, correlations and intermediate quantities from the example likelihood. |
| [Cosmic shear](EXAMPLE_EVALUATE2.ipynb) | Shear spectra and correlations with the source-only YAML settings. |
| [Covariance](EXAMPLE_EVALUATE_COVARIANCE.ipynb) | Separate halo terms, component maps, matrix diagnostics and scale selection. |

> [!NOTE]
> Choose the Python kernel from the activated Cocoa environment. Restart it
> after recompiling an interface.

# Running the tests <a name="tests"></a>

The tests check NLA/TATT vectors, input validation, cache consistency and
covariance assembly; [the test guide](tests/README.md) explains their scope.
We assume the project is enabled in `set_installation_options.sh` and its
covariance interface is compiled, the Cocoa Conda environment is active,
the shell is Bash, and the current folder is `cocoa/Cocoa/`.

**Step :one:**: activate Cocoa's private environment.

```bash
source start_cocoa.sh
```

**Step :two:**: run the data-vector tests.

```bash
python -m pytest ./projects/des_y6/tests/data_vector
```

**Step :three:**: run the separate covariance tests.

```bash
python -m pytest ./projects/des_y6/tests/covariance
```

# Appendix <a name="appendix"></a>

## FAQ: Which inputs do the examples use? <a name="inputs"></a>

The active [dataset descriptor](data/DESY6.dataset) preserves four source
bins, six lens bins and 26 angular bins from 2.5 to 995.267926 arcmin.
Both evaluate examples write the same ordering; the source-only example
sets unselected galaxy entries to zero.

| Block | Entries before cuts |
|---|---:|
| Cosmic shear $`\xi_+`$ | 260 |
| Cosmic shear $`\xi_-`$ | 260 |
| Galaxy–galaxy lensing $`\gamma_t`$ | 624 |
| Galaxy clustering $`w(\theta)`$ | 156 |

The n(z) files store **lower bin edges**; retain
`photoz_zmid_convention: 0`. The optional `DESY6.mask` selects 541 entries,
while the active dummy descriptor selects all 1,300. The supplied 1,690-entry
`DESY6.cov` has no verified row mapping to either selection.

## FAQ: Which accuracy settings are available? <a name="accuracy"></a>

| Data-vector setting | Control |
|---|---|
| `accuracyboost` | Overall interpolation-table refinement. |
| `integration_accuracy` | Line-of-sight quadrature refinement. |
| `internal_accuracyboost` | C-FAST-PT convolution grid refinement. |
| `nonlimber_accuracyboost` | Non-Limber distance grid refinement. |
| `pk_z_refinement` | Nested redshift refinement of the power tables. |
| `adopt_limber_gg` | Clustering Limber switch. |
| `adopt_limber_gs` | Galaxy–shear Limber switch. |
| `photoz_interpolation_type` | n(z) interpolation method. |
| `photoz_zmid_convention` | n(z) coordinate convention. |

The likelihood uses growth at $`k=0.05\,\mathrm{Mpc}^{-1}`$ and supplies both
total-matter and cold-matter-plus-baryon power to the core. Ordinary
data-vector tables retain 1,500 wavenumbers at boost 1. The
[covariance power refinement](covariance/README.md#accuracy) is separate.

## FAQ: What does the model include? <a name="model"></a>

- **Data vectors:** preserve the project's cosmology, nuisance priors and photo-z shifts.
- **Gaussian covariance:** supports non-Limber clustering/galaxy–shear spectra and NLA/TATT.
- **SSC and cNG:** use massless neutrinos, zero IA, linear galaxy bias and Limber projection.

The data-vector YAMLs select NLA; `IA_model: 1` selects TATT with its
additional amplitudes in [the source parameters](likelihood/params_source.yaml).
The covariance YAML selects zero IA and a spherical-cap footprint. It is
an analogous forecast, not a reconstruction of the DES survey likelihood.
The DES Y6 calibration-mode likelihood and a trained hybrid emulator are
not distributed here.

Data-vector spectra use their own wrappers; covariance spectra use separate
all-pairs calculations.

The [original Cosmosis comparison](https://github.com/joaoreboucas1/y6_code_comparison)
used altered spin prefactors. This project now uses the shared core's
standard prefactors; the earlier comparison does not validate the current
settings.
