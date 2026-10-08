# Table of contents

1. [Overview](#overview)
2. [Running the production CLI](#command-line)
3. [Running the notebook](#notebook)
4. [Output files and figures](#outputs)
5. [Running the tests](#tests)
6. [Appendix](#appendix)
   1. [FAQ: Which survey inputs are used?](#survey-inputs)
   2. [FAQ: Which measurement layout is used?](#layout)
   3. [FAQ: What does the calculation include?](#model)
   4. [FAQ: Which accuracy settings are available?](#accuracy-settings)
   5. [FAQ: How can users check numerical accuracy?](#accuracy)

# Overview <a name="overview"></a>

The [evaluate YAML](../EXAMPLE_EVALUATE_COVARIANCE.yaml) and
[notebook](../EXAMPLE_EVALUATE_COVARIANCE.ipynb) generate an analogous DES Y6
3×2pt covariance. Gaussian (G), super-sample (SSC), connected non-Gaussian
(cNG) and total matrices are saved separately. The default real-space
forecast has **1,300 entries before cuts**.

> [!WARNING]
> **CLI for production; notebook wrappers for exploration.** The CLI calls
> the production interface. The notebook exposes intermediate arrays through
> Armadillo, pybind11 and CARMA, with conversion and rearrangement overhead.
> Both routes use the same C kernels and survey settings.

> [!IMPORTANT]
> This forecast is not the supplied DES survey likelihood covariance.
> The active dataset contains dummy data and a 1,300-entry identity
> covariance. The separate `DESY6.cov` has 1,690 entries and no verified
> mapping to this layout. It is not loaded, truncated or overwritten.

# Running the production CLI <a name="command-line"></a>

We assume Cocoa and this project are installed, the Cocoa Conda environment
is active, the shell is Bash, and the current folder is `cocoa/Cocoa/`.
Covariance generation is omitted by the default build.

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

**Step :six:**: compute the YAML's covariance.

```bash
python ./projects/des_y6/covariance/compute_covariance.py ./projects/des_y6/EXAMPLE_EVALUATE_COVARIANCE.yaml
```

- `--output PATH`: select another NPZ output file.
- `--overwrite`: replace an existing output explicitly.

The YAML uses Cobaya's reader and `theory`, `params` and `sampler: evaluate`
blocks to specify one fixed cosmology. No MCMC sampler runs. Relative paths
start in `cocoa/Cocoa/`. Set threads only through `OMP_NUM_THREADS`; the
runner fixes BLAS to one thread.

> [!NOTE]
> To retain covariance generation across sessions, comment out
> `export IGNORE_COSMOLIKE_DES_Y6_COVARIANCE=1` in `set_installation_options.sh`.
> Likelihood evaluation with a supplied covariance remains available when
> covariance generation is omitted.

# Running the notebook <a name="notebook"></a>

We assume the project is enabled in `set_installation_options.sh` and its
covariance interface is compiled, the Cocoa Conda environment is active,
the shell is Bash, and the current folder is `cocoa/Cocoa/`.

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

**Step :four:**: open the printed URL and select
`projects/des_y6/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

> [!NOTE]
> Select the Python kernel from the activated Cocoa environment and restart
> it after recompilation. The notebook takes its OpenMP team size from
> `OMP_NUM_THREADS` and keeps BLAS at one thread.

# Output files and figures <a name="outputs"></a>

| File under `covariance/` | Contents |
|---|---|
| `production_covariance.npz` | CLI result: full G, SSC, cNG, total, signal, coordinates, ordering, settings and stage timings. |
| `forecast_real.npz` | Notebook result: full matrices and resolved inputs. |
| `forecast_selected.npz` | Notebook result after the optional 541-entry selection, with original row indices. |
| `forecast_camb.npz` | Notebook's prepared CAMB power and background arrays. |

Generated NPZ files in this folder are ignored by Git. The final notebook
cell replaces its own output files; files under `data/` remain unchanged.

| Notebook figure | Interpretation |
|---|---|
| Matter trispectra | Separate 1h, combined 2h, 3h, 4h and summed terms before survey projection. |
| Total correlation matrix | Total covariance divided by its diagonal rms products. |
| Component maps | G, SSC and cNG, normalized by the total diagonal rms products. |

The notebook reports matrix positivity before and after applying
`DESY6.mask` to both axes. Positivity is a matrix diagnostic; it does not
establish physical fidelity or integration convergence. The
[test guide](../tests/covariance/README.md) describes the reduced assembly
checks separately.

# Running the tests <a name="tests"></a>

See the [test guide](../tests/covariance/README.md) for the small adapter
checks and their limits. They do not establish full-survey convergence.

We assume Cocoa and this project are installed, the Cocoa Conda environment
is active, the shell is Bash, and the current folder is `cocoa/Cocoa/`.

**Step :one:**: activate Cocoa.

```bash
source start_cocoa.sh
```

**Step :two:**: enable the covariance build.

```bash
unset IGNORE_COSMOLIKE_DES_Y6_COVARIANCE
```

**Step :three:**: compile.

```bash
source ./projects/des_y6/scripts/compile_des_y6.sh
```

**Step :four:**: run the covariance checks.

```bash
python -m pytest ./projects/des_y6/tests/covariance
```


# Appendix <a name="appendix"></a>

## FAQ: Which survey inputs are used? <a name="survey-inputs"></a>

The adapter uses the published final-sample per-bin densities and shape
dispersions from [DES Y6 cosmological constraints, Table 2](https://arxiv.org/html/2601.14559v1#S2.T2).
Bin numbers follow the columns in the supplied n(z) files.

| Source bin | Effective density (arcmin⁻²) | Per-component shape dispersion |
|---|---:|---:|
| 1 | 2.05 | 0.265 |
| 2 | 2.10 | 0.287 |
| 3 | 2.14 | 0.282 |
| 4 | 2.32 | 0.347 |

| Lens bin | Density (arcmin⁻²) | Fiducial linear bias |
|---|---:|---:|
| 1 | 0.128 | 1.54 |
| 2 | 0.092 | 1.81 |
| 3 | 0.097 | 1.85 |
| 4 | 0.123 | 1.76 |
| 5 | 0.096 | 1.93 |
| 6 | 0.097 | 1.90 |

The bias values follow the project's fiducial and
[DES Y6 framework, Table 3](https://arxiv.org/html/2601.14859v1).
Shape dispersion is already per component, using the convention in
[the shear catalog, Section 4.6](https://arxiv.org/html/2501.05665v2#S4.SS6);
no extra square-root-of-two conversion is applied.

| Input | Adopted value or convention |
|---|---|
| Joint footprint area | 4,031.04 deg² |
| Footprint model | Spherical cap with that area |
| Lens redshift shapes | `data/DESY6_lens.nz`, six bins |
| Source redshift shapes | `data/DESY6_source.nz`, four bins |
| n(z) coordinate convention | Lower redshift edges, `photoz_zmid: 0` |
| Photo-z shifts | Zero |
| Multiplicative shear calibration | Zero |

The area comes from [the DES Y6 joint footprint paper](https://arxiv.org/html/2509.07943v2).
A spherical cap retains its area, not the survey's holes or angular mask
power. Normalized n(z) tables specify radial shapes; their integrals do not
supply the catalog number densities above. The original redshift files and
their tails are preserved; see [the input inventory](../data/README.md).

## FAQ: Which measurement layout is used? <a name="layout"></a>

The real-space forecast follows the active `DESY6.dataset` binning, with
26 logarithmic angular bins from 2.5 to 995.267926 arcmin. This is the
repository's layout, not a claim to reconstruct the published likelihood.

| Block in output order | Tomographic pairs | Entries |
|---|---:|---:|
| Cosmic shear $`\xi_+`$ | 10 source pairs | 260 |
| Cosmic shear $`\xi_-`$ | 10 source pairs | 260 |
| Galaxy–galaxy lensing $`\gamma_t`$ | All 24 lens-source pairs | 624 |
| Galaxy clustering $`w(\theta)`$ | Six lens auto-correlations | 156 |

The active dummy likelihood uses `ones.mask`. The notebook's optional
`DESY6.mask` selection retains 541 forecast entries and applies to this
real-space ordering only. Neither choice activates the separate 1,690-entry
covariance. The alternative `DESY6_dummy.dataset` ends at 250 arcmin and
therefore defines a different layout.

Setting `covariance.space: fourier` selects an illustrative 600-entry
forecast: 40 spectra in 15 integer multipole bands spanning 30–4,000.
It has no separate $`\xi_-`$ block. Do not apply the real-space mask to it.

## FAQ: What does the calculation include? <a name="model"></a>

| Forecast input | Default |
|---|---|
| `omegam` | 0.319 |
| `omegab` | 0.049 |
| `H0` | 67 km s⁻¹ Mpc⁻¹ |
| `ns` | 0.96 |
| `As_1e9` | 2.1 |
| `w` | −1 |
| `w0pwa` | −1, representing $`w_0+w_a`$ |
| `mnu` | 0 eV |
| `halofit_version` | `takahashi` |
| `gaussian.nonlimber` | `true` |
| `gaussian.ia` | `none` |

- **Gaussian:** non-Limber clustering and galaxy–shear spectra are available.
- **Gaussian:** shear–shear and higher-order TATT spectra retain Limber projection.
- **SSC and cNG:** retain the zero-IA, linearly biased, Limber matter model.
- **All components:** use massless neutrinos and zero magnification/RSD.

The forecast's galaxy factors are linear bias, not an HOD tracer model.
Real-space transformations use the shared full-sky kernels. The massless
forecast and Takahashi power prescription differ from the data-vector YAML's
massive-neutrino HMcode settings.

Gaussian IA can be set to `NLA` with `A1`, or `TATT` with `A1`, `A2` and
`B_TA`. Each amplitude is a scalar or a four-element source-bin list.
A list specifies per-bin amplitudes, not an amplitude/redshift-slope pair.
The core supplies its growth dependence; these choices do not add IA to
SSC or cNG.

Matter-halo integrals use the shared production mass panels from
$`10^{-40}`$ to $`10^{17}`$ solar masses/h. Wynn extrapolation estimates the
remaining low-mass tail of $`I_{11}`$, with residual completion of its
zero-wavenumber response. Higher moments retain direct integrals. See
[the core covariance documentation](https://github.com/CosmoLike/cocoa-cosmolike-core/tree/main/cosmolike/covariances)
for that prescription and its limits.

## FAQ: Which accuracy settings are available? <a name="accuracy-settings"></a>

Baseline values come from [`default.yaml`](default.yaml). Override them in
the evaluate YAML's `covariance` block or pass them to `survey.configuration`.
Reinitialize notebook inputs after changing the settings.

| Control | Baseline | Purpose |
|---|---:|---|
| `accuracy_boost` | `1` | Overall table and cutoff refinement; 1, 2, 4 or 8. |
| `integration_accuracy` | `0` | Independent quadrature level, 0 through 4. |
| `power_accuracyboost` | `8` | Subdivisions of input log-k intervals; multiplied by the global boost. |
| `ell_max` | `100000` | Real-space transform cutoff; Fourier measurement bands stay fixed. |
| `mask_ell_max` | `32768` | Survey-footprint spectrum cutoff. |
| `ng_ell_intervals` | `127` | Base log-multipole intervals for non-Gaussian interpolation. |
| `non_gaussian_accuracyboost` | `1` | Refinement of the non-Gaussian multipole grid. |
| `window_accuracyboost` | `1` | Lensing-window interpolation refinement. |
| `core_accuracyboost` | `1` | Shared core interpolation refinement for this calculation. |
| `response_step` | `5e-05` | Half-width of the log-k response derivative; divided by the global boost. |
| `nonlimber_lmax` | `1000` | Gaussian gg/gs non-Limber correction cutoff. |
| `nonlimber_accuracyboost` | `1` | Gaussian non-Limber distance-grid refinement. |

| `integration_accuracy` | Ordinary nodes per panel | Wynn-tail nodes per panel |
|---|---:|---:|
| 0 | 96 | 32 |
| 1 | 128 | 64 |
| 2 | 256 | 128 |
| 3 | 512 | 256 |
| 4 | 1024 | 512 |

The default power refinement prepares 11,993 samples from 1,500 inputs for
linear, nonlinear and cold-matter power. It refines interpolation, not the
Boltzmann solution itself, and does not affect ordinary data-vector runs.
Thread counts are environment settings, not accuracy or YAML keys.

Compare G, SSC, cNG and total, including off-diagonal elements, positivity
and generalized variance ratios. Keep the cosmology, catalog and measured
bins fixed while testing one control at a time. Integration, interpolation,
cutoff and parameter-error convergence are separate questions.

## FAQ: How can users check numerical accuracy? <a name="accuracy"></a>

The baseline lives in [`default.yaml`](default.yaml). Override its controls
inside the evaluate YAML's `covariance` block and save each result to a
separate output. Reinitialize notebook inputs after changing accuracy.

Compare G, SSC, cNG and total separately, including off-diagonal entries,
positivity and generalized variance ratios. Keep the cosmology, catalog
inputs and measurement bins fixed while varying one numerical control at a
time. Check quadrature, interpolation, transform cutoffs and non-Limber
sampling independently. A higher setting or agreement in a small assembly
test does not establish DES Y6 full-matrix or Fisher convergence.
