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
| CLI and notebook code paths agree bitwise | Production and wrapper backends, six-entry forecast |
| OpenMP outputs agree bitwise | One versus eight threads, one BLAS thread |
| Saved total, row indices and survey settings agree with memory | NPZ output |

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


The [LSST shared component suite](https://github.com/CosmoLike/cocoa_lsst_y1/tree/main/tests/covariance)
checks the common kernels independently. This project uses the same shared
`check_project_forecast` helper as the other survey adapters.
