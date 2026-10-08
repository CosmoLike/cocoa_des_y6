# DES Y6 input files

The files in this directory are preserved from the project input release.
Forecast construction does not overwrite them.

| File | Role |
|---|---|
| `DESY6.dataset` | Active example: six lens bins, four source bins, 26 angles from 2.5 to 995.267926 arcmin. |
| `dummy.modelvector`, `dummy.cov`, `ones.mask` | 1,300-entry placeholder vector, identity covariance and uncut selection. |
| `DESY6_lens.nz`, `DESY6_source.nz` | Six/four n(z) columns, with lower redshift edges in the first column. |
| `DESY6.mask` | Optional real-space selection: 541 of the 1,300 entries. |
| `DESY6.cov` | Separate 1,690-entry covariance; not selected by the active descriptor. |
| `DESY6_dummy.dataset` | Alternative dummy descriptor ending at 250 arcmin; a different angular layout. |
| `baryons_logPkR.h5` | Simulation power-ratio tables for optional baryon calculations. |

The 1,300-entry order is ξ+ (260), ξ− (260), γt (624), w (156).
`DESY6.mask` retains 152, 58, 272 and 59 entries in those groups. All
24 lens-source pairs and six lens auto-correlations are present before cuts.

The separate 1,690-entry covariance cannot be combined with that vector by
truncation. No local metadata establishes its tomographic mapping. The
likelihood checks index bounds before passing files to the C++ layer;
using incompatible files raises an error.

Both n(z) files have 299 samples, separated by Δz = 0.01. The generator
subtracts 0.005 from the original centers, so use the **lower-edge** reading
convention (`photoz_zmid_convention: 0`). The original tiny negative lens
tails and terminal source-bin values have not been edited.

The data vector is a placeholder: neither its likelihood value nor a
comparison to its unit covariance measures agreement with DES observations.
The [forecast catalogue assumptions](../covariance/README.md#survey-inputs)
are recorded separately from these file-layout facts.
