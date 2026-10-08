"""This module holds the DES Y6 choices of the shared covariance forecast.

It is the project adapter of cosmolike_notebook_utils.covariance: the
shared code holds the numerical algorithms, and this module supplies the
DES Y6 facts through three functions. configuration returns the
cosmology, the survey description (6 lens and 4 source n(z) files, number
densities, shape noise, footprint area, bins) and the accuracy settings;
initialize installs them in the compiled interface; compute returns the
Gaussian (G), super-sample (SSC) and connected non-Gaussian (cNG)
covariances and their total. compute_covariance.py, the covariance
notebook and the covariance tests call these functions. The example is a
forecast with massless neutrinos and explicit Gaussian non-Limber and IA
choices; it does not reproduce the project's frozen likelihood.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    The covariance README records the catalog assumptions and their
    sources. The n(z) files give the shape of each redshift distribution;
    the number densities below, not the file normalization, set the noise.

    Arguments:
        accuracy_boost = None uses default.yaml; 1, 2, 4 or 8 refines it
        gaussian = optional mapping of the Gaussian-part model: nonlimber
                   (bool), ia (none, NLA or TATT), A1, A2 and B_TA (see
                   forecast.gaussian_model)
        accuracy_overrides = named internal controls from default.yaml,
                   given as keyword arguments (**accuracy_overrides
                   collects them into a dict)

    Returns:
        dict of fully resolved settings, including the unboosted accuracy
        parameters.

    Raises:
        ValueError for invalid accuracy settings (load_covariance_accuracy)
        or an invalid Gaussian model (gaussian_model).
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # 16 logarithmic edges from ell = 30 to 4001, rounded to integers, make
    # 15 bands; band b covers band_first[b] <= ell <= band_last[b]. These
    # inclusive bands count every integer multipole once.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once.
    # As_1e9 = 1e9 A_s, w0pwa = w0 + wa (so wa = 0), and mnu = 0 eV: the
    # shared forecast requires massless neutrinos.
    settings = {
        "cosmology": {
            "omegam": 0.319,
            "omegab": 0.049,
            "H0": 67.0,
            "ns": 0.96,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # File columns describe radial shapes; these flags fix their z
        # convention: cubic-spline interpolation (0) of a z column that
        # holds lower bin edges (0), as in the likelihood.
        "lens_file": "data/DESY6_lens.nz",
        "source_file": "data/DESY6_source.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # Measured pair and band choices are fixed during accuracy
        # refinement. excluded_gammat = [] keeps all 24 lens-source pairs;
        # lnm_edges are the halo-mass integration panels, ln(M/[Msun/h]).
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # DES Y6 cosmology Table 2 (arXiv:2601.14559): densities per
        # arcmin^2 and per-component dispersion. The joint footprint
        # is 4031.04 deg^2 (arXiv:2509.07943). See covariance/README.md.
        # bias holds one linear bias per lens bin, the ref centers of
        # likelihood/params_lens.yaml.
        "area_deg2": 4031.04,
        "lens_density_arcmin2": [0.128, 0.092, 0.097, 0.123, 0.096, 0.097],
        "source_density_arcmin2": [2.05, 2.10, 2.14, 2.32],
        "sigma_e_component": [0.265, 0.287, 0.282, 0.347],
        "bias": [1.54, 1.81, 1.85, 1.76, 1.93, 1.90],

        # 27 edges of the 26 logarithmic angular bins of DESY6.dataset.
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=995.2679263837432, num=27),
        # Radial panels of the covariance's Limber integrals, as scale-factor
        # edges a = 1/(1+z) for z = 3.1 down to 1e-5: a increases from the
        # distant boundary toward the observer.
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_des_y6_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Raises:
        As initialize_forecast: RuntimeError when the build lacks
        covariance support; FileNotFoundError or ValueError for missing or
        invalid inputs, before the interface state changes.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    Arguments:
        interface = the compiled project module, already initialized
        settings = the mapping configuration() returns
        space = "real" (angular bins) or "fourier" (E-mode bandpowers)
        rows = optional int32 [n_rows, 3] table (type, A, B) of the
               measured rows to keep; None keeps the full layout
        progress = optional function called as progress(stage,
                   elapsed_seconds)
        backend = None for the notebook wrappers, interface.covariance for
                  the direct bindings the command line uses
    Returns:
        the shared forecast dict: covariance components, mean signals,
        coordinates and resolved settings. The full real-space layout has
        1300 entries; the Fourier layout has 600: 15 bandpowers for each of
        the 10 shear, 24 lens-source and 6 clustering pairs.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
