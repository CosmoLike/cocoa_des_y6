"""This module defines the cobaya likelihood shared by the DES Y6 combinations.

The class _cosmolike_prototype_base computes ln L for one probe selection of
the DES Y6 real-space 3x2pt data vector. Five thin subclasses, one module
each, choose the selection: cosmic_shear ("xi"), combo_2x2pt, combo_3x2pt,
combo_xi_gg and combo_xi_ggl. cobaya imports them as
cobaya.likelihoods.des_y6.<module> because start_cocoa.sh links this folder
into cobaya's likelihood folder. The module is not run directly.

Terms:
  cobaya = the Python framework that runs the analysis. It reads a yaml file,
      builds the theory codes (CAMB or emulators) and the likelihoods, and
      asks each likelihood for ln L at every parameter point a sampler
      proposes.
  ln L = -chi2/2 with chi2 = (d - m)^T C^-1 (d - m): d is the data vector,
      m the model vector and C the covariance (constant terms dropped).
  3x2pt = three kinds of angular two-point correlation functions: cosmic
      shear xi_+(theta) and xi_-(theta) between source-galaxy shapes,
      galaxy-galaxy lensing gamma_t(theta) between lens positions and source
      shapes, and galaxy clustering w(theta) between lens positions.
  tomographic bin = a slice of a galaxy sample in photometric redshift,
      described by its redshift distribution n(z); DES Y6 has 4 source bins
      and 6 lens bins (data/DESY6.dataset).
  ci = the compiled C++ module cosmolike_des_y6_interface, built from
      interface/interface.cpp. It keeps the survey and cosmology state in C
      global variables shared by the whole Python process.

Data-vector layout, 26 angular bins per correlation function:
  xi_+ (10 source pairs, 260 entries), xi_- (260), gamma_t (6 x 4
  lens-source pairs, 624), w (6 lens bins, 156): 1300 entries. Entries
  removed by the mask or by the probe selection are zero.

One evaluation:
  initialize, once: read the dataset descriptor, check the file layouts,
      build the redshift and wavenumber grids, configure ci;
  get_requirements, once: name the theory quantities cobaya must compute;
  logp, at every point: set_cosmo_related sends P(k, z), the growth factor
      and the distances to ci, set_lens_related and set_source_related send
      the nuisance parameters, and ci returns the model vector and chi2.

Theory paths, chosen by the yaml option use_emulator:
  0 = CAMB computes the matter power spectra and distances; cosmolike
      projects them into the data vector;
  1 = emulator theories supply the xi, gamma_t and w vectors; cosmolike adds
      the shear-calibration and point-mass terms and applies the mask;
  2 = emulators replace CAMB for the background and the matter power
      spectra (the hybrid EXAMPLE_EMUL2 examples); cosmolike projects them.

Units: cobaya supplies k in 1/Mpc, P(k) in Mpc^3 and distances in Mpc; ci
receives k in h/Mpc, P(k) in (Mpc/h)^3 and distances in Mpc/h.
"""

# A __future__ import must precede every other statement; only the module
# docstring and comments may come before it. Under Python 3 these three
# change nothing: absolute imports, true division and print() are the default.
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools
from collections.abc import Mapping

# cobaya's likelihood base class and error type, getdist's ini-file reader,
# and the file-layout check of this package
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile
from .dataset_validation import validate_layout

import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time spent inside a `with timer(label):` block.

  contextlib.contextmanager turns this generator function (a function that
  pauses at `yield`) into a context manager, the object a `with` statement
  uses: the code before `yield` runs when the block starts, the block runs
  at the `yield`, and the code after it runs when the block ends.

  Arguments:
    label = text printed before the elapsed time

  Side effects:
    prints "<label>: <seconds>s" when the block ends.
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

import cosmolike_des_y6_interface as ci

# OpenMP thread count for cosmolike, read once when this module is imported;
# 1 when OMP_NUM_THREADS is not set.
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Return fn wrapped so that cosmolike's OpenMP thread count is reset first.

    OpenMP is the C library that runs cosmolike's loops on several CPU cores
    (threads). Each parallel loop uses the count omp_get_max_threads()
    returns, which starts at OMP_NUM_THREADS. A library loaded in the same
    process can call omp_set_num_threads(1), and every later cosmolike loop
    started from that thread then runs on one core. The wrapper therefore
    calls ci.set_omp_threads(COSMOLIKE_OMP_THREADS) before each call of fn;
    that call also sets the BLAS linear-algebra library to one thread
    (interface.cpp).

    with_omp_threads is a decorator: `@with_omp_threads` above a method
    definition replaces the method by the returned wrapper. functools.wraps
    copies the name and the docstring of fn onto the wrapper.

    Arguments:
      fn = the function or method to wrap

    Returns:
      wrapper, a function that takes the same arguments as fn and returns
      what fn returns.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Reset the OpenMP thread count, then call fn with the same arguments.

        *args collects the positional and **kwargs the keyword arguments of
        the call, so any call is forwarded unchanged.
        """
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every DES nuisance-parameter name in the yaml files (DES_M1,
# DES_DZ_S1, DES_A1_1, DES_B1_1, DES_PM1, ...).
survey = "DES"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Compute the DES Y6 3x2pt ln L for the probe selection of a subclass.

  DataSetLikelihood is the cobaya base class of a likelihood whose input
  files are listed in a dataset descriptor: an ini file of key = value lines
  (data/DESY6.dataset), read here with getdist's IniFile. Before cobaya
  calls initialize, it copies every option of the likelihood's yaml block
  (the defaults in <combination>.yaml merged with the user's yaml) onto the
  instance, so self.accuracyboost is the yaml's accuracyboost.

  Options (yaml keys):
    path, data_file = folder and name of the dataset descriptor;
      data_vector_file and mask_file, when not null, replace the
      descriptor's data_file and mask_file entries
    accuracyboost, integration_accuracy, internal_accuracyboost,
      nonlimber_accuracyboost, pk_z_refinement, lmax, kmax_boltzmann =
      numerical controls (project README, accuracy settings)
    photoz_interpolation_type = n(z) interpolation: 0 cubic spline,
      1 linear, 2 Steffen (a cubic that does not overshoot the samples)
    photoz_zmid_convention = meaning of the n(z) files' z column: 0 lower
      bin edges (the DES Y6 files), 1 bin midpoints
    adopt_limber_gs, adopt_limber_gg = projection of galaxy-galaxy lensing
      and galaxy clustering: 1 Limber approximation at every multipole,
      0 exact (non-Limber) projection below the core's multipole limit
    include_HOD_GX = 1 halo-model galaxy power (needs adopt_limber_gg = 1);
      include_halo_IA = 1 halo-model intrinsic alignment (IA)
    IA_model = 0 NLA (nonlinear alignment: shapes follow the tidal field
      linearly, amplitude A1), 1 TATT (adds tidal torquing, amplitude A2,
      and the density weighting b_TA)
    IA_redshift_evolution = redshift dependence of the IA amplitudes (the
      codes are listed in <combination>.yaml)
    IA_code = source of the one-loop IA and bias spectra: 0 the C FAST-PT,
      1 the Python FAST-PT theory (FAST-PT evaluates perturbation-theory
      integrals with fast Fourier transforms)
    bias_model = six integers, the redshift model of (b1, b2, bs2, b3,
      bmag, bK); 0 = one constant value per lens bin
    non_linear_emul = nonlinear P(k): 0 the linear P(k), 1 EuclidEmulator2,
      2 CAMB's nonlinear model
    use_emulator = theory path (module docstring)
    external_nz_modeling = 1 sends the n(z) tables from Python at every
      evaluation, so a user function can modify them
    external_baryon_suppression, add_baryons_on_dv, which_bsims_add_on_dv,
      use_baryon_pca, create_baryon_pca, baryon_pca_select_sims,
      filename_baryon_pca = baryonic-feedback options (initialize)
    print_datavector, print_datavector_file = write the model vector to a
      text file at every evaluation
    fixed_params = parameters to fix (get_modified_defaults)
    debug = true sets cosmolike's log level to debug instead of info
  """

  @classmethod
  def get_modified_defaults(cls, defaults, input_options={}):
    """Apply the yaml option `fixed_params` to the default parameters.

    cobaya calls this class method (a method that receives the class, cls,
    instead of an instance) when it reads the defaults of a combination
    (its yaml file, e.g. combo_xi_gg.yaml), before it merges them with the
    user's yaml. The parameters of a combination come from
    `params: !defaults [params_lens, params_source]`, and the `!defaults`
    tag builds the whole `params` mapping from those files, so the same
    yaml cannot change one entry of it. A combination or a user yaml that
    fixes some of these parameters lists them under `fixed_params` instead,
    as name: value or name: {info}. The shipped DES Y6 combination files
    set no `fixed_params`, so their defaults pass through unchanged.
    Each entry replaces the parameter's default info with the cobaya merge
    rule: a value drops prior, ref and proposal and keeps the other keys
    (the latex label). A user yaml can override `fixed_params` like any
    other option of the likelihood.

    Arguments:
      defaults = the combination's default options (dict, `params`
                 included), changed in place
      input_options = the user's options for this likelihood (dict)

    Returns:
      defaults, with each parameter of `fixed_params` replaced.
    """
    fixed = input_options.get("fixed_params", defaults.get("fixed_params"))
    params = defaults.get("params") or {}
    for p, info in (fixed or {}).items():
      old = params.get(p)
      new = {}
      if isinstance(old, Mapping):
        # keep every key of the default info except the sampling ones
        for key, value in old.items():
          if key not in ("prior", "ref", "proposal"):
            new[key] = value
      if isinstance(info, Mapping):
        new.update(info)
      else:
        new["value"] = info
      params[p] = new
    if params:
      defaults["params"] = params
    return defaults

  def initialize(self, probe):
    """Read the dataset descriptor, build the grids and configure cosmolike.

    cobaya calls initialize once, after it has copied the yaml options onto
    the instance; each subclass passes its probe selection. In order: read
    the file names and the binning from the descriptor; check that the
    data, mask and covariance files share the full layout; build the
    redshift and wavenumber grids; send the binning, the probe selection,
    the numerical and model choices, the n(z), and the data, mask and
    covariance files to ci; resolve the baryon options.

    Arguments:
      probe = probe selection for ci.init_probes: "xi", "2x2pt", "3x2pt",
              "xi_gg" or "xi_ggl"

    Raises:
      LoggedError (cobaya's error type, which also writes to the log) when
      pk_z_refinement is not a positive integer; ValueError from
      validate_layout when an input file does not match the full layout.

    Side effects:
      sets self.probe, the file names, the bin counts and the grids
      z_interp_1D, z_interp_2D, z_interp_2D_camb and log10k_interp_2D;
      may switch off conflicting baryon options; replaces the global state
      of ci.
    """
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    if self.data_vector_file is None:
      self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    if self.mask_file is None:
      self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    # Probe selection masks a full-layout vector; all input files must
    # therefore describe every bin, even for the shear-only likelihood.
    # Full length = ntheta x (xi_+ and xi_- pairs, ns (ns + 1) in total,
    # + gamma_t pairs nl ns + w auto-correlations nl), with ns source and
    # nl lens bins: 26 x (20 + 24 + 6) = 1300 for data/DESY6.dataset.
    size = self.ntheta * (self.source_ntomo*(self.source_ntomo+1)
                          + self.lens_ntomo*self.source_ntomo + self.lens_ntomo)
    validate_layout(data_file=self.data_vector_file, mask_file=self.mask_file,
                    cov_file=self.cov_file, size=size)

    # ------------------------------------------------------------------------   
    # z_interp_1D: redshift nodes of the comoving-distance table and of the
    # growth factor (set_cosmo_related), in three uniform blocks: [0, 3)
    # with max(100, 0.8 tmp) nodes (dz = 0.003 at accuracyboost = 1),
    # [3, 50.1) with max(100, 0.4 tmp) nodes, and [1070, 1100] with
    # max(50, 0.1 tmp) nodes, which brackets the last-scattering surface
    # (z about 1090).
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # Keep the P(k,z) grids nested: refining a uniform block multiplies
    # its interval count and retains every original sample. The CAMB
    # transfer calculation remains on its own fixed 140-redshift grid;
    # these extra nodes sample cobaya's smooth interpolator through it,
    # below z = 50.
    # getattr(self, name, default) reads a yaml option and returns default
    # when the yaml block does not define it.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    # m = refinement factor of the z_interp_2D blocks: the accuracy boost
    # rounded up to a power of two and capped at 16 (grids of different
    # boosts are then nested in each other), times pk_z_refinement. m = 1
    # gives 105 nodes on [0, 3) (dz = 0.029) and 35 on [3, 49.99]: 140.
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False),
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module caps the number of requested redshifts at
    # 256, so the list handed to CAMB through the Pk_interpolator
    # requirement stays at this boost-independent 140-node grid (the
    # m = 1 grid above). The denser nested nodes only re-evaluate the
    # smooth z-spline cobaya builds through these transfer redshifts when
    # the cosmolike tables are filled, so raising the boost refines only
    # that resampling, and the CAMB side never exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False),
                                            np.linspace(3.0,49.99,35)),axis=0)
    
    # log10 of the wavenumber in 1/Mpc: 1250 + 250 accuracyboost nodes from
    # k = 1.0e-5 to 1e2 1/Mpc (1500 nodes at accuracyboost = 1).
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # The init_* calls below configure ci for this likelihood: defaults
    # first (initial_setup), then the probe selection and the angular
    # binning, ntheta logarithmic bins from theta_min to theta_max (arcmin).
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # Number of comoving-distance (chi) samples of the non-Limber integrals,
    # refined on top of the accuracy boost; narrow lens bins need it (see
    # init_nonlimber_accuracy_boost). The non-Limber projection evaluates
    # its Bessel-function integrals with FFTLog, a fast Fourier transform on
    # a logarithmic grid.
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    # Projection of galaxy-galaxy lensing (gs) and galaxy clustering (gg).
    # 1 = Limber approximation: C_ell reads P(k) only at k = (ell + 1/2)/chi,
    # accurate at high ell and for broad radial kernels; 0 = the exact
    # (non-Limber) projection below the core's multipole limit.
    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 1)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model galaxy power (HOD, the
    # halo occupation distribution: mean galaxy count per halo mass);
    # always set, so a model never inherits the previous model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Halo statistics use cold dark matter + baryons. The emulator path
    # has no separate cb spectrum and uses the documented small-scale ratio.
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    # use_emulator = 1: emulator theories supply the xi, gamma_t and w
    # vectors, and cosmolike adds only the point-mass and shear-calibration
    # terms; it needs the n(z), the data files and a reduced accuracy boost.
    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)  
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling: Python keeps the n(z) tables (self.lens_nz,
      # self.source_nz) and sends a copy at every evaluation, so a user
      # function can modify them (set_lens_related, set_source_related);
      # otherwise cosmolike reads the files once, here.
      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 

      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # NLA (IA_model = 0) uses no FAST-PT intrinsic-alignment terms, so
        # a request for the Python FAST-PT theory (IA_code = 1) is switched
        # to the C implementation, which also computes any one-loop bias
        # term; get_requirements, which cobaya calls after initialize, then
        # asks for no IA_PS. The yaml value changes without a message.
        self.IA_code = 0
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe not in ("xi", "3x2pt_ss_sk_sk", "2x2pt_ss_sk"):
        # bias_model = redshift model of (b1, b2, bs2, b3, bmag, bK); 0 = one
        # constant value per lens bin (cosmolike bias.c). Of the probes
        # listed in the condition only "xi" occurs in this project, so
        # every selection with lens galaxies sets the bias model.
        ci.init_bias(bias_model=self.bias_model)

      if self.non_linear_emul == 1:
        # EuclidEmulator2 predicts the nonlinear boost P_nl/P_lin; its
        # object is built once here and reused at every evaluation.
        self.emulator = ee2.PyEuclidEmulator()

      # Baryonic-feedback prescriptions:
      #   external_baryon_suppression = a theory (for example BCEmu) gives
      #     S(k, z) = P_baryons/P_gravity-only, which multiplies the
      #     nonlinear P(k, z) (set_cosmo_related);
      #   add_baryons_on_dv = cosmolike multiplies its matter power spectrum
      #     by the ratio P_hydro/P_gravity-only of the hydrodynamical
      #     simulation which_bsims_add_on_dv (all_sims_hdf5_file);
      #   use_baryon_pca = the model vector gets sum_i Q_i PC_i, with PC_i
      #     the principal components in baryon_pca_file and Q_i the
      #     parameters DES_BARYON_Q1..Q4;
      #   create_baryon_pca = compute those components from the simulations
      #     baryon_pca_select_sims and write them (internal_get_datavector).
      # Conflicts are resolved without a message: create_baryon_pca turns
      # off external_baryon_suppression and use_baryon_pca and skips
      # add_baryons_on_dv; otherwise external_baryon_suppression turns
      # use_baryon_pca and add_baryons_on_dv off.
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      # four principal-component amplitudes, DES_BARYON_Q1 to
      # DES_BARYON_Q4 (params_source.yaml)
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Return the theory quantities cobaya must compute for this likelihood.

    Each entry of the returned dict is a requirement: the key names a
    quantity a theory code (CAMB or an emulator) provides, the value holds
    its options or None. cobaya calls get_requirements once, after
    initialize; at every point it then has the theory codes compute these
    quantities before logp runs, and the likelihood reads them through
    self.provider.

    Returns:
      dict, by use_emulator:
        1 = the emulated blocks of the probe ('cosmic_shear', 'ggl',
            'wtheta'), with H0 and the comoving distance on z_interp_1D
            when the probe has galaxy-galaxy lensing (point-mass term);
            None for any other probe name
        2 = As, H0, omegam, omegab, mnu, w and wa, the linear and nonlinear
            total-matter P(k, z) on z_interp_2D_camb up to
            k_max = kmax_boltzmann x accuracyboost [1/Mpc], and the
            comoving distance [Mpc] on z_interp_1D
        any other value (0) = as for 2 without mnu, w and wa (requested
            only for EuclidEmulator2), plus omnuh2, the cold dark matter +
            baryon spectrum ("delta_nonu"), a CMB temperature C_l request,
            and, when enabled, the baryon suppression and the Python
            FAST-PT spectra
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      return {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "wa": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
      }
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # The baryon theory computes the suppression factor S(k, z) on the
      # (z, k) values requested here: z_interp_2D, and k in 1/Mpc
      # (log10k_interp_2D is in 1/Mpc; the theory converts to h/Mpc if it
      # needs to).
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # IA_code = 1: the one-loop IA and galaxy-bias spectra come from the
      # Python FAST-PT theory.
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 (non_linear_emul = 1) reads these parameters too.
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # Keep both fields available to the likelihood and direct halo readers.
      # CAMB obtains them from the same transfer-function calculation.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Send the matter power spectra, growth factor and distances to ci.

    Runs at every evaluation; with_omp_threads first restores the OpenMP
    thread count. This is a hot path: the numpy operations act on whole
    arrays. With use_emulator = 1 only the comoving distances are sent: the
    emulated vectors need nothing else. Otherwise, in order: read the
    linear P(k, z) from cobaya's interpolator; build the nonlinear P(k, z)
    chosen by non_linear_emul; apply the external baryon suppression when
    enabled; sample the growth factor; call ci.set_cosmology; with
    IA_code = 1, install the Python FAST-PT tables.

    Shapes and units:
      PKL.logP(z_interp_2D, k) [n_z, n_k]: ln P in Mpc^3, k in 1/Mpc
        -> flatten(order='F') [n_z n_k]: Fortran (column-major) order
           lists the array column by column, so the z index varies fastest
        -> + ln h^3: ln P in (Mpc/h)^3 (lnPL, lnPNL; lnPL_cb alike)
      log10k_interp_2D - log10 h: log10 k with k in h/Mpc
      G_growth [n_growth] on z_growth, the z_interp_1D nodes up to
        z_interp_2D[-1]
      chi = comoving distance [Mpc] x h, in Mpc/h, on z_interp_1D

      legend: n_z = len(z_interp_2D), n_k = len(log10k_interp_2D),
              n_growth = len(z_growth)

    Raises:
      LoggedError for a non_linear_emul other than 0, 1 or 2.

    Side effects:
      replaces the cosmology state of ci.
    """
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      # PKL = cobaya's interpolator of the linear total-matter P(k, z): a
      # bicubic spline in (z, ln k) through CAMB's output; outside CAMB's
      # k range it continues as a power law (a straight line in ln k,
      # ln P) from k = 1e-6 up to 250 accuracyboost 1/Mpc.
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      # ln P_lin in (Mpc/h)^3 on the (z, k) grid, flattened with the z
      # index fastest, the layout ci.set_cosmology expects.
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 0:
        # non_linear_emul = 0: the linear spectrum fills the nonlinear slot
        # too. A copy, because the external baryon suppression below
        # changes lnPNL in place and must leave lnPL unchanged.
        lnPNL = lnPL.copy()
      elif self.non_linear_emul == 1:
        # EuclidEmulator2 (EE2) predicts the boost B(k, z) = P_nl/P_lin
        # from these cosmological parameters.
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EE2 covers z <= 10 and 8.73e-3 <= k <= 9.41 h/Mpc: the boost is
        # requested at the redshifts below 10 and at len(log10k_interp_2D)
        # log-spaced k from 10^-2.0589 = 8.73e-3 to 10^0.973 = 9.40 h/Mpc.
        # get_boost2 returns those k (kbt) and one boost row per redshift.
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # ln B, linear in log10 k, moved onto the likelihood's k grid in
        # h/Mpc (tmp [n_z10, n_k], n_z10 = number of redshifts below 10).
        # Above 9.40 h/Mpc the straight line is extrapolated; below
        # 8.73e-3 h/Mpc the next line sets ln B = 0 (B = 1): those large
        # scales are linear.
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        # lnbt [n_z, n_k]: ln B on the full grid, 0 (B = 1) on z >= 10
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # CAMB's nonlinear spectrum (the model halofit_version names in the
        # yaml) covers every redshift; it is kept on z >= 10 below.
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # On z < 10, ln P_nl = ln P_lin + ln B. Both vectors are reshaped to
        # [n_z, n_k] (order='F' undoes the flattening); (z < 10)[:, None] is
        # a column of booleans broadcast across k; np.where picks, element
        # by element, from the first array where True; ravel(order='F')
        # flattens the result back.
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        # CAMB's nonlinear spectrum at every redshift, with the grid,
        # flattening and units of lnPL.
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", self.non_linear_emul)

      # G(z) = D(z) (1 + z): the linear growth factor D divided by a, its
      # value in a matter-dominated universe. At fixed k,
      # D(z)/D(0) = sqrt(P_lin(z, k)/P_lin(0, k)).
      # G on the dense 1D z grid (clipped to the P(k) interpolator range):
      # cosmolike reads G linearly in z, and on the coarse 2D grid
      # (dz ~ 0.03) the linear read misses D by up to 9e-5 and the
      # growth rate f = 1 - (1+z) dlnG/dz (the slope of the table) by
      # 1%; on the 1D grid (dz = 0.003) by 1e-6 and 0.2%. PKL is a cubic
      # spline in z through CAMB's transfer redshifts, so this asks CAMB
      # for no extra redshifts (about 0.1 ms per evaluation). The table
      # stays divided by G at the last z_2D node (z_growth ends below
      # it); cosmolike's growfac divides by G(0), so D(z=0) = 1.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # G is sampled at growth_k (default 0.05/Mpc), a sub-horizon scale.
      # At k = 5e-4/Mpc (about 2 H0/c) CAMB's dark-energy perturbations
      # change the growth by 0.5-0.9% at w != -1 (z = 0.5 to 2), while every
      # reader of G (IA amplitudes, one-loop D^4, sigma(M, z), the growth
      # rate f) describes sub-horizon modes; with 0.06 eV neutrinos the
      # growth varies by 0.03% above 0.05/Mpc (cosmolike_core skill,
      # references/growth_factor_measurements.md)
      growth_k = float(getattr(self, "growth_k", 0.05))
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # External baryon suppression: the baryon theory returns a dict that
      # maps each requested redshift to S(k) = P_baryons/P_gravity-only on
      # the requested k grid, with its calibration mask already applied;
      # ln S is added to ln P_nl at that redshift. Redshifts are matched by
      # exact float equality with the dict keys. If reading the result
      # fails, the error is logged and the evaluation continues without
      # suppression.
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              # lnPNL[i::n_z] selects every k at redshift index i, since the
              # z index varies fastest in the flattened vector.
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code = 1: one-loop IA and galaxy-bias spectra at z = 0 from the
      # Python FAST-PT theory, k in h/Mpc and P in (Mpc/h)^3. FPTIA [12, N]
      # holds ten IA spectra, then the k row (row -2) and P_lin; FPTbias
      # [8, N] holds six bias spectra, k and P_lin; N = number of k values.
      # Set after ci.set_cosmology, which draws a new cosmology.random, the
      # tag cosmolike's caches compare to detect a cosmology change.
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
    else:
      # use_emulator = 1: only the distance table is needed (point mass).
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so direct halo readers can be used
    even after a likelihood evaluation that did not count halos.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra at wavenumbers far above the neutrino
        free-streaming wavenumber, where the neutrinos do not cluster; an
        approximation on cluster scales. Its measured size is in
        projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Send the source-galaxy nuisance parameters to ci.

    Runs at every evaluation. For each of the source_ntomo source bins i
    (counted from 1 in the names) it sends the shear calibration DES_Mi;
    unless use_emulator = 1, also the photo-z shift DES_DZ_Si and the IA
    amplitudes DES_A1_i, DES_A2_i and DES_BTA_i. With
    IA_redshift_evolution = 3 (the shipped default) the core reads only
    the first entries of the IA lists: A1[0] and A1[1] are the amplitude
    and the exponent eta of the tidal-alignment power law in (1+z)/1.62,
    A2[0] and A2[1] the same for tidal torquing, and B_TA[0] = b_TA; with
    2, the lists hold one value per source bin. A missing parameter counts
    as 0.

    Arguments:
      params = parameter values, name -> float; **params collects every
               keyword argument of the call into this dict

    Side effects:
      replaces the source nuisance state of ci; with external_nz_modeling,
      also the source n(z).
    """
    ntomo = self.source_ntomo
    # Each list below holds one value per bin: the inner list builds the
    # names (DES_M1, ..., DES_M<ntomo>), the outer one reads each value
    # from params, 0 when absent.
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # The n(z) table is sent at every point of the chain, so a user
        # function of the nuisance parameters can modify it (for example,
        # to add outliers): (1) copy the array, so self.source_nz keeps the
        # fiducial n(z); (2) modify the copy; (3) send it with
        # ci.set_source_sample.
        source_nz_local = self.source_nz.copy()

        # A modifying function goes here, for example
        # source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # The photo-z shifts DES_DZ_Si are still applied to the sent n(z);
        # remove this call when the user function models the shifts itself.
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Send the lens-galaxy nuisance parameters to ci.

    Runs at every evaluation. For each of the lens_ntomo lens bins i it
    sends the point-mass amplitude DES_PMi; unless use_emulator = 1, also
    the galaxy bias parameters DES_B1_i (linear), DES_B2_i (quadratic),
    DES_BMAG_i (magnification), DES_B3NL_i (third order) and DES_BK_i
    (nonlocal), and the photo-z shift DES_DZ_Li. A missing parameter
    counts as 0, except DES_B1_i, which counts as 1. The point mass
    absorbs the lensing signal of the mass enclosed below the smallest
    modeled scale: outside that mass gamma_t falls as 1/theta^2, so the
    core adds a term proportional to DES_PMi/theta^2 (PointMass::get_pm).

    Arguments:
      params = parameter values, name -> float

    Side effects:
      replaces the lens nuisance state of ci; with external_nz_modeling,
      also the lens n(z).
    """
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # The n(z) table is sent at every point of the chain, so a user
        # function of the nuisance parameters can modify it (for example,
        # to add outliers): (1) copy the array, so self.lens_nz keeps the
        # fiducial n(z); (2) modify the copy; (3) send it with
        # ci.set_lens_sample.
        lens_nz_local = self.lens_nz.copy()

        # A modifying function goes here, for example
        # lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # The photo-z shifts DES_DZ_Li are still applied to the sent n(z);
        # remove this call when the user function models the shifts itself.
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    """Return ln L = -chi2/2 for a model vector.

    ci.compute_chi2 forms chi2 = (d - m)^T C^-1 (d - m) from the entries
    the mask keeps, with the data vector d and the covariance C read at
    initialize.

    Arguments:
      datavector = model vector m, a 1-D float array of the full layout

    Returns:
      ln L, a float (constant terms dropped).
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params):
    """Return ln L at the current point; cobaya calls this at every step.

    cobaya passes this likelihood's input parameters as keyword arguments,
    collected by **params into a dict, after the theory codes have
    computed the requirements for the same point.

    Arguments:
      params = parameter values, name -> float

    Returns:
      ln L, a float.
    """
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Return the model vector at the current point.

    Arguments:
      params = parameter values, name -> float

    Returns:
      float64 numpy array of the full layout (1300 entries for
      data/DESY6.dataset); entries removed by the mask or by the probe
      selection are zero.

    Side effects:
      updates the state of ci; with print_datavector, writes the vector to
      print_datavector_file.
    """
    if self.use_emulator == 1:
      dv = self.internal_get_datavector_emulator(**params)
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector_emulator(self, **params):
    """Assemble the model vector from emulated probes (use_emulator = 1).

    The emulator theories supply the cosmic-shear block (xi_+ then xi_-),
    the gamma_t block and the w block. Each block of the probe selection
    is copied to its place in the full layout, and the other entries stay
    zero. ci then adds the shear calibration (DES_Mi), the point-mass term
    when any DES_PMi is nonzero, and, with use_baryon_pca, the baryon
    principal components with amplitudes DES_BARYON_Q1..Q4; it also sets
    the masked entries to zero.

    Layout: sizes = ci.compute_data_vector_3x2pt_real_sizes() holds the
    block lengths [n_xi, n_gammat, n_w]; the blocks start at 0, n_xi and
    n_xi + n_gammat.

    Arguments:
      params = parameter values, name -> float

    Returns:
      float64 numpy array of the full layout.

    Raises:
      ValueError when an emulated block has the wrong length or the probe
      is unknown.

    Side effects:
      with print_datavector, writes the vector to print_datavector_file.
    """
    # ---------------------------------------------------------------
    # Shear calibration (DES_Mi) and point mass (DES_PMi) are fast
    # parameters: cobaya changes them without rerunning the theory codes,
    # and they are never emulated. The point-mass term enters only probes
    # with gamma_t and needs the lens state and the distances;
    # all(v == 0 for v in PM) is True when every amplitude is zero.
    PM = [params.get(p,0) for p in [survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)]]
    if self.probe not in ("xi", "xi_gg") and not all(v == 0 for v in PM):
      self.set_lens_related(**params)
      self.set_cosmo_related()
    self.set_source_related(**params)
    # ---------------------------------------------------------------

    sizes = ci.compute_data_vector_3x2pt_real_sizes()
    total_size = int(np.sum(sizes))
    dv = np.zeros(total_size, dtype='float64') 
    
    if self.probe == "xi":
      tmp = self.provider.get_cosmic_shear()
      if (len(tmp) != sizes[0]):
        raise ValueError(f'Incompatible Sizes (Emulator Cosmic Shear)')
      dv[0:sizes[0]] = tmp[0:sizes[0]]
    elif self.probe == "xi_ggl":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1]):
        raise ValueError(f'Incompatible Sizes (Emulator xi_ggl)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
    elif self.probe == "3x2pt":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "xi_gg":
      tmp1 = self.provider.get_cosmic_shear()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "2x2pt": 
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    else:
      raise ValueError(f'Unknown probe')

    if not self.use_baryon_pca: 
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=1)
    else:
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=1)
    dv = np.array(dv, dtype='float64')
    
    # Two text columns, as in the data files: the entry index and the value;
    # fmt = '%d', '%1.8e' is a tuple of the two column formats.
    if self.print_datavector:
      size = len(dv)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = dv
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return dv

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the model vector with cosmolike (use_emulator 0 or 2).

    The cosmology goes to ci first, then the lens state (every probe
    except "xi") and the source state; ci then projects the spectra into
    the full layout, with zeros at the masked entries. With
    create_baryon_pca, the principal components of the baryonic
    contamination over the simulations baryon_pca_select_sims are computed
    and written to filename_baryon_pca at every evaluation. With
    use_baryon_pca, sum_i Q_i PC_i is added (Q_i = DES_BARYON_Q1..Q4).

    Arguments:
      params = parameter values, name -> float

    Returns:
      the model vector as a list of floats of the full layout
      (get_datavector converts it to a numpy array).

    Side effects:
      updates the state of ci; writes files with create_baryon_pca or
      print_datavector.
    """
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    # Two text columns, as in the data files: the entry index and the value.
    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
