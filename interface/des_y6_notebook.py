"""This module prepares the DES Y6 notebook examples through their likelihood.

build_model reads an example yaml (EXAMPLE_EVALUATE1.yaml or
EXAMPLE_EVALUATE2.yaml), which supplies the cosmology, the nuisance
parameters and the numerical settings, and builds a cobaya Model: the
object that holds the theory code (CAMB) and the project's likelihood and
evaluates them at a parameter point. evaluate runs that model at the
yaml's fiducial point, so the CAMB tables and the cosmolike state are
prepared exactly as in a likelihood run, and then copies the intermediate
angular spectra and correlation functions from the compiled notebook
wrappers into numpy arrays for the shared plots.

Use one model per notebook kernel (the Python process behind a notebook):
the compiled interface keeps its cosmolike state in C global variables
shared by the whole process. These examples use a dummy observed vector
and unit covariance; their likelihood values have no interpretation as a
fit to DES measurements.
"""

import os
from pathlib import Path

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

import cosmolike_des_y6_interface as ci


def build_model(example):
    """Build one DES Y6 model and read its explicitly specified fiducial point.

    Arguments:
      example = 1 for the full 3x2pt example, 2 for cosmic shear.

    Returns:
      (model, point): a Cobaya Model and its sampled-parameter dictionary.

    Raises:
      ValueError for an unsupported example or an omitted fiducial parameter.
      RuntimeError when the model does not have the expected DES Y6 likelihood.

    The function initializes compiled Cosmolike state. It disables data-vector
    file writing; all physics and numerical settings remain those of the YAML.
    """
    if example not in (1, 2):
        raise ValueError(f"example={example}: choose 1 (3x2pt) or 2 (shear)")
    # This file is interface/des_y6_notebook.py: parents[1] of its path is
    # the project folder, and two levels above the project is Cocoa/.
    project = Path(__file__).resolve().parents[1]
    cocoa = project.parents[1]
    input_file = project / f"EXAMPLE_EVALUATE{example}.yaml"
    info = yaml_load_file(file_name=str(input_file))
    overrides = info["sampler"]["evaluate"]["override"]
    if example == 1:
        likelihood_name = "des_y6.combo_3x2pt"
    else:
        likelihood_name = "des_y6.cosmic_shear"
    if list(info["likelihood"]) != [likelihood_name]:
        raise RuntimeError(
            f"{input_file} must contain only {likelihood_name}; "
            "use the matching project example"
        )
    # The likelihood reads its files from the project's data folder and
    # writes no model-vector file; CAMB comes from Cocoa's external modules.
    # get_model only builds the model, so the sampler and output blocks go
    # (pop with a default removes a key that may be absent); stop_at_error
    # makes a failed evaluation raise instead of returning -inf.
    info["likelihood"][likelihood_name]["path"] = str(project / "data")
    info["likelihood"][likelihood_name]["print_datavector"] = False
    info["theory"]["camb"]["path"] = str(cocoa / "external_modules/code/CAMB")
    info.pop("sampler")
    info.pop("output", None)
    info["debug"] = False
    info["timing"] = False
    info["stop_at_error"] = True
    model = get_model(info_or_yaml_or_file=info)
    point = {}
    for name in model.parameterization.sampled_params():
        if name not in overrides:
            raise ValueError(
                f"{input_file}: evaluate.override has no value for {name}; "
                "specify every sampled fiducial parameter before running"
            )
        point[name] = overrides[name]
    return model, point


def evaluate(model, point, ell, clustering=False):
    """Evaluate a YAML-initialized model and copy its notebook-wrapper arrays.

    Arguments:
      model = the model returned by build_model, alone in this Python process.
      point = dictionary containing every sampled parameter of that model.
      ell = one-dimensional array of positive integer multipoles.
      clustering = True for example 1; False for the shear-only example 2.

    Returns:
      A dictionary with dimensionless ell, C_ss and C_BB angular spectra,
      theta_arcmin, xi_plus, xi_minus, and the masked data vector. Shear arrays
      have shape (n_ell or n_theta, 4, 4). With clustering=True, C_gs and gammat
      have shape (n_ell or n_theta, 6, 4); C_gg and wtheta have shape
      (n_ell or n_theta, 6, 6), with only clustering auto-bin entries filled.
      All correlations are dimensionless. Each returned array owns its data.

    Raises:
      ValueError for invalid multipoles or thread settings.
      RuntimeError for a non-finite model evaluation or wrapper output.

    This call changes the compiled cosmology and nuisance state. The shear and
    galaxy-shear spectra are Limber diagnostics; clustering spectra and the
    real-space galaxy correlations follow the likelihood's Limber switches.
    """
    multipoles = np.array(ell, dtype=np.float64, copy=True)
    if multipoles.ndim != 1 or multipoles.size == 0:
        raise ValueError("ell must be a nonempty one-dimensional multipole array")
    if not np.all(np.isfinite(multipoles)) or np.any(multipoles < 1.0):
        raise ValueError("ell must contain finite positive multipoles")
    if np.any(multipoles != np.floor(multipoles)):
        raise ValueError("ell must contain integer multipoles for non-Limber lookup")
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    if threads < 1:
        raise ValueError(
            f"OMP_NUM_THREADS={threads}: set a positive integer before starting this notebook"
        )
    # Restore cosmolike's OpenMP thread count before evaluating: a library
    # loaded in the notebook can lower it (with_omp_threads in the
    # likelihood explains why).
    ci.set_omp_threads(n=threads)
    # cached=False recomputes even when the point equals the previous one,
    # so the compiled state read below belongs to this evaluation.
    loglikes = model.loglikes(
        params_values=point,
        return_derived=False,
        cached=False,
    )
    if not np.all(np.isfinite(loglikes)):
        raise RuntimeError("DES Y6 evaluation is non-finite; inspect the model inputs")
    shear_ee, shear_bb = ci.C_ss_tomo_limber(l=multipoles)
    xi_plus, xi_minus = ci.xi_pm_tomo()
    result = {
        "ell": multipoles,
        "C_ss": shear_ee,
        "C_BB": shear_bb,
        "theta_arcmin": ci.get_binning_real_space(),
        "xi_plus": xi_plus,
        "xi_minus": xi_minus,
        "data_vector": ci.compute_data_vector_masked(),
    }
    if clustering:
        likelihood = model.likelihood["des_y6.combo_3x2pt"]
        result["C_gs"] = ci.C_gs_tomo_limber(l=multipoles)
        if likelihood.adopt_limber_gg:
            result["C_gg"] = ci.C_gg_tomo_limber(l=multipoles)
        else:
            result["C_gg"] = ci.C_gg_tomo(l=multipoles)
        result["gammat"] = ci.w_gammat_tomo()
        result["wtheta"] = ci.w_gg_tomo()
    for name, values in result.items():
        array = np.array(values, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"DES Y6 wrapper output {name} contains non-finite values")
        result[name] = array
    return result
