"""This script computes and saves the DES Y6 covariance matrix.

The covariance of the 3x2pt data vector is the sum of a Gaussian part (G),
the super-sample covariance (SSC: the response to density modes larger
than the survey) and the connected non-Gaussian part (cNG). The numerical
work runs in C through the production bindings of the compiled interface;
run_covariance (cosmolike_notebook_utils.covariance.command_line) reads
the evaluate yaml, prepares CAMB and the project through
des_y6_covariance.py, and writes an .npz archive (numpy's file of named
arrays).

From an activated Cocoa installation:
    python projects/des_y6/covariance/compute_covariance.py \
        projects/des_y6/EXAMPLE_EVALUATE_COVARIANCE.yaml

See --help and covariance/README.md for space and accuracy options.
The numerical model and survey settings are shared with the notebook.
"""

import os
from pathlib import Path
import sys

# One thread for each linear-algebra library behind numpy (OpenBLAS, MKL,
# Apple's vecLib): they read these variables when they load, so the values
# are set before the first numpy import. Cosmolike's own OpenMP threads
# follow OMP_NUM_THREADS from the environment.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; COBAYA_NOMPI = 1 keeps it from starting MPI (the library that runs
# cooperating processes), and no sampler runs for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# Put cosmolike_core (the shared notebook package) and the compiled
# interface folder first on sys.path, the list of folders Python searches
# on import; project.parents[1] is Cocoa/. des_y6_covariance.py is found
# because Python puts this script's own folder on sys.path.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_des_y6_interface as ci
import des_y6_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# True only when this file runs as a script, not when it is imported.
if __name__ == "__main__":
    # default_space="real": angular bins; joint=False: the galaxy 3x2pt
    # adapter (True selects the cluster 6x2pt+N adapter).
    run_covariance(
        interface=ci, survey=survey, default_space="real", joint=False,
    )
