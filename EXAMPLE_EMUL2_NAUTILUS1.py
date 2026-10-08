"""This script runs the des_y6 hybrid Nautilus sampling for example 1 (3x2pt).

"Hybrid" means use_emulator: 2: trained emulators replace CAMB for the
background expansion and the matter power spectra, and cosmolike still
computes the survey projections. The run reads EXAMPLE_EMUL2_EVALUATE1.yaml
(des_y6.combo_3x2pt) and samples the posterior by nested sampling with the
Nautilus package, which also estimates the Bayesian evidence. The shared
runner run() in cosmolike_core/cocoa_hybrid_sampling.py documents the method
and every command option; the output files go to the project's chains/
folder.

Run from Cocoa/ after source start_cocoa.sh, for example
    python ./projects/des_y6/EXAMPLE_EMUL2_NAUTILUS1.py --check
which evaluates the fiducial point and stops; the project README gives the
MPI commands (several cooperating processes, one model each).
"""

from pathlib import Path
import sys

# Put cosmolike_core, which holds cocoa_hybrid_sampling, first on
# sys.path (the list of folders Python searches on import);
# project.parents[1] is Cocoa/.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# True only when this file runs as a script, not when it is imported.
if __name__ == "__main__":
    run(mode="nautilus", project=project, example=1)
