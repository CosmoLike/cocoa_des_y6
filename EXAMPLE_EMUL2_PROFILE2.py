"""This script runs the des_y6 hybrid profile for example 2 (cosmic shear).

"Hybrid" means use_emulator: 2: trained emulators replace CAMB for the
background expansion and the matter power spectra, and cosmolike still
computes the survey projections. The run reads EXAMPLE_EMUL2_EVALUATE2.yaml
(des_y6.cosmic_shear) and fixes one sampled parameter at each value of a
grid centered on a saved minimization (--minfile, required) and minimizes -2
log posterior over the other parameters; the minima against the fixed value
form the profile. The shared runner run() in
cosmolike_core/cocoa_hybrid_sampling.py documents the method and every
command option; the output files go to the project's chains/ folder.

Run from Cocoa/ after source start_cocoa.sh, for example
    python ./projects/des_y6/EXAMPLE_EMUL2_PROFILE2.py --check
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
    run(mode="profile", project=project, example=2)
