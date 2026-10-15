"""Regenerate the n(z) reconstruction and bin-area figures from saved test results.

Run from an active Cocoa environment. This reads the committed probability
audit; it neither changes frozen references nor reruns a cosmology.
"""
from pathlib import Path
import sys

# From tests/studies, the fourth parent is the Cocoa runtime folder.
# This keeps imports independent of the terminal working directory.
STUDIES = Path(__file__).resolve().parent
CORE = STUDIES.parents[3] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(CORE))
from cocoa_nz_figures import plot_nz_reconstruction


if __name__ == "__main__":
    directory = STUDIES / "results"
    plot_nz_reconstruction(
        report_file=directory / "nz_reconstruction.json",
        arrays_file=directory / "nz_probability_data.npz",
        output_directory=directory)
