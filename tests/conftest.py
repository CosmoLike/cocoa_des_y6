"""This pytest configuration fixes the thread counts before any library loads.

pytest imports a conftest.py before the test modules of its folder and of
the folders below, so these settings reach every test. OpenMP (cosmolike's
parallel C loops) and the BLAS linear-algebra libraries behind numpy read
their thread counts once, when they load, so the variables are set before
any numerical import: four OpenMP threads, the count the shared harness
requires (cocoa_testing.REQUIRED_OMP_THREADS), one BLAS thread, and
COBAYA_NOMPI = 1, which keeps cobaya from using MPI. The tests folder goes
first on sys.path, so `import cocoa_test_utils` works from the subfolders.
"""

from pathlib import Path
import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
