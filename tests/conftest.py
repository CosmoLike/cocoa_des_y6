"""Test environment setup before importing numerical libraries."""

from pathlib import Path
import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
