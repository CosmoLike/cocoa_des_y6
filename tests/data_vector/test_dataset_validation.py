"""Reject incompatible file indices before initializing the C++ likelihood.

Tiny files exercise the layout contract without parsing a survey covariance
or importing the likelihood, CAMB or the compiled project interface.
"""

import importlib.util

import numpy as np
import pytest

import cocoa_test_utils as u

# Load dataset_validation.py by file path, as a module named
# des_y6_dataset_validation, without importing the likelihood package:
# spec_from_file_location describes where and how to load the file,
# module_from_spec creates an empty module, and exec_module runs the file
# inside it.
module_path = u.PROJECT_DIR/"likelihood/dataset_validation.py"
specification = importlib.util.spec_from_file_location(
    name="des_y6_dataset_validation", location=module_path)
validation = importlib.util.module_from_spec(specification)
specification.loader.exec_module(validation)


@pytest.fixture
def layout_files(tmp_path):
    """Write three measurements with a valid data/mask/covariance layout.

    A pytest fixture: pytest calls it and passes its return value to every
    test that names layout_files as an argument.

    Arguments:
      tmp_path = a temporary folder pytest creates for this test

    Returns:
      dict of validate_layout's keyword arguments: the three file paths
      and size = 3. The covariance rows are (i, j, C_ij): one diagonal row
      per index plus two off-diagonal rows.
    """
    data_file = tmp_path/"data.txt"
    mask_file = tmp_path/"mask.txt"
    cov_file = tmp_path/"cov.txt"
    values = np.column_stack((np.arange(3), np.ones(3)))
    np.savetxt(fname=data_file, X=values)
    np.savetxt(fname=mask_file, X=values)
    covariance = np.array([
        [0, 0, 2],
        [0, 1, 0.1],
        [1, 1, 3],
        [1, 2, 0.2],
        [2, 2, 4],
    ])
    np.savetxt(fname=cov_file, X=covariance)
    return {
        "data_file": data_file,
        "mask_file": mask_file,
        "cov_file": cov_file,
        "size": 3,
    }


def test_matching_layout_is_accepted(layout_files):
    """Accept files whose indices match the full layout.

    validate_layout(**layout_files) passes the dict entries as keyword
    arguments.
    """
    validation.validate_layout(**layout_files)


@pytest.mark.parametrize("field", ("data_file", "mask_file"))
def test_short_data_or_mask_is_rejected(layout_files, field):
    """Reject a data or mask file that lacks the last index.

    Arguments:
      layout_files = the valid files of the fixture above
      field = "data_file" or "mask_file", the file to shorten (one test
              run each)
    """
    shortened = np.array([[0, 1], [1, 1]])
    np.savetxt(fname=layout_files[field], X=shortened)
    with pytest.raises(ValueError, match="expected consecutive indices 0 to 2"):
        validation.validate_layout(**layout_files)


def test_reordered_mask_is_rejected(layout_files):
    """Reject a mask whose indices are out of order."""
    reordered = np.array([[0, 1], [2, 1], [1, 1]])
    np.savetxt(fname=layout_files["mask_file"], X=reordered)
    with pytest.raises(ValueError, match="expected consecutive indices 0 to 2"):
        validation.validate_layout(**layout_files)


@pytest.mark.parametrize("index", (-1, 3, 1.5, np.nan, np.inf))
def test_invalid_covariance_index_is_rejected(layout_files, index):
    """Reject a negative, too large, fractional, NaN or infinite index.

    Arguments:
      layout_files = the valid files of the fixture above
      index = the invalid value placed in one covariance row (one test run
              each)
    """
    covariance = np.array([
        [0, 0, 2],
        [1, 1, 3],
        [2, 2, 4],
        [index, 1, 0.1],
    ])
    np.savetxt(fname=layout_files["cov_file"], X=covariance)
    with pytest.raises(ValueError, match="covariance indices must be integers in 0 to 2"):
        validation.validate_layout(**layout_files)


def test_missing_diagonal_is_rejected(layout_files):
    """Reject a covariance without the diagonal row (2, 2)."""
    covariance = np.array([[0, 0, 2], [1, 1, 3], [1, 2, 0.2]])
    np.savetxt(fname=layout_files["cov_file"], X=covariance)
    with pytest.raises(ValueError, match="expected exactly one diagonal entry per measurement"):
        validation.validate_layout(**layout_files)


def test_duplicate_diagonal_is_rejected(layout_files):
    """Reject a covariance that lists the diagonal row (2, 2) twice."""
    covariance = np.array([[0, 0, 2], [1, 1, 3], [2, 2, 4], [2, 2, 4]])
    np.savetxt(fname=layout_files["cov_file"], X=covariance)
    with pytest.raises(ValueError, match="expected exactly one diagonal entry per measurement"):
        validation.validate_layout(**layout_files)
