"""Reject incompatible file indices before initializing the C++ likelihood.

Tiny files exercise the layout contract without parsing a survey covariance
or importing the likelihood, CAMB or the compiled project interface.
"""

import importlib.util

import numpy as np
import pytest

import cocoa_test_utils as u

module_path = u.PROJECT_DIR/"likelihood/dataset_validation.py"
specification = importlib.util.spec_from_file_location(
    name="des_y6_dataset_validation", location=module_path)
validation = importlib.util.module_from_spec(specification)
specification.loader.exec_module(validation)


@pytest.fixture
def layout_files(tmp_path):
    """Three measurements with a valid indexed data/mask/covariance layout."""
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
    validation.validate_layout(**layout_files)


@pytest.mark.parametrize("field", ("data_file", "mask_file"))
def test_short_data_or_mask_is_rejected(layout_files, field):
    shortened = np.array([[0, 1], [1, 1]])
    np.savetxt(fname=layout_files[field], X=shortened)
    with pytest.raises(ValueError, match="expected consecutive indices 0 to 2"):
        validation.validate_layout(**layout_files)


def test_reordered_mask_is_rejected(layout_files):
    reordered = np.array([[0, 1], [2, 1], [1, 1]])
    np.savetxt(fname=layout_files["mask_file"], X=reordered)
    with pytest.raises(ValueError, match="expected consecutive indices 0 to 2"):
        validation.validate_layout(**layout_files)


@pytest.mark.parametrize("index", (-1, 3, 1.5, np.nan, np.inf))
def test_invalid_covariance_index_is_rejected(layout_files, index):
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
    covariance = np.array([[0, 0, 2], [1, 1, 3], [1, 2, 0.2]])
    np.savetxt(fname=layout_files["cov_file"], X=covariance)
    with pytest.raises(ValueError, match="expected exactly one diagonal entry per measurement"):
        validation.validate_layout(**layout_files)


def test_duplicate_diagonal_is_rejected(layout_files):
    covariance = np.array([[0, 0, 2], [1, 1, 3], [2, 2, 4], [2, 2, 4]])
    np.savetxt(fname=layout_files["cov_file"], X=covariance)
    with pytest.raises(ValueError, match="expected exactly one diagonal entry per measurement"):
        validation.validate_layout(**layout_files)
