"""Parse the DES Y6 covariance YAML without running CAMB or a covariance."""

import os
import sys

import pytest

import cocoa_test_utils as u

sys.path.insert(0, str(u.PROJECT_DIR/"covariance"))

from cobaya.yaml import yaml_dump, yaml_load_file
from cosmolike_notebook_utils.covariance.command_line import load_run_configuration
import des_y6_covariance as survey


def test_covariance_yaml_uses_environment_threads():
    """The project example has no YAML thread count; the environment wins."""
    u.require_cocoa_environment()
    filename = u.PROJECT_DIR/"EXAMPLE_EVALUATE_COVARIANCE.yaml"
    _, run = load_run_configuration(filename=filename, survey=survey)
    info = yaml_load_file(file_name=str(filename))
    assert "threads" not in info["covariance"]
    assert run["threads"] == int(os.environ["OMP_NUM_THREADS"])
    assert run["space"] == "real"


def test_covariance_yaml_rejects_thread_key(tmp_path):
    """A YAML thread key must fail even when its value matches the environment."""
    u.require_cocoa_environment()
    source = u.PROJECT_DIR/"EXAMPLE_EVALUATE_COVARIANCE.yaml"
    info = yaml_load_file(file_name=str(source))
    info["covariance"]["threads"] = int(os.environ["OMP_NUM_THREADS"])
    filename = tmp_path/"threads.yaml"
    filename.write_text(yaml_dump(info))
    with pytest.raises(ValueError, match="remove covariance.threads; set OMP_NUM_THREADS"):
        load_run_configuration(filename=filename, survey=survey)
