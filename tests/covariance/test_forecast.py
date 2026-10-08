"""Exercise the DES Y6 forecast adapter with a small numerical workload.

The adapter is covariance/des_y6_covariance.py: its configuration,
initialize and compute functions bind the DES Y6 survey choices to the
shared covariance code. check_project_forecast (cosmolike_core,
cocoa_covariance_testing.py) runs it and checks the result.
"""

import sys
import cocoa_test_utils as u

# des_y6_covariance.py sits in covariance/, which is not a package: put the
# folder on sys.path so the import below finds it.
sys.path.insert(0, str(u.PROJECT_DIR/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_des_y6_interface as interface
import des_y6_covariance as survey


def test_forecast_adapter(tmp_path):
    """Check layouts, components, one/eight threads, CLI/wrapper and saving.

    This checks assembly, not survey convergence or DES Y6 data fidelity.
    CLI = the command-line interface, compute_covariance.py.

    Arguments:
      tmp_path = a temporary folder pytest creates for this test
    """
    check_project_forecast(
        interface=interface,
        survey=survey,
        expected_sizes=(1300, 600),
        directory=tmp_path,
    )
