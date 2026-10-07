"""Exercise the DES Y6 forecast adapter with a small numerical workload."""

import sys
import cocoa_test_utils as u

sys.path.insert(0, str(u.PROJECT_DIR/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_des_y6_interface as interface
import des_y6_covariance as survey


def test_forecast_adapter(tmp_path):
    """Check layouts, components, one/eight threads, CLI/wrapper and saving.

    This checks assembly, not survey convergence or DES Y6 data fidelity.
    """
    check_project_forecast(
        interface=interface,
        survey=survey,
        expected_sizes=(1300, 600),
        directory=tmp_path,
    )
