"""Skip the optional covariance sector only when its bindings are absent."""

import pytest
import cocoa_test_utils as u
import cosmolike_des_y6_interface as interface


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    if not interface.has_covariance:
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DES_Y6_COVARIANCE after start_cocoa.sh, "
            "then recompile DES Y6."
        )
