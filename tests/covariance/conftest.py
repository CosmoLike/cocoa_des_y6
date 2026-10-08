"""Skip the optional covariance sector only when its bindings are absent."""

import pytest
# Importing cocoa_test_utils puts the project's interface folder on
# sys.path, which the next import needs; the name u is not used otherwise.
import cocoa_test_utils as u
import cosmolike_des_y6_interface as interface


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip every covariance test when the interface lacks covariance support.

    A pytest fixture is a function pytest runs around tests: autouse=True
    applies it to every test in this folder, and scope="session" runs it
    once per test run. interface.has_covariance is False when DES Y6 was
    compiled with IGNORE_COSMOLIKE_DES_Y6_COVARIANCE set.
    """
    if not interface.has_covariance:
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DES_Y6_COVARIANCE after start_cocoa.sh, "
            "then recompile DES Y6."
        )
