"""Check that the frozen contract describes the selected DES Y6 layout."""

import numpy as np

import cocoa_test_utils as u


def test_referenced_inputs_and_layout():
    u.verify_frozen()
    data = u.FROZEN_DIR/"data"
    assert not (data/"DESY6.cov").exists()
    assert not (data/"DESY6.mask").exists()
    descriptor = (data/"DESY6.dataset").read_text()
    assert "data_file = dummy.modelvector" in descriptor
    assert "cov_file = dummy.cov" in descriptor
    assert "mask_file = ones.mask" in descriptor
    lens = np.loadtxt(fname=data/"DESY6_lens.nz")
    source = np.loadtxt(fname=data/"DESY6_source.nz")
    assert lens.shape[1] == 7
    assert source.shape[1] == 5
    for table in (lens, source):
        assert np.all(np.isfinite(table))
        assert np.all(np.diff(table[:, 0]) > 0.0)
        assert np.all(np.sum(table[:, 1:], axis=0) > 0.0)
    # The supplied lens file has tiny negative interpolation tails. Preserve
    # the survey inputs exactly instead of clipping them for a test.
    for filename in ("DESY6_lens.nz", "DESY6_source.nz"):
        assert (data/filename).read_bytes() == (u.PROJECT_DIR/"data"/filename).read_bytes()
    for filename in ("dummy.modelvector", "ones.mask"):
        table = np.loadtxt(fname=data/filename)
        assert table.shape == (1300, 2)
        np.testing.assert_array_equal(table[:, 0], np.arange(1300))


def test_tatt_reference_exercises_second_order_terms():
    u.verify_frozen()
    nla = np.load(file=u.FROZEN_DIR/"example1_nla.npy", allow_pickle=False)
    tatt = np.load(file=u.FROZEN_DIR/"example1_tatt.npy", allow_pickle=False)
    assert not np.array_equal(nla[:1144], tatt[:1144])
    # Galaxy auto-correlations have no intrinsic-alignment contribution.
    np.testing.assert_array_equal(nla[1144:], tatt[1144:])
