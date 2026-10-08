"""This module defines combo_xi_gg, the DES Y6 shear plus clustering likelihood.

The selection "xi_gg" combines cosmic shear xi_+(theta) and xi_-(theta)
with galaxy clustering w(theta); the galaxy-galaxy lensing entries of the
1300-entry model vector stay zero. _cosmolike_prototype_base does all the
work, combo_xi_gg.yaml holds the default options, and a cobaya yaml
selects this likelihood as des_y6.combo_xi_gg.
"""
from cobaya.likelihoods.des_y6._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y6_interface as ci
import numpy as np

class combo_xi_gg(_cosmolike_prototype_base):
  """Select cosmic shear and clustering, xi and w; other methods are inherited."""
  def initialize(self):
    """Initialize the shared likelihood with the probe selection "xi_gg".

    super(combo_xi_gg, self) reaches the parent class,
    _cosmolike_prototype_base, whose initialize does the work.
    """
    super(combo_xi_gg,self).initialize(probe="xi_gg")