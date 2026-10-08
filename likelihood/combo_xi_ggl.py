"""This module defines combo_xi_ggl, the DES Y6 shear plus lensing likelihood.

The selection "xi_ggl" combines cosmic shear xi_+(theta) and xi_-(theta)
with galaxy-galaxy lensing gamma_t(theta); the clustering entries of the
1300-entry model vector stay zero. _cosmolike_prototype_base does all the
work, combo_xi_ggl.yaml holds the default options, and a cobaya yaml
selects this likelihood as des_y6.combo_xi_ggl.
"""
from cobaya.likelihoods.des_y6._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y6_interface as ci
import numpy as np

class combo_xi_ggl(_cosmolike_prototype_base):
  """Select cosmic shear and lensing, xi and gamma_t; other methods are inherited."""
  def initialize(self):
    """Initialize the shared likelihood with the probe selection "xi_ggl".

    super(combo_xi_ggl, self) reaches the parent class,
    _cosmolike_prototype_base, whose initialize does the work.
    """
    super(combo_xi_ggl,self).initialize(probe="xi_ggl")