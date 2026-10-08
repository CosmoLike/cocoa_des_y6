"""This module defines combo_3x2pt, the DES Y6 3x2pt likelihood.

3x2pt combines cosmic shear xi_+(theta) and xi_-(theta), galaxy-galaxy
lensing gamma_t(theta) and galaxy clustering w(theta): every entry of the
1300-entry model vector. _cosmolike_prototype_base does all the work,
combo_3x2pt.yaml holds the default options, and a cobaya yaml selects this
likelihood as des_y6.combo_3x2pt.
"""
from cobaya.likelihoods.des_y6._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y6_interface as ci
import numpy as np

class combo_3x2pt(_cosmolike_prototype_base):
  """Select all three probes, xi, gamma_t and w; other methods are inherited."""
  def initialize(self):
    """Initialize the shared likelihood with the probe selection "3x2pt".

    super(combo_3x2pt, self) reaches the parent class,
    _cosmolike_prototype_base, whose initialize does the work.
    """
    super(combo_3x2pt,self).initialize(probe="3x2pt")
