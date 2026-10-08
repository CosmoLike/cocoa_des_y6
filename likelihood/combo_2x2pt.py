"""This module defines combo_2x2pt, the DES Y6 2x2pt likelihood.

2x2pt combines galaxy-galaxy lensing gamma_t(theta) and galaxy clustering
w(theta); the cosmic-shear entries of the 1300-entry model vector stay
zero. _cosmolike_prototype_base does all the work, combo_2x2pt.yaml holds
the default options, and a cobaya yaml selects this likelihood as
des_y6.combo_2x2pt.
"""
from cobaya.likelihoods.des_y6._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y6_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  """Select the 2x2pt probes, gamma_t and w; other methods are inherited."""
  def initialize(self):
    """Initialize the shared likelihood with the probe selection "2x2pt".

    super(combo_2x2pt, self) reaches the parent class,
    _cosmolike_prototype_base, whose initialize does the work.
    """
    super(combo_2x2pt,self).initialize(probe="2x2pt")