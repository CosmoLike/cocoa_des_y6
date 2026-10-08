"""This module defines cosmic_shear, the DES Y6 cosmic-shear likelihood.

The selection "xi" keeps cosmic shear xi_+(theta) and xi_-(theta), the
correlations of source-galaxy shapes; the galaxy entries (gamma_t and w)
of the 1300-entry model vector stay zero. _cosmolike_prototype_base does
all the work, cosmic_shear.yaml holds the default options (source-galaxy
parameters only), and a cobaya yaml selects this likelihood as
des_y6.cosmic_shear.
"""
from cobaya.likelihoods.des_y6._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y6_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """Select cosmic shear, xi_+ and xi_-; other methods are inherited."""
  def initialize(self):
    """Initialize the shared likelihood with the probe selection "xi".

    super(cosmic_shear, self) reaches the parent class,
    _cosmolike_prototype_base, whose initialize does the work.
    """
    super(cosmic_shear,self).initialize(probe="xi")
