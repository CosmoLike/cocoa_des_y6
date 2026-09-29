# Cocoa DES-Y6

Implementation of DES-Y6 in Cocoa. 

## Installation

The same as a standard Cocoa project:
1. Clone this repository inside the `cocoa/Cocoa/projects/` directory, and name it `des_y6`
1. Activate your Cocoa environment, `conda activate cocoa && source start_cocoa.sh`
1. From the `Cocoa/` directory, run the command: `source projects/des_y6/scripts/compile_des_y6.sh`

## Running

For now, I do not publish the DES-Y6 data vector and covariance matrix, but this repository can still compute data vectors at given cosmologies. One example is in `EXAMPLE_EVALUATE1.yaml`. After running `cobaya-run projects/des_y6/EXAMPLE_EVALUATE1.yaml`, you should have a file `EXAMPLE.modelvector`.

## Code Validation

This repository has been validated against Cosmosis for a single data vector, see the [code comparison repository](https://github.com/joaoreboucas1/y6_code_comparison); comparison chains are still ongoing. Agreement between Cosmosis and Cocoa is only achieved if the variables `ell_prefactor2` and `ell_bmag_prefactor` are set to 1.0 in `external_modules/cosmolike/cosmo2D.c`, as shown in the comparison repository. 