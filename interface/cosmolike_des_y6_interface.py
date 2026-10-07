"""Explain a missing build when Python cannot find the compiled DES Y6 module.

Python prefers the adjacent extension module when it is installed. This
fallback must never load another survey's library under the DES Y6 name.
"""

raise ImportError(
    "DES Y6 is not compiled. From Cocoa/, enable DES Y6 in "
    "set_installation_options.sh, source start_cocoa.sh, then source "
    "projects/des_y6/scripts/compile_des_y6.sh."
)
