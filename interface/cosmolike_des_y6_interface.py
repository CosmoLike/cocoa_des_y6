"""This fallback module explains a missing build of the DES Y6 interface.

The real cosmolike_des_y6_interface is an extension module: a compiled C++
library (cosmolike_des_y6_interface.so, built from interface.cpp by
scripts/compile_des_y6.sh) that Python imports like a .py file. When the
library and this file sit in the same folder, Python imports the library,
because its import search tries extension modules before source files.
This file is therefore imported only when the library is absent, and it
stops at once with an ImportError that says how to build the library. It
must never load another survey's library under the DES Y6 name.
"""

raise ImportError(
    "DES Y6 is not compiled. From Cocoa/, enable DES Y6 in "
    "set_installation_options.sh, source start_cocoa.sh, then source "
    "projects/des_y6/scripts/compile_des_y6.sh."
)
