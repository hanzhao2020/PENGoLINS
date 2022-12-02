"""
The "transfer_matrix" module
----------------------------
contains class that builds the transfer matrix of unknowns and 
their derivatives.
"""

import os
from petsc4py import PETSc
from dolfin import *

path_to_script_dir = os.path.dirname(os.path.realpath(__file__))

cpp_file = open(path_to_script_dir+"/cpp/transfer_matrix.cpp","r")
cpp_code = cpp_file.read()
cpp_file.close()

module = compile_cpp_code(cpp_code,include_dirs=[path_to_script_dir+"/cpp",])

def create_transfer_matrix(V1, V2, deriv=0):
    """
    Return the transfer matrix of function between function spaces
    ``V1`` and ``V2``.

    Parameters
    ----------
    V1 : dolfin FunctionSpace
    V2 : dolfin FunctionSpace
    deriv : int, default is 0

    Returns
    -------
    res : dolfin PETScMatrix
    """
    return module.PETScDMCollectionCustom.create_transfer_matrix(V1, V2, deriv)


if __name__ == "__main__":
    pass