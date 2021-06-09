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

def create_transfer_matrix(V1, V2):
    """
    Return the transfer matrix of function between function spaces
    ``V1`` and ``V2``.

    Parameters
    ----------
    V1 : dolfin FunctionSpace
    V2 : dolfin FunctionSpace

    Returns
    -------
    res : dolfin PETScMatrix
    """
    return module.PETScDMCollectionTemp.create_transfer_matrix(V1, V2)

def create_transfer_matrix_partial_derivative(V1, V2, partial_dir):
    """
    Return the transfer matrix of partial derivative of function 
    between function spaces ``V1`` and ``V2``.

    Parameters
    ----------
    V1 : dolfin FunctionSpace
    V2 : dolfin FunctionSpace
    partial_dir : int
        Direction of partial derivative.

    Returns
    -------
    res : dolfin PETScMatrix
    """
    return module.PETScDMCollectionTemp.\
        create_transfer_matrix_partial_derivative(V1, V2, partial_dir)



if __name__ == "__main__":
    pass


# """
# The "transfer_matrix" module
# ----------------------------
# contains class that builds the transfer matrix of unknowns and 
# their derivatives.
# """

# import os
# from dolfin import *

# path_to_script_dir = os.path.dirname(os.path.realpath(__file__))

# cpp_file = open(path_to_script_dir+"/cpp/transfer_matrix.cpp","r")
# cpp_code = cpp_file.read()
# cpp_file.close()

# module = compile_cpp_code(cpp_code,include_dirs=[path_to_script_dir+"/cpp",])

# def create_transfer_matrix(V1, V2):
#     """
#     Return the transfer matrix of function between function spaces
#     ``V1`` and ``V2``.

#     Parameters
#     ----------
#     V1 : dolfin FunctionSpace
#     V2 : dolfin FunctionSpace

#     Returns
#     -------
#     res : dolfin PETScMatrix
#     """
#     return module.transfer_matrix.create_transfer_matrix(V1, V2)

# def create_transfer_matrix_partial_derivative(V1, V2, partial_dir):
#     """
#     Return the transfer matrix of partial derivative of function 
#     between function spaces ``V1`` and ``V2``.

#     Parameters
#     ----------
#     V1 : dolfin FunctionSpace
#     V2 : dolfin FunctionSpace
#     partial_dir : int
#         Direction of partial derivative.

#     Returns
#     -------
#     res : dolfin PETScMatrix
#     """
#     return module.transfer_matrix.\
#         create_transfer_matrix_partial_derivative(V1, V2, partial_dir)

# if __name__ == "__main__":
#     pass


