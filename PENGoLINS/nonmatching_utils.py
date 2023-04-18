"""
The "nonmatching_utils" module
------------------------------
contains functions that can be used to compute coupling 
of non-matching.
"""

import os
from math import *
import numpy as np 

from mpi4py import MPI as pyMPI
from petsc4py import PETSc
from tIGAr.common import *
from tIGAr.BSplines import *

from PENGoLINS.transfer_matrix import *
from PENGoLINS.math_utils import *

DOLFIN_CONSTANT = function.constant.Constant
DOLFIN_FUNCTION = function.function.Function
DOLFIN_VECTOR = cpp.la.Vector
DOLFIN_MATRIX = cpp.la.Matrix
DOLFIN_PETSCVECTOR = cpp.la.PETScVector
DOLFIN_PETSCMATRIX = cpp.la.PETScMatrix
PETSC4PY_VECTOR = PETSc.Vec
PETSC4PY_MATRIX = PETSc.Mat

SAVE_PATH = "./"

def v2p(v):
    """
    Convert "dolfin.cpp.la.PETScVector" to 
    "petsc4py.PETSc.Vec".
    """
    return as_backend_type(v).vec()

def m2p(A):
    """
    Convert "dolfin.cpp.la.PETScMatrix" to 
    "petsc4py.PETSc.Mat".
    """
    return as_backend_type(A).mat()

def arg2v(x):
    """
    Convert dolfin Function or dolfin Vector to petsc4py.PETSc.Vec.
    """
    if isinstance(x, DOLFIN_FUNCTION):
        x_PETSc = x.vector().vec()
    elif isinstance(x, DOLFIN_PETSCVECTOR):
        x_PETSc = v2p(x)
    elif isinstance(x, DOLFIN_VECTOR):
        # x_PETSc = zero_petsc_vec(x.size())
        # x_PETSc.setArray(x[:])
        # x_PETSc.assemble()
        x_PETSc = v2p(x)
    elif isinstance(x, PETSC4PY_VECTOR):
        x_PETSc = x
    else:
        if mpirank == 0:
            raise TypeError("Type " + str(type(x)) + " is not supported.")
    return x_PETSc

def arg2m(A):
    """
    Convert dolfin Matrix to petsc4py.PETSc.Mat.
    """
    if isinstance(A, DOLFIN_PETSCMATRIX):
        A_PETSc = m2p(A)
    elif isinstance(A, DOLFIN_MATRIX):
        A_PETSc = m2p(A)
    elif isinstance(A, PETSC4PY_MATRIX):
        A_PETSc = A
    else:
        if mpirank == 0:
            raise TypeError("Type " + str(type(A)) + " is not supported.")
    return A_PETSc

def A_x(A, x):
    """
    Compute b = A*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec

    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    # b_PETSc = PETSc.Vec(comm)
    # b_PETSc.create(comm=comm)
    # b_PETSc.setSizes(A_PETSc.getSizes()[0])  # Contains local size
    # b_PETSc.setUp()
    # # b_PETSc.assemble()
    b_PETSc = A_PETSc.createVecLeft()
    A_PETSc.mult(x_PETSc, b_PETSc)
    # b_PETSc.ghostUpdate()
    b_PETSc.assemble()
    return b_PETSc

def A_x_b(A, x, b):
    """
    Compute "Ax = b".

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    b : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    """
    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    b_PETSc = arg2v(b)
    A_PETSc.mult(x_PETSc, b_PETSc)
    # b_PETSc.ghostUpdate()
    b_PETSc.assemble()

def AT_x(A, x):
    """
    Compute b = A^T*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec

    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    # b_PETSc = PETSc.Vec(A_PETSc.getComm())
    # b_PETSc.create(A_PETSc.getComm())
    # # # Raised error in parallel since local size is not passed
    # # b_PETSc.setSizes(A_PETSc.getSizes()[1][1]) 
    # b_PETSc.setSizes(A_PETSc.getSizes()[1])
    # b_PETSc.setUp()
    # # b_PETSc.assemble()
    b_PETSc = A_PETSc.createVecRight()
    A_PETSc.multTranspose(x_PETSc, b_PETSc)
    # b_PETSc.ghostUpdate()
    b_PETSc.assemble()
    return b_PETSc

def AT_x_b(A, x, b):
    """
    Compute b = A^T*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    b : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    """
    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    b_PETSc = arg2v(b)
    A_PETSc.multTranspose(x_PETSc, b_PETSc)
    # b_PETSc.ghostUpdate()
    b_PETSc.assemble()

def AT_R_B(A, R, B):
    """
    Compute "A^T*R*B". A,R and B are "petsc4py.PETSc.Mat".

    Parameters
    -----------
    A : petsc4py.PETSc.Mat
    R : petsc4py.PETSc.Mat
    B : petsc4py.PETSc.Mat

    Returns
    -------
    ATRB : petsc4py.PETSc.Mat
    """
    ATRB = A.transposeMatMult(R).matMult(B)
    return ATRB

def apply_bcs_vec(spline, v):
    """
    Apply boundary conditions of ``spline`` to vector ``v``.

    Parameters
    ----------
    spline : ExtractedSpline
    v : dolfin Function, dolfin Vector, dolfin PETScVector
        or PETSc.Vec
    """
    v_PETSc = arg2v(v)
    v_PETSc.setValues(spline.zeroDofs, 
        zeros(spline.zeroDofs.getLocalSize()))
    v_PETSc.assemblyBegin()
    v_PETSc.assemblyEnd()

def apply_bcs_mat(spline0, A, spline1=None, diag=1):
    """
    Apply boundary conditions of ``spline0`` and ``spline1``(if given) 
    to matrix ``A``.

    Parameters
    ----------
    spline0 : ExtractedSpline
    A : dolfin Matrix, dolfin PETScMatrix, PETSc.Mat
    spline1 : ExtrctedSpline, default is None
    diag : int, optional, default is 1
        Values put in all diagonals of eliminated rows.
        1 is for diagonal blocks, 0 is for off-diagonal blocks.
    """
    A_PETSc = arg2m(A)

    if spline1 is None:
        A_PETSc.zeroRowsColumns(spline0.zeroDofs, diag=diag)
    else:
        A_PETSc.zeroRows(spline0.zeroDofs, diag=diag)
        A_PETSc = A_PETSc.copy()
        A_PETSc.transpose()
        A_PETSc.zeroRows(spline1.zeroDofs, diag=diag)
        A_PETSc.transpose()

    A_PETSc.assemblyBegin()
    A_PETSc.assemblyEnd()
    return A_PETSc

def IGA2FE(spline, u_IGA, applyBCs=False):
    """
    Convert the DoFs of ``u_IGA`` from IGA space to FE space. 

    Parameters
    ----------
    spline : ExtractedSpline
    u_IGA : dolfin Function, dolfin Vector, dolfin PETScVector
        or PETSc.Vec
    applyBCs : bool, optional, default is False

    Returns
    -------
    u_FE : dolfin PETScVector
    """
    if applyBCs:
        apply_bcs_vec(spline, u_IGA)
    u_FE_PETSc = A_x(spline.M, u_IGA)
    u_FE = PETScVector(u_FE_PETSc)
    return u_FE

def FE2IGA(spline, u_FE, applyBCs=True):
    """
    Convert the DoFs of``u_F`` from FE space to IGA space.

    Parameters
    ----------
    spline : ExtractedSpline
    u_FE : dolfin Function, dolfin Vector, dolfin PETScVector
        or PETSc.Vec
    applyBCs : bool, optional

    Returns
    -------
    u_IGA : dolfin PETScVector
    """
    u_IGA_PETSc = AT_x(spline.M, u_FE)
    u_IGA = PETScVector(u_IGA_PETSc)
    if applyBCs:
        apply_bcs_vec(spline, u_IGA)
    return u_IGA

def zero_petsc_vec(vec_sizes, vec_type=None, comm=worldcomm):
    """
    Create zero PETSc vector of size ``vec_sizes``.

    Parameters
    ----------
    vec_sizes : int or list of ints with local and global sizes
    vec_type : str, optional, default is None
        For available types, see petsc4py.PETSc.Vec.Type.
    comm : MPI communicator

    Returns
    -------
    v : petsc4py.PETSc.Vec
    """
    v = PETSc.Vec(comm)
    v.create(comm=comm)
    if vec_type is None:
        if mpisize == 1:
            vec_type = 'seq'
        else:
            vec_type = 'mpi'
    v.setType(vec_type)
    v.setSizes(vec_sizes)
    v.setUp()
    v.assemble()
    return v

def zero_petsc_mat(row_sizes, col_sizes, mat_type=None, 
                   PREALLOC=500, comm=worldcomm):
    """
    Create zeros PETSc matrix with shape (``row_sizes``, ``col_sizes``).

    Parameters
    ----------
    row_sizes : int or list of ints with local and global sizes
    col_sizes : int or list of ints with local and global sizes
    mat_type : str, optional, default is None
        For available types, see petsc4py.PETSc.Mat.Type
    comm : MPI communicator

    Returns
    -------
    A : petsc4py.PETSc.Mat
    """
    A = PETSc.Mat(comm)
    A.create(comm=comm)
    if mat_type is None:
        if mpisize == 1:
            mat_type = 'seqaij'
        else:
            mat_type = 'mpiaij'
    A.setType(mat_type)
    A.setSizes([row_sizes, col_sizes])
    A.setPreallocationNNZ([PREALLOC, PREALLOC])
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    A.setUp()
    A.assemble()
    return A

def penalty_residual(PE, mortar_funcs):
    """
    Compute the differentiation of penalty energy ``PE`` w.r.t. 
    variables ``vars1`` and ``vars2``.

    Parameters
    -----------
    PE : ufl Form
    mortar_funcs : list of dolfin Functions

    Returns
    -------
    Rm_list : list of ufl Forms, rank 2, 2x2
    """
    Rm_list = [[None for j in range(len(mortar_funcs[i]))] 
              for i in range(len(mortar_funcs))]
    for i in range(len(mortar_funcs)):
        for j in range(len(mortar_funcs[i])):
            Rm_list[i][j] = derivative(PE, mortar_funcs[i][j])
    return Rm_list

def penalty_residual_deriv(Rm_list, mortar_funcs):
    """
    Compute the Jacobian of residuals of penalty energy "Rm_list"
    w.r.t. variables "vars1" and "vars2".

    Parameters
    -----------
    Rm_list : list of ufl Forms, rank 2, 2x2
    mortar_funcs : list of dolfin Functions

    Returns
    -------
    dRm_dum_list : list of ufl Forms, rank 4, 2x2x2x2
    """
    dRm_dum_list = [[[[None for l in range(len(mortar_funcs[k]))]
                     for j in range(len(Rm_list[i]))]
                     for k in range(len(mortar_funcs))]
                     for i in range(len(Rm_list))]

    for i in range(len(Rm_list)):
        for k in range(len(mortar_funcs)):
            if k == i:
                for j in range(len(Rm_list[i])):
                    for l in range(j, len(mortar_funcs[k])):
                        dRm_dum_list[i][k][j][l] = \
                            derivative(Rm_list[i][j], mortar_funcs[k][l])
                            #str(i) + str(j) + " " + str(k) + str(l)
            elif k > i:
                for j in range(len(Rm_list[i])):
                    for l in range(len(mortar_funcs[k])):
                        dRm_dum_list[i][k][j][l] = \
                            derivative(Rm_list[i][j], mortar_funcs[k][l])
                            # str(i) + str(j) + " " + str(k) + str(l)
    return dRm_dum_list

def transfer_penalty_residual(Rm_list, A_list):
    """
    Compute the contribution of the residuals of the penalty terms on 
    RHS using transfer matrices.

    Parameters
    ----------
    Rm_list : list of ufl Forms, rank 2, 2x2
    A_list : list of dolfin PETScMatrix

    Returns
    -------
    R_FE : list of petsc4py.PETSc.Vecs, size of 2
    """
    R_FE = [None for i in range(len(Rm_list))]
    for i in range(len(Rm_list)):
        for j in range(len(Rm_list[i])):
            if R_FE[i] is not None:
                R_FE[i] += AT_x(A_list[i][j], assemble(Rm_list[i][j]))
            else:
                R_FE[i] = AT_x(A_list[i][j], assemble(Rm_list[i][j]))
    return R_FE

def transfer_penalty_residual_deriv(dRm_dum_list, A_list):
    """
    Compute the contribution of the Jacobian of the penalty terms on 
    LHS using transfer matrix.

    Parameters
    ----------
    dRm_dum_list : list of ufl Forms, rank 4, 2x2x2x2
    A_list : list of dolfin PETScMatrix

    Returns
    -------
    dR_du_FE : list of petsc4py.PETSc.Mats, rank 2, size of 2x2
    """
    dRm_dum = [[[[None for l in range(len(dRm_dum_list[i][j][k]))]
                       for k in range(len(dRm_dum_list[i][j]))]
                       for j in range(len(dRm_dum_list[i]))]
                       for i in range(len(dRm_dum_list))]
    dR_du_FE = [[None for j in range(len(dRm_dum_list[i]))]
                       for i in range(len(dRm_dum_list))]

    for i in range(len(dRm_dum_list)):
        for j in range(len(dRm_dum_list[i])):
            if i == j:
                for k in range(len(dRm_dum_list[i][j])):
                    for l in range(k, len(dRm_dum_list[i][j][k])):
                        dRm_dum[i][j][k][l] = m2p(
                            assemble(dRm_dum_list[i][j][k][l]))
            elif i < j:
                for k in range(len(dRm_dum_list[i][j])):
                    for l in range(len(dRm_dum_list[i][j][k])):
                        dRm_dum[i][j][k][l] = m2p(
                            assemble(dRm_dum_list[i][j][k][l]))

    for i in range(len(dRm_dum_list)):
        for j in range(len(dRm_dum_list)):
            if i == j:
                for k in range(len(dRm_dum_list[i][j])):
                    for l in range(k, len(dRm_dum_list[i][j][k])):
                        dR_du_temp = AT_R_B(m2p(A_list[i][k]), 
                            dRm_dum[i][j][k][l], m2p(A_list[j][l]))
                        if k == l:
                            if dR_du_FE[i][j] is None: 
                                dR_du_FE[i][j] = dR_du_temp
                            else:
                                dR_du_FE[i][j] += dR_du_temp
                        elif k < l:
                            # Add the off-diagonal contribution twice
                            # due to symmetry of the block matrix
                            dR_du_temp_T = dR_du_temp.copy()
                            dR_du_temp_T.transpose()
                            dR_du_FE[i][j] += dR_du_temp
                            dR_du_FE[i][j] += dR_du_temp_T
            elif i < j:
                for k in range(len(dRm_dum_list[i][j])):
                    for l in range(len(dRm_dum_list[i][j][k])):
                        dR_du_temp = AT_R_B(m2p(A_list[i][k]), 
                            dRm_dum[i][j][k][l], m2p(A_list[j][l]))
                        if dR_du_FE[i][j] is None: 
                                dR_du_FE[i][j] = dR_du_temp
                        else:
                            dR_du_FE[i][j] += dR_du_temp
            elif i > j:
                # # Leave lower off-diagonal block as none due to symmetry
                # dR_du_temp = dR_du_FE[j][i].copy()
                # dR_du_FE[i][j] = dR_du_temp.transpose()
                pass
    return dR_du_FE

def R2IGA(splines, R):
    """
    Convert residuals ``R`` from FE to IGA space.
    
    Parameters
    ----------
    splines : list of ExtractedSplines
    R : list of dolfin PETScVectors

    Returns
    -------
    R_IGA : list of dolfin PETScVectors
    """
    R_IGA = [None, None]
    for i in range(len(R)):
        R_IGA[i] = v2p(FE2IGA(splines[i], R[i]))
    return R_IGA

def dRdu2IGA(splines, dR_du):
    """
    Convert Jacobians "dR_du" from FE to IGA space.

    Parameters
    ----------
    splines : list of ExtractedSplines
    dR_du : list of petsc4py.PETSc.Mats, rank 2

    Returns
    -------
    dRdu_IGA : list of petsc4py.PETSc.Mats, rank 2
    """
    dRdu_IGA = [[None for i in range(len(dR_du[0]))] \
                      for j in range(len(dR_du))]
    for i in range(len(dR_du)):
        for j in range(len(dR_du[i])):
            if i == j:
                diag = 1
            else:
                diag = 0
            dRdu_IGA[i][j] = AT_R_B(m2p(splines[i].M), dR_du[i][j], 
                                    m2p(splines[j].M))
            apply_bcs_mat(splines[i], dRdu_IGA[i][j], splines[j], diag=diag)
    return dRdu_IGA

def create_nest_PETScVec(v_list, comm=worldcomm):
    """
    Create nest petsc4py.PETSc.Vec from ``v_list``.
    comm : mpi4py.MPI.Intracomm, optional

    Parameters
    ----------
    v_list : list of PETSc.Vecs
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    v : petsc4py.PETSc.Vec
    """
    v = PETSc.Vec(comm)
    v.createNest(v_list, comm=comm)
    v.setUp()
    v.assemble()
    return v

def create_nest_PETScMat(A_list, PREALLOC=500, comm=worldcomm):
    """
    Create nest PETSc.Mat from ``A_list``.

    Parameters
    ----------
    A_list : list of petsc4py.PETSc.Mats, rank 2
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    A : petsc4py.PETSc.Mat
    """
    A = PETSc.Mat(comm)
    A.createNest(A_list, comm=comm)
    A.setPreallocationNNZ([PREALLOC, PREALLOC])
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    A.setUp()
    A.assemble()
    return A

def create_aijmat_from_nestmat(A, A_list, PREALLOC=500, 
                               csr=True, comm=worldcomm):
    """
    Create an AIJ type PETSc matrix from given NEST PETSc 
    matrix ``A`` and its list of submatrices ``A_list`` by
    setting values from ``A`` to new matrix ``A_new``.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat of type nest
    A_list : list of petsc4py.PETSc.Mats, rank 2
    PREALLOC : int, optional, default is 500
    csr : bool, optional, default is True
        If csr is True, get values from submatrices and 
        set values to global matrix in csr format.
    comm : mpi4py.MPI.Intracomm, optional

    Returns
    -------
    A_new: petsc4py.PETSc.Mat of type aij
    """
    # if mpirank == 0:
    #     if csr:
    #         print("Creating aij PETSc matrix from nest PETSc matrix "
    #               "in csr format ...")
    #     else:
    #         print("Creating aij PETSc matrix from nest PETSc matrix "
    #               "in dense format ...")

    # Get information of global nest matrix
    A_size_row, A_size_col = A.getSizes()
    A_range_row = A.getOwnershipRange()
    A_range_col = A.getOwnershipRangeColumn()
    A_range_col_allgather = comm.allgather(A_range_col)

    # Get information of submatrices
    A_sub_size_row_list = []
    A_sub_size_col_list = []
    A_sub_size_col_allgather_list = []
    A_sub_range_row_list = []
    A_sub_range_col_list = []
    A_sub_range_col_allgather_list = []

    for i in range(A.getNestSize()[0]):
        sub_mat = A.getNestSubMatrix(i,i)
        A_sub_size_row_list += [sub_mat.getSizes()[0],]
        A_sub_size_col_list += [sub_mat.getSizes()[1],]
        A_sub_size_col_allgather_list += \
            [comm.allgather(A_sub_size_col_list[-1]),]
        A_sub_range_row_list += [sub_mat.getOwnershipRange(),]
        A_sub_range_col_list += [sub_mat.getOwnershipRangeColumn(),]
        A_sub_range_col_allgather_list += \
            [comm.allgather(A_sub_range_col_list[-1]),]

    # Create new aij global matrix
    A_new = PETSc.Mat(comm)
    A_new.createAIJ(A.getSizes(), comm=comm)
    A_new.setPreallocationNNZ([PREALLOC, PREALLOC])
    A_new.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    A_new.setUp()
    A_new.assemble()

    # Set values to A_new
    ind_off_local_row = 0
    ind_off_global_row = A_range_row[0]
    ind_off_global_col = A_range_col[0]
    ind_off_global_col_all = comm.allgather(ind_off_global_col)

    for i in range(A.getNestSize()[0]):
        ind_off_local_col = np.zeros(MPI.size(comm))
        for j in range(A.getNestSize()[1]):
            sub_mat_size_row = A_sub_size_row_list[i]
            sub_mat_size_col = A_sub_size_col_list[j]
            sub_mat_range_row = A_sub_range_row_list[i]
            sub_mat_range_col = A_sub_range_col_list[j]
            sub_mat_size_col_all = A_sub_size_col_allgather_list[j]
            sub_mat_range_col_all = A_sub_range_col_allgather_list[j]

            if A_list[i][j] is not None:
                sub_mat = A_list[i][j]

                if csr:
                    # Get and set values in csr format
                    mat_indptr, mat_indices, mat_vals_csr = \
                        sub_mat.getValuesCSR()
                    mat_indices_global = mat_indices.copy()

                    for col_range_iter in range(MPI.size(comm)):
                        # Indices in current column ownership range
                        ind_right = np.where(mat_indices >= 
                            sub_mat_range_col_all[col_range_iter][0])
                        ind_left = np.where(mat_indices < 
                            sub_mat_range_col_all[col_range_iter][1])
                        ind_int = np.intersect1d(ind_right, ind_left)
                        # Indices offset
                        ind_off_local_col_temp \
                            = ind_off_local_col[col_range_iter]
                        ind_off_global_col_temp \
                            = ind_off_global_col_all[col_range_iter]
                        ind_off_zero = \
                            -sub_mat_range_col_all[col_range_iter][0]
                        # Indices in global matrix
                        mat_indices_global[ind_int] = mat_indices[ind_int]   \
                                                    + ind_off_local_col_temp \
                                                    + ind_off_global_col_temp\
                                                    + ind_off_zero
                    # Create indptr in global level
                    row_ind = np.arange(ind_off_local_row+ind_off_global_row, 
                                        ind_off_local_row+ind_off_global_row \
                                        +sub_mat_size_row[0])
                    mat_indptr_pre = np.zeros(row_ind[0]-A_range_row[0], 
                                              dtype='int32')
                    mat_indptr_post = np.ones(A_range_row[-1]-row_ind[-1]-1, 
                                              dtype='int32')*mat_indptr[-1]
                    mat_indptr_full = np.concatenate([mat_indptr_pre, 
                        mat_indptr, mat_indptr_post], dtype='int32')
                    # Set values in csr format
                    A_new.setValuesCSR(mat_indptr_full, mat_indices_global, 
                                       mat_vals_csr)
                else:
                    # Get and set values in dense format
                    sub_ind_row = np.arange(sub_mat_range_row[0], 
                                  sub_mat_range_row[1], dtype='int32')
                    sub_ind_col = np.arange(0, sub_mat_size_col[1], 
                                            dtype='int32')
                    mat_vals = sub_mat.getValues(sub_ind_row, sub_ind_col)
                    # Segment submatrix according to column ownership range
                    # and create column indices in global level
                    mat_vals_subs = []
                    col_ind_subs = []
                    for col_range_iter in range(MPI.size(comm)):
                        mat_vals_subs += [mat_vals[:,
                            sub_mat_range_col_all[col_range_iter][0]:
                            sub_mat_range_col_all[col_range_iter][1]]]
                        col_ind_subs += [np.arange(
                            ind_off_global_col_all[col_range_iter] \
                            + ind_off_local_col[col_range_iter],
                            ind_off_global_col_all[col_range_iter] \
                            + ind_off_local_col[col_range_iter] \
                            + sub_mat_size_col_all[col_range_iter][0], 
                            dtype='int32')]
                    # Create row indices for submatrix in global level
                    row_ind = np.arange(
                              ind_off_local_row + ind_off_global_row, 
                              ind_off_local_row + ind_off_global_row \
                              + sub_mat_size_row[0], dtype='int32')
                    # Set values in dense format
                    for col_range_iter in range(MPI.size(comm)):
                        A_new.setValues(row_ind, col_ind_subs[col_range_iter], 
                                        mat_vals_subs[col_range_iter])
            # Update column indices offset 
            for col_range_iter in range(MPI.size(comm)):
                ind_off_local_col[col_range_iter] \
                    += sub_mat_size_col_all[col_range_iter][0]
        # Update row indices offset
        ind_off_local_row += sub_mat_size_row[0]

    A_new.setUp()
    A_new.assemble()
    return A_new

def ksp_solve(A, x, b, ksp_type=PETSc.KSP.Type.CG, 
              pc_type=PETSc.PC.Type.FIELDSPLIT, 
              fieldsplit_type="additive",
              fieldsplit_ksp_type=PETSc.KSP.Type.PREONLY,
              fieldsplit_pc_type=PETSc.PC.Type.LU, 
              rtol=1e-15, max_it=100000,
              ksp_view=False, monitor_residual=False):
    """
    Solve "Ax=b" using PETSc Krylov solver.

    Parameters
    ----------
    A : PETSc.Mat
    x : PETSc.Vec
    b : PETSc.Vec
    ksp_type : str, default is "cg"
        KSP solver type, for addtioner type, see PETSc.KSP.Type
    pc_type : str, default is "fieldsplit"
        PETSc preconditioner type, for additional preconditioner 
        type, see PETSc.PC.Type
    fieldsplit_type : str, default is "additive"
        Only needed if preconditioner is "fieldsplit". {"additive", 
        "multiplicative", "symmetric_multiplicative", "schur"}
    fieldsplit_ksp_type : str, default is "cg"
    fieldsplit_pc_type : str, default is "lu"
    rtol : float, default is 1e-15
    max_it : int, default is 100000
    ksp_view : bool, default is False
    monitor_residual : bool, default is False
    """
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)
    pc = ksp.getPC()

    if ksp_view:
        PETScOptions.set('ksp_view')
    if monitor_residual: 
        PETScOptions.set('ksp_monitor_true_residual')

    PETScOptions.set('pc_type', pc_type)

    if pc_type == PETSc.PC.Type.FIELDSPLIT:
        nest_size = A.getNestSize()[0]
        PETScOptions.set('pc_type', 'fieldsplit')
        PETScOptions.set('pc_fieldsplit_type', 'additive')
        for i in range(nest_size):
            fieldsplit_ksp_name = "fieldsplit_"+str(i)+"_ksp_type"
            fieldsplit_pc_name = "fieldsplit_"+str(i)+"_pc_type"
            PETScOptions.set(fieldsplit_ksp_name, fieldsplit_ksp_type)
            PETScOptions.set(fieldsplit_pc_name, fieldsplit_pc_type)

        fields = []
        for i in range(nest_size):
            fields += [(str(i), A.getNestISs()[0][i]),]
        pc.setFieldSplitIS(*fields)
    else:
        pc.setType(pc_type)

    ksp.setTolerances(rtol=rtol)
    ksp.max_it = max_it
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)

    if ksp.getResidualNorm() < rtol:
        if mpirank == 0:
            print("KSP solver successfully converged with {} "
                  "iterations.".format(ksp.getIterationNumber()))
    else:
        if mpirank == 0:
            print("KSP solver didn't converge for relative tolerance {} "
                  "and max iteration {}. Consider using larger max "
                  "iterations or smaller relative tolerance."
                  .format(ksp.getTolerances()[0], ksp.max_it))

def solve_nonmatching_mat(A, x, b, solver="direct", 
                          ksp_type=PETSc.KSP.Type.CG, 
                          pc_type=PETSc.PC.Type.FIELDSPLIT, 
                          fieldsplit_type="additive",
                          fieldsplit_ksp_type=PETSc.KSP.Type.PREONLY,
                          fieldsplit_pc_type=PETSc.PC.Type.LU, 
                          rtol=1e-15, max_it=100000,
                          ksp_view=False, monitor_residual=False):
    """
    Solve system "Ax=b", where ``A`` is the LHS non-matching matrix.

    Parameters
    ----------
    A : nest PETSc.Mat
    x : nest PETSc.Vec
    b : nest PETSc.Vec
    solver : str, {"ksp", "direct"} or user defined solver, 
        Default is "ksp", which is petsc4py PETSc KSP solver
    ksp_type : str, default is "cg"
        KSP solver type, for addtioner type, see PETSc.KSP.Type
    pc_type : str, default is "fieldsplit"
        PETSc preconditioner type, for additional preconditioner 
        type, see PETSc.PC.Type
    fieldsplit_type : str, default is "additive"
        Only needed if preconditioner is "fieldsplit". {"additive", 
        "multiplicative", "symmetric_multiplicative", "schur"}
    fieldsplit_ksp_type : str, default is "cg"
    fieldsplit_pc_type : str, default is "lu"
    rtol : float, default is 1e-15
    max_it : int, default is 100000
    ksp_view : bool, default is False
    monitor_residual : bool, default is False
    """
    if not isinstance(A, PETSC4PY_MATRIX):
        if mpirank == 0:
            raise TypeError("Type "+str(type(A))+" is not supported yet.")

    if solver == "direct":
        solve(PETScMatrix(A), PETScVector(x), PETScVector(b), "mumps")
    elif solver == 'ksp':
        # ksp solver works in parallel
        ksp_solve(A, x, b, ksp_type=ksp_type, pc_type=pc_type, 
                  fieldsplit_type=fieldsplit_type,
                  fieldsplit_ksp_type=fieldsplit_ksp_type,
                  fieldsplit_pc_type=fieldsplit_pc_type, 
                  rtol=rtol, max_it=max_it, ksp_view=ksp_view, 
                  monitor_residual=monitor_residual)
    else:
        # Keep an option for user customized solver
        solver.ksp().setOperators(A=A)
        solver.ksp().solve(b, x)
        solver.ksp().reset()

def save_results(spline, u, index, file_name="u", save_path=SAVE_PATH, 
                 folder="results/", save_cpfuncs=True, comm=None):
    """
    Save results to .pvd file.

    Parameters
    ----------
    spline : ExtractedSpline
    u : dolfin Function
    index : int, index of the file
    file_name : str, optional, default is "u".
    save_path : str, optional
    folder : str, optional, default is "results/"
    save_cpfuncs : bool, optional
        If True, save spline.cpFuncs to pvd file. Default is True
    comm : mpi4py.MPI.Intracomm, optional, default is None.
    

    To view the saved files in Paraview:
    ------------------------------------------------------------------------------
    (F0_0/F0_3-coordsX)*iHat + (F0_1/F0_3-coordsY)*jHat + (F0_2/F0_3-coordsZ)*kHat
    (u0_0/F0_3)*iHat + (u0_1/F0_3)*jHat + (u0_2/F0_3)*kHat
    ------------------------------------------------------------------------------
    (F1_0/F1_3-coordsX)*iHat + (F1_1/F1_3-coordsY)*jHat + (F1_2/F1_3-coordsZ)*kHat
    (u1_0/F1_3)*iHat + (u1_1/F1_3)*jHat + (u1_2/F1_3)*kHat
    ------------------------------------------------------------------------------
    (F2_0/F2_3-coordsX)*iHat + (F2_1/F2_3-coordsY)*jHat + (F2_2/F2_3-coordsZ)*kHat
    (u2_0/F2_3)*iHat + (u2_1/F2_3)*jHat + (u2_2/F2_3)*kHat
    ------------------------------------------------------------------------------
    (F3_0/F3_3-coordsX)*iHat + (F3_1/F3_3-coordsY)*jHat + (F3_2/F3_3-coordsZ)*kHat
    (u3_0/F3_3)*iHat + (u3_1/F3_3)*jHat + (u3_2/F3_3)*kHat
    ------------------------------------------------------------------------------
    for index = 0, 1, 2, 3, etc.
    """
    u_split = u.split()

    if comm is None:
        comm = spline.comm

    if len(u_split) == 0:
        name_disp = file_name + str(index)
        u.rename(name_disp,name_disp)
        File(comm, save_path + folder + name_disp + "_file.pvd") << u
    else:
        for i in range(len(u_split)):
            name_disp = file_name + str(index) + "_" + str(i)
            u_split[i].rename(name_disp,name_disp)
            File(comm, save_path + folder + name_disp 
                + "_file.pvd") << u_split[i]

    if save_cpfuncs:
        for i in range(spline.nsd+1):
            name_control_mesh = "F" + str(index) + "_" + str(i)
            spline.cpFuncs[i].rename(name_control_mesh, name_control_mesh)
            File(comm, save_path + folder + name_control_mesh 
                + "_file.pvd") << spline.cpFuncs[i]

def save_cpfuncs(cpfuncs, index, save_path=SAVE_PATH, folder="results/", 
                 comm=worldcomm):
    """
    Save control point functions ``cpfuncs`` of an ExtractedSpline.

    Parameters
    -----------
    cpfuncs : list of dolfin Functions, or spline.cpFuns
    index : int, index of the save file
    save_path : str, optional

    To view save file in Paraview:
    -----------------------------------------
    (cpfuncs0_0/cpfuncs0_3-coordsX)*iHat \
    + (cpfuncs0_1/cpfuncs0_3-coordsY)*jHat \
    + (cpfuncs0_2/cpfuncs0_3-coordsZ)*kHat
    -----------------------------------------
    (cpfuncs1_0/cpfuncs1_3-coordsX)*iHat \
    + (cpfuncs1_1/cpfuncs1_3-coordsY)*jHat \
    + (cpfuncs1_2/cpfuncs1_3-coordsZ)*kHat
    -----------------------------------------
    (cpfuncs2_0/cpfuncs2_3-coordsX)*iHat \
    + (cpfuncs2_1/cpfuncs2_3-coordsY)*jHat \
    + (cpfuncs2_2/cpfuncs2_3-coordsZ)*kHat
    -----------------------------------------
    (cpfuncs3_0/cpfuncs3_3-coordsX)*iHat \
    + (cpfuncs3_1/cpfuncs3_3-coordsY)*jHat \
    + (cpfuncs3_2/cpfuncs3_3-coordsZ)*kHat
    -----------------------------------------
    for index = 0, 1, 2, 3, etc.
    """
    for i in range(len(cpfuncs)):
        name_control_mesh = "cpfuncs" + str(index) + "_" + str(i)
        cpfuncs[i].rename(name_control_mesh, name_control_mesh)
        File(comm, save_path + folder + name_control_mesh 
            + "_file.pvd") << cpfuncs[i]

def generate_interpolated_data(data, num_pts):
    """
    Given initial data ``data`` and specify the number of points 
    ``num_pts``, return the nearly evenly interpolated data.

    Parameters
    ----------
    data : ndarray
    num_pts : int

    Returns
    -------
    interp_data : ndarray
    """
    if data.ndim == 1:
        data = np.array([data]).transpose()
    rows, cols = data.shape

    if rows > num_pts:
        if mpirank == 0:
            print("Generating interpolated data ...")
            print("Number of points to interpolate {} is smaller than the "
                  "number of given points {}, removing points from data to "
                  "match the number of points.".format(num_pts, rows))
        num_remove = rows - num_pts
        remove_ind = np.linspace(1, rows-2, num_remove, dtype=int)
        interp_data = np.delete(data, remove_ind, axis=0)
    
    elif rows == num_pts:
        interp_data = data

    else:
        num_insert = num_pts - rows
        num_interval = rows - 1
        interp_data = np.zeros((num_pts, cols))

        num1 = round(num_insert/num_interval)
        num_insert_element = np.ones(num_interval).astype(int)*int(num1)
        round_num = int(num1*num_interval)
        diff = int(round_num - num_insert)

        if diff > 0:
            for i in range(abs(int(diff))):
                num_insert_element[i] -= 1
        elif diff < 0:
            for i in range(abs(int(diff))):
                num_insert_element[i] += 1

        num_pts_element = num_insert_element + 1

        for i in range(num_interval):
            for j in range(cols):
                if i == num_interval-1:
                    interp_data[np.sum(num_pts_element[0:i]):num_pts, j] \
                    = np.linspace(data[i,j], data[i+1,j], 
                                  num_pts_element[i]+1)[0:]
                else:
                    interp_data[np.sum(num_pts_element[0:i]):np.sum(\
                        num_pts_element[0:i+1]), j] = np.linspace(data[i,j], \
                        data[i+1,j], num_pts_element[i]+1)[0:-1]

    return interp_data

def generate_mortar_mesh(pts=None, num_el=None, data=None, comm=worldcomm):
    """
    Create topologically 1D, geometrically 1, 2 or 3D mortar mesh with 
    a single row of elements connecting them using given data points.

    Parameters
    ----------
    pts : ndarray or None, optional 
        Locations of nodes of mortar mesh
    num_el : int or None, optional 
        number of elements of mortar mesh
    data : ndarray or None, optional 
        Locations of nodes of mortar mesh. If ``data`` is not given, 
        ``pts`` and ``num_el`` are required.
    comm : mpi4py.MPI.Intarcomm, optional

    Returns
    -------
    mesh : dolfin Mesh
    """
    if data is not None:
        data = data
    else:
        data = generate_interpolated_data(pts, num_el+1)

    MESH_FILE_NAME = generateMeshXMLFileName(comm)

    if MPI.rank(comm) == 0:

        if data.ndim == 1:
            data = np.array([data]).transpose()
        rows, cols = data.shape

        dim = cols
        nverts = rows
        nel = nverts - 1

        fs = '<?xml version="1.0" encoding="UTF-8"?>' + "\n"
        fs += '<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">' + "\n"
        fs += '<mesh celltype="interval" dim="' + str(dim) + '">' + "\n"

        fs += '<vertices size="' + str(nverts) + '">' + "\n"
        if dim == 1:
            for i in range(nverts):
                x0 = repr(data[i,0])
                fs += '<vertex index="' + str(i) + '" x="' + x0 + '"/>' + "\n"
            fs += '</vertices>' + "\n"

        elif dim == 2:
            for i in range(nverts):
                x0 = repr(data[i,0])
                y0 = repr(data[i,1])
                fs += '<vertex index="' + str(i) + '" x="' + x0 \
                   + '" y="' + y0 + '"/>' + "\n"
            fs += '</vertices>' + "\n"

        elif dim == 3:
            for i in range(nverts):
                x0 = repr(data[i,0])
                y0 = repr(data[i,1])
                z0 = repr(data[i,2])
                fs += '<vertex index="' + str(i) + '" x="' + x0 \
                   + '" y="' + y0 + '" z="' + z0 +'"/>' + "\n"
            fs += '</vertices>' + "\n"

        else:
            raise ValueError("Unsupported parametric"
                " dimension: {}".format(dim))

        fs += '<cells size="' + str(nel) + '">' + "\n"
        for i in range(nel):
            v0 = str(i)
            v1 = str(i+1)
            fs += '<interval index="' + str(i) + '" v0="' + v0 + '" v1="' \
                + v1 + '"/>' + "\n"

        fs += '</cells></mesh></dolfin>'

        f = open(MESH_FILE_NAME,'w')
        f.write(fs)
        f.close()
        
    MPI.barrier(comm)    
    mesh = Mesh(comm, MESH_FILE_NAME)

    if MPI.rank(comm) == 0:
        os.remove(MESH_FILE_NAME)

    return mesh

def deformed_position(spline, xi, u_hom):
    """
    Return deformed position of a point in physical domain by giving 
    the parametric location and displacement.

    Parameters
    ----------
    spline : ExtractedSpline
    xi : ndarray, parametric location of point
    u_hom : dolfin Function, displacement of point

    Returns
    -------
    position : ndarray
    """
    position = np.zeros(3)
    for i in range(3):
        # # Direct evaluation doesn't work in parallel
        # position[i] = (spline.cpFuncs[i](xi)
        #     +u_hom(xi)[i])/spline.cpFuncs[3](xi)
        cp_funcs_i = eval_func(spline.mesh, spline.cpFuncs[i], xi)
        u_hom_i = eval_func(spline.mesh, u_hom[i], xi)
        w = eval_func(spline.mesh, spline.cpFuncs[3], xi)
        position[i] = (cp_funcs_i + u_hom_i)/w
    return position

def undeformed_position(spline, xi):
    """
    Return undeformed position of a point in the physical domain by 
    specifying the parametric location.

    Parameters
    ----------
    spline : ExtractedSpline
    xi : ndarray, parametric location of point

    Returns
    -------
    position : ndarray
    """
    position = np.zeros(3)
    for i in range(3):
        # position[i] = spline.cpFuncs[i](xi)/spline.cpFuncs[3](xi)
        cp_funcs_i = eval_func(spline.mesh, spline.cpFuncs[i], xi)
        w = eval_func(spline.mesh, spline.cpFuncs[3], xi)
        position[i] = cp_funcs_i/w
    return position

def compute_line_Jacobian(X):
    """
    Compute line Jacobian for mortar mesh.

    Parameters
    ----------
    X : dolfin ListTensor 
        Geometric mapping of the mortar mesh

    Returns
    -------
    line_Jacobian : ufl Sqrt
    """
    dXdxi = grad(X)
    line_Jacobian = sqrt(tr(dXdxi*dXdxi.T))
    return line_Jacobian

def move_mortar_mesh(mortar_mesh, mesh_location):
    """
    Move the mortar mesh to a specified location.

    Parameters
    ----------
    mortar_mesh : dolfin Mesh
    mesh_location : ndarray or DOLFIN function
    """
    if isinstance(mesh_location, DOLFIN_FUNCTION):
        set_coordinates(mortar_mesh.geometry(), mesh_location)
    elif isinstance(mesh_location, np.ndarray):
        Vm = VectorFunctionSpace(mortar_mesh, 'CG', 1)
        um = Function(Vm)

        num_node = int(um.vector().vec().getSizes()[1]\
                       /mortar_mesh.geometric_dimension())
        if num_node == mesh_location.shape[0]:
            mesh_location_data = mesh_location
        else:
            mesh_location_data = generate_interpolated_data(mesh_location, 
                                                            num_node)

        mesh_location_flat = mesh_location_data[::-1].reshape(-1, 1)
        v2p(um.vector()).setValues(np.arange(mesh_location_flat.size, 
                                   dtype='int32'), mesh_location_flat)
        v2p(um.vector()).ghostUpdate()
        set_coordinates(mortar_mesh.geometry(), um)
    else:
        raise TypeError("Type "+str(type(mesh_location))+ 
                        " is not supported for mesh_location.")

def spline_mesh_phy_coordinates(spline, reshape=True):
    """
    Return the physical coordiantes of the spline mesh.

    Parameters
    ----------
    spline : ExtractedSpline
    reshape : bool, optional

    Returns
    -------
    phy_coordinates : ndarray
    """
    para_coordinates = spline.mesh.coordinates()
    phy_coordinates = np.zeros((para_coordinates.shape[0], 3))
    for i in range(para_coordinates.shape[0]):
        for j in range(3):
            phy_coordinates[i, j] = spline.F[j](para_coordinates[i])
    if reshape:
        num_rows = int(np.where(para_coordinates[:,0]==0.)[0][1])
        num_cols = int(para_coordinates.shape[0]/num_rows)
        phy_coordinates = phy_coordinates.reshape(num_rows, num_cols, 3)
    return phy_coordinates

def tangent_components(mesh):
    """
    Compute tangent vector components of a 1D interval mesh
    in 2D parameteric space.

    Parameters
    ----------
    mesh : dolfin mesh
    """
    mesh_coord = mesh.coordinates()
    mesh_x_coord = mesh_coord[:,0]
    mesh_y_coord = mesh_coord[:,1]

    x_coord_grad = np.gradient(mesh_x_coord)
    y_coord_grad = np.gradient(mesh_y_coord)

    V = FunctionSpace(mesh, 'CG', 1)
    t1 = Function(V)
    t2 = Function(V)

    t1.vector().set_local(x_coord_grad[::-1])
    t2.vector().set_local(y_coord_grad[::-1])

    return t1, t2

def spline_mesh_size(spline):
    """
    Compute the mesh size in the physical space of 
    ExtractedSpline.

    Parameters
    ----------
    spline : ExtractedSpline

    Returns
    -------
    h : ufl math functions
    """
    # dxi_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
    # dX_dxi = grad(spline.F)
    # dX_dxiHat = dX_dxi*dxi_dxiHat
    # h = sqrt(tr(dX_dxiHat*dX_dxiHat.T))
    h_para = CellDiameter(spline.mesh)
    dX_dxi = grad(spline.F)
    h = h_para*sqrt(tr(dX_dxi*dX_dxi.T))
    return h

def point_in_mesh(mesh, xi):
    """
    Check if a point of location ``xi`` is inside mesh.

    Parameters
    ----------
    mesh : dolfin mesh
    xi : ndarray

    Returns
    -------
    res : bool
    """
    return len(mesh.bounding_box_tree()\
        .compute_entity_collisions(Point(xi)))>0

def eval_func(mesh, f, xi, allreduce=True):
    """
    Evaluate function ``f`` at point ``xi``, where ``f`` has 
    domain of ``mesh``. This function is used in parallel.

    Parameters
    ----------
    mesh : dolfin mesh
    f : dolfin Function
    xi : ndarray
    allreduce : bool, optional, default is True.
    """
    pt_in_mesh = point_in_mesh(mesh, xi)
    pt_in_mesh_allgather = worldcomm.allgather(pt_in_mesh)

    if pt_in_mesh:
        res = f(xi)
    else:
        if len(f.ufl_shape) > 0:
            res = np.zeros(f.ufl_shape[0])
        else:
            res = 0.

    if allreduce:
        res = worldcomm.allreduce(res, op=pyMPI.SUM)\
              /pt_in_mesh_allgather.count(True)

    return res

def lumped_project(to_project, V):
    """
    Project a ufl expression to function space with lumped mass
    
    Parameters
    ----------
    to_project : ufl expression to project
    V : dolfin FunctionSpace

    Returns
    -------
    u : dolfin Function
    """
    u = Function(V)
    v = TestFunction(V)
    if len(u.ufl_shape) > 0:
        lhs = assemble(inner(Constant((1.,)*u.ufl_shape[0]), v)*dx)        
    else:
        lhs = assemble(inner(Constant(1.,), v)*dx)
    rhs = assemble(inner(to_project, v)*dx)
    as_backend_type(u.vector()).vec().\
        pointwiseDivide(as_backend_type(rhs).vec(),
                        as_backend_type(lhs).vec())
    return u

def spline_project(spline, V, to_project, lump_mass=False):
    """
    Project an expression to a user defined function space ``V``.

    Parameters
    ----------
    spline : tIGAr extracted spline
    V : dolfin function space
    to_project : ufl expression
    lump_mass : bool, default is False

    Returns
    -------
    retval : dolfin function
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    rhs_form = inner(to_project, v)*spline.dx.meas
    retval = Function(V)
    if lump_mass:
        if len(to_project.ufl_shape) > 0:
            num_field = to_project.ufl_shape[0]
            lhs_form = inner(Constant((1.0)*num_field))*spline.dx.meas
        else:
            lhs_form = inner(Constant((1.0)),v)*spline.dx.meas
        A = assemble(lhs_form)
        b = assemble(rhs_form)
        as_backend_type(retval.vector()).vec()\
            .pointwiseDivide(as_backend_type(b).vec(),
                             as_backend_type(A).vec())
    else:
        lhs_form = inner(u,v)*spline.dx.meas
        A = assemble(lhs_form)
        b = assemble(rhs_form)
        solve(A, retval.vector(), b)
    return retval

def spline_project_scalar(spline, to_project, rationalize=True, 
                          lump_mass=False):
    """
    Project a scalar field expression to FE function space 
    ``spline.V_control``

    Parameters
    ----------
    spline : tIGAr extracted spline
    to_project : ufl expression
    rationalize : bool, default is True
    lump_mass : bool, default is False
    """
    u = TrialFunction(spline.V_control)
    v = TestFunction(spline.V_control)
    u = spline.rationalize(u)
    v = spline.rationalize(v)
    rhs_form = inner(to_project, v)*spline.dx
    retval = Function(spline.V_control)
    if lump_mass:
        iga_dofs = PETScVector(zero_petsc_vec(m2p(spline.M_control).sizes[1]))
        lhs_form = inner(Constant((1.0)),v)*spline.dx
        lhs_vec_FE = assemble(lhs_form)
        lhs_vec = AT_x(m2p(spline.M_control), v2p(lhs_vec_FE))
        rhs_vec_FE = assemble(rhs_form)
        rhs_vec = AT_x(m2p(spline.M_control), v2p(rhs_vec_FE))
        as_backend_type(iga_dofs.vector()).vec()\
            .pointwiseDivide(as_backend_type(rho_vec),
                             as_backend_type(lhs_vec))
        retval.vector()[:] = spline.M_control*iga_dofs
    else:
        iga_dofs = PETScVector(zero_petsc_vec(m2p(spline.M_control).sizes[1]))
        lhs_form = inner(u,v)*spline.dx
        A_FE = assemble(lhs_form)
        A_IGA = AT_R_B(m2p(spline.M_control), m2p(A_FE), 
                       m2p(spline.M_control))
        b_FE = assemble(rhs_form)
        b_IGA = AT_x(m2p(spline.M_control), v2p(b_FE))
        solve(PETScMatrix(A_IGA), iga_dofs, PETScVector(b_IGA), "mumps")
        retval.vector()[:] = spline.M_control*iga_dofs
    if rationalize:
        as_backend_type(retval.vector()).vec()\
            .pointwiseDivide(as_backend_type(retval.vector()).vec(),
                as_backend_type(spline.cpFuncs[-1].vector()).vec())
    return retval


# import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D 

# def set_axes_equal(ax):
#     """
#     Set the 3D plot axes equal.

#     Parameters
#     ----------
#     ax : Axes3DSubplot
#     """
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()
#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)
#     plot_radius = 0.5*max([x_range, y_range, z_range])
#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# def plot3D(data, style="-", color=None, label=None, 
#            linewidth=1, fig=None, axes_equal=False,
#            x_label="x", y_label="y", z_label="z"):
#     """
#     3D plot.

#     Parameters
#     ----------
#     data : ndarray, 3-dimensional
#     style : str, optional
#     color : str, optional
#     label : str, optional
#     linewidth : int, optional
#     fig : matplotlib Figure, optional
#     axes_equal : bool, optional
#     xlabel : str, optional
#     ylabel : str, optional
#     zlabel : str, optional
#     """
#     if fig is None:
#         fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot(data[:,0], data[:,1], data[:,2], style, 
#             color=color, label=label, linewidth=linewidth)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_zlabel(z_label)
#     if axes_equal:
#         set_axes_equal(ax)

if __name__ == "__main__":
    pass