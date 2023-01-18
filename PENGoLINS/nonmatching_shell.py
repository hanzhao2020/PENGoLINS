"""
The "nonmatching_shell" module
------------------------------
contains functions that penalize displacement and rotational continuity 
of non-matching interface.
"""

from PENGoLINS.nonmatching_utils import *
from PENGoLINS.parametric_loc import *
from ufl import Jacobian

# Shell problem definitions
from ShNAPr.kinematics import *
from ShNAPr.SVK import *
from ShNAPr.hyperelastic import *

def create_transfer_matrix_list(V1, V2, deriv=1):
    """
    Create a list that contains the transfer matrices of 
    unknowns and their derivatives.

    Parameters
    ----------
    V1 : dolfin FunctionSpace
    V2 : dolfin FunctionSpace
    deriv : int, creat transfer matrices from derivative
        order 0 to ``deriv``. Default is 1.

    Returns
    -------
    matrix_list : list of dolfin PETScMatrices
    """
    matrix_list = []
    for i in range(deriv+1):
        matrix_list += [create_transfer_matrix(V1,V2,i)]
    return matrix_list

def transfer_mortar_u(u, um, A):
    """
    Transfer spline patch's displacements to mortar mesh.

    Parameters
    ----------
    u : dolfin Function, spline patch's displacement
    um : list of dolfin Function, mortar mesh's 
         displacement and first derivatives
    A : list of dolfin PETScMatrices for displacement
        and its first derivative
    """
    for i in range(len(um)):
        m2p(A[i]).mult(v2p(u.vector()), v2p(um[i].vector()))

def transfer_mortar_cpfuns(spline, mortar_cpfuncs, A_control):
    """
    Transfer ``spline.cpFuns`` and its derivatives to the 
    function space of mortar mesh. 

    Parameters
    ----------
    spline : ExtractedSpline
    mortar_cpfuns : list of dolfin Functions for mortar mesh's
        control point functions, four components.
    A_control : list of dolfin PETScMatrices
    """
    for i in range(len(mortar_cpfuncs)):
        for j in range(len(mortar_cpfuncs[i])):
            m2p(A_control[i]).mult(v2p(spline.cpFuncs[j].vector()), 
                                   v2p(mortar_cpfuncs[i][j].vector()))

def create_geometrical_mapping(spline, cpfuncs):
    """
    Create the geometric mapping using ``cp_fucns`` and 
    its derivatives.

    Parameters
    ----------
    spline : ExtractedSpline
    cpfuncs : list of dolfin Functions

    Returns
    -------
    F : dolfin ListTensor
    dFdxi : dolfin ListTensor
    """
    cpfuncs_deriv0 = cpfuncs[0]
    cpfuncs_deriv10 = []
    cpfuncs_deriv11 = []
    for i in range(len(cpfuncs[1])):
        cpfuncs_deriv10 += [cpfuncs[1][i][0]]
        cpfuncs_deriv11 += [cpfuncs[1][i][1]]

    # Reference configuration for mortar mesh
    F_list = []
    dFdxi0_list = []
    dFdxi1_list = []
    for i in range(len(cpfuncs[0])-1):
        # Components of reference configuration of mortar mesh
        F_list += [cpfuncs_deriv0[i]/cpfuncs_deriv0[-1]]
        # Components of derivatives of reference configuration
        # of mortar mesh, using quotient rule to compute the 
        # derivatives: d(f/g) = (df*g-f*dg)/g**2.
        dFdxi0_list += [(cpfuncs_deriv10[i]*cpfuncs_deriv0[-1] \
                        -cpfuncs_deriv0[i]*cpfuncs_deriv10[-1])\
                        /(cpfuncs_deriv0[-1]*cpfuncs_deriv0[-1]),]
        dFdxi1_list += [(cpfuncs_deriv11[i]*cpfuncs_deriv0[-1] \
                        -cpfuncs_deriv0[i]*cpfuncs_deriv11[-1])\
                        /(cpfuncs_deriv0[-1]*cpfuncs_deriv0[-1]),]

    F = as_vector(F_list)
    dFdxi = as_tensor([[dFdxi0_list[0], dFdxi1_list[0]],\
                       [dFdxi0_list[1], dFdxi1_list[1]],\
                       [dFdxi0_list[2], dFdxi1_list[2]]])
    return F, dFdxi

def physical_configuration(F, dFdxi, cpfuncs, um):
    """
    Return the physical configuration or the deformed state
    of the mortar mesh.

    Parameters
    ----------
    F : dolfin ListTensor
    dFdxi : dolfin ListTensor
    cpfuncs : list of dolfin Functions
    um : list of dolfin Functions

    Returns
    -------
    x : dolfin ListTensor
    dxdxi : dolfin ListTensor
    """
    um_deriv0 = um[0]
    um_deriv10 = as_vector([um[1][0], um[1][2], um[1][4]])
    um_deriv11 = as_vector([um[1][1], um[1][3], um[1][5]])

    cpfuncs_deriv0 = cpfuncs[0]
    cpfuncs_deriv10 = []
    cpfuncs_deriv11 = []
    for i in range(len(cpfuncs[1])):
        cpfuncs_deriv10 += [cpfuncs[1][i][0]]
        cpfuncs_deriv11 += [cpfuncs[1][i][1]]

    u = um_deriv0/cpfuncs_deriv0[-1]
    dudxi0 = (um_deriv10*cpfuncs_deriv0[-1] - um_deriv0\
           *cpfuncs_deriv10[-1])/(cpfuncs_deriv0[-1]**2)
    dudxi1 = (um_deriv11*cpfuncs_deriv0[-1] - um_deriv0\
           *cpfuncs_deriv11[-1])/(cpfuncs_deriv0[-1]**2)
    dudxi = as_tensor([[dudxi0[0], dudxi1[0]],
                       [dudxi0[1], dudxi1[1]],
                       [dudxi0[2], dudxi1[2]]])
    # Current configure in physical coordinates `xm` and 
    # its parametric gradient `dxmdxi`
    x = u + F
    dxdxi = dudxi + dFdxi
    return x, dxdxi

def interface_geometry(dxdxi):
    """
    Compute the curvilinear basis "a2" and normal vector "an".

    Parameters
    ----------
    dxdxi : dolfin ListTensor

    Returns
    -------
    a0 : dolfin ListTensor
    a1 : dolfin ListTensor
    a2 : dolfin ListTensor
    """
    # Curvilinear basis 'a0', 'a1' and 'a2'
    a0 = as_vector([dxdxi[0,0], dxdxi[1,0], dxdxi[2,0]])
    a1 = as_vector([dxdxi[0,1], dxdxi[1,1], dxdxi[2,1]])
    a2 = unit(cross(a0, a1))
    return a0, a1, a2

def interface_orthonormal_basis(dxdxi):
    """
    Return orthonormal basis `e_i` for i = 1,2,3.

    Parameters
    ----------
    dxdxi : dolfin ListTensor
    e0 : dolfin ListTensor
    e1 : dolfin ListTensor
    a2 : dolfin ListTensor
    """
    a0, a1, a2 = interface_geometry(dxdxi)
    e0, e1 = orthonormalize2D(a0, a1)
    return e0, e1, a2

def project_normal_vector_onto_tangent_space(to_project, e0, e1):
    """
    Project a normal vector on to the tangent space of a surface
    """
    res = inner(to_project, e0)*e0 + inner(to_project, e1)*e1
    return res

def penalty_displacement_integrand(alpha_d, u0m_hom, u1m_hom):
    """
    Penalization of displacements on the non-matching 
    interface between two splines.
    
    Parameters
    ----------
    alpha_d : ufl.algebra.Division
    u0m_hom : dolfin Function
    u1m_hom : dolfin Function

    Return
    ------
    W_pd : ufl Form
    """
    W_pd_int = 0.5*alpha_d*((u0m_hom-u1m_hom)**2)
    return W_pd_int

def penalty_rotation_integrand(mortar_mesh, alpha_r, 
                               dX0dxi, dx0dxi, dX1dxi, dx1dxi,
                               proj_tan=True):
    """
    Penalization of rotation on the non-matching interface 
    between two splines.

    Parameters
    ----------
    mortar_mesh : dolfin Mesh
    alpha_r : ufl.algebra.Division
    dX0dxi : dolfin ListTensor
    dx0dxi : dolfin ListTensor
    dX1dxi : dolfin ListTensor
    dx1dxi : dolfin ListTensor

    Return
    ------
    W_pr : ufl Form
    """
    # Orthonormal basis for patch 0
    a00, a01, a02 = interface_geometry(dx0dxi)
    A00, A01, A02 = interface_geometry(dX0dxi)

    # Orthonormal basis for patch 1
    a10, a11, a12 = interface_geometry(dx1dxi)
    A10, A11, A12 = interface_geometry(dX1dxi)

    xi = SpatialCoordinate(mortar_mesh)
    t = Jacobian(xi)
    if proj_tan:
        # lump mass projection of tangent vector
        Vm = FunctionSpace(mortar_mesh, 'CG', 1)
        t0 = lumped_project(t[0,0], Vm)
        t1 = lumped_project(t[1,0], Vm)
    else:
        t0, t1 = t[0,0], t[1,0]

    at0 = t0*a00 + t1*a01
    At0 = t0*A00 + t1*A01

    an1 = cross(a02, at0)/sqrt(inner(at0, at0))
    An1 = cross(A02, At0)/sqrt(inner(At0, At0))

    W_pr_int = 0.5*alpha_r*((inner(a02, a12) - inner(A02, A12))**2 
             + (inner(an1, a12) - inner(An1, A12))**2)

    return W_pr_int

# def penalty_rotation_in_paper(alpha_r, dX1dxi, dx1dxi, dX2dxi, dx2dxi, 
#                               line_Jacobian=None, dx_m=None):
#     """
#     Penalization of rotation on the non-matching interface 
#     between two splines.

#     Parameters
#     ----------
#     alpha_r : ufl.algebra.Division
#     u1m_hom : dolfin Function
#     u2m_hom : dolfin Function
#     dXdxi : dolfin ListTensor
#     line_Jacobian : dolfin Function or None, optional
#     dx_m : ufl Measure or None, optional

#     Return
#     ------
#     W_pr : ufl Form
#     """
#     # print("New penatly rotation")
#     if line_Jacobian is None:
#         line_Jacobian = Constant(1.)
#     if dx_m is None:
#         dx_m = dx

#     # Orthonormal basis for patch 1
#     e11, e21, e31 = interface_orthonormal_basis(dx1dxi)
#     E11, E21, E31 = interface_orthonormal_basis(dX1dxi)

#     # Orthonormal basis for patch 2
#     e12, e22, e32 = interface_orthonormal_basis(dx2dxi)
#     E12, E22, E32 = interface_orthonormal_basis(dX2dxi)

#     pe = project_normal_vector_onto_tangent_space(e32, e11, e21)
#     pE = project_normal_vector_onto_tangent_space(E32, E11, E21)

#     W_pr = 0.5*alpha_r*((dot(e31, e32)-dot(E31, E32))**2 \
#            + (sqrt(dot(pe, pe)+DOLFIN_EPS)-sqrt(dot(pE, pE)))**2) \
#            *line_Jacobian*dx_m

#     # # Expand the above formulation to increase stability
#     # W_pr = 0.5*alpha_r*((dot(e31, e32)-dot(E31, E32))**2 \
#     #      + (dot(pe, pe) + dot(pE, pE) 
#     #         - (2*sqrt(dot(pe, pe)+DOLFIN_EPS)*sqrt(dot(pE,pE))) )
#     #      )*line_Jacobian*dx_m

#     return W_pr

# def penalty_rotation_old(alpha_r, dX1dxi, dx1dxi, dX2dxi, dx2dxi, 
#                          line_Jacobian=None, dx_m=None):
#     """
#     Penalization of rotation on the non-matching interface 
#     between two splines.

#     Parameters
#     ----------
#     alpha_r : ufl.algebra.Division
#     u1m_hom : dolfin Function
#     u2m_hom : dolfin Function
#     dXdxi : dolfin ListTensor
#     line_Jacobian : dolfin Function or None, optional
#     dx_m : ufl Measure or None, optional

#     Return
#     ------
#     W_pr : ufl Form
#     """
#     # print("Old penatly rotation")
#     # For patch 1
#     if line_Jacobian is None:
#         line_Jacobian = Constant(1.)
#     if dx_m is None:
#         dx_m = dx
#     e11, e21, e31 = interface_orthonormal_basis(dx1dxi)
#     E11, E21, E31 = interface_orthonormal_basis(dX1dxi)

#     # For patch 2
#     a12, a22, a32 = interface_geometry(dx2dxi)
#     A12, A22, A32 = interface_geometry(dX2dxi)

#     W_pr = 0.5*alpha_r*((dot(e11, a32)-dot(E11, A32))**2 \
#          + (dot(e21, a32)-dot(E21, A32))**2
#          + (dot(e31, a32)-dot(E31, A32))**2)*line_Jacobian*dx_m
#     return W_pr

def penalty_energy(spline0, spline1, u0, u1, 
                   mortar_mesh, mortar_funcs, mortar_cpfuncs, 
                   A, A_control, alpha_d, alpha_r, 
                   dx_m=None, metadata=None, proj_tan=True):
    """
    Penalization of displacement and rotation of non-matching interface 
    between two extracted splines.
    
    Parameters
    ----------
    spline0 : ExtractedSpline
    spline1 : ExtractedSpline
    u0 : dolfin Function, displacement of spline0
    u1 : dolfin Function, displacement of splint1
    mortar_mesh : dolfin Mesh
    mortar_funcs : list of dolfin Functions, mortar mesh's displacement
        and first derivatives on two sides
    mortar_cpfuncs : list of dolfin Functions, mortar mesh's control
        point functions and first derivatives on two sides
    A : list of dolfin PETScMatrices, transfer matrices for displacements
    A_control : list of dolfin PETScMatrices, transfer matrices
        for control point functions
    alpha_d : ufl.algebra.Division
    alpha_r : ufl.algebra.Division
    dx_m : ufl Measure or None
    metadata : dict, metadata for dolfin integration measure
    proj_tan : bool, project mortar mesh's tangent vector 
        onto a CG1 function space with lumped mass

    Returns
    -------
    W_p : ufl Form
    """
    splines = [spline0, spline1]
    spline_u = [u0, u1]
    mortar_X = []
    mortar_dXdxi = []
    mortar_x = []
    mortar_dxdxi = []
    for side in range(len(mortar_funcs)):
        transfer_mortar_u(spline_u[side], mortar_funcs[side], A[side])
        transfer_mortar_cpfuns(splines[side], mortar_cpfuncs[side], 
                               A_control[side])
        X_temp, dXdxi_temp = create_geometrical_mapping(splines[side], 
                             mortar_cpfuncs[side])
        mortar_X += [X_temp]
        mortar_dXdxi += [dXdxi_temp]
        x_temp, dxdxi_temp = physical_configuration(X_temp, dXdxi_temp, 
                             mortar_cpfuncs[side], mortar_funcs[side])
        mortar_x += [x_temp]
        mortar_dxdxi += [dxdxi_temp]

    if dx_m is not None:
        dx_m = dx_m
    else:
        if metadata is not None:
            dx_m = dx(domain=mortar_mesh, metadata=metadata)
        else:
            dx_m = dx(domain=mortar_mesh, metadata={'quadrature_degree': 0, 
                                          'quadrature_scheme': 'vertex'})

    line_Jacobian = compute_line_Jacobian(mortar_X[1])
    if line_Jacobian == 0.:
        line_Jacobian = sqrt(tr(mortar_dXdxi[1]*mortar_dXdxi[1].T))

    # Penalty of displacement
    W_pd_int = penalty_displacement_integrand(alpha_d, mortar_funcs[0][0], 
                                              mortar_funcs[1][0])
    # Penalty of rotation
    W_pr_int = penalty_rotation_integrand(mortar_mesh, alpha_r, 
                                          mortar_dXdxi[0], mortar_dxdxi[0], 
                                          mortar_dXdxi[1], mortar_dxdxi[1],
                                          proj_tan)

    W_p = (W_pd_int + W_pr_int)*line_Jacobian*dx_m
    return W_p

def SVK_residual(spline, u_hom, z_hom, E, nu, h, dWext=None):
    """
    PDE residaul of Kirchhoff--Love shell using the St.Venant--Kirchhoff 
    (SVK) material model.

    Parameters
    ----------
    spline : ExtractedSpline
    u_hom : dolfin Function
    z_hom : dolfin Argument
    E : dolfin Constant
    h : dolfin Constant
    dWext : dolfin Form

    Returns
    -------
    res : dolfin Form
    """
    X = spline.F
    x = X + spline.rationalize(u_hom)
    z = spline.rationalize(z_hom)

    Wint = surfaceEnergyDensitySVK(spline, X, x, E, nu, h)*spline.dx
    dWint = derivative(Wint, u_hom, z_hom)
    if dWext is None:
        res = dWint
    else:
        res = dWint - dWext
    return res

def hyperelastic_residual(spline, u_hom, z_hom, h, psi_el, 
                          dWext=None, quad_pts=4):
    """
    PDE residual for Kirchhoff--Love shell using incompressible 
    hyperelastic model.

    Parameters
    ----------
    spline : ExtractedSpline
    u_hom : dolfin Function
    z_hom : dolfin Argument
    h : dolfin Constant
    psi_el : function, elastic strain-energy functional
    dWext : dolfin Form
    quad_pts : int

    Returns
    -------
    res: dolfin Form
    """
    X = spline.F
    x = X + spline.rationalize(u_hom)
    dxi2 = throughThicknessMeasure(quad_pts, h)
    psi = incompressiblePotentialKL(spline, X, x, psi_el)
    Wint = psi*dxi2*spline.dx
    dWint = derivative(Wint, u_hom, z_hom)
    if dWext is None:
        res = dWint
    else:
        res = dWint - dWext
    return res


if __name__ == "__main__":
    pass