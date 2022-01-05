"""
The "nonmatching_shell" module
------------------------------
contains functions that penalize displacement and rotational continuity 
of non-matching interface.
"""

from PENGoLINS.nonmatching_utils import *
from PENGoLINS.parametric_loc import *

# Shell problem definitions
from ShNAPr.kinematics import *
from ShNAPr.SVK import *
from ShNAPr.hyperelastic import *

def create_transfer_matrix_list(V1, V2, dV2=None):
    """
    Create a list that contains the transfer matrices of 
    unknowns and their derivatives.

    Parameters
    ----------
    V1 : dolfin FunctionSpace
    V2 : dolfin FunctionSpace
    dV2 : dolfin FunctionSpace, optional
        If dV2 is not None, create the transfer matrices 
        of the derivatives.

    Returns
    -------
    matrix_list : list of dolfin PETScMatrices
    """
    matrix_list = []
    # A12 = PETScDMCollection.create_transfer_matrix(V1,V2)
    A12 = create_transfer_matrix(V1,V2)
    matrix_list += [A12,]

    if dV2 is not None: 
        # Matrices to trasfer derivatives
        dim = dV2.mesh().geometric_dimension()
        for i in range(dim):
            dA12 = create_transfer_matrix_partial_derivative(V1, dV2, i)
            matrix_list += [dA12,]

    return matrix_list

def transfer_cpfuns(spline, V_control, dV_control, A_control):
    """
    Transfer ``spline.cpFuns`` and its derivatives to the 
    function space of mortar mesh. 

    Parameters
    ----------
    spline : ExtractedSpline
    V_control : dolfin FunctionSpace
    dV_control : dolfin FunctionSpace
    A_control : list of dolfin PETScMatrices

    Returns
    -------
    cpfuncs : PETSc.Vec
    cpfuncs_dxi1 : PETSc.Vec
    cpfuncs_dxi2 : PETSc.Vec
    """
    nsd = spline.nsd
    cpfuncs = []
    cpfuncs_dxi1 = []
    cpfuncs_dxi2 = []
    for i in range(nsd+1):
        cpfuncs += [Function(V_control),]
        cpfuncs_dxi1 += [Function(dV_control),]
        cpfuncs_dxi2 += [Function(dV_control),]
        m2p(A_control[0]).mult(spline.cpFuncs[i].vector().vec(), 
                               cpfuncs[i].vector().vec())
        m2p(A_control[1]).mult(spline.cpFuncs[i].vector().vec(), 
                               cpfuncs_dxi1[i].vector().vec())
        m2p(A_control[2]).mult(spline.cpFuncs[i].vector().vec(), 
                               cpfuncs_dxi2[i].vector().vec())
    return cpfuncs, cpfuncs_dxi1, cpfuncs_dxi2

def create_geometrical_mapping(spline, cpfuncs, cpfuncs_dxi1, cpfuncs_dxi2):
    """
    Create the geometric mapping using ``cp_fucns`` and 
    its derivatives.

    Parameters
    ----------
    spline : ExtractedSpline
    cpfuncs : PETSc.Vec
    cpfuncs_dxi1 : PETSc.Vec
    cpfuncs_dxi2 : PETSc.Vec

    Returns
    -------
    F : dolfin ListTensor
    dFdxi : dolfin ListTensor
    """
    nsd = spline.nsd
    # Reference configuration construction on mortar mesh.
    components = []
    components_dxi1 = []
    components_dxi2 = []
    for i in range(nsd):
        # Components of reference configuration of 'mesh_m'.
        components += [cpfuncs[i]/cpfuncs[nsd],]
        # Components of derivatives of reference configuration
        # of 'mesh_m', using quotient rule to compute the 
        # derivatives: d(f/g) = (df*g-f*dg)/g**2. 
        components_dxi1 += [(cpfuncs_dxi1[i]*cpfuncs[nsd] -\
            cpfuncs[i]*cpfuncs_dxi1[nsd])/(cpfuncs[nsd]*cpfuncs[nsd]), ]
        components_dxi2 += [(cpfuncs_dxi2[i]*cpfuncs[nsd] -\
            cpfuncs[i]*cpfuncs_dxi2[nsd])/(cpfuncs[nsd]*cpfuncs[nsd]), ]

    F = as_vector(components)
    dFdxi = as_tensor([[components_dxi1[0], components_dxi2[0]],\
                       [components_dxi1[1], components_dxi2[1]],\
                       [components_dxi1[2], components_dxi2[2]]])
    return F, dFdxi

def physical_configuration(cpfuncs, cpfuncs_dxi1, cpfuncs_dxi2, 
                           F, dFdxi, mortar_vars):
    """
    Return the physical configuration or the deformed state
    of the mortar mesh.

    Parameters
    ----------
    cpfuncs : PETSc.Vec
    cpfuncs_dxi1 : PETSc.Vec
    cpfuncs_dxi2 : PETSc.Vec
    F : dolfin ListTensor
    dFdxi : dolfin ListTensor
    mortar_vars : list of dolfin Functions

    Returns
    -------
    x : dolfin ListTensor
    dxdxi : dolfin ListTensor
    """
    u = mortar_vars[0]/cpfuncs[-1]
    dudxi1 = (mortar_vars[1]*cpfuncs[-1] - mortar_vars[0]\
           *cpfuncs_dxi1[-1])/(cpfuncs[-1]**2)
    dudxi2 = (mortar_vars[2]*cpfuncs[-1] - mortar_vars[0]\
           *cpfuncs_dxi2[-1])/(cpfuncs[-1]**2)
    dudxi = as_tensor([[dudxi1[0], dudxi2[0]],\
                       [dudxi1[1], dudxi2[1]],\
                       [dudxi1[2], dudxi2[2]]])
    # Current configure in physical coordinates 'xm' and 
    # its parametric gradient 'dxmdxi'
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

def project_normal_vector_onto_tangent_space(to_project, e1, e2):
    """
    Project a normal vector on to the tangent space of a surface
    """
    res = inner(to_project, e1)*e1 + inner(to_project, e2)*e2
    return res

def penalty_displacement(alpha_d, u1m_hom, u2m_hom, 
                         line_Jacobian=None, dx_m=None):
    """
    Penalization of displacements on the non-matching 
    interface between two splines.
    
    Parameters
    ----------
    alpha_d : ufl.algebra.Division
    u1m_hom : dolfin Function
    u2m_hom : dolfin Function
    # dXdxi : dolfin ListTensor
    line_Jacobian : dolfin Function or None, optional
    dx_m : ufl Measure

    Return
    ------
    W_pd : ufl Form
    """
    if line_Jacobian is None:
        line_Jacobian = Constant(1.)
    if dx_m is None:
        dx_m = dx
    W_pd = 0.5*alpha_d*((u1m_hom-u2m_hom)**2)*line_Jacobian*dx_m
    return W_pd

def penalty_rotation(alpha_r, dX1dxi, dx1dxi, dX2dxi, dx2dxi, 
                     line_Jacobian=None, dx_m=None):
    """
    Penalization of rotation on the non-matching interface 
    between two splines.

    Parameters
    ----------
    alpha_r : ufl.algebra.Division
    u1m_hom : dolfin Function
    u2m_hom : dolfin Function
    dXdxi : dolfin ListTensor
    line_Jacobian : dolfin Function or None, optional
    dx_m : ufl Measure or None, optional

    Return
    ------
    W_pd : ufl Form
    """

    if line_Jacobian is None:
        line_Jacobian = Constant(1.)
    if dx_m is None:
        dx_m = dx

    # Orthonormal basis for patch 1
    e11, e21, e31 = interface_orthonormal_basis(dx1dxi)
    E11, E21, E31 = interface_orthonormal_basis(dX1dxi)

    # Orthonormal basis for patch 2
    e12, e22, e32 = interface_orthonormal_basis(dx2dxi)
    E12, E22, E32 = interface_orthonormal_basis(dX2dxi)

    pe = project_normal_vector_onto_tangent_space(e32, e11, e21)
    pE = project_normal_vector_onto_tangent_space(E32, E11, E21)

    W_pr = 0.5*alpha_r*((dot(e31, e32)-dot(E31, E32))**2 \
           + (sqrt(dot(pe, pe)+DOLFIN_EPS)-sqrt(dot(pE, pE)))**2) \
           *line_Jacobian*dx_m

    return W_pr

def penalty_energy(spline1, spline2, mortar_mesh, Vm_control, dVm_control, 
                   A1_control, A2_control, alpha_d, alpha_r, 
                   mortar_vars1, mortar_vars2, dx_m=None, metadata=None):
    """
    Penalization of displacement and rotation of non-matching interface 
    between two extracted splines.
    
    Parameters
    ----------
    spline1 : ExtractedSpline
    spline2 : ExtractedSpline
    Vm_control : dolfin FunctionSpace
    dVm_control : dolfin FunctionSpace
    A1_control : list of dolfin PETScMatrices
    A2_control : list of dolfin PETScMatrices
    alpha_d : ufl.algebra.Division
    alpha_r : ufl.algebra.Division
    mortar_vars1 : list of dolfin Functions
    mortar_vars2 : list of dolfin Functions
    dx_m : ufl Measure or None
    quadrature_degree : int, default is 2.

    Returns
    -------
    W_p : ufl Form
    """
    cpfuncs1, cpfuncs1_dxi1, cpfuncs1_dxi2 = transfer_cpfuns(spline1, 
                                             Vm_control, dVm_control, 
                                             A1_control)
    X1, dX1dxi = create_geometrical_mapping(spline1, cpfuncs1, cpfuncs1_dxi1, 
                                            cpfuncs1_dxi2)
    x1, dx1dxi = physical_configuration(cpfuncs1, cpfuncs1_dxi1, 
                                        cpfuncs1_dxi2, X1, dX1dxi, 
                                        mortar_vars1)

    cpfuncs2, cpfuncs2_dxi1, cpfuncs2_dxi2 = transfer_cpfuns(spline2, 
                                             Vm_control, dVm_control, 
                                             A2_control)
    X2, dX2dxi = create_geometrical_mapping(spline2, cpfuncs2, cpfuncs2_dxi1, 
                                            cpfuncs2_dxi2)
    x2, dx2dxi = physical_configuration(cpfuncs2, cpfuncs2_dxi1, 
                                        cpfuncs2_dxi2, X2, dX2dxi, 
                                        mortar_vars2)
    if dx_m is not None:
        dx_m = dx_m
    else:
        if metadata is not None:
            dx_m = dx(domain=mortar_mesh, metadata=metadata)
        else:
            dx_m = dx(domain=mortar_mesh, metadata={"quadrature_degree":2})

    line_Jacobian = compute_line_Jacobian(X2)
    if line_Jacobian == 0.:
        line_Jacobian = sqrt(tr(dX2dxi*dX2dxi.T))

    # Penalty of displacement
    W_pd = penalty_displacement(alpha_d, mortar_vars1[0], mortar_vars2[0], 
                                line_Jacobian, dx_m)
    # Penalty of rotation
    W_pr = penalty_rotation(alpha_r, dX1dxi, dx1dxi, dX2dxi, dx2dxi, 
                            line_Jacobian, dx_m)
    W_p = W_pd + W_pr
    return W_p

def SVK_residual(spline, u_hom, z_hom, E, nu, h, dWext):
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
    res = dWint - dWext
    return res

def hyperelastic_residual(spline, u_hom, z_hom, E, nu, h, dWext, quad_pts=4):
    """
    PDE residual for Kirchhoff--Love shell using incompressible 
    hyperelastic model.

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
    res: dolfin Form
    """
    mu = E/(2*(1+nu))
    X = spline.F
    x = X + spline.rationalize(u_hom)
    def psi_el(E_):
        # Neo-Hookean potential
        C = 2.0*E_ + Identity(3)
        I1 = tr(C)
        return 0.5*mu*(I1 - 3.0)
    dxi2 = throughThicknessMeasure(quad_pts, h)
    psi = incompressiblePotentialKL(spline, X, x, psi_el)
    Wint = psi*dxi2*spline.dx
    dWint = derivative(Wint, u_hom, z_hom)
    res = dWint - dWext
    return res

if __name__ == "__main__":
    pass