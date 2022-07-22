from PENGoLINS.nonmatching_coupling import *

def rotational_mat(alpha):
    """
    Rotational matrix, input ``alpha`` is in degree.
    """
    phi = Constant(np.deg2rad(alpha))
    T = as_matrix([[cos(phi)**2, sin(phi)**2, sin(phi)*cos(phi)],
                   [sin(phi)**2, cos(phi)**2, -sin(phi)*cos(phi)],
                   [-2*sin(phi)*cos(phi), 2*sin(phi)*cos(phi), 
                    cos(phi)**2-sin(phi)**2]])
    return T

def orthotropic_mat(E1, E2, nu12, nu21, G12):
    """
    Return Orthotropic material matrix.
    """
    mat_denom = 1-nu12*nu21
    D_ort = as_matrix([[E1/mat_denom, nu21*E1/mat_denom, 0],
                       [nu12*E2/mat_denom, E2/mat_denom, 0],
                       [0, 0, G12]])
    return D_ort

def laminate_ABD_mat(n_ply, h_th, D_ort, fiber_angles):
    """
    Compute laminated A,B and D material matrices.

    Parameters
    ----------
    n_ply : int, the number of ply
    h_th : float or ufl Constant, laminate total thickness
    D_ort : ufl matrix or list, unrotated orthotropic material matrix
    fiber_angles : list of float, fiber angles for all layers in degree

    Returns
    -------
    (A, B, D) : Extensional, extensional-bending coupling and 
                bending material matrices
    """
    assert n_ply == len(fiber_angles)
    if isinstance(D_ort, list):
        assert n_ply == len(D_ort)
    # Assume all layers have the same thickness
    ply_th = h_th/n_ply

    A = 0
    B = 0
    D = 0

    for i in range(n_ply):
        T_mat = rotational_mat(fiber_angles[i])
        if isinstance(D_ort, list):
            D_bar = T_mat.T*D_ort[i]*T_mat
        else:
            D_bar = T_mat.T*D_ort*T_mat
        # Distance from the centroid of the ``i``-th ply to the 
        # mid-plane of the laminate
        z = ply_th*(i+1 - n_ply/2. - 1/2.)
        A += D_bar*ply_th
        B += D_bar*ply_th*z
        D += D_bar*(ply_th*z**2. + ply_th**3/12.)
    return (A, B, D)


def surfaceEnergyDensitySVKLaminate(spline, X, x, h_th, A_mat, B_mat, D_mat, 
                                    membrane=False,
                                    membranePrestress=Constant(((0,0),(0,0))),
                                    bendingPrestress=Constant(((0,0),(0,0)))):
    """
    Elastic energy for laminated shell pre unit area. For details,
    see function ``surfaceEnergyDensitySVK`` 
    """
    A0,A1,A2,_,A,B = surfaceGeometry(spline,X)
    a0,a1,a2,_,a,b = surfaceGeometry(spline,x)
    epsilon = 0.5*(a - A)
    kappa = B - b
    epsilonBar = covariantRank2TensorToCartesian2D(epsilon,A,A0,A1)
    kappaBar = covariantRank2TensorToCartesian2D(kappa,A,A0,A1)

    nBar = A_mat*voigt2D(epsilonBar) + B_mat*voigt2D(kappaBar)
    mBar = B_mat*voigt2D(epsilonBar) + D_mat*voigt2D(kappaBar)
    
    # DO NOT add prestresses directly to nBar and mBar, then plug them into
    # the standard formula for energy.  The resulting prestress will be off
    # by a factor of two.
        
    Wint = inner(voigt2D(epsilonBar),0.5*nBar
                 + voigt2D(membranePrestress,strain=False))
    if(not membrane):
        Wint += inner(voigt2D(kappaBar),0.5*mBar
                      + voigt2D(bendingPrestress,strain=False))
    return Wint

def SVK_residual_laminate(spline, u_hom, z_hom, h, 
                          A_mat, B_mat, D_mat, dWext):
    """
    PDE residaul of Kirchhoff--Love laminated shell using the 
    St.Venant--Kirchhoff (SVK) material model.

    Parameters
    ----------
    spline : ExtractedSpline
    u_hom : dolfin Function
    z_hom : dolfin Argument
    h : dolfin Constant
    A_mat : extensional matrix
    B_mat : coupling matrix
    D_mat : bending matrix
    dWext : dolfin Form

    Returns
    -------
    res : dolfin Form
    """
    X = spline.F
    x = X + spline.rationalize(u_hom)
    z = spline.rationalize(z_hom)

    Wint = surfaceEnergyDensitySVKLaminate(spline, X, x, h,
                                      A_mat, B_mat, D_mat)*spline.dx
    dWint = derivative(Wint, u_hom, z_hom)
    res = dWint - dWext
    return res

class NonMatchingCouplingLaminate(NonMatchingCoupling):
    """
    Coupling of non-matching problem for laminated shells.
    """
    def __init__(self, splines, h_th, A_mat, B_mat, D_mat, 
                 num_field=3, contact=None, int_measure_metadata=None, 
                 comm=None):
        """
        Parameters
        ----------
        splines : list of ExtractedSpline
        h_th : ufl Constant or list of ufl Constant.
        A_mat : ufl matrix, extensional matrix
        B_mat : ufl matrix, extensional-bending coupling matrix
        D_mat : ufl matrix, bending matrix
        num_field : int, optional
            Number of field of the unknowns. Default is 3.
        contact : ShNAPr.contact.ShellContactContext, optional
        int_measure_metadata : dict, optional
        comm : mpi4py.MPI.Intracomm, optional, default is None.
        """

        super().__init__(splines, E=None, h_th=h_th, nu=None,
                         num_field=num_field, contact=contact,
                         int_measure_metadata=int_measure_metadata,
                         comm=comm)
        self._init_ABD_matrices(A_mat, B_mat, D_mat)

    def _init_ABD_matrices(self, A_mat, B_mat, D_mat):
        if isinstance(A_mat, list):
            self.A_mat = A_mat  # Extensional matrix
            if len(self.A_mat) != self.num_splines:
                if MPI.rank(self.comm) == 0:
                    raise AssertionError("Length of extensional matrix list"
                        " doesn't match with the number of splines.")
        else:
            self.A_mat = [A_mat for i in range(self.num_splines)]

        if isinstance(B_mat, list):
            self.B_mat = B_mat  # Coupling matrix
            if len(self.B_mat) != self.num_splines:
                if MPI.rank(self.comm) == 0:
                    raise AssertionError("Length of coupling matrix list"
                        " doesn't match with the number of splines.")
        else:
            self.B_mat = [B_mat for i in range(self.num_splines)]

        if isinstance(D_mat, list):
            self.D_mat = D_mat  # Bending matrix
            if len(self.D_mat) != self.num_splines:
                if MPI.rank(self.comm) == 0:
                    raise AssertionError("Length of bending matrix list"
                        " doesn't match with the number of splines.")
        else:
            self.D_mat = [D_mat for i in range(self.num_splines)]

    def max_matij(self, A):
        """
        Get maximum entry for a given ufl matrix ``A``.
        """
        mat_shape = A.ufl_shape
        max_val = A[0,0]
        for i in range(mat_shape[0]):
            for j in range(mat_shape[1]):
                max_val = max_value(A[i,j], max_val)
        return max_val

    def penalty_parameters(self, h_th=None, method='minimum'):
        """
        Create lists for pealty paramters of displacement and rotation
        for laminated shells.
        """
        self._init_properties(E=None, h_th=h_th, nu=None)

        self.alpha_d_list = []
        self.alpha_r_list = []

        for i in range(self.num_interfaces):
            s_ind0, s_ind1 = self.mapping_list[i]
            
            if self.h_th_is_function:
                A_x_b(self.transfer_matrices_thickness_list[i][0],
                      self.h_th[s_ind0].vector(), 
                      self.mortar_h_th[i][0].vector())
                A_x_b(self.transfer_matrices_thickness_list[i][1],
                      self.h_th[s_ind1].vector(), 
                      self.mortar_h_th[i][1].vector())
                h_th0 = self.mortar_h_th[i][0]
                h_th1 = self.mortar_h_th[i][1]
            else:
                h_th0 = self.h_th[s_ind0]
                h_th1 = self.h_th[s_ind1]

            # # Use symbolic max Aij in the penalty parameters will
            # # solw the code performance.
            # max_Aij0 = self.max_matij(self.A_mat[s_ind0])
            # max_Aij1 = self.max_matij(self.A_mat[s_ind1])
            # max_Dij0 = self.max_matij(self.D_mat[s_ind0])
            # max_Dij1 = self.max_matij(self.D_mat[s_ind1])

            max_Aij0_sym = self.max_matij(self.A_mat[s_ind0])
            max_Aij1_sym = self.max_matij(self.A_mat[s_ind1])
            max_Dij0_sym = self.max_matij(self.D_mat[s_ind0])
            max_Dij1_sym = self.max_matij(self.D_mat[s_ind1])
            max_Aij0 = project(max_Aij0_sym, self.Vms_control[i])
            max_Aij1 = project(max_Aij1_sym, self.Vms_control[i])
            max_Dij0 = project(max_Dij1_sym, self.dVms_control[i])
            max_Dij1 = project(max_Dij1_sym, self.dVms_control[i])

            if method == 'minimum':
                alpha_d = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*min_value(max_Aij0, max_Aij1)
                alpha_r = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*min_value(max_Dij0, max_Dij1)
            elif method == 'maximum':
                alpha_d = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*max_value(max_Aij0, max_Aij1)
                alpha_r = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*max_value(max_Dij0, max_Dij1)
            elif method == 'average':
                alpha_d = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*(max_Aij0+max_Aij1)*0.5
                alpha_r = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*(max_Dij0+max_Dij1)*0.5
            else:
                raise TypeError("Penalty method:", method, 
                                "is not supported.")
            self.alpha_d_list += [alpha_d,]
            self.alpha_r_list += [alpha_r,]