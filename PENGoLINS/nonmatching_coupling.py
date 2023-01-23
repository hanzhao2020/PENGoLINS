"""
The "nonmatcing_coupling" module
--------------------------------
contains class that sets up and solves the non-matching of 
coupling with multiple spline patches.
"""

from PENGoLINS.nonmatching_shell import *
from ufl import min_value
from ufl import max_value

class NonMatchingCoupling(object):
    """
    Class sets up the system of coupling of non-matching with 
    multiple spline patches.
    """
    def __init__(self, splines, E, h_th, nu, num_field=3, 
                 int_V_family='CG', int_V_degree=1,
                 int_dx_metadata=None, contact=None, comm=None):
        """
        Pass the list of splines and number of element for 
        each spline and other parameters to initialize the 
        coupling of non-matching problem.
        
        Parameters
        ----------
        splines : list of ExtractedSplines
        E : ufl Constant or list, Young's modulus
        h_th : ufl Constant or list, thickness of the splines
        nu : ufl Constant or list, Poisson's ratio
        num_field : int, optional
            Number of field of the unknowns. Default is 3.
        int_V_family : str, optional, element family for 
            mortar meshes. Default is 'CG'.
        int_V_degree : int, optional, default is 1.
        int_dx_metadata : dict, optional
            Metadata information for integration measure of 
            intersection curves. Default is vertex quadrature
            with degree 0.
        contact : ShNAPr.contact.ShellContactContext, optional
        comm : mpi4py.MPI.Intracomm, optional, default is None.
        """
        self.splines = splines
        self.num_splines = len(splines)
        self.num_field = num_field
        self.para_dim = splines[0].mesh.geometric_dimension()
        self._init_properties(E, h_th, nu)

        self.int_V_family = int_V_family
        self.int_V_degree = int_V_degree

        self.spline_funcs = [Function(spline.V) for spline in self.splines]
        self.spline_test_funcs = [TestFunction(spline.V) 
                                  for spline in self.splines]

        if comm is None:
            self.comm = self.splines[0].comm
        else:
            self.comm = comm
        
        if int_dx_metadata is None:
            self.int_dx_metadata = {'quadrature_degree': 0, 
                                         'quadrature_scheme': 'vertex'}
            # self.int_dx_metadata = {"quadrature_degree":2}
        else:
            self.int_dx_metadata = int_dx_metadata

        self.contact = contact
        self.residuals = None
        self.residuals_deriv = None
        self.point_sources = None
        self.point_source_inds = None

    def global_zero_dofs(self):
        """
        Returns the global zero DoFs for the non-matching system.

        Returns
        -------
        zero_dofs: PETSc IS
        """
        zero_dofs_list = []
        ind_off = 0
        for i in range(self.num_splines):
            zero_dofs_list += [self.splines[i].zeroDofs.getIndices()+ind_off,]
            ind_off += self.splines[i].M.size(1)
        zero_dofs_array = np.concatenate(zero_dofs_list, 
                                         axis=0, dtype='int32')
        zero_dofs = PETSc.IS(self.comm)
        zero_dofs.createGeneral(zero_dofs_array)
        return zero_dofs

    def _init_properties(self, E=None, h_th=None, nu=None):
        """
        Initialize geometric and material properties. For internal use.

        Parameters
        ----------
        E : ufl Constant or list, Young's modulus, default is None
        h_th : ufl Constant or list, thickness of the splines,
            default is None
        nu : ufl Constant or list, Poisson's ratio, default is None
        """
        if E is not None:
            if isinstance(E, list):
                self.E = E  # Young's modulus
                if len(self.E) != self.num_splines:
                    if MPI.rank(self.comm) == 0:
                        raise AssertionError("Length of Young's modulus list"
                            " doesn't match with the number of splines.")
            else:
                self.E = [E for i in range(self.num_splines)]

        if h_th is not None:
            if isinstance(h_th, list):
                self.h_th = h_th  # Thickness of the spline surfaces
                if len(self.h_th) != self.num_splines:
                    if MPI.rank(self.comm) == 0:
                        raise AssertionError("Length of shell thickness list"
                            " doesn't match with the number of splines.")
            else:
                self.h_th = [h_th for i in range(self.num_splines)]

        if nu is not None:
            if isinstance(nu, list):
                self.nu = nu  # Poisson's ratio
                if len(self.nu) != self.num_splines:
                    if MPI.rank(self.comm) == 0:
                        raise AssertionError("Length of Poisson's ratio list"
                            " doesn't match with the number of splines.")
            else:
                self.nu = [nu for i in range(self.num_splines)]

    def create_mortar_meshes(self, mortar_nels, mortar_coords=None):
        """
        Create mortar meshes for non-matching with multiple patches.

        Parameters
        ----------
        mortar_nels : list of ints
            Contains number of elements for all mortar meshes.
        mortar_coords : list of ndarrays, optional
            Contains points of location for all mortar meshes.
            Default is None, corresponds to coordinate 
            [[0,0],[0,1]].
        """
        self.num_intersections = len(mortar_nels)
        if mortar_coords is None:
            mortar_coords = [np.array([[0.,0.],[0.,1.]]),]\
                            *self.num_intersections
        self.mortar_meshes = [generate_mortar_mesh(mortar_coords[i], 
                              mortar_nels[i], comm=self.comm) 
                              for i in range(self.num_intersections)]

    def _create_mortar_func_spaces(self):
        """
        Vms are vector function spaces with dimension of ``num_field``
        Vms_control are scalar function spaces
        dVms are vector function space with dimension of  
        ``num_field``*``para_dim``.
        dVms_control are vector function spaces with dimension of 
        ``para_dim``
        """
        self.Vms = []
        self.Vms_control = []
        self.dVms = []
        self.dVms_control = []

        for mortar_mesh in self.mortar_meshes:
            self.Vms += [VectorFunctionSpace(mortar_mesh, self.int_V_family, 
                         self.int_V_degree, dim=self.num_field)]
            self.Vms_control += [FunctionSpace(mortar_mesh, self.int_V_family, 
                         self.int_V_degree)]
            self.dVms += [VectorFunctionSpace(mortar_mesh, self.int_V_family, 
                         self.int_V_degree, dim=self.num_field*self.para_dim)]
            self.dVms_control += [VectorFunctionSpace(mortar_mesh, self.int_V_family, 
                         self.int_V_degree, dim=self.para_dim)]

    def _create_mortar_funcs(self):
        """
        mortar_funcs are mortar meshes' displacements and displacement 
        derivatives on two sides.
        mortar_cpfuncs are mortar meshes' control point functions
        (four components) on two sides.
        """
        # For one element in ``mortar_funcs``:
        # [[um0, dum0], [um1, dum1]]
        self.mortar_funcs = [[] for i in range(self.num_intersections)]
        # For one element in ``mortar_cpfuncs``:
        # [[[cp00, cp01, cp02, cp03], [dcp00, dcp01, dcp02, dcp03]], 
        #  [[cp10, cp11, cp12, cp13], [dcp10, dcp11, dcp12, dcp13]]]
        self.mortar_cpfuncs = [[] for i in range(self.num_intersections)]
        for mortar_ind in range(self.num_intersections):
            for side in range(2):
                self.mortar_funcs[mortar_ind] += \
                    [[Function(self.Vms[mortar_ind]),
                      Function(self.dVms[mortar_ind])]]
                self.mortar_cpfuncs[mortar_ind] += [[[],[]]]
                for field in range(self.num_field+1):
                    self.mortar_cpfuncs[mortar_ind][side][0] += \
                        [Function(self.Vms_control[mortar_ind])]
                    self.mortar_cpfuncs[mortar_ind][side][1] += \
                        [Function(self.dVms_control[mortar_ind])]

    def mortar_meshes_setup(self, mapping_list, mortar_parametric_coords, 
                            penalty_coefficient=1000, 
                            penalty_method="minimum"):
        """
        Set up coupling of non-matching system for mortar meshes.

        Parameters
        ----------
        mapping_list : list of ints
        mortar_parametric_coords : list of ndarrays
        penalty_coefficient : float, optional, default is 1000
        penalty_method : str, {'minimum', 'maximum', 'average'}
        """
        assert self.num_intersections == len(mapping_list)
        self._create_mortar_func_spaces()
        self._create_mortar_funcs()

        self.mortar_parametric_coords = mortar_parametric_coords
        self.mapping_list = mapping_list
        self.penalty_coefficient = penalty_coefficient
        self.transfer_matrices_list = []
        self.transfer_matrices_control_list = []
        self.transfer_matrices_linear_list = []
        self.h0m_list = []
        self.h1m_list = []
        self.hm_avg_list = []

        # Create transfer matrices for displacements and cpfuncs
        for i in range(self.num_intersections):
            transfer_matrices = [None, None]
            transfer_matrices_control = [None, None]
            transfer_matrices_linear = [None, None]
            for j in range(len(self.mapping_list[i])):
                move_mortar_mesh(self.mortar_meshes[i], 
                                 mortar_parametric_coords[i][j])
                # Create transfer matrices
                transfer_matrices[j] = create_transfer_matrix_list(
                    self.splines[self.mapping_list[i][j]].V, self.Vms[i], 1)
                transfer_matrices_control[j] = create_transfer_matrix_list(
                    self.splines[self.mapping_list[i][j]].V_control, 
                    self.Vms_control[i], 1)
                transfer_matrices_linear[j] = create_transfer_matrix(
                    self.splines[self.mapping_list[i][j]].V_linear,
                    self.Vms_control[i])
            move_mortar_mesh(self.mortar_meshes[i], 
                             mortar_parametric_coords[i][0])

            # Store transfer matrices in lists for future use
            self.transfer_matrices_list += [transfer_matrices,]
            self.transfer_matrices_control_list += [transfer_matrices_control]
            self.transfer_matrices_linear_list += [transfer_matrices_linear,]

            # Compute element length
            s_ind0, s_ind1 = self.mapping_list[i]
            h0 = spline_mesh_size(self.splines[s_ind0])
            h0_func = self.splines[s_ind0]\
                .projectScalarOntoLinears(h0, lumpMass=True)
            h0m = Function(self.Vms_control[i])
            A_x_b(transfer_matrices_linear[0], h0_func.vector(), h0m.vector())
            h1 = spline_mesh_size(self.splines[s_ind1])
            h1_func = self.splines[s_ind1]\
                .projectScalarOntoLinears(h1, lumpMass=True)
            h1m = Function(self.Vms_control[i])
            A_x_b(transfer_matrices_linear[1], h1_func.vector(), h1m.vector())
            hm_avg = 0.5*(h0m+h1m)
            self.h0m_list += [h0m]
            self.h1m_list += [h1m]
            self.hm_avg_list += [hm_avg,]

        self.penalty_parameters(method=penalty_method)
        self.mortar_mesh_symexp()

    def penalty_parameters(self, E=None, h_th=None, nu=None, 
                           method='minimum'):
        """
        Create lists for pealty paramters for displacement and rotation.

        Parameters
        ----------
        E : ufl Constant or list, Young's modulus
        h_th : ufl Constant or list, thickness of the splines
        nu : ufl Constant or list, Poisson's ratio
        method: str, {'minimum', 'maximum', 'average'}
        """
        self._init_properties(E, h_th, nu)

        self.alpha_d_list = []
        self.alpha_r_list = []

        for i in range(self.num_intersections):
            s_ind0, s_ind1 = self.mapping_list[i]

            # # Original implementation
            # # Use "Minimum" method for spline patches with different
            # # material properties. 
            # # For other methods, see Herrema et al. Section 4.2
            # # For uniform isotropic material:
            # max_Aij0 = float(self.E[s_ind0]*self.h_th[s_ind0]\
            #            /(1-self.nu[s_ind0]**2))
            # max_Aij1 = float(self.E[s_ind1]*self.h_th[s_ind1]\
            #            /(1-self.nu[s_ind1]**2))
            # alpha_d = Constant(self.penalty_coefficient)\
            #           /self.hm_avg_list[i]*min(max_Aij0, max_Aij1)
            # max_Dij0 = float(self.E[s_ind0]*self.h_th[s_ind0]**3\
            #            /(12*(1-self.nu[s_ind0]**2)))
            # max_Dij1 = float(self.E[s_ind1]*self.h_th[s_ind1]**3\
            #            /(12*(1-self.nu[s_ind1]**2)))
            # alpha_r = Constant(self.penalty_coefficient)\
            #           /self.hm_avg_list[i]*min(max_Dij0, max_Dij1)
            # self.alpha_d_list += [alpha_d,]
            # self.alpha_r_list += [alpha_r,]

            h_th0 = self.h_th[s_ind0]
            h_th1 = self.h_th[s_ind1]

            max_Aij0 = self.E[s_ind0]*h_th0\
                       /(1-self.nu[s_ind0]**2)
            max_Aij1 = self.E[s_ind1]*h_th1\
                       /(1-self.nu[s_ind1]**2)
            max_Dij0 = self.E[s_ind0]*h_th0**3\
                       /(12*(1-self.nu[s_ind0]**2))
            max_Dij1 = self.E[s_ind1]*h_th1**3\
                       /(12*(1-self.nu[s_ind1]**2))

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

    def mortar_mesh_symexp(self):
        """
        Compute RHS non-matching residuals and LHS non-matching derivatives
        in dolfin Forms
        """
        # list for UFL forms
        self.Rm_symexp_list = [None for i in range(self.num_intersections)]
        # self.dRm_dum_symexp_list = [None for i in 
        #                           range(self.num_intersections)]
        # list for Dolfin forms
        self.Rm_list = [None for i in range(self.num_intersections)]
        self.dRm_dum_list = [None for i in range(self.num_intersections)]
        dx_m = dx(metadata=self.int_dx_metadata)
        for i in range(self.num_intersections):
            s_ind0, s_ind1 = self.mapping_list[i]
            self.PE = penalty_energy(self.splines[s_ind0], 
                self.splines[s_ind1], self.spline_funcs[s_ind0], 
                self.spline_funcs[s_ind1], self.mortar_meshes[i], 
                self.mortar_funcs[i], self.mortar_cpfuncs[i], 
                self.transfer_matrices_list[i],
                self.transfer_matrices_control_list[i],
                self.alpha_d_list[i], self.alpha_r_list[i], 
                dx_m=dx_m)

            # An initial check for penalty energy, if ``PE``is nan,
            # raise RuntimeError.
            PE_value = assemble(self.PE)
            if PE_value is nan:
                if MPI.rank(self.comm) == 0:
                    raise RuntimeError("Penalty energy value is nan between "
                          "splines {:2d} and {:2d}.".format(
                          self.mapping_list[i][0], self.mapping_list[i][1]))

            Rm_temp = penalty_residual(self.PE, self.mortar_funcs[i])
            dRm_dum_temp = penalty_residual_deriv(Rm_temp, 
                                                  self.mortar_funcs[i])

            Rm_temp_to_assemble = [[Form(Rm_ij) for Rm_ij in Rm_i] 
                                    for Rm_i in Rm_temp]
            dRm_dum_temp_to_assemble = [[[[Form(dRm_dum_ijkl) 
                                           if dRm_dum_ijkl is not None 
                                           else None 
                                           for dRm_dum_ijkl in dRm_dum_ijk]
                                           for dRm_dum_ijk in dRm_dum_ij]
                                           for dRm_dum_ij in dRm_dum_i]
                                           for dRm_dum_i in dRm_dum_temp]
            self.Rm_symexp_list[i] = Rm_temp
            # self.dRm_dum_symexp_list[i] = dRm_dum_temp
            self.Rm_list[i] = Rm_temp_to_assemble
            self.dRm_dum_list[i] = dRm_dum_temp_to_assemble

    def set_residuals(self, residuals, residuals_deriv=None):
        """
        Specify the shell residuals.

        Parameters
        ----------
        residuals : list of ufl forms
        residuals_deriv : list of ufl forms or None, default is None
        """
        if residuals_deriv is None:
            residuals_deriv = [derivative(residuals[i], self.spline_funcs[i]) 
                                          for i in range(self.num_splines)]

        # self.residuals = residuals
        # self.residuals_deriv = residuals_deriv

        self.residuals = [Form(res) for res in residuals]
        self.residuals_deriv = [Form(res_deriv) 
                                for res_deriv in residuals_deriv]

    def set_point_sources(self, point_sources, point_source_inds):
        """
        Specify point load for shell patches.

        Parameters
        ----------
        point_sources : list of dolfin PointSources
        point_source_inds : list of ints, spline indices where
            point sources are applied
        """
        self.point_sources = point_sources
        self.point_source_inds = point_source_inds

    def assemble_nonmatching(self, assemble_nonmatching_LHS=True):
        """
        Assemble the non-matching system.
        """
        ## Step 1: assemble residuals of ExtractedSplines 
        # and derivatives.
        if self.residuals is None:
            if MPI.rank(self.comm) == 0:
                raise RuntimeError("Shell residuals are not specified.") 
        if self.residuals_deriv is None:
            if MPI.rank(self.comm) == 0:
                raise RuntimeError("Derivatives of shell residuals are "
                                   "not specified.")

        # Compute contributions from shell residuals and derivatives
        R_FE = []
        dR_du_FE = []
        for i in range(self.num_splines):
            R_assemble = assemble(self.residuals[i])
            dR_du_assemble = assemble(self.residuals_deriv[i])
            if self.point_sources is not None:
                for j, ps_ind in enumerate(self.point_source_inds):
                    if ps_ind == i:
                        R_assemble_temp = R_assemble.copy()
                        R_assemble_temp.zero()
                        self.point_sources[j].apply(R_assemble_temp)
                        R_assemble = R_assemble - R_assemble_temp
            R_FE += [v2p(R_assemble),]
            dR_du_FE += [m2p(dR_du_assemble),]

        ## Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        Rm_FE = [None for i1 in range(self.num_splines)]
        if assemble_nonmatching_LHS:
            dRm_dum_FE = [[None for i1 in range(self.num_splines)] 
                                for i2 in range(self.num_splines)]

        # Compute non-matching contributions ``Rm_FE`` and 
        # ``dRm_dum_FE``.
        for i in range(self.num_intersections):
            Rm = transfer_penalty_residual(self.Rm_list[i], 
                      self.transfer_matrices_list[i])
            dRm_dum = transfer_penalty_residual_deriv(
                           self.dRm_dum_list[i],  
                           self.transfer_matrices_list[i])

            for j in range(len(Rm)):
                if Rm_FE[self.mapping_list[i][j]] is not None:
                    Rm_FE[self.mapping_list[i][j]] += Rm[j]
                else:
                    Rm_FE[self.mapping_list[i][j]] = Rm[j]
                if assemble_nonmatching_LHS:
                    for k in range(j, len(dRm_dum[j])):
                        if dRm_dum_FE[self.mapping_list[i][j]]\
                           [self.mapping_list[i][k]] is not None:
                            dRm_dum_FE[self.mapping_list[i][j]]\
                                [self.mapping_list[i][k]] += dRm_dum[j][k]
                        else:
                            dRm_dum_FE[self.mapping_list[i][j]]\
                                [self.mapping_list[i][k]] = dRm_dum[j][k]

        # Filling lower triangle blocks of non-matching derivatives
        for i in range(self.num_splines-1):
            for j in range(i+1, self.num_splines):
                if dRm_dum_FE[i][j] is not None:
                    dRm_dum_temp = dRm_dum_FE[i][j].copy()
                    dRm_dum_temp.transpose()
                    dRm_dum_FE[j][i] = dRm_dum_temp

        ## Step 3: add spline residuals and non-matching 
        # contribution together
        Rt_FE = [None for i1 in range(self.num_splines)]
        dRt_dut_FE = [[None for i1 in range(self.num_splines)] 
                           for i2 in range(self.num_splines)]
        for i in range(self.num_splines):
            for j in range(self.num_splines):
                if i == j:
                    if Rm_FE[i] is not None:
                        Rt_FE[i] = R_FE[i] + Rm_FE[i]
                        dRt_dut_FE[i][i] = dR_du_FE[i] + dRm_dum_FE[i][i]
                    else:
                        Rt_FE[i] = R_FE[i]
                        dRt_dut_FE[i][i] = dR_du_FE[i]
                else:
                    dRt_dut_FE[i][j] = dRm_dum_FE[i][j]

        ## Step 4: add contact contributions if contact is given
        if self.contact is not None:
            Kcs, Fcs = self.contact.assembleContact(self.spline_funcs, 
                                                    output_PETSc=True)
            for i in range(self.num_splines):
                for j in range(self.num_splines):
                    if i == j:
                        if Fcs[i] is not None:
                            Rt_FE[i] += Fcs[i]
                            dRt_dut_FE[i][i] += Kcs[i][i]
                    else:
                        if dRt_dut_FE[i][j] is not None:
                            if Kcs[i][j] is not None:
                                dRt_dut_FE[i][j] += Kcs[i][j]
                        else:
                            if Kcs[i][j] is not None:
                                dRt_dut_FE[i][j] = Kcs[i][j]

        return dRt_dut_FE, Rt_FE

    def extract_nonmatching_system(self, Rt_FE, dRt_dut_FE):
        """
        Extract matrix and vector to IGA space.

        Parameters
        ----------
        Rt_FE : list of assembled vectors
        dRt_dut_FE : list of assembled matrices

        Returns
        -------
        b : petsc4py.PETSc.Vec
            LHS of non-matching system
        A : petsc4py.PETSc.Mat
            RHS of non-matching system
        """
        Rt_IGA = []
        dRt_dut_IGA = []
        for i in range(self.num_splines):
            Rt_IGA += [v2p(FE2IGA(self.splines[i], Rt_FE[i], True)),]
            # Rt_IGA += [AT_x(self.splines[i].M, Rt_FE[i]),]
            dRt_dut_IGA += [[],]
            for j in range(self.num_splines):
                if dRt_dut_FE[i][j] is not None:
                    dRm_dum_IGA_temp = AT_R_B(m2p(self.splines[i].M), 
                                  dRt_dut_FE[i][j], m2p(self.splines[j].M))
                    if i==j:
                        dRm_dum_IGA_temp = apply_bcs_mat(self.splines[i], 
                                           dRm_dum_IGA_temp, diag=1)
                    else:
                        dRm_dum_IGA_temp = apply_bcs_mat(self.splines[i], 
                                           dRm_dum_IGA_temp, self.splines[j], 
                                           diag=0)
                else:
                    dRm_dum_IGA_temp = None

                dRt_dut_IGA[i] += [dRm_dum_IGA_temp,]

        self.A_list = dRt_dut_IGA
        self.b_list = Rt_IGA

        self.b = create_nest_PETScVec(Rt_IGA, comm=self.comm)
        self.A = create_nest_PETScMat(dRt_dut_IGA, comm=self.comm)

        return self.A, self.b

    def update_mortar_funcs(self):
        """
        Update values in ``motar_funcs`` from ``spline_funcs``.
        """
        for i in range(len(self.transfer_matrices_list)):
            for j in range(len(self.transfer_matrices_list[i])):
                for k in range(len(self.transfer_matrices_list[i][j])):
                    A_x_b(self.transfer_matrices_list[i][j][k], 
                          self.spline_funcs[self.mapping_list[i][j]].\
                          vector(), self.mortar_funcs[i][j][k].vector())

    def solve_linear_nonmatching_problem(self, solver="direct", 
                                ksp_type=PETSc.KSP.Type.CG, 
                                pc_type=PETSc.PC.Type.FIELDSPLIT, 
                                fieldsplit_type="additive",
                                fieldsplit_ksp_type=PETSc.KSP.Type.PREONLY,
                                fieldsplit_pc_type=PETSc.PC.Type.LU, 
                                rtol=1e-15, max_it=100000, ksp_view=False, 
                                monitor_residual=False):
        """
        Solve the linear non-matching system.

        Parameters
        ----------
        solver : {'ksp', 'direct'}, or user defined solver. 
            For 'ksp', the non-matching system will be solved by 
            petsc4py PETSc KSP solver of type 'cg' with preconditioner
            'fieldsplit' of type 'additive'. Default is 'direct'.
        ksp_type : str, default is "cg"
            KSP solver type, for additional type, see PETSc.KSP.Type
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

        Returns
        -------
        self.spline_funcs : list of dolfin functions
        """
        for i in range(self.num_splines):
            self.spline_funcs[i].vector().zero()

        dRt_dut_FE, Rt_FE = self.assemble_nonmatching()
        self.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

        if solver == "direct":
            # In parallel, create a new aij matrix that have the 
            # same entries with original nest matrix to solve
            # it using dolfin direct solver.
            if MPI.size(self.comm) == 1:
                self.A.convert("seqaij")
            else:
                self.A = create_aijmat_from_nestmat(self.A, self.A_list, 
                                                    comm=self.comm)

        if solver == "ksp" and pc_type != PETSc.PC.Type.FIELDSPLIT:
            # Only use "fieldsplit" preconditioner for nest matrix
            self.A = create_aijmat_from_nestmat(self.A, self.A_list, 
                                                comm=self.comm)

        self.u_list = []
        for i in range(self.num_splines):
            self.u_list += [zero_petsc_vec(self.splines[i].M.size(1), 
                                      comm=self.splines[i].comm),]
        self.u = create_nest_PETScVec(self.u_list, comm=self.comm)
        solve_nonmatching_mat(self.A, self.u, -self.b, solver=solver, 
                              ksp_type=ksp_type, pc_type=pc_type, 
                              fieldsplit_type=fieldsplit_type,
                              fieldsplit_ksp_type=fieldsplit_ksp_type,
                              fieldsplit_pc_type=fieldsplit_pc_type, 
                              rtol=rtol, max_it=max_it, ksp_view=ksp_view, 
                              monitor_residual=monitor_residual)
        
        for i in range(self.num_splines):
            self.splines[i].M.mat().mult(self.u_list[i], 
                                         self.spline_funcs[i].vector().vec())
            v2p(self.spline_funcs[i].vector()).ghostUpdate()
            v2p(self.spline_funcs[i].vector()).assemble()
        return self.spline_funcs

    def solve_nonlinear_nonmatching_problem(self, solver="direct", 
                                ref_error=None, rtol=1e-3, max_it=20,
                                zero_mortar_funcs=True, 
                                ksp_type=PETSc.KSP.Type.CG, 
                                pc_type=PETSc.PC.Type.FIELDSPLIT, 
                                fieldsplit_type="additive",
                                fieldsplit_ksp_type=PETSc.KSP.Type.PREONLY,
                                fieldsplit_pc_type=PETSc.PC.Type.LU, 
                                ksp_rtol=1e-15, ksp_max_it=100000,
                                ksp_view=False, ksp_monitor_residual=False, 
                                iga_dofs=False, modified_Newton=False,
                                LHS_nm_assemble_times=1):
        """
        Solve the nonlinear non-matching system using Newton's method.

        Parameters
        ----------
        solver : {"ksp", "direct"} or user defined solver
            The linear solver inside Newton's iteration, default is "direct".
        ref_error : float, optional, default is None
        rtol : float, optional, default is 1e-3
            Relative tolerance for Newton's iteration
        max_it: int, optional, default is 20
            Maximum iteration for Newton's iteration
        zero_mortar_funcs : bool, optional, default is True
            Set dolfin functions on mortar meshes to zero before the 
            start of Newton's iteration, (helpful for convergence).
        ksp_type : str, default is "cg"
            KSP solver type, for additional type, see PETSc.KSP.Type
        pc_type : str, default is "fieldsplit"
            PETSc preconditioner type, for additional preconditioner 
            type, see PETSc.PC.Type
        fieldsplit_type : str, default is "additive"
            Only needed if preconditioner is "fieldsplit". {"additive", 
            "multiplicative", "symmetric_multiplicative", "schur"}
        fieldsplit_ksp_type : str, default is "cg"
        fieldsplit_pc_type : str, default is "lu"
        ksp_rtol : float, default is 1e-15
            Relative tolerance for PETSc KSP solver
        ksp_max_it : int, default is 100000
            Maximum iteration for PETSc KSP solver
        ksp_view : bool, default is False
        ksp_monitor_residual : bool, default is False
        iga_dofs : bool, default is False
            If True, return nonlinear solution in IGA DoFs
        modified_Newton : bool, default is False
            If True, assemble the LHS non-matching contribution 
            ``LHS_nm_assemble_times`` times for each nonlinear
            solve.
        LHS_nm_assemble_times : int, default is 1

        Returns
        -------
        self.spline_funcs : list of dolfin functions
        """

        # Zero out values in mortar mesh functions if True
        if zero_mortar_funcs:
            for i in range(len(self.transfer_matrices_list)):
                for j in range(len(self.transfer_matrices_list[i])):
                    for k in range(len(self.transfer_matrices_list[i][j])):
                            self.mortar_funcs[i][j][k].interpolate(Constant(
                                (0.,)*len(self.mortar_funcs[i][j][k])))

        # If iga_dofs is True, only starts from zero displacements,
        # this argument is designed for solving nonlinear 
        # displacements in IGA DoFs in optimization problem.
        if iga_dofs:
            u_iga_list = []
            for i in range(self.num_splines):
                u_FE_temp = Function(self.splines[i].V)
                u_iga_list += [v2p(multTranspose(self.splines[i].M,
                                   u_FE_temp.vector())),]
                self.spline_funcs[i].interpolate(Constant((0.,0.,0.)))
            u_iga = create_nest_PETScVec(u_iga_list, comm=self.comm)
                
        for newton_iter in range(max_it+1):
            if modified_Newton:
                if newton_iter < LHS_nm_assemble_times:
                    assemble_nonmatching_LHS = True
                else:
                    assemble_nonmatching_LHS = False
            else:
                assemble_nonmatching_LHS = True
            dRt_dut_FE, Rt_FE = self.assemble_nonmatching(
                                assemble_nonmatching_LHS)
            self.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

            if solver == "direct":
                if MPI.size(self.comm) == 1:
                    self.A.convert("seqaij")
                else:
                    self.A = create_aijmat_from_nestmat(self.A, self.A_list, 
                                                        comm=self.comm)

            if solver == "ksp" and pc_type != PETSc.PC.Type.FIELDSPLIT:
                self.A = create_aijmat_from_nestmat(self.A, self.A_list, 
                                                    comm=self.comm)

            current_norm = self.b.norm()

            if newton_iter==0 and ref_error is None:
                ref_error = current_norm

            rel_norm = current_norm/ref_error
            if newton_iter >= 0:
                if MPI.rank(self.comm) == 0:
                    print("Solver iteration: {}, relative norm: {:.12}."
                          .format(newton_iter, rel_norm))
                sys.stdout.flush()

            if rel_norm < rtol:
                if MPI.rank(self.comm) == 0:
                    print("Newton's iteration finished in {} "
                          "iterations (relative tolerance: {})."
                          .format(newton_iter, rtol))
                break

            if newton_iter == max_it:
                if MPI.rank(self.comm) == 0:
                    raise StopIteration("Nonlinear solver failed to "
                          "converge in {} iterations.".format(max_it))

            du_list = []
            du_IGA_list = []
            for i in range(self.num_splines):
                du_list += [Function(self.splines[i].V),]
                du_IGA_list += [zero_petsc_vec(self.splines[i].M.size(1), 
                                               comm=self.splines[i].comm)]
            du = create_nest_PETScVec(du_IGA_list, comm=self.comm)

            solve_nonmatching_mat(self.A, du, -self.b, solver=solver,
                                  ksp_type=ksp_type, pc_type=pc_type, 
                                  fieldsplit_type=fieldsplit_type,
                                  fieldsplit_ksp_type=fieldsplit_ksp_type,
                                  fieldsplit_pc_type=fieldsplit_pc_type, 
                                  rtol=ksp_rtol, max_it=ksp_max_it, 
                                  ksp_view=ksp_view, 
                                  monitor_residual=ksp_monitor_residual)

            if iga_dofs:
                u_iga += du

            for i in range(self.num_splines):
                self.splines[i].M.mat().mult(du_IGA_list[i], 
                                             du_list[i].vector().vec())
                self.spline_funcs[i].assign(self.spline_funcs[i]+du_list[i])
                v2p(du_list[i].vector()).ghostUpdate()

            self.update_mortar_funcs()
            # for i in range(len(self.transfer_matrices_list)):
            #     for j in range(len(self.transfer_matrices_list[i])):
            #         for k in range(len(self.transfer_matrices_list[i][j])):
            #             A_x_b(self.transfer_matrices_list[i][j][k], 
            #                   self.spline_funcs[self.mapping_list[i][j]].\
            #                   vector(), self.mortar_funcs[i][j][k].vector())

        if iga_dofs:
            return self.spline_funcs, u_iga
        else:
            return self.spline_funcs


class NonMatchingNonlinearProblem(NonlinearProblem):
    """
    A customized subclass of dolfin ``NonlinearProlem`` to make use of 
    ``PETScSNESSolver``. Note: Experimental function, solver can
    only converge in serial.
    """
    def __init__(self, problem, **kwargs):
        """
        Initialization of ``NonMatchingNonlinearProblem``.

        Parameters
        ----------
        problem : instance of ``NonMatchingCoupling``
        """
        super(NonMatchingNonlinearProblem, self).__init__(**kwargs)
        self.problem = problem
        self.u_list = []
        for i in range(self.problem.num_splines):
            self.u_list += [zero_petsc_vec(self.problem.splines[i].M.size(1), 
                                         comm=self.problem.splines[i].comm),]
        self.u = create_nest_PETScVec(self.u_list, comm=self.problem.comm)

    def form(self, A, P, b, x):
        """
        Update solution.
        """
        dRt_dut_FE, Rt_FE = self.problem.assemble_nonmatching()
        self.A, self.b = self.problem.extract_nonmatching_system(
                         Rt_FE, dRt_dut_FE)

        if MPI.size(self.problem.comm) == 1:
            self.A.convert("seqaij")
        else:
            self.A = create_aijmat_from_nestmat(self.A, self.problem.A_list, 
                                                comm=self.problem.comm)
        x.vec().copy(result=self.u)

        for i in range(self.problem.num_splines):
            v2p(self.problem.spline_funcs[i].vector()).ghostUpdate()
            self.problem.splines[i].M.mat().mult(self.u_list[i], 
                self.problem.spline_funcs[i].vector().vec())

        # Update mortar mesh dolfin Functions
        for i in range(len(self.problem.transfer_matrices_list)):
            for j in range(len(self.problem.transfer_matrices_list[i])):
                for k in range(len(self.problem.transfer_matrices_list[i][j])):
                    A_x_b(self.problem.transfer_matrices_list[i][j][k], 
                          self.problem.spline_funcs[
                          self.problem.mapping_list[i][j]].vector(), 
                          self.problem.mortar_funcs[i][j][k].vector())

    def F(self, b, x):
        """
        Update residual.
        """
        b.vec().setSizes(self.b.getSizes())
        b.vec().setUp()
        b.vec().assemble()
        self.b.copy(result=b.vec())
        return b

    def J(self, A, x):
        """
        Update Jacobian.
        """
        A.mat().setSizes(self.A.getSizes())
        A.mat().setUp()
        A.mat().assemble()
        self.A.copy(result=A.mat())
        return A

class NonMatchingNonlinearSolver:
    """
    Nonlinear solver for the non-matching problem.
    """
    def __init__(self, nonlinear_problem, solver):
        """
        Initialization of ``NonMatchingNonlinearSolver``.

        Parameters
        ----------
        nonlinear_problem : instance of ``NonMatchingNonlinearProblem``
        solver : PETSc SNES Solver
        """
        self.nonlinear_problem = nonlinear_problem
        self.solver = solver 

    def solve(self):
        """
        Solve the non-matching nonlinear problem.
        """
        temp_vec = PETScVector(self.nonlinear_problem.u.copy())
        self.solver.solve(self.nonlinear_problem, temp_vec)
        temp_vec.vec().copy(result=self.nonlinear_problem.u)


if __name__ == "__main__":
    pass