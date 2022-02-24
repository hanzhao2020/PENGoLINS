"""
The "nonmatcing_coupling" module
--------------------------------
contains class that sets up and solves the non-matching of 
coupling with multiple spline patches.
"""

from PENGoLINS.nonmatching_shell import *

class NonMatchingCoupling(object):
    """
    Class sets up the system of coupling of non-matching with 
    multiple spline patches.
    """
    def __init__(self, splines, E, h_th, nu, num_field=3, 
                 contact=None, int_measure_metadata=None, 
                 comm=worldcomm):
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
        contact : ShNAPr.contact.ShellContactContext, optional
        int_measure_metadata : dict, optional
            Metadata information for integration measure of 
            intersection curves. Default is vertex quadrature
            with degree 0.
        comm : mpi4py.MPI.Intracomm, optional
        """
        self.comm = comm
        self.splines = splines
        self.num_splines = len(splines)
        self.num_field = num_field
        self.geom_dim = splines[0].mesh.geometric_dimension()

        if isinstance(E, list):
            self.E = E  # Young's modulus
            if len(self.E) != self.num_splines:
                if mpirank == 0:
                    raise AssertionError("Length of Young's modulus list "
                        "doesn't match with the number of splines.")
        else:
            self.E = [E for i in range(self.num_splines)]

        if isinstance(h_th, list):
            self.h_th = h_th  # Thickness of the spline surfaces
            if len(self.h_th) != self.num_splines:
                if mpirank == 0:
                    raise AssertionError("Length of shell thickness list "
                        "doesn't match with the number of splines.")
        else:
            self.h_th = [h_th for i in range(self.num_splines)]

        if isinstance(nu, list):
            self.nu = nu  # Poisson's ratio
            if len(self.nu) != self.num_splines:
                if mpirank == 0:
                    raise AssertionError("Length of Poisson's ratio list "
                        "doesn't match with the number of splines.")
        else:
            self.nu = [nu for i in range(self.num_splines)]

        self.spline_funcs = [Function(spline.V) for spline in self.splines]
        self.spline_test_funcs = [TestFunction(spline.V) 
                                  for spline in self.splines]

        if int_measure_metadata is None:
            self.int_measure_metadata = {'quadrature_degree': 0, 
                                         'quadrature_scheme': 'vertex'}
            # self.int_measure_metadata = {"quadrature_degree":2}
        else:
            self.int_measure_metadata = int_measure_metadata

        self.contact = contact
        self.residuals = None
        self.deriv_residuals = None

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
        self.num_interfaces = len(mortar_nels)
        if mortar_coords is None:
            mortar_coords = [np.array([[0.,0.],[0.,1.]]),]\
                            *self.num_interfaces
        self.mortar_meshes = [generate_mortar_mesh(mortar_coords[i], 
                              mortar_nels[i], comm=self.comm) 
                              for i in range(self.num_interfaces)]

    def create_mortar_funcs(self, family, degree):
        """
        Create function spaces, function spaces of control points 
        and functions for all mortar meshes.

        Parameters
        ----------
        family : str, specification of the element family.
        degree : int
        """
        # Function space for dolfin functions of mortar meshes
        if self.num_field == 1:
            self.Vms = [FunctionSpace(mortar_mesh, family, degree) for 
                        mortar_mesh in self.mortar_meshes]
        else:
            self.Vms = [VectorFunctionSpace(mortar_mesh, family, degree, 
                        dim=self.num_field) for mortar_mesh in \
                        self.mortar_meshes]

        # Function space for control points information of mortar meshes
        self.Vms_control = [FunctionSpace(mortar_mesh, family, degree) for 
                            mortar_mesh in self.mortar_meshes]
        # Mortar meshes' functions
        self.mortar_funcs = [[Function(Vm), Function(Vm)] for Vm in self.Vms]

    def create_mortar_funcs_derivative(self, family, degree):
        """
        Create the derivative of function spaces, derivative of 
        function spaces of control points and derivative of 
        function of mortar meshes.

        Parameters
        ----------
        family : str, specification of the element family.
        degree : int    
        """
        self.dVms = []
        self.dVms_control = []
        self.mortar_funcs_dxi = [[] for i in range(self.geom_dim)]

        for i in range(self.num_interfaces):
            if self.num_field == 1:
                self.dVms += [FunctionSpace(self.mortar_meshes[i], 
                                            family, degree),]
            else:
                self.dVms += [VectorFunctionSpace(self.mortar_meshes[i], 
                                                 family, degree, 
                                                 dim=self.num_field),]

            self.dVms_control += [FunctionSpace(self.mortar_meshes[i], 
                                                family, degree),]

            for j in range(self.geom_dim):
                self.mortar_funcs_dxi[j] += [[Function(self.dVms[i]), 
                                              Function(self.dVms[i])],]

    def __create_mortar_vars(self):
        """
        Rearrange the order of functions and their derivatives 
        of mortar meshes. For internal use.
        """
        self.mortar_vars = []
        for i in range(self.num_interfaces):
            self.mortar_vars += [[[],[]],]
            for j in range(2):
                self.mortar_vars[i][j] += [self.mortar_funcs[i][j],]
                for k in range(self.geom_dim):
                    self.mortar_vars[i][j] += [self.mortar_funcs_dxi[k][i][j]]

    def mortar_meshes_setup(self, mapping_list, mortar_parametric_coords, 
                            penalty_coefficient=1000):
        """
        Set up coupling of non-matching system for mortar meshes.

        Parameters
        ----------
        mapping_list : list of ints
        mortar_parametric_coords : list of ndarrays
        penalty_coefficient : float, optional, default is 1000
        """
        self.__create_mortar_vars()
        self.mapping_list = mapping_list

        self.transfer_matrices_list = []
        self.transfer_matrices_control_list = []
        self.transfer_matrices_linear_list = []
        self.hm_avg_list = []
        self.alpha_d_list = []
        self.alpha_r_list = []

        for i in range(self.num_interfaces):
            transfer_matrices = [[], []]
            transfer_matrices_control = [[], []]
            transfer_matrices_linear = [[], []]
            for j in range(len(self.mapping_list[i])):

                move_mortar_mesh(self.mortar_meshes[i], 
                                 mortar_parametric_coords[i][j])
                # Create transfer matrices
                transfer_matrices[j] = create_transfer_matrix_list(
                    self.splines[self.mapping_list[i][j]].V, 
                    self.Vms[i], self.dVms[i])
                transfer_matrices_control[j] = create_transfer_matrix_list(
                    self.splines[self.mapping_list[i][j]].V_control, 
                    self.Vms_control[i], self.dVms_control[i])
                transfer_matrices_linear[j] = create_transfer_matrix(
                    self.splines[self.mapping_list[i][j]].V_linear,
                    self.Vms_control[i])

            # Store transfers in lists for future use
            self.transfer_matrices_list += [transfer_matrices,]
            self.transfer_matrices_control_list += [transfer_matrices_control]
            self.transfer_matrices_linear_list += [transfer_matrices_linear,]

            s_ind0, s_ind1 = mapping_list[i]
            # Compute element length
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
            self.hm_avg_list += [hm_avg,]

            # Use "Minimum" method for spline patches with different
            # material properties. 
            # For other methods, see Herrema et al. Section 4.2
            # For uniform isotropic material:
            max_Aij0 = float(self.E[s_ind0]*self.h_th[s_ind0]\
                       /(1-self.nu[s_ind0]**2))
            max_Aij1 = float(self.E[s_ind1]*self.h_th[s_ind1]\
                       /(1-self.nu[s_ind1]**2))
            alpha_d = Constant(penalty_coefficient)/hm_avg\
                      *min(max_Aij0, max_Aij1)
            max_Dij0 = float(self.E[s_ind0]*self.h_th[s_ind0]**3\
                       /(12*(1-self.nu[s_ind0]**2)))
            max_Dij1 = float(self.E[s_ind1]*self.h_th[s_ind1]**3\
                       /(12*(1-self.nu[s_ind1]**2)))
            alpha_r = Constant(penalty_coefficient)/hm_avg\
                      *min(max_Dij0, max_Dij1)
            self.alpha_d_list += [alpha_d,]
            self.alpha_r_list += [alpha_r,]

    def set_residuals(self, residuals, deriv_residuals=None,  
                      point_sources=None, point_source_inds=None):
        """
        Specify the shell residuals.

        Parameters
        ----------
        residuals : list of ufl forms
        deriv_residuals : list of ufl forms or None, default is None
        point_sources : list of dolfin PointSources, default is None
        point_source_inds : list of inds, default is None
        """
        if deriv_residuals is None:
            deriv_residuals = [derivative(residuals[i], self.spline_funcs[i]) 
                                          for i in range(self.num_splines)]

        # Convert residuals and derivatives from ufl.form.From
        # to dolfin.fem.form.Form
        self.residuals = [Form(res) for res in residuals]
        self.deriv_residuals = [Form(Dres) for Dres in deriv_residuals]

        if point_sources is None and point_source_inds is not None:
            if mpirank == 0:
                raise RuntimeError("``point_sources`` has to be given ", 
                                    "if ``point_source_inds`` is given.")
        elif point_sources is not None and point_source_inds is None:
            if mpirank == 0:
                raise RuntimeError("``point_source_inds`` has to be given ", 
                                    "if ``point_sources`` is given.")
        self.point_sources = point_sources
        self.point_source_inds = point_source_inds

    def assemble_nonmatching(self):
        """
        Assemble the non-matching system.
        """
        # Step 1: assemble residuals of ExtractedSplines 
        # and derivatives.
        if self.residuals is None:
            if mpirank == 0:
                raise RuntimeError("Shell residuals are not specified.") 
        if self.deriv_residuals is None:
            if mpirank == 0:
                raise RuntimeError("Derivatives of shell residuals are "
                                   "not specified.")

        # Compute contributions from shell residuals and derivatives
        R_FE = []
        dR_du_FE = []
        for i in range(self.num_splines):
            R_assemble = assemble(self.residuals[i])
            dR_du_assemble = assemble(self.deriv_residuals[i])
            if self.point_sources is not None:
                for j, ps_ind in enumerate(self.point_source_inds):
                    if ps_ind == i:
                        self.point_sources[j].apply(R_assemble)
            R_FE += [v2p(R_assemble),]
            dR_du_FE += [m2p(dR_du_assemble),]

        # Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        Rm_FE = [None for i1 in range(self.num_splines)]
        dRm_dum_FE = [[None for i1 in range(self.num_splines)] 
                      for i2 in range(self.num_splines)]

        # Compute non-matching contributions ``Rm_FE`` and 
        # ``dRm_dum_FE``.
        for i in range(self.num_interfaces):
            dx_m = dx(domain=self.mortar_meshes[i], 
                      metadata=self.int_measure_metadata)
            PE = penalty_energy(self.splines[self.mapping_list[i][0]], 
                self.splines[self.mapping_list[i][1]], self.mortar_meshes[i],
                self.Vms_control[i], self.dVms_control[i], 
                self.transfer_matrices_control_list[i][0], 
                self.transfer_matrices_control_list[i][1], 
                self.alpha_d_list[i], self.alpha_r_list[i], 
                self.mortar_vars[i][0], self.mortar_vars[i][1], 
                dx_m=dx_m)

            # An initial check for penalty energy, if ``PE``is nan,
            # raise RuntimeError.
            PE_value = assemble(PE)
            if PE_value is nan:
                if mpirank == 0:
                    raise RuntimeError("Penalty energy value is nan between "
                          "splines {:2d} and {:2d}.".format(
                          self.mapping_list[i][0], self.mapping_list[i][1]))

            Rm_list = penalty_differentiation(PE, 
                self.mortar_vars[i][0], self.mortar_vars[i][1])
            Rm = transfer_penalty_differentiation(Rm_list, 
                self.transfer_matrices_list[i][0], 
                self.transfer_matrices_list[i][1])
            dRm_dum_list = penalty_linearization(Rm_list, 
                self.mortar_vars[i][0], self.mortar_vars[i][1])
            dRm_dum = transfer_penalty_linearization(dRm_dum_list, 
                self.transfer_matrices_list[i][0], 
                self.transfer_matrices_list[i][1])

            for j in range(len(dRm_dum)):
                if Rm_FE[self.mapping_list[i][j]] is not None:
                    Rm_FE[self.mapping_list[i][j]] += Rm[j]
                else:
                    Rm_FE[self.mapping_list[i][j]] = Rm[j]

                for k in range(len(dRm_dum[j])):
                    if dRm_dum_FE[self.mapping_list[i][j]]\
                       [self.mapping_list[i][k]] is not None:
                        dRm_dum_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] += dRm_dum[j][k]
                    else:
                        dRm_dum_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] = dRm_dum[j][k]

        # Step 3: add spline residuals and non-matching contribution together
        for i in range(self.num_splines):
            if Rm_FE[i] is not None:
                Rm_FE[i] += R_FE[i]
                dRm_dum_FE[i][i] += dR_du_FE[i]
            else:
                Rm_FE[i] = R_FE[i]
                dRm_dum_FE[i][i] = dR_du_FE[i]

        # Step 4: add contact contributions if contact is given
        if self.contact is not None:
            Kcs, Fcs = self.contact.assembleContact(self.spline_funcs, 
                                                    output_PETSc=True)
            for i in range(self.num_splines):
                if Rm_FE[i] is None:
                    Rm_FE[i] = Fcs[i]
                elif Rm_FE[i] is not None and Fcs[i] is not None:
                    Rm_FE[i] += Fcs[i]
                for j in range(self.num_splines):
                    if dRm_dum_FE[i][j] is None:
                        dRm_dum_FE[i][j] = Kcs[i][j]
                    elif dRm_dum_FE[i][j] is not None \
                        and Kcs[i][j] is not None:
                        dRm_dum_FE[i][j] += Kcs[i][j]

        # self.dR_du_FE = dR_du_FE
        # self.R_FE = R_FE
        # self.dRm_dum_FE = dRm_dum_FE
        # self.Rm_FE = Rm_FE

        return dRm_dum_FE, Rm_FE

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
            Rt_IGA += [v2p(FE2IGA(self.splines[i], Rt_FE[i])),]
            # Rt_IGA += [AT_x(self.splines[i].M, Rt_FE[i]),]
            dRt_dut_IGA += [[],]
            for j in range(self.num_splines):
                if dRt_dut_FE[i][j] is not None:
                    dRm_dum_IGA_temp = AT_R_B(m2p(self.splines[i].M), 
                                  dRt_dut_FE[i][j], m2p(self.splines[j].M))

                    if i==j:
                        apply_bcs_mat(self.splines[i], dRm_dum_IGA_temp, 
                                      diag=1)
                    else:
                        apply_bcs_mat(self.splines[i], dRm_dum_IGA_temp, 
                                      diag=0)
                else:
                    dRm_dum_IGA_temp = None

                dRt_dut_IGA[i] += [dRm_dum_IGA_temp,]

        self.A_list = dRt_dut_IGA
        self.b_list = Rt_IGA

        self.b = create_nest_PETScVec(Rt_IGA, comm=self.comm)
        self.A = create_nest_PETScMat(dRt_dut_IGA, comm=self.comm)

        return self.A, self.b

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
            # it using Dolfin direct solver.
            if mpisize == 1:
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
                                ksp_view=False, ksp_monitor_residual=False):
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

        Returns
        -------
        self.spline_funcs : list of dolfin functions
        """

        # Zero out values in mortar mesh functions if True
        if zero_mortar_funcs:
            for i in range(len(self.transfer_matrices_list)):
                for j in range(len(self.transfer_matrices_list[i])):
                    for k in range(len(self.transfer_matrices_list[i][j])):
                            self.mortar_vars[i][j][k].interpolate(
                                                      Constant((0.,0.,0.)))
        
        for newton_iter in range(max_it+1):

            dRt_dut_FE, Rt_FE = self.assemble_nonmatching()
            self.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

            if solver == "direct":
                if mpisize == 1:
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
                    if mpirank == 0:
                        print("Solver iteration: {}, relative norm: {:.12}."
                            .format(newton_iter, rel_norm))
                sys.stdout.flush()

            if rel_norm < rtol:
                if MPI.rank(self.comm) == 0:
                    if mpirank == 0:
                        print("Newton's iteration finished in {} "
                              "iterations (relative tolerance: {})."
                              .format(newton_iter, rtol))
                break

            if newton_iter == max_it:
                if mpirank == 0:
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

            for i in range(self.num_splines):
                self.splines[i].M.mat().mult(du_IGA_list[i], 
                                             du_list[i].vector().vec())
                self.spline_funcs[i].assign(self.spline_funcs[i]+du_list[i])
                v2p(du_list[i].vector()).ghostUpdate()

            for i in range(len(self.transfer_matrices_list)):
                for j in range(len(self.transfer_matrices_list[i])):
                    for k in range(len(self.transfer_matrices_list[i][j])):
                        A_x_b(self.transfer_matrices_list[i][j][k], 
                            self.spline_funcs[
                                self.mapping_list[i][j]].vector(), 
                            self.mortar_vars[i][j][k].vector())

        return self.spline_funcs


class NonMatchingNonlinearProblem(NonlinearProblem):
    """
    A customized subclass of dolfin ``NonlinearProlem`` to make use of 
    ``PETScSNESSolver``.
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
        self.A.convert('seqaij')

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
                        self.problem.mortar_vars[i][j][k].vector())

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