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
                 int_measure_metadata=None, residuals=None, 
                 contact=None, transfer_derivative=True, 
                 comm=worldcomm):
        """
        Pass the list of splines and number of element for 
        each spline and other parameters to initialize the 
        coupling of non-matching problem.
        
        Parameters
        ----------
        splines : list of ExtractedSplines
        E : ufl Constant, Young's modulus
        h_th : ufl Constant, thickness of the splines
        nu : ufl Constant, Poisson's ratio
        num_field : int, optional
            Number of field of the unknowns. Default is 3.
        transfer_derivative : bool, optional, default is True.
        """
        if not isinstance(comm, type(worldcomm)):
            self.comm = worldcomm
        else:
            self.comm = comm

        self.splines = splines
        self.num_splines = len(splines)
        self.num_field = num_field
        self.geom_dim = splines[0].mesh.geometric_dimension()

        if isinstance(E, list):
            self.E = E  # Young's modulus
            if len(self.E) != self.num_splines:
                raise AssertionError("Length of Young's modulus list "
                    "doesn't match with the number of splines.")
        else:
            self.E = [E for i in range(self.num_splines)]

        if isinstance(h_th, list):
            self.h_th = h_th  # Thickness of the splines
            if len(self.h_th) != self.num_splines:
                raise AssertionError("Length of shell thickness list "
                    "doesn't match with the number of splines.")
        else:
            self.h_th = [h_th for i in range(self.num_splines)]

        if isinstance(nu, list):
            self.nu = nu  # Poisson's ratio
            if len(self.nu) != self.num_splines:
                raise AssertionError("Length of Poisson's ratio list "
                    "doesn't match with the number of splines.")
        else:
            self.nu = [nu for i in range(self.num_splines)]

        self.transfer_derivative = transfer_derivative

        self.spline_funcs = [Function(spline.V) for spline in self.splines]
        self.spline_test_funcs = [TestFunction(spline.V) 
                                  for spline in self.splines]

        if int_measure_metadata is None:
            self.int_measure_metadata = {"quadrature_degree":2}
        else:
            self.int_measure_metadata = int_measure_metadata

        self.residuals = residuals
        if self.residuals is not None:
            self.Dres = [derivative(self.residuals[i], self.spline_funcs[i]) 
                         for i in range(self.num_splines)]
        else:
            self.Dres = None
        self.contact = contact

    def create_mortar_meshes(self, num_el_list, mortar_pts_list):
        """
        Create mortar meshes for non-matching with multiple patches.

        Parameters
        ----------
        num_el_list : list of ints
            Contains number of elements for all mortar meshes.
        "mortar_pts_list" : list of ndarrays
            Contains points of location for all mortar meshes.
        """
        self.num_interfaces = len(num_el_list)
        self.mortar_meshes = [generate_mortar_mesh(mortar_pts_list[i], 
                              num_el_list[i], comm=self.comm) 
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
        if self.num_field == 1:
            self.Vms = [FunctionSpace(mortar_mesh, family, degree) for 
                        mortar_mesh in self.mortar_meshes]
        else:
            self.Vms = [VectorFunctionSpace(mortar_mesh, family, degree, 
                dim=self.num_field) for mortar_mesh in self.mortar_meshes]

        self.Vms_control = [FunctionSpace(mortar_mesh, family, degree) for 
                            mortar_mesh in self.mortar_meshes]
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
        of mortar meshes.
        """
        self.mortar_vars = []
        for i in range(self.num_interfaces):
            self.mortar_vars += [[[],[]],]
            for j in range(2):
                self.mortar_vars[i][j] += [self.mortar_funcs[i][j],]
                if self.transfer_derivative:
                    for k in range(self.geom_dim):
                        self.mortar_vars[i][j] += \
                            [self.mortar_funcs_dxi[k][i][j],]

    def mortar_meshes_setup(self, mapping_list, mortar_meshes_locations, 
                            penalty_coefficient=1e3):
        """
        Set up coupling of non-matching system for mortar meshes.

        Parameters
        ----------
        mapping_list : list of ints
        mortar_meshes_locations : list of ndarrays
        penalty_coefficient : float, optional, default is 1e3
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
                                 mortar_meshes_locations[i][j])
                # Create transfer matrices
                if self.transfer_derivative:
                    transfer_matrices[j] = create_transfer_matrix_list(
                        self.splines[self.mapping_list[i][j]].V, 
                        self.Vms[i], self.dVms[i])
                    transfer_matrices_control[j] = create_transfer_matrix_list(
                        self.splines[self.mapping_list[i][j]].V_control, 
                        self.Vms_control[i], self.dVms_control[i])
                    transfer_matrices_linear[j] = create_transfer_matrix(
                        self.splines[self.mapping_list[i][j]].V_linear,
                        self.Vms_control[i])
                else:
                    transfer_matrices[j] = create_transfer_matrix_list(
                        self.splines[self.mapping_list[i][j]].V, self.Vms[i])
                    transfer_matrices_control[j] = create_transfer_matrix_list(
                        self.splines[self.mapping_list[i][j]].V_control, 
                        self.Vms_control[i])
                    transfer_matrices_linear[j] = create_transfer_matrix(
                        self.splines[self.mapping_list[i][j]].V_linear,
                        self.Vms_control[i])

            self.transfer_matrices_list += [transfer_matrices,]
            self.transfer_matrices_control_list += [transfer_matrices_control,]
            self.transfer_matrices_linear_list += [transfer_matrices_linear,]

            s_ind0, s_ind1 = mapping_list[i]

            # Compute element length
            h0 = spline_mesh_size(self.splines[s_ind0])
            h0_func = project(h0, self.splines[s_ind0].V_linear)
            h0m = A_x(transfer_matrices_linear[0], h0_func)
            h1 = spline_mesh_size(self.splines[s_ind1])
            h1_func = project(h1, self.splines[s_ind1].V_linear)
            h1m = A_x(transfer_matrices_linear[1], h1_func)
            h_avg = 0.5*(h0m+h1m)
            hm_avg = Function(self.Vms_control[i])
            hm_avg.vector().set_local(h_avg.getArray()[::-1])
            self.hm_avg_list += [hm_avg,]

            # Use "Minimum" method for spline patches with different
            # material properties.
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

    def set_residuals(self, residuals, Dres=None,  
                      point_sources=None, point_source_inds=None):
        """
        Specify the shell residuals.

        Parameters
        ----------
        residuals : list of ufl forms
        Dres : list of ufl forms or None, default is None
        point_sources : list of dolfin PointSources, default is None
        point_source_inds : list of inds, default is None
        """
        self.residuals = residuals
        if Dres is not None:
            self.Dres = Dres
        else:
            self.Dres = [derivative(self.residuals[i], self.spline_funcs[i]) 
                         for i in range(self.num_splines)]

        if point_sources is None and point_source_inds is not None:
            raise RuntimeError("``point_sources`` has to be given ", 
                                "if ``point_source_inds`` is given.")
        elif point_sources is not None and point_source_inds is None:
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
            raise RuntimeError("Shell residuals are not specified.") 
        R_FE = []
        dR_du_FE = []
        for i in range(self.num_splines):
            R_assemble = assemble(self.residuals[i])
            dR_du_assemble = assemble(self.Dres[i])
            if self.point_sources is not None:
                for j, ps_ind in enumerate(self.point_source_inds):
                    if ps_ind == i:
                        self.point_sources[j].apply(R_assemble)
            R_FE += [v2p(R_assemble),]
            dR_du_FE += [m2p(dR_du_assemble),]

        # Step 2: assemble non-matching contributions
        Rm_FE = [None for i1 in range(self.num_splines)]
        dRm_dum_FE = [[None for i1 in range(self.num_splines)] 
                      for i2 in range(self.num_splines)]

        for i in range(self.num_interfaces):
            dx_m = dx(domain=self.mortar_meshes[i], 
                      metadata=self.int_measure_metadata)
            self.PE = penalty_energy(self.splines[self.mapping_list[i][0]], 
                self.splines[self.mapping_list[i][1]], self.mortar_meshes[i],
                self.Vms_control[i], self.dVms_control[i], 
                self.transfer_matrices_control_list[i][0], 
                self.transfer_matrices_control_list[i][1], 
                self.alpha_d_list[i], self.alpha_r_list[i], 
                self.mortar_vars[i][0], self.mortar_vars[i][1], 
                dx_m=dx_m)
            Rm_list = penalty_differentiation(self.PE, 
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

        # Step 3: add spline residuls and non-matching contribution together
        for i in range(self.num_splines):
            Rm_FE[i] += R_FE[i]
            dRm_dum_FE[i][i] += dR_du_FE[i]

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
            dRt_dut_IGA += [[],]
            for j in range(self.num_splines):
                if dRt_dut_FE[i][j] is not None:
                    dRm_dum_IGA_temp = AT_R_B(m2p(self.splines[i].M), 
                                  dRt_dut_FE[i][j], m2p(self.splines[j].M))
                    if i==j:
                        diag = 1
                    else:
                        diag = 0
                    apply_bcs_mat(self.splines[i], dRm_dum_IGA_temp, 
                                  self.splines[j], diag=diag)
                else:
                    dRm_dum_IGA_temp = None
                dRt_dut_IGA[i] += [dRm_dum_IGA_temp,]

        b = create_nested_PETScVec(Rt_IGA, comm=self.comm)
        A = create_nested_PETScMat(dRt_dut_IGA, comm=self.comm)
        return A, b

    def solve_linear_nonmatching_problem(self, solver=None):
        """
        Solve the linear non-matching system.

        Parameters
        ----------
        solver : {'KSP'}, optional, if None, use dolfin solver

        Returns
        -------
        self.spline_funcs : list of dolfin functions
        """
        dRt_dut_FE, Rt_FE = self.assemble_nonmatching()
        A, b = self.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

        u_list = []
        for i in range(self.num_splines):
            u_list += [zero_petsc_vec(self.splines[i].M.size(1), 
                                      comm=self.splines[i].comm),]
        u = create_nested_PETScVec(u_list, comm=self.comm)

        solve_nested_mat(A, u, -b, solver=solver)
        
        for i in range(self.num_splines):
            v2p(self.spline_funcs[i].vector()).ghostUpdate()
            self.splines[i].M.mat().mult(u_list[i], 
                                         self.spline_funcs[i].vector().vec())
        return self.spline_funcs

    def solve_nonlinear_nonmatching_problem(self, solver=None, ref_error=None,
                                            rel_tol=1e-3, max_iter=20,
                                            zero_mortar_funcs=True):
        """
        Solve the nonlinear non-matching system.

        Parameters
        ----------
        solver : {'KSP'}, optional, if None, use dolfin solver
        ref_error : float, optional, default is None
        rel_tol : float, optional, default is 1e-3

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
        
        for newton_iter in range(max_iter+1):

            dRt_dut_FE, Rt_FE = self.assemble_nonmatching()
            A, b = self.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

            current_norm = b.norm()

            if newton_iter==0 and ref_error is None:
                ref_error = current_norm

            rel_norm = current_norm/ref_error
            if newton_iter > 0:
                if MPI.rank(self.comm) == 0:
                    print("Solver iteration: {}, relative norm: {:.12}."\
                        .format(newton_iter, rel_norm))
                sys.stdout.flush()

            if rel_norm < rel_tol:
                if MPI.rank(self.comm) == 0:
                    print("Newton's iteration finished in {} iterations "
                        "(relative tolerance: {}).".format(newton_iter, 
                                                           rel_tol))
                break

            if newton_iter == max_iter:
                raise StopIteration("Nonlinear solver failed to converge "
                    "in {} iterations.".format(max_iter))

            du_list = []
            du_IGA_list = []
            for i in range(self.num_splines):
                du_list += [Function(self.splines[i].V),]
                du_IGA_list += [zero_petsc_vec(self.splines[i].M.size(1), 
                                               comm=self.splines[i].comm)]
            du = create_nested_PETScVec(du_IGA_list, comm=self.comm)

            solve_nested_mat(A, du, -b, solver=solver)

            for i in range(self.num_splines):
                v2p(du_list[i].vector()).ghostUpdate()
                self.splines[i].M.mat().mult(du_IGA_list[i], 
                                             du_list[i].vector().vec())
                self.spline_funcs[i].assign(self.spline_funcs[i]+du_list[i])

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
    A customed subclass of dolfin ``NonlinearProlem`` to make use of 
    ``PETScSNESSolver``.
    """
    def __init__(self, problem, **kwargs):
        """
        Initilization of ``NonMatchingNonlinearProblem``.

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
        self.u = create_nested_PETScVec(self.u_list, comm=self.problem.comm)

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
        Initilization of ``NonMatchingNonlinearSolver``.

        Parameters
        ----------
        nonlinear_problem : instance of ``NonMatchingNonlinearProble``
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