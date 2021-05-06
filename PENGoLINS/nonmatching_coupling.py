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
                 transfer_derivative=True):
        """
        Pass the list of splines and number of element for 
        each spline and other parameters to initialize the 
        coupling of non-matching problem.
        
        Parameters
        ----------
        splines : list of ExtractedSplines
        E : ufl Constant, Young's modulus
        h_th : ufl Constant, thickness of the splines
        nu : ufl Constant, Passion's ratio
        num_field : int, optional
            Number of field of the unknowns. Default is 3.
        transfer_derivative : bool, optional, default is True.
        """
        # A list that contains all non-matching splines
        self.splines = splines
        self.E = E  # Young's modulus 
        self.h_th = h_th  # Thickness of the splines
        self.nu = nu  # Passion's ratio
        self.num_field = num_field
        self.geometric_dimension = splines[0].mesh.geometric_dimension()
        self.num_splines = len(splines)
        self.transfer_derivative = transfer_derivative
        self.spline_funcs = [Function(spline.V) for spline in self.splines]
        self.spline_test_funcs = [TestFunction(spline.V) for spline in \
                                  self.splines]

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
            num_el_list[i]) for i in range(self.num_interfaces)]

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
            self.Vms = [FunctionSpace(mortar_mesh, family, degree) for \
                mortar_mesh in self.mortar_meshes]
        else:
            self.Vms = [VectorFunctionSpace(mortar_mesh, family, degree, 
                dim=self.num_field) for mortar_mesh in self.mortar_meshes]

        self.Vms_control = [FunctionSpace(mortar_mesh, family, degree) for \
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
        self.mortar_funcs_dxi = [[] for i in range(self.geometric_dimension)]

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

            for j in range(self.geometric_dimension):
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
                    for k in range(self.geometric_dimension):
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
            # print("Mortar mesh index:", i)
            transfer_matrices = [[], []]
            transfer_matrices_control = [[], []]
            transfer_matrices_linear = [[], []]
            for j in range(len(self.mapping_list[i])):
                move_mortar_mesh(self.mortar_meshes[i], 
                                 mortar_meshes_locations[i][j])
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

            h0 = spline_mesh_size(self.splines[self.mapping_list[i][0]])
            h0_func = project(h0, 
                self.splines[self.mapping_list[i][0]].V_linear)
            h0m = A_x(transfer_matrices_linear[0], h0_func)

            h1 = spline_mesh_size(self.splines[self.mapping_list[i][1]])
            h1_func = project(h1, 
                self.splines[self.mapping_list[i][1]].V_linear)
            h1m = A_x(transfer_matrices_linear[1], h1_func)
            h_avg = 0.5*(h0m+h1m)

            hm_avg = Function(self.Vms_control[i])
            hm_avg.vector().set_local(h_avg.getArray()[::-1])
            self.hm_avg_list += [hm_avg,]

            alpha_d = Constant(penalty_coefficient)*self.E*self.h_th\
                    /(hm_avg*(1-self.nu**2))
            alpha_r = Constant(penalty_coefficient)*self.E*self.h_th**3\
                    /(12*hm_avg*(1-self.nu**2))
            self.alpha_d_list += [alpha_d,]
            self.alpha_r_list += [alpha_r,]

    def penalty_setup(self, dx_m_list=None, quadrature_degree=2):
        """
        Set up penalty terms and corresponding linearizations.

        Parameters
        ----------
        dx_m_list : list of ufl Measure or None
        quadrature_degree : int, default is 2.
        """
        self.Rm_IGA = []
        self.dRm_dum_IGA = []
        for i in range(self.num_interfaces):
            if dx_m_list is None:
                dx_m = None
            else:
                dx_m = dx_m_list[i]
            self.PE = penalty_energy(self.splines[self.mapping_list[i][0]], 
                self.splines[self.mapping_list[i][1]], self.mortar_meshes[i],
                self.Vms_control[i], self.dVms_control[i], 
                self.transfer_matrices_control_list[i][0], 
                self.transfer_matrices_control_list[i][1], 
                self.alpha_d_list[i], self.alpha_r_list[i], 
                self.mortar_vars[i][0], self.mortar_vars[i][1], 
                dx_m=dx_m, quadrature_degree=quadrature_degree)

            Rm_list = penalty_differentiation(self.PE, 
                self.mortar_vars[i][0], self.mortar_vars[i][1])
            Rm = transfer_penalty_differentiation(Rm_list, 
                self.transfer_matrices_list[i][0], 
                self.transfer_matrices_list[i][1])
            self.Rm_IGA += [R2IGA([self.splines[self.mapping_list[i][0]], 
                self.splines[self.mapping_list[i][1]]], Rm),]

            dRm_dum_list = penalty_linearization(Rm_list, 
                self.mortar_vars[i][0], self.mortar_vars[i][1])
            dRm_dum = transfer_penalty_linearization(dRm_dum_list, 
                self.transfer_matrices_list[i][0], 
                self.transfer_matrices_list[i][1])
            self.dRm_dum_IGA += [dRdu2IGA([self.splines[self.mapping_list\
                [i][0]], self.splines[self.mapping_list[i][1]]], dRm_dum),]

    # def splines_setup0(self, source_terms, point_sources=None, 
    #                   point_source_ind=None):
    #     """
    #     PDE residuals that using the SVK module for all splines. 
        
    #     Parameters
    #     ----------
    #     source_terms : list of ufl Forms
    #     point_sources : list of dolfin PointSources, optional
    #         Point load for splines.
    #     point_source_ind : list of ints, optional
    #         Point load for indices of splines.
    #     """
    #     self.source_terms = source_terms
    #     self.point_sources = point_sources
    #     self.point_source_ind = point_source_ind

    #     self.R = []
    #     self.R_IGA = []
    #     self.dR_du_IGA = []
    #     for i in range(self.num_splines):
    #         self.R += [SVK_residual(self.splines[i], self.spline_funcs[i], 
    #                                self.spline_test_funcs[i], self.E, self.nu, 
    #                                self.h_th, source_terms[i]),]
    #         self.R_IGA += [v2p(self.splines[i].assembleVector(self.R[i])),]
    #         if point_sources is not None:
    #             for j in range(len(point_source_ind)):
    #                 if i == point_source_ind[j]:
    #                     R_assemble = assemble(self.R[i])
    #                     point_sources[j].apply(R_assemble)
    #                     self.R_IGA[i] += v2p(self.splines[i].\
    #                         extractVector(R_assemble))
    #         self.dR_du_IGA += [m2p(self.splines[i].assembleMatrix(
    #             derivative(self.R[i], self.spline_funcs[i]))),]

    def splines_setup(self, residuals, point_sources=None, 
                      point_source_ind=None):
        """
        PDE residuals that using the SVK module for all splines. 
        
        Parameters
        ----------
        residuals: list of ufl Forms
        point_sources : list of dolfin PointSources, optional
            Point load for splines.
        point_source_ind : list of ints, optional
            Indices of point load for splines.
        """
        self.R = residuals
        self.point_sources = point_sources
        self.point_source_ind = point_source_ind

        self.R_IGA = []
        self.dR_du_IGA = []
        ps_count = 0
        if point_sources is not None:
            for i in range(self.num_splines):
                for j in range(len(self.point_source_ind)):
                    if i == j:
                        R_assemble = assemble(self.R[i])
                        point_sources[ps_count].apply(R_assemble)
                        self.R_IGA += [v2p(self.splines[i].extractVector(
                            R_assemble)),]
                        ps_count += 1
                    else:
                        self.R_IGA += [v2p(self.splines[i].assembleVector(
                            self.R[i])),]
                self.dR_du_IGA += [m2p(self.splines[i].assembleMatrix(
                    derivative(self.R[i], self.spline_funcs[i]))),]
        else:
            for i in range(self.num_splines):
                self.R_IGA += [v2p(self.splines[i].assembleVector(self.R[i])),]
                self.dR_du_IGA += [m2p(self.splines[i].assembleMatrix(
                    derivative(self.R[i], self.spline_funcs[i]))),]
           
    def nonmatching_setup(self):
        """
        Set up the system that solves the coupling of non-matching 
        problem.
        """
        self.A_list = []
        b_list_ = []
        self.b_list = []
        self.u_IGA_list = []
        self.u_list = []
        self.mat_sizes = [self.dR_du_IGA[i].size[0] for i in \
                           range(self.num_splines)]

        for i in range(self.num_splines):
            self.A_list += [[],]
            for j in range(self.num_splines):
                self.A_list[i] += [zero_petsc_mat(self.mat_sizes[i], 
                                                  self.mat_sizes[j]),]
            b_list_ += [zero_petsc_vec(self.mat_sizes[i]),]
            self.u_IGA_list += [FE2IGA(self.splines[i], 
                                       self.spline_funcs[i]),]
            self.u_list += [v2p(self.u_IGA_list[i]),]
            self.A_list[i][i] += self.dR_du_IGA[i]
            b_list_[i] += self.R_IGA[i]

        for i in range(self.num_interfaces):
            for j in range(2):
                for k in range(2):
                    self.A_list[self.mapping_list[i][j]]\
                        [self.mapping_list[i][k]] += self.dRm_dum_IGA[i][j][k]
                b_list_[self.mapping_list[i][j]] += self.Rm_IGA[i][j]

        for i in range(self.num_splines):
            self.b_list += [-b_list_[i],]

        self.A = create_nested_PETScMat(self.A_list)
        self.u = create_nested_PETScVec(self.u_list)
        self.b = create_nested_PETScVec(self.b_list)

    def solve_linear_nonmatching_system(self, solver=None):
        """
        Solve the linear non-matching system.

        Parameters
        ----------
        solver : {'KSP'}, optional, if None, use dolfin solver

        Returns
        -------
        self.spline_funcs : list of dolfin functions
        """
        solve_nested_mat(self.A, self.u, self.b, solver=solver)
        for i in range(self.num_splines):
            v2p(self.spline_funcs[i].vector()).ghostUpdate()
            self.spline_funcs[i].vector().set_local(IGA2FE(
                self.splines[i], self.u_IGA_list[i])[:])
        return self.spline_funcs

    def solve_nonlinear_nonmatching_system(self, solver=None, ref_error=None,
                                           rel_tol=1e-5, max_iter=20):
        """
        Solve the nonlinear non-matching system.

        Parameters
        ----------
        solver : {'KSP'}, optional, if None, use dolfin solver
        ref_error : float, optional, default is None
        rel_tol : float, optional, default is 1e-5

        Returns
        -------
        self.spline_funcs : list of dolfin functions
        """
        self.rel_norm_list = []

        for newton_iter in range(max_iter+1):
            current_norm = self.b.norm()

            if newton_iter==0 and ref_error is None:
                ref_error = current_norm

            rel_norm = current_norm/ref_error
            if newton_iter > 0:
                self.rel_norm_list += [rel_norm,]
                print("Solver iteration: {}, relative norm: {:.12}."\
                    .format(newton_iter, rel_norm))
                sys.stdout.flush()

            if rel_norm < rel_tol:
                print("Newton's iteration finished in {} iterations "
                    "(relative tolerance: {}).".format(newton_iter, rel_tol))
                self.num_iter_converge = newton_iter
                break

            if newton_iter == max_iter:
                raise StopIteration("Nonlinear solver failed to converge "
                    "in {} iterations.".format(max_iter))

            du_list = []
            du_IGA_list = []
            du_IGA_petsc_list = []
            for i in range(self.num_splines):
                du_list += [Function(self.splines[i].V),]
                du_IGA_list += [FE2IGA(self.splines[i], du_list[i]),]
                du_IGA_petsc_list += [v2p(du_IGA_list[i]),]
            du = create_nested_PETScVec(du_IGA_petsc_list)

            solve_nested_mat(self.A, du, self.b)

            for i in range(self.num_splines):
                v2p(du_list[i].vector()).ghostUpdate()
                du_list[i].vector().set_local(IGA2FE(self.splines[i], 
                                                     du_IGA_list[i])[:])
                self.spline_funcs[i].assign(self.spline_funcs[i]+du_list[i])

            for i in range(len(self.transfer_matrices_list)):
                for j in range(len(self.transfer_matrices_list[i])):
                    for k in range(len(self.transfer_matrices_list[i][j])):
                        A_x_b(self.transfer_matrices_list[i][j][k], 
                            func2v(self.spline_funcs[self.mapping_list[i][j]]), 
                            func2v(self.mortar_vars[i][j][k]))

            self.penalty_setup()
            self.splines_setup(self.R, self.point_sources, 
                               self.point_source_ind)
            self.nonmatching_setup()

        return self.spline_funcs

    def nonmatching_residual(self):
        """
        Compute the residual of the non-matching system.
        """
        self.penalty_setup()
        self.R_IGA = []
        ps_count = 0
        if self.point_sources is not None:
            for i in range(self.num_splines):
                for j in range(len(self.point_source_ind)):
                    if i == j:
                        R_assemble = assemble(self.R[i])
                        self.point_sources[ps_count].apply(R_assemble)
                        self.R_IGA += [v2p(self.splines[i].extractVector(
                            R_assemble)),]
                        ps_count += 1
                    else:
                        self.R_IGA += [v2p(self.splines[i].assembleVector(
                            self.R[i])),]
        else:
            for i in range(self.num_splines):
                self.R_IGA += [v2p(self.splines[i].assembleVector(self.R[i])),]

        b_list_ = []
        self.b_list = []

        for i in range(self.num_splines):
            b_list_ += [zero_petsc_vec(self.mat_sizes[i]),]
            b_list_[i] += self.R_IGA[i]

        for i in range(self.num_interfaces):
            for j in range(2):
                b_list_[self.mapping_list[i][j]] += self.Rm_IGA[i][j]

        for i in range(self.num_splines):
            self.b_list += [-b_list_[i],]

        b_ = create_nested_PETScVec(self.b_list)
        self.b = zero_petsc_vec(b_.size)
        self.b.setArray(b_.getArray())
        return self.b

    def nonmatching_Jacobian(self):
        """
        Compute the Jacobian of non-matching system.
        """
        self.penalty_setup()
        self.dR_du_IGA = []
        for i in range(self.num_splines):
            self.dR_du_IGA += [m2p(self.splines[i].assembleMatrix(
                derivative(self.R[i], self.spline_funcs[i]))),]
        self.A_list = []

        for i in range(self.num_splines):
            self.A_list += [[],]
            for j in range(self.num_splines):
                self.A_list[i] += [zero_petsc_mat(self.mat_sizes[i], 
                                                  self.mat_sizes[j]),]
            self.A_list[i][i] += self.dR_du_IGA[i]

        for i in range(self.num_interfaces):
            for j in range(2):
                for k in range(2):
                    self.A_list[self.mapping_list[i][j]]\
                        [self.mapping_list[i][k]] += self.dRm_dum_IGA[i][j][k]

        self.A = create_nested_PETScMat(self.A_list)
        self.A.convert('seqaij')
        return self.A

# TODO: The class ``NonMatchingNonlineraProblem`` failed to converge for 
#       nearly linear problem even though the residual and Jacobian are
#       correct (verified in a manually written Newtion's iteration).
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

    def form(self, A, P, b, x):
        """
        Update solution.
        """
        print("Running self.form")
        # Update solution in IGA function space
        x.vec().copy(result=self.problem.u)
        print("x norm:", x.vec().norm())
        # print("u norm:", self.problem.u.norm())

        for i in range(self.problem.num_splines):
            v2p(self.problem.spline_funcs[i].vector()).ghostUpdate()
            self.problem.spline_funcs[i].vector().set_local(IGA2FE(
                    self.problem.splines[i], 
                    self.problem.u_IGA_list[i])[:])
        # Update mortar mesh dolfin Functions
        for i in range(len(self.problem.transfer_matrices_list)):
            for j in range(len(self.problem.transfer_matrices_list[i])):
                for k in range(len(self.problem.transfer_matrices_list[i][j])):
                    A_x_b(self.problem.transfer_matrices_list[i][j][k], 
                        func2v(self.problem.spline_funcs[
                            self.problem.mapping_list[i][j]]), 
                        func2v(self.problem.mortar_vars[i][j][k]))

    def F(self, b, x):
        """
        Update residual.
        """
        print("Running self.F")
        res = self.problem.nonmatching_residual()
        b.vec().setSizes(self.problem.b.getSizes())
        b.vec().setUp()
        b.vec().assemble()
        self.problem.b.copy(result=b.vec())
        print("Res norm:", self.problem.b.norm())
        return b

    def J(self, A, x):
        """
        Update Jacobian.
        """
        print("Running self.J")
        Dres = self.problem.nonmatching_Jacobian()
        A.mat().setSizes(self.problem.A.getSizes())
        A.mat().setUp()
        A.mat().assemble()
        # print("Type of nonmatching.A", type(self.problem.A), "---",
        #     self.problem.A.type)
        # print("Size of nonmatching.A", self.problem.A.getSizes())
        self.problem.A.copy(result=A.mat())
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
        temp_vec = PETScVector(self.nonlinear_problem.problem.u.copy())
        print("temp_vec norm:", norm(temp_vec))
        self.solver.solve(self.nonlinear_problem, temp_vec)
        temp_vec.vec().copy(result=self.nonlinear_problem.problem.u)


if __name__ == "__main__":
    pass