from PENGoLINS.nonmatching_coupling import *
import matplotlib.pyplot as plt

def create_surf(pts, num_el0, num_el1,p):
    knots0 = np.linspace(0,1,num_el0+1)[1:-1]
    knots1 = np.linspace(0,1,num_el1+1)[1:-1]
    L1 = line(pts[0],pts[1])
    L2 = line(pts[2],pts[3])
    srf = ruled(L1,L2)
    deg0, deg1 = srf.degree 
    srf.elevate(0,p-deg0)
    srf.elevate(1,p-deg1)
    srf.refine(0,knots0)
    srf.refine(1,knots1)
    return srf

def create_spline(srf, num_field, BCs=[0,1]):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)

    for field in range(num_field):
        scalar_spline = spline_generator.getScalarSpline(field)
        for para_direction in range(2):
            if BCs[para_direction] == 1:
                side = 0 # only consider fixing the 0 side
                side_dofs = scalar_spline.getSideDofs(para_direction, 
                                                      side, nLayers=1)
                spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 3*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

E = Constant(1.0e7)
nu = Constant(0.)
h_th = Constant(0.1)
tip_load = -10.

L = 20.
w = 2.
h = 2.
num_field = 3

pts0 = [[-w/2., 0., 0.], [w/2., 0., 0.],\
        [-w/2., L, 0.], [w/2., L, 0.]]
pts1 = [[0., 0., 0.], [0.,0.,-h],\
        [0., L, 0.], [0., L, -h]]


pc_list = [1.0e3,]
# Uncomment the full pc_list to test different penalty coefficients
# pc_list = [1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 1.0e3, 1.0e5, 1.0e7]
# num_el_list = [8, 16, 32, 64, 128, 256]
z_disp_list = []
theta_free_list = []
theta_load_list = []
theta_twist_list = []

for penalty_coefficient in pc_list:
# for num_el in num_el_list:

    num_el = 10
    # penalty_coefficient = 1e3
    print("Number of elements:", num_el)
    print("Penalty coefficient:", penalty_coefficient)
    p = 3
    num_el0 = num_el
    num_el1 = num_el + 1
    p0 = p
    p1 = p

    print("Creating geometry...")
    srf0 = create_surf(pts0, int(num_el0/2), num_el0, p0)
    srf1 = create_surf(pts1, int(num_el1/2), num_el1, p1)
    spline0 = create_spline(srf0, num_field, BCs=[0,1])
    spline1 = create_spline(srf1, num_field, BCs=[0,1])

    splines = [spline0, spline1]
    problem = NonMatchingCoupling(splines, E, h_th, nu, comm=selfcomm)

    mortar0_pts = np.array([[0.,0.],[0.,1.]])
    mortar_pts = [mortar0_pts]
    mortar_nels = [2*num_el0]

    problem.create_mortar_meshes(mortar_nels, mortar_pts)
    problem.create_mortar_funcs('CG',1)
    problem.create_mortar_funcs_derivative('CG',1)

    mapping_list = [[0,1],]
    physical_locations = [np.array([[0.,0.,0.],[0.,20.,0.]]),]

    mortar_mesh_locations = [[]]*problem.num_interfaces
    for i in range(problem.num_interfaces):
        for j in range(2):
            mortar_mesh_locations[i] += [interface_parametric_location(
                                         splines[mapping_list[i][j]], 
                                         problem.mortar_meshes[i], 
                                         physical_locations[i]),]

    problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                                penalty_coefficient)

    source_terms = []
    residuals = []
    f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])
    ps0 = PointSource(spline0.V.sub(2), Point(1.,1.), -tip_load)
    ps_list = [ps0,]
    ps_ind = [0,]

    for i in range(len(splines)):
        source_terms += [inner(f0, problem.splines[i].rationalize(\
            problem.spline_test_funcs[i]))*problem.splines[i].dx,]
        residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
            problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]

    problem.set_residuals(residuals, point_sources=ps_list, 
                          point_source_inds=ps_ind)
    problem.solve_linear_nonmatching_problem()

    ########## Measure displacement and angles of intertest ##########
    # z-displacement at load point
    xi = array([1.0, 1.0])
    z_disp = -problem.spline_funcs[0](xi)[2]/spline0.cpFuncs[3](xi)
    print("Displacement at load point in z direction = {:8.6f}."\
          .format(z_disp))

    # Measure angle between two patches at the end
    EPS = 1.0e-2
    xi1_load = [array([0.5, 1.0]), array([0.5+EPS, 1.0])]
    xi1_free = [array([0.5, 1.0]), array([0.5-EPS, 1.0])]
    xi2 = [array([0., 1.]), array([0.+EPS, 1.])]
    xis = [xi1_load, xi1_free, xi2]

    pts1_load, pts1_free = [], []
    pts2_vectical = []
    pts_end = [pts1_load, pts1_free, pts2_vectical]
    splines_test = [spline0, spline0, spline1]
    y_homs_test = [problem.spline_funcs[0], problem.spline_funcs[0], 
                   problem.spline_funcs[1]]

    for i in range(len(xis)):
        for j in range(2):
            pts_end[i] += [deformed_position(splines_test[i], 
                xis[i][j], y_homs_test[i]),]

    vec1_load = pts1_load[1] - pts1_load[0]
    vec1_free = pts1_free[1] - pts1_free[0]
    vec2 = pts2_vectical[1] - pts2_vectical[0]

    theta_load = vec_angle(vec1_load, vec2)
    theta_free = vec_angle(vec1_free, vec2)
    print("Angle between the end of two "
        "patches (load end) = {:8.6f}.".format(theta_load))
    print("Angle between the end of two"
        " patches (free end) = {:8.6f}.".format(theta_free))

    # Measure twist angle between two ends of vertical patches
    xi2_pin = [array([0., 0.]), array([0.+EPS, 0.])]
    xi2_free = [array([0., 1.]), array([0.+EPS, 1.])]
    pts2_pin = []
    pts2_free = []

    xis_twist = [xi2_pin, xi2_free]
    pts_twist = [pts2_pin, pts2_free]

    for i in range(len(xis_twist)):
        for j in range(2):
            pts_twist[i] += [deformed_position(spline1, 
                xis_twist[i][j], problem.spline_funcs[1])]

    vec2_pin = pts2_pin[1] - pts2_pin[0]
    vec2_free = pts2_free[1] - pts2_free[0]
    theta_twist = vec_angle(vec2_pin, vec2_free)
    print("Twist angle for vertical patch = {:8.6f}.".format(theta_twist))

    z_disp_list += [z_disp]
    theta_free_list += [theta_free,]
    theta_load_list += [theta_load,]
    theta_twist_list += [theta_twist]
    print("")

SAVE_PATH = "./"
# SAVE_PATH = "/home/han/Documents/test_results/"
for i in range(len(splines)):
    save_results(splines[i], problem.spline_funcs[i], i, 
        save_path=SAVE_PATH, save_cpfuncs=True, comm=selfcomm)

if len(pc_list) > 1:
    # # Plot angle w.r.t. the penalty coefficient
    plt.figure()
    plt.plot(pc_list, z_disp_list, '-*', 
            label="Vertical displacement on load point")
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel("Penalty coefficient")
    plt.ylabel("Displacement")
    plt.title("Displacement for T-beam problem with 2 patches")

    plt.figure()
    plt.plot(pc_list, theta_load_list, '-*', label="Angle of the load side")
    plt.plot(pc_list, theta_free_list, '-*', label="Angle of the free side")
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel("Penalty coefficient")
    plt.ylabel("Angle")
    plt.title("Angle for T-beam problem with 2 patches")

    plt.figure()
    plt.plot(pc_list, theta_twist_list, '-*', label="Twist angle of vertical path")
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel("Penalty coefficient")
    plt.ylabel("Twist")
    plt.title("Twist for T-beam problem with 2 patches")
    plt.show()


# R_FE, dR_du_FE = problem.assemble_residuals()
# Rm_FE, dRm_dum_FE = problem.assemble_nonmatching()
# Rt_FE, dRt_dut_FE = problem.setup_nonmatching_system(R_FE, dR_du_FE,
#                                                   Rm_FE, dRm_dum_FE)
# b, A = problem.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

# u_IGA_list = []
# u_list = []
# for i in range(problem.num_splines):
#     u_IGA_list += [FE2IGA(problem.splines[i], problem.spline_funcs[i]),]
#     u_list += [v2p(u_IGA_list[i]),]

# u = create_nested_PETScVec(u_list, comm=problem.comm)

# solve_nested_mat(A, u, -b, solver=None)

"""
Visualization with Paraview:
 
1. Load output files for one extracted spline and apply an AppendAttributes 
   filter to combine them.
2. Apply Calculator filter to the AppendAttributes with the formula

    (F0_0/F0_3-coordsX)*iHat + (F0_1/F0_3-coordsY)*jHat + (F0_2/F0_3-coordsZ)*kHat

   for index=0 to get the undeformed configuration, then apply WarpByVector
   filter to the Calculator with scale 1.
3. Apply another Calculator filter to WarpByVector with the formula

    (u0_0/F0_3)*iHat + (u0_1/F0_3)*jHat + (u0_2/F0_3)*kHat

   for index=0 to get the displacement, then apply WarpByVector filter to the 
   Calculator. 

Note: for spline patches with index other than 0, replace ``u0`` and ``F0``
by ``ui`` and ``Fi`` with corresponding index i. 
"""