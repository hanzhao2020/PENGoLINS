from tIGAr.NURBS import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.igakit_utils import *
from igakit.io import VTK
import matplotlib.pyplot as plt

# Geometry creation using igakit
def create_roof_srf(num_el, p, R, angle_lim=[50,130], z_lim=[0,1]):
    # angle = (math.radians(angle_lim[0]), math.radians(angle_lim[1]))
    angle = (angle_lim[0], angle_lim[1])
    C = circle(center=[0,0,z_lim[0]], radius=R, angle=angle)
    T = circle(center=[0,0,z_lim[1]], radius=R, angle=angle)
    S = ruled(C,T)
    deg1, deg2 = S.degree
    S.elevate(0,p-deg1)
    S.elevate(1,p-deg2)
    new_knots = np.linspace(0,1,num_el+1)[1:-1]
    S.refine(0,new_knots)
    S.refine(1,new_knots)
    return S

# Extracted spline creation
def create_spline(srf, num_field=3, BCs=[1,1]):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)

    for field in range(0,3):
        scalar_spline = spline_generator.getScalarSpline(field)
        parametric_direction = 0
        for side in [0,1]:
            side_dofs = scalar_spline.getSideDofs(parametric_direction, side)
            if BCs[side] == 1:
                spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 2*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Geometric and material paramters
R = 2540
L = 508
theta = 0.1
P_max = 3000

h_th = Constant(12.7)
E = Constant(3102.75)
nu = Constant(0.3)
p = 3

# force_ratio = 0.0877
# QoI_ref = 0.693

# Central angles in radians
angles = [np.pi/2-theta, np.pi/2-theta/2, np.pi/2+theta/2, np.pi/2+theta]
angle_lim_list = [angles[0:2], angles[1:3], angles[2:4]]*3
z_lims = [0, L/4, 3*L/4, L]
z_lim_list = [z_lims[0:2]]*3 + [z_lims[1:3]]*3 + [z_lims[2:4]]*3

penalty_coefficient = 1.0e3

if MPI.rank(worldcomm) == 0:
    print("Penalty coefficient:", penalty_coefficient)
num_srf = 9
num_el = 8
p = 3  # Spline order

# Number of elements for splines in one side
num_el0 = num_el
num_el1 = num_el - 2
num_el2 = num_el - 1
num_el3 = num_el + 2
num_el4 = num_el + 1
num_el5 = num_el + 3
num_el6 = num_el - 1
num_el7 = num_el
num_el8 = num_el - 2
spline_nels = [num_el0, num_el1, num_el2, 
               num_el3, num_el4, num_el5, 
               num_el6, num_el7, num_el8]

bcs_list = [[1,0],[0,0],[0,1]]*3

nurbs_srfs = []
splines = []
total_dofs = 0

if MPI.rank(worldcomm) == 0:
    print("Creating geometry...")

for i in range(num_srf):
    nurbs_srfs += [create_roof_srf(spline_nels[i], p, R, 
        angle_lim=angle_lim_list[i], z_lim=z_lim_list[i]),]
    total_dofs += nurbs_srfs[i].control.shape[0]\
               *nurbs_srfs[i].control.shape[1]*3
    splines += [create_spline(nurbs_srfs[i], BCs=bcs_list[i]),]

if MPI.rank(worldcomm) == 0:
    print("Number of surfaces:", num_srf)
    print("Total DoFs:", total_dofs)

if MPI.rank(worldcomm) == 0:
    print("Starting analysis...")
# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
# Mortar meshes' parameters
mapping_list = [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],
                [0,3],[3,6],[1,4],[4,7],[2,5],[5,8]]
num_interfaces = len(mapping_list)

# Mortar mesh parametric locations
h_mortar_locs = [np.array([[0., 1.], [1., 1.]]), 
                 np.array([[0., 0.], [1., 0.]])]
v_mortar_locs = [np.array([[1., 0.], [1., 1.]]),
                 np.array([[0., 0.], [0., 1.]])]
mortar_nels = []
mortar_mesh_locations = []
for j in range(num_interfaces):
    mortar_nels += [3*(spline_nels[mapping_list[j][0]]\
                    +spline_nels[mapping_list[j][1]])]
    if j < 6:
        mortar_mesh_locations += [v_mortar_locs]
    else:
        mortar_mesh_locations += [h_mortar_locs]

problem.create_mortar_meshes(mortar_nels)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)
problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations, 
                            penalty_coefficient)

force_ratio_ref = [0.0877, 0.1980, 0.3473, 0.4686, 0.5647, 0.6381,
                    0.6908, 0.7246, 0.7412]
w_ref = [0.693, 1.638, 3.087, 4.477, 5.802, 7.057,
              8.237, 9.339, 10.358]

force_ratio_list = np.array(force_ratio_ref)
w_list = []

ps_w = splines[4].cpFuncs[3](np.array([0.5,0.5]))

for step, force_ratio in enumerate(force_ratio_list):

    f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])

    ps_ind = [4,]
    ps0 = PointSource(splines[ps_ind[0]].V.sub(1), Point(.5,.5), 
                      force_ratio*P_max/ps_w)
    ps_list = [ps0,]

    source_terms = []
    residuals = []
    for i in range(len(splines)):
        source_terms += [inner(f0, problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*problem.splines[i].dx]
        residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
            problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]
    problem.set_residuals(residuals, point_sources=ps_list,
                          point_source_inds=ps_ind)

    if mpirank == 0:
        print("Solving linear non-matching problem...")
    problem.solve_nonlinear_nonmatching_problem(solver="direct")

    # Print quantity of interest
    spline_ind = 4
    xi = array([0.5,0.5])
    disp_y_hom = eval_func(problem.splines[spline_ind].mesh, 
                       problem.spline_funcs[spline_ind][1], xi)
    w = eval_func(problem.splines[spline_ind].mesh, 
                  problem.splines[spline_ind].cpFuncs[3], xi)
    QoI = -disp_y_hom/w
    if mpirank == 0:
        print("Quantity of interest for patch {} = {:10.8f} (Reference "
              "value = {}).".format(spline_ind, QoI, w_ref[step]))

    w_list += [QoI,]



point_load_ref = np.array(force_ratio_ref)*P_max
point_load_list = np.array(force_ratio_list)*P_max

point_load_ref = np.insert(point_load_ref, 0, 0)
point_load_list = np.insert(point_load_list, 0, 0)
w_ref = np.insert(w_ref, 0, 0)
w_list = np.insert(w_list, 0, 0)
data_file = 'hinged_cylin_iso_nonmatching.npz'
np.savez(data_file, force_ref=point_load_ref, w_ref=w_ref,
                    force=point_load_list, w=w_list)

plt.figure()
plt.plot(w_ref, point_load_ref, "o--", mfc='none', linewidth=0.5, 
         color='black', label='Reference')
plt.plot(w_list, point_load_list, "-*", 
         color='tab:blue', label='Isotropic material')
plt.xlabel("Central deflection")
plt.ylabel("Point load")
plt.legend()
plt.show()

SAVE_PATH = "./"
for i in range(problem.num_splines):
    save_results(problem.splines[i], problem.spline_funcs[i], i, 
                save_cpfuncs=True, save_path=SAVE_PATH, comm=problem.comm)

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