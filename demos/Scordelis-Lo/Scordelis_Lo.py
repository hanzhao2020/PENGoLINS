from tIGAr.NURBS import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.igakit_utils import *

# Geometry creation using igakit
def create_roof_srf(num_el, p, R, angle_lim=[50,130], z_lim=[0,1]):
    angle = (math.radians(angle_lim[0]), math.radians(angle_lim[1]))
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
def create_spline(srf, num_field=3, BCs=[1,1], fix_z_node=False):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)

    for field in range(0,2):
        scalar_spline = spline_generator.getScalarSpline(field)
        parametric_direction = 1
        for side in [0,1]:
            side_dofs = scalar_spline.getSideDofs(parametric_direction, side)
            if BCs[side] == 1:
                spline_generator.addZeroDofs(field, side_dofs)

    if fix_z_node:
        # Pin z displacement for one control point to eliminate rigid mode.
        field = 2
        spline_generator.addZeroDofs(field, [0,])

    quad_deg = 2*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Geometric and material paramters
L = 50.0
R = 25.0
theta = 40.0 # in degrees
h_th = Constant(0.25) # thickness
E = Constant(4.32e8)
nu = Constant(0.0)
areal_force_density = Constant(90.0)
f = as_vector([Constant(0.0), -areal_force_density, Constant(0.0)])
QoI_ref = 0.3006
num_srf = 9

angles = [50, 80, 100, 130]  # Central angles in degree
angle_lim_list = [angles[0:2], angles[1:3], angles[2:4]]*3
z_lims = [0, L/4, 3*L/4, L]
z_lim_list = [z_lims[0:2]]*3 + [z_lims[1:3]]*3 + [z_lims[2:4]]*3

penalty_coefficient = 1.0e3

if MPI.rank(worldcomm) == 0:
    print("Penalty coefficient:", penalty_coefficient)
num_el = 6
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

# Dirichlet BCs
bc0 = [1,0]
bc1 = [0,0]
bc2 = [0,1]
bcs_list = [bc0]*3 + [bc1]*3 + [bc2]*3

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

    if i == 0:
        splines += [create_spline(nurbs_srfs[i], BCs=bcs_list[i], 
            fix_z_node=True),]
    else:
        splines += [create_spline(nurbs_srfs[i], BCs=bcs_list[i]),]

if MPI.rank(worldcomm) == 0:
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
problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations, 
                            penalty_coefficient)
source_terms = []
residuals = []
for i in range(len(splines)):
    source_terms += [inner(f, problem.splines[i].rationalize(
    problem.spline_test_funcs[i]))*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]
problem.set_residuals(residuals)

if mpirank == 0:
    print("Solving linear non-matching problem...")
problem.solve_linear_nonmatching_problem(solver="direct")

# Check the quantity of interest on both sides
xi_list = [array([0.0, 0.5]), array([1.0, 0.5])]
spline_inds = [3,5]
for j in range(len(spline_inds)):
    xi = xi_list[j]
    disp_y_hom = eval_func(problem.splines[spline_inds[j]].mesh, 
                       problem.spline_funcs[spline_inds[j]][1], xi)
    w = eval_func(problem.splines[spline_inds[j]].mesh, 
                  problem.splines[spline_inds[j]].cpFuncs[3], xi)
    QoI_temp = -disp_y_hom/w
    if mpirank == 0:
        print("Quantity of interest for patch {} = {:10.8f} (Reference "
              "value = {}).".format(spline_inds[j], QoI_temp, QoI_ref))

# Compute von Mises stress
if mpirank == 0:
    print("Computing von Mises stresses...")
von_Mises_tops = []
von_Mises_bots = []
for i in range(problem.num_splines):
    spline_stress = ShellStressSVK(problem.splines[i], 
                                   problem.spline_funcs[i],
                                   E, nu, h_th, linearize=True)
    # von Mises stresses on top surfaces
    von_Mises_top = spline_stress.vonMisesStress(h_th/2)
    von_Mises_top_proj = problem.splines[i].projectScalarOntoLinears(
                            von_Mises_top, lumpMass=False)
    v2p(von_Mises_top_proj.vector()).ghostUpdate()
    von_Mises_tops += [von_Mises_top_proj]
    # von Mises stresses on bottom surfaces
    von_Mises_bot = spline_stress.vonMisesStress(-h_th/2)
    von_Mises_bot_proj = problem.splines[i].projectScalarOntoLinears(
                            von_Mises_bot, lumpMass=False)
    v2p(von_Mises_bot_proj.vector()).ghostUpdate()
    von_Mises_bots += [von_Mises_bot_proj]

SAVE_PATH = "./"
for i in range(problem.num_splines):
    save_results(problem.splines[i], problem.spline_funcs[i], i, 
                save_cpfuncs=True, save_path=SAVE_PATH, comm=problem.comm)
    von_Mises_tops[i].rename("von_Mises_top_"+str(i), 
                             "von_Mises_top_"+str(i))
    File(SAVE_PATH+"results/von_Mises_top_"+str(i)+".pvd") \
        << von_Mises_tops[i]
    von_Mises_bots[i].rename("von_Mises_bot_"+str(i), 
                             "von_Mises_bot_"+str(i))
    File(SAVE_PATH+"results/von_Mises_bot_"+str(i)+".pvd") \
        << von_Mises_bots[i]

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