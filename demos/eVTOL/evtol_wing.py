"""
The required eVTOL geometry can be downloaded from:

    https://drive.google.com/file/d/1xpY8ACQlodmwkUZsiEQvTZPcUu-uezgi/view?usp=sharing

and extracted using the command "tar -xvzf eVTOL_wing_structure.tgz".
"""
from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *

SAVE_PATH = "./"

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=2)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                        setBCs=None, side=0, direction=0, index=0):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

save_disp = True
save_stress = True
# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 5 m.
geom_scale = 2.54e-5  # Convert current length unit to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th = Constant(3.0e-3)  # Thickness of surfaces, m

p = 3  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "eVTOL_wing_structure.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
evtol_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

# Outer skin indices: list(range(12, 18))
# Spars indices: [78, 92, 79]
# Ribs indices: list(range(80, 92))
wing_indices = list(range(12, 18)) + [78, 92, 79]  + list(range(80, 92))
wing_surfaces = [evtol_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

num_pts_eval = [16]*num_surfs
u_insert_list = [8]*num_surfs
v_insert_list = [8]*num_surfs
ref_level_list = [1]*num_surfs
for i in [4,5]:
    if ref_level_list[i] > 4:
        ref_level_list[i] = 2
    elif ref_level_list[i] <=4 and ref_level_list[i] >= 1:
        ref_level_list[i] = 1

u_num_insert = []
v_num_insert = []
for i in range(len(u_insert_list)):
    u_num_insert += [ref_level_list[i]*u_insert_list[i]]
    v_num_insert += [ref_level_list[i]*v_insert_list[i]]

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(wing_surfaces, reparametrize=True, 
                                refine=True)
preprocessor.reparametrize_BSpline_surfaces(num_pts_eval, num_pts_eval,
                                            geom_scale=geom_scale,
                                            remove_dense_knots=True,
                                            rtol=1e-4)
preprocessor.refine_BSpline_surfaces(p, p, u_num_insert, v_num_insert, 
                                     correct_element_shape=True)
preprocessor.compute_intersections(mortar_refine=2)

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", preprocessor.num_intersections_all)

# # Display B-spline surfaces and intersections using 
# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# preprocessor.display_surfaces(display, save_fig=True)
# preprocessor.display_intersections(display, save_fig=True)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
for i in range(num_surfs):
    if i in [0, 1]:
        # Apply clamped BC to surfaces near root
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i], 
                 setBCs=clampedBC, side=0, direction=0)
        splines += [spline,]
    else:
        spline = OCCBSpline2tIGArSpline(
                 preprocessor.BSpline_surfs_refine[i])
        splines += [spline,]

# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
problem.create_mortar_meshes(preprocessor.mortar_nels)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)

if mpirank == 0:
    print("Setting up mortar meshes...")

problem.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

# Define magnitude of load
weight = 3000 # Take-off weight, kg
wing_vol = 0
for i in range(num_surfs):
    wing_vol += assemble(h_th*Constant(1.)*splines[i].dx)
load = Constant(weight/2/wing_vol)  # N/m^3
f1 = as_vector([Constant(0.0), Constant(0.0), load])

# Distributed downward load
loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    source_terms += [inner(loads[i], problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*h_th*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]
problem.set_residuals(residuals)

if mpirank == 0:
    print("Solving linear non-matching problem...")

problem.solve_linear_nonmatching_problem()

# print out vertical displacement at the tip of trailing edge
right_srf_ind = 3
xi = array([1, 1])
z_disp_hom = eval_func(problem.splines[right_srf_ind].mesh, 
                       problem.spline_funcs[right_srf_ind][2], xi)
w = eval_func(problem.splines[right_srf_ind].mesh, 
              problem.splines[right_srf_ind].cpFuncs[3], xi)
QoI = z_disp_hom/w

if mpirank == 0:
    print("Trailing edge tip vertical displacement: {:10.8f}.\n".format(QoI))

# Compute von Mises stress
if mpirank == 0:
    print("Computing von Mises stresses...")

von_Mises_tops = []
for i in range(problem.num_splines):
    spline_stress = ShellStressSVK(problem.splines[i], 
                                   problem.spline_funcs[i],
                                   E, nu, h_th, linearize=True, 
                                   G_det_min=5e-2)
    # von Mises stresses on top surfaces
    von_Mises_top = spline_stress.vonMisesStress(h_th/2)
    von_Mises_top_proj = problem.splines[i].projectScalarOntoLinears(
                         von_Mises_top, lumpMass=True)
    von_Mises_tops += [von_Mises_top_proj]

if mpirank == 0:
    print("Saving results...")

if save_disp:
    for i in range(problem.num_splines):
        save_results(splines[i], problem.spline_funcs[i], i, 
                     save_path=SAVE_PATH, folder="results/", 
                     save_cpfuncs=True, comm=worldcomm)
if save_stress:
    for i in range(problem.num_splines):
        von_Mises_tops[i].rename("von_Mises_top_"+str(i), 
                                 "von_Mises_top_"+str(i))
        File(SAVE_PATH+"results/von_Mises_top_"+str(i)+".pvd") \
            << von_Mises_tops[i]