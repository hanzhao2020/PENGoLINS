"""
The required eVTOL geometry can be downloaded form:

    https://drive.google.com/file/d/1xpY8ACQlodmwkUZsiEQvTZPcUu-uezgi/view?usp=sharing

and extracted using the command "tar -xvzf eVTOL.tgz".
"""

from PENGoLINS.nonmatching_coupling import *
from igakit.io import VTK

SAVE_PATH = "./"

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=2)
        spline_generator.addZeroDofs(field, side_dofs)

def ikNURBS2tIGArspline(ikNURBS, num_field=3, quad_deg_const=4, 
                        setBCs=None, side=0, direction=0, index=0):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    quad_deg = ikNURBS.degree[0]*quad_deg_const
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)

    spline_mesh = NURBSControlMesh(ikNURBS, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 5 m.
geom_scale = 2.54e-5 # 1.e-3  # Convert mm to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.35)  # Poisson's ratio
h_th = Constant(3.0e-3)  # Thickness of surfaces, m

p = 3  # spline order
num_field = 3
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "eVTOL_wing_structure.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
evtol_surfaces = [topoface2surface(face, BSpline=True) 
                  for face in igs_shapes]

# Outer skin indices: list(range(12, 18))
# Spars indices: [78, 92, 79]
# Ribs indices: list(range(80, 92))
wing_indices = list(range(12, 18)) + [78, 92, 79] + list(range(80, 92))
wing_surfaces = []
for ind in wing_indices:
    wing_surfaces += [evtol_surfaces[ind],]

num_srfs = len(wing_surfaces)
print("Number of surfaces:", num_srfs)

num_pts_eval = [16]*num_srfs
# Knots insertion for outer skin; spars; ribs
u_insert_list = [16, 15, 14, 13, 1, 1] \
              + [16, 17, 18] + [4]*12  
v_insert_list = [8, 7, 6, 5, 12, 11] \
              + [1]*3 + [1]*12
ref_level = 1

u_num_insert = [i*ref_level for i in u_insert_list]
v_num_insert = [i*ref_level for i in v_insert_list]

reconstructed_srfs = []
ikNURBS_srfs = []
for i in range(num_srfs):
    recon_bs_surface = reconstruct_BSpline_surface(wing_surfaces[i], 
                       u_num_eval=num_pts_eval[i], v_num_eval=num_pts_eval[i], 
                       tol3D=1e-3, geom_scale=geom_scale)
    reconstructed_srfs += [recon_bs_surface]
    ikNURBS_srfs += [BSpline_surface2ikNURBS(recon_bs_surface, p=p, 
                      u_num_insert=u_num_insert[i], 
                      v_num_insert=v_num_insert[i]),]

# # Save igakit NURBS to vtk format
# for i in range(num_srfs):
#     ik_srfs_filenames = "evtol_wing"+str(i)+".vtk"
#     VTK().write(SAVE_PATH+"evtol_srfs/"+ik_srfs_filenames, ikNURBS_srfs[i])

total_cp = 0
for i in range(num_srfs):
    total_cp += ikNURBS_srfs[i].control.shape[0]\
               *ikNURBS_srfs[i].control.shape[1]
print("Total DoFs:", total_cp*3)

print("Computing non-matching interfaces...")
mapping_list = []
mortar_nels = [] # Number of elements for mortar meshes
intersection_curves = [] # List of intersection curves
intersections_para_coords = [] 
for i in range(num_srfs):
    for j in range(i+1, num_srfs):
        bs_intersect = BSplineSurfacesIntersections(reconstructed_srfs[i], 
                                                    reconstructed_srfs[j], 
                                                    rtol=1e-4)
        if bs_intersect.num_intersections > 0:
            mapping_list += [[i, j],]*bs_intersect.num_intersections
            mortar_nels += [np.max([np.max(ikNURBS_srfs[i].control.shape), 
                            np.max(ikNURBS_srfs[j].control.shape)])*2,]\
                           *bs_intersect.num_intersections
            intersection_curves += bs_intersect.intersections
            intersections_para_coords += \
                bs_intersect.get_parametric_coordinates(
                    num_pts=int((mortar_nels[-1])*1.1))

num_interfaces = len(mapping_list)
print("Number of non-matching interfaces:", num_interfaces)

# # Display the surfaces and non-matching interfaces
# display, start_display, add_menu, add_function_to_menu = init_display()
# for i in range(num_srfs):
#     display.DisplayShape(reconstructed_srfs[i])

# display, start_display, add_menu, add_function_to_menu = init_display()
# for i in range(len(intersection_curves)):
#     display.DisplayShape(intersection_curves[i], color='BLUE')

print("Creating splines...")
splines = []
for i in range(num_srfs):
    if i in [0, 1]:
        # Apply clamped BCs to outer surfaces
        spline = ikNURBS2tIGArspline(ikNURBS_srfs[i], setBCs=clampedBC, 
                 side=0, direction=0)
        splines += [spline,]
    else:
        spline = ikNURBS2tIGArspline(ikNURBS_srfs[i])
        splines += [spline,]

# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th, nu, comm=selfcomm)
problem.create_mortar_meshes(mortar_nels)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)

print("Setting up mortar meshes...")
problem.mortar_meshes_setup(mapping_list, intersections_para_coords, 
                            penalty_coefficient)

# Define magnitude of load
weight = 3000 # kg
wing_vol = 0
for i in range(num_srfs):
    wing_vol += assemble(h_th*Constant(1.)*splines[i].dx)
load = Constant(weight/2/wing_vol)  # N/m^3
f1 = as_vector([Constant(0.0), Constant(0.0), load])

# Distributed downward load
loads = [f1]*num_srfs
source_terms = []
residuals = []
for i in range(len(splines)):
    source_terms += [inner(loads[i], problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*h_th*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]
problem.set_residuals(residuals)

print("Solving linear non-matching problem...")
problem.solve_linear_nonmatching_problem()

# print out vertical displacement at the tip of trailing edge
right_srf_ind = 3
xi = array([1, 1])
QoI = problem.spline_funcs[right_srf_ind](xi)[2]\
      /problem.splines[right_srf_ind].cpFuncs[3](xi)
print("Trailing edge tip vertical displacement: {:10.8f}.\n".format(QoI))

# Compute von Mises stress
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
                            von_Mises_top, lumpMass=True)
    von_Mises_tops += [von_Mises_top_proj]
    # von Mises stresses on bottom surfaces
    von_Mises_bot = spline_stress.vonMisesStress(-h_th/2)
    von_Mises_bot_proj = problem.splines[i].projectScalarOntoLinears(
                            von_Mises_bot, lumpMass=True)
    von_Mises_bots += [von_Mises_bot_proj]

print("Saving results...")
for i in range(problem.num_splines):
    save_results(splines[i], problem.spline_funcs[i], i, 
                 save_path=SAVE_PATH, folder="results/", 
                 save_cpfuncs=True, comm=selfcomm)

    von_Mises_tops[i].rename("von_Mises_top_"+str(i), 
                             "von_Mises_top_"+str(i))
    File(SAVE_PATH+"results/von_Mises_top_"+str(i)+".pvd") \
        << von_Mises_tops[i]

    von_Mises_bots[i].rename("von_Mises_bot_"+str(i), 
                             "von_Mises_bot_"+str(i))
    File(SAVE_PATH+"results/von_Mises_bot_"+str(i)+".pvd") \
        << von_Mises_bots[i]