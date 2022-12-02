from os import path
from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *

parameters["std_out_all_processes"] = False

SAVE_PATH = "./"

def zero_bc(spline_generator, direction=0, side=0, n_layers=2):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, 
                                              nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=2, 
                           zero_bcs=None, direction=0, side=0,
                           zero_domain=None, fields=[0,1,2], index=0):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)
    quad_deg = surface.UDegree()*quad_deg_const
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)
    if zero_bcs is not None:
        zero_bcs(spline_generator, direction, side)
    if zero_domain is not None:
        for i in fields:
            spline_generator.addZeroDofsByLocation(zero_domain(), i)
    # spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Define material and geometric parameters 
h_th = Constant(0.04)  # Thickness
E = Constant(1e7)
nu = Constant(0.4)
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "./nonmatching_bicuspid.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)

bicuspid_leaflets = [topoface2surface(face, BSpline=True) 
                     for face in igs_shapes]
num_surfs = len(bicuspid_leaflets)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

# Geometry preprocessing and surface-surface intersections computation
preprocessor = OCCPreprocessing(bicuspid_leaflets, 
                                reparametrize=False, refine=False)
print("Computing intersections...")
preprocessor.compute_intersections(rtol=1e-1, mortar_refine=2)
num_interfaces = preprocessor.num_intersections_all

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", num_interfaces)

# # Display bicuspid valve leaflets and intersections
# display, _, _, _ = init_display()
# preprocessor.display_surfaces(display, transparency=0.4, show_bdry=False)
# preprocessor.display_intersections(display, color='RED', show_triedron=True)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
bcs_funcs = [zero_bc,]*3 + [None,]*4
bcs = [[1,0],]*3 + [[None, None]]*4
for i in range(num_surfs):
    splines += [OCCBSpline2tIGArSpline(bicuspid_leaflets[i], 
                                       zero_bcs=bcs_funcs[i], 
                                       direction=bcs[i][0], 
                                       side=bcs[i][1], index=i),]

# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
problem.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

problem.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)
source_terms = []
residuals = []
pressure = Constant(1e3)

# Follower pressure
for i in range(len(splines)):
    A0,A1,A2,dA2,A,B = surfaceGeometry(problem.splines[i], 
                                       problem.splines[i].F)
    a0,a1,a2,da2,a,b = surfaceGeometry(problem.splines[i], 
                       problem.splines[i].F+problem.spline_funcs[i])
    source_terms += [(pressure)*sqrt(det(a)/det(A))\
                     *inner(a2,problem.splines[i].rationalize(
                     problem.spline_test_funcs[i]))*problem.splines[i].dx,]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]

# # Distributed upward load
# f1 = as_vector([Constant(0.), Constant(0.), pressure])
# loads = [f1]*num_surfs
# source_terms = []
# residuals = []
# for i in range(num_surfs):
#     source_terms += [inner(loads[i], problem.splines[i].rationalize(
#         problem.spline_test_funcs[i]))*h_th*problem.splines[i].dx]
#     residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
#         problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]

problem.set_residuals(residuals)

if mpirank == 0:
    print("Solving linear non-matching problem...")
problem.solve_nonlinear_nonmatching_problem(rtol=1e-2, max_it=100)

print("Saving results...")
for i in range(len(splines)):
    save_results(splines[i], problem.spline_funcs[i], i, 
                 save_cpfuncs=True, save_path=SAVE_PATH)