"""
The required eVTOL geometry can be downloaded form:

    https://drive.google.com/file/d/1YO6UDztRqY4r2u7CnshALAUO93iPaa46/view

and extracted using the command "tar -xvzf eVTOL.tgz".
"""

from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.occ_utils import *

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
                        setBCs=None, index=0):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    quad_deg = ikNURBS.degree[0]*quad_deg_const
    spline_mesh = NURBSControlMesh(ikNURBS, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator)
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction is around 5 m.
geom_scale = 1.e-3  # Convert mm to m
E = Constant(68e9)  # Young's modulus, Pa
nu = Constant(0.0)  # Poisson's ratio
h_th = Constant(3.0e-3)  # Thickness of surfaces, m
arealForceDensity = Constant(1.e2)  # N/m^2, Pa
f0 = as_vector([Constant(0.0), Constant(0.0), Constant(0.0)])
f1 = as_vector([Constant(0.0), Constant(0.0), -arealForceDensity])

p = 3  # spline order
num_field = 3
penalty_coefficient = 1.0e3

print("Importing geometry...")
# For stp format geometry
filename_stp = "eVTOL.stp"
stp_shapes = read_stp_file(filename_stp, as_compound=False)
evtol_surfaces = [topoface2surface(face, bspline=True) 
                  for face in stp_shapes]

# # For igs format geometry
# filename_igs = "eVTOL_in.igs"
# igs_shapes = read_igs_file(filename_igs, as_compound=False)
# evtol_surfaces = [topoface2surface(face, bspline=True) 
#                   for face in igs_shapes]

wing_surfaces = evtol_surfaces[10:18]  # 10:18 are the indices of wing
num_srfs = len(wing_surfaces)
print("Number of surfaces:", num_srfs)

bs_ind = list(range(num_srfs))
num_pts_eval = [16]*num_srfs
num_insert_min = 24
u_num_insert = list(range(num_insert_min+num_srfs, num_insert_min,-1))
v_num_insert = list(range(num_insert_min+num_srfs, num_insert_min,-1))
u_ref_level = 1
v_ref_level = 1
u_num_insert = [int(i*u_ref_level) for i in u_num_insert]
v_num_insert = [int(i*v_ref_level) for i in v_num_insert]

# # Insert random different numbers of knots between ``insert_min`` and 
# # ``insert_max`` to make the surfaces unmatched
# u_num_insert = []
# v_num_insert = []
# insert_min = 20
# insert_max = 40
# for i in range(len(bs_ind)):
#     u_temp = np.random.randint(insert_min, insert_max)
#     v_temp = np.random.randint(insert_min, insert_max)
#     while u_temp in u_num_insert:
#         u_temp = np.random.randint(insert_min, insert_max)
#     u_num_insert += [u_temp,]
#     while v_temp in v_num_insert:
#         v_temp = np.random.randint(insert_min, insert_max)
#     v_num_insert += [v_temp]

reconstructed_srfs = []
ikNURBS_srfs = []
for i in range(num_srfs):
    recon_bs_surface = reconstruct_occ_Bspline_surface(
                       wing_surfaces[bs_ind[i]], u_num_eval=num_pts_eval[i], 
                       v_num_eval=num_pts_eval[i], tol3D=1e-3, 
                       geom_scale=geom_scale)
    reconstructed_srfs += [recon_bs_surface,]
    ikNURBS_srfs += [Bspline_surface2ikNURBS_refine(recon_bs_surface, p, 
                      u_num_insert=u_num_insert[i], 
                      v_num_insert=v_num_insert[i]),]
    num_row, num_col, _ = ikNURBS_srfs[i].control.shape
    control_w = ikNURBS_srfs[i].control
    control_coord = np.zeros((num_row, num_col, 3))
    for m in range(num_row):
        for n in range(num_col):
            control_coord[m,n,:] = control_w[m,n,:3]/control_w[m,n,3]

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
intersection_curves = []
interface_phy_coords = []
interface_phy_coords_proj = []
for i in range(num_srfs):
    for j in range(i+1, num_srfs):
        bs_connect = BSplineSurfacesConnectedEdges(reconstructed_srfs[i], 
                                                   reconstructed_srfs[j])
        if bs_connect.num_connected_edges > 0:
            mapping_list += [[i, j],]*bs_connect.num_connected_edges
            intersection_curves += bs_connect.connected_edges
            connected_edge_coords = bs_connect.connected_edge_coords(20)
            connected_edge_coords_proj = []
            for k in range(len(connected_edge_coords)):
                connected_edge_coords[k] = connected_edge_coords[k]
                connected_edge_coords_proj += [[project_locations_on_surface(
                                                connected_edge_coords[k],
                                                reconstructed_srfs[i]), 
                                                project_locations_on_surface(
                                                connected_edge_coords[k],
                                                reconstructed_srfs[j])],]
            interface_phy_coords += connected_edge_coords
            interface_phy_coords_proj += connected_edge_coords_proj

num_interfaces = len(mapping_list)
print("Number of non-matching interfaces:", num_interfaces)

# # Display the non-matching interfaces
# display, start_display, add_menu, add_function_to_menu = init_display()
# # for i in range(len(evtol_indices)):
# #     display.DisplayShape(reconstructed_srfs[i])
# for i in range(len(intersection_curves)):
#     display.DisplayShape(intersection_curves[i], color='BLUE')

print("Creating splines...")
splines = []
for i in range(num_srfs):
    if i < 2:
        spline = ikNURBS2tIGArspline(ikNURBS_srfs[i], setBCs=clampedBC)
        splines += [spline,]
    else:
        spline = ikNURBS2tIGArspline(ikNURBS_srfs[i])
        splines += [spline,]

problem = NonMatchingCoupling(splines, E, h_th, nu, comm=selfcomm)
mortar_nels = []
mortar_pts = []
for i in range(num_interfaces):
    mortar0_pts = np.array([[0.,0.],[0.,1.]])
    mortar_pts += [mortar0_pts,]
    mortar_nels += [np.max([np.max(ikNURBS_srfs[mapping_list[i][0]].control.shape),
                    np.max(ikNURBS_srfs[mapping_list[i][1]].control.shape)])*2,]

problem.create_mortar_meshes(mortar_nels, mortar_pts)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)

print("Computing non-matching interface parametric locations...")
max_iter = 100
rtol = 1e-9
print_res = False
interp_phy_loc = False
r = 0.7
edge_tol = 1e-4
mortar_meshes_locations_newton = []

for i in range(num_interfaces):
    # print("physical_locations index:", i)
    parametric_location0 = interface_parametric_location(
        problem.splines[mapping_list[i][0]], problem.mortar_meshes[i], 
        interface_phy_coords_proj[i][0], max_iter=max_iter, rtol=rtol, 
        print_res=print_res, interp_phy_loc=interp_phy_loc, r=r, 
        edge_tol=edge_tol)
    parametric_location1 = interface_parametric_location(
        problem.splines[mapping_list[i][1]], problem.mortar_meshes[i], 
        interface_phy_coords_proj[i][1], max_iter=max_iter, rtol=rtol, 
        print_res=print_res, interp_phy_loc=interp_phy_loc, r=r, 
        edge_tol=edge_tol)
    mortar_meshes_locations_newton += \
        [[parametric_location0, parametric_location1],]

print("Setting up mortar meshes...")
problem.mortar_meshes_setup(mapping_list, mortar_meshes_locations_newton, 
                            penalty_coefficient)

# Distributive downward load
loads = [f1]*len(bs_ind)
right_srf_ind = 1
source_terms = []
residuals = []
for i in range(len(splines)):
    source_terms += [inner(loads[i], problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]

# ps0 = PointSource(problem.splines[right_srf_ind].V.sub(2), 
#                   Point(1.,1.), -tip_load)
problem.set_residuals(residuals) #, [ps0,], [right_srf_ind,])

print("Solving linear non-matching system...")
problem.solve_linear_nonmatching_system()

print("Saving results...")
for i in range(len(splines)):
    save_results(splines[i], problem.spline_funcs[i], i, 
                 save_path=SAVE_PATH, folder="results/", 
                 save_cpfuncs=True, comm=selfcomm)
    
xi = array([1, 1])
QoI = -problem.spline_funcs[right_srf_ind](xi)[2]\
      /problem.splines[right_srf_ind].cpFuncs[3](xi)
print("Bottom right vertical displacement: {:10.8f}.\n".format(QoI))