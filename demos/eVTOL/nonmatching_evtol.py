"""
The required eVTOL geometry can be downloaded form:

    https://drive.google.com/file/d/1YO6UDztRqY4r2u7CnshALAUO93iPaa46/view

and extracted using the command "tar -xvzf eVTOL.tgz".
"""

from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.occ_utils import *

SAVE_PATH = "./"
# SAVE_PATH = "/home/han/Documents/test_results/"

def zero_bc(spline_generator, direction=0, side=0, n_layers=2):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, 
                                              nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def ikNURBS2tIGArspline(ikNURBS, num_field=3, quad_deg_const=4, 
                        zero_bcs=None, direction=0, side=0,
                        zero_domain=None, fields=[0,1,2], index=0,):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    quad_deg = ikNURBS.degree[0]*quad_deg_const
    spline_mesh = NURBSControlMesh(ikNURBS, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)
    if zero_bcs is not None:
        zero_bcs(spline_generator, direction, side)
    if zero_domain is not None:
        for i in fields:
            spline_generator.addZeroDofsByLocation(zero_domain(), i)
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)
    spline_generator.writeExtraction(DIR)
    # spline = ExtractedSpline(DIR, quad_deg)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Define parameters
# Scale down the geometry using ``geom_scale`` to make the length 
# of the wing in the span-wise direction is around 5 m.
geom_scale = 1.e-3
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
filename_stp = "eVTOL.stp"
stp_shapes = read_stp_file(filename_stp, as_compound=False)
evtol_surfaces = [topoface2surface(face, bspline=True) \
                    for face in stp_shapes]

evtol_indices = [
                 0, 1, 2, 3, 4, 5, 6, 7,  # Fuselage
                 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  # Right wing
                 20, 21, 22, 23, 24, 25, 26, 27,  # Left wing
                 # 28, 29, 30, 31, 32, 33, 34, 35,  # Right inner nacelle
                 # 36, 37, 38, 39, 40, 41, 42, 43,  # Left inner nacelle
                 # 44, 45, 46, 47, 48, 49, 50, 51,  # Right outer nacelle
                 # 52, 53, 54, 55, 56, 57, 58, 59,  # Left outer nacelle
                 60, 61, 62, 63, 64, 65,  # Vertical tail
                 66, 67, 68, 69, 70, 71,  # Right tail
                 74, 75, 76, 77,  # Left tail
                 ]

evtol_srfs_test = [evtol_surfaces[i] for i in evtol_indices]
num_srfs = len(evtol_srfs_test)
print("Number of surfaces:", num_srfs)

# # Display the surfaces
# display, start_display, add_menu, add_function_to_menu = init_display()
# for i in range(len(evtol_indices)):
# # for i in range(4,8):
#     display.DisplayShape(evtol_srfs_test[i])

bs_ind = list(range(num_srfs))
num_pts_eval = [22]*len(bs_ind)

min_insert = 16
iter_num = 8
u_num_insert = list(range(min_insert+iter_num, min_insert, -1)) \
             *(num_srfs//iter_num) + list(range(min_insert, min_insert \
             + num_srfs%iter_num))
v_num_insert = list(range(min_insert+iter_num, min_insert, -1)) \
             *(num_srfs//iter_num) + list(range(min_insert, min_insert \
             + num_srfs%iter_num))
u_ref_level = 1.
v_ref_level = 1.
u_num_insert = [int(i*u_ref_level) for i in u_num_insert]
v_num_insert = [int(i*v_ref_level) for i in v_num_insert]

reconstructed_srfs = []
ikNURBS_srfs = []
for i in range(num_srfs):
    recon_bs_surface = reconstruct_occ_Bspline_surface(
                       evtol_srfs_test[bs_ind[i]], u_num_eval=num_pts_eval[i], 
                       v_num_eval=num_pts_eval[i], tol3D=1e-3, 
                       geom_scale=geom_scale)
    reconstructed_srfs += [recon_bs_surface,]
    ikNURBS_srfs += [Bspline_surface2ikNURBS_refine(recon_bs_surface, p, 
                      u_num_insert=u_num_insert[i], 
                      v_num_insert=v_num_insert[i]),]

total_cp = 0
for i in range(len(ikNURBS_srfs)):
    total_cp += ikNURBS_srfs[i].control.shape[0]\
               *ikNURBS_srfs[i].control.shape[1]
print("Total DoFs:", total_cp*3)

# for i in range(num_srfs):
#     ik_srfs_filenames = "evtol_srfs"+str(i)+".vtk"
#     VTK().write(SAVE_PATH+"evtol_srfs/"+ik_srfs_filenames, ikNURBS_srfs[i])
# exit()

print("Computing non-matching interfaces...")
mapping_list = []
intersection_curves = []
interface_phy_coords = []
interface_phy_coords_proj = []
for i in range(num_srfs):
    for j in range(i+1, num_srfs):
        bs_intersect = BSplineSurfacesIntersections(reconstructed_srfs[i], 
                                                    reconstructed_srfs[j], 
                                                    rtol=1e-4)
        if bs_intersect.num_intersections > 0:
            mapping_list += [[i, j],]*bs_intersect.num_intersections
            intersection_curves += bs_intersect.intersections
            intersection_coords = bs_intersect.intersection_coords(30)
            intersection_coords_proj = []
            for k in range(len(intersection_coords)):
                intersection_coords[k] = intersection_coords[k]
                intersection_coords_proj += [[project_locations_on_surface(
                                              intersection_coords[k],
                                              reconstructed_srfs[i]), 
                                              project_locations_on_surface(
                                              intersection_coords[k],
                                              reconstructed_srfs[j])],]
            interface_phy_coords += intersection_coords
            interface_phy_coords_proj += intersection_coords_proj

num_interfaces = len(mapping_list)
print("Number of non-matching interfaces:", num_interfaces)

# # # Display the non-matching interfaces
# display, start_display, add_menu, add_function_to_menu = init_display()
# # display.SetBackgroundImage("./white_background.jpeg", stretch=True)
# # for i in range(len(evtol_indices)):
# #     display.DisplayShape(reconstructed_srfs[i])
# for i in range(len(intersection_curves)):
#     display.DisplayShape(intersection_curves[i], color='BLUE')

ikNURBS0_cp = ikNURBS_srfs[0].control
value_to_fix = DOLFIN_EPS*6e13 + np.min(ikNURBS0_cp[:,:,2])

class zero_domain(SubDomain):
    def inside(self, x, on_boundary):
        return x[2] < value_to_fix

print("Creating splines....")
splines = []
bc_indices = [0, 1]
for i in range(num_srfs):
    if i in bc_indices:
        spline = ikNURBS2tIGArspline(ikNURBS_srfs[i], #zero_bcs=zero_bc, 
                                     # direction=0, side=1, index=i)
                                     zero_domain=zero_domain, index=i)
    else:
        spline = ikNURBS2tIGArspline(ikNURBS_srfs[i], index=i)
    splines += [spline]

################## Creating non-matching problem #########################
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
edge_tol = 1e-3
mortar_meshes_locations_newton = []
for i in range(num_interfaces):
    # print("Non-matching interface index:", i)
    # print("Side 0")
    parametric_location0 = interface_parametric_location(
        problem.splines[mapping_list[i][0]], problem.mortar_meshes[i], 
        interface_phy_coords_proj[i][0], max_iter=max_iter, rtol=rtol, 
        print_res=print_res, interp_phy_loc=interp_phy_loc, r=r, 
        edge_tol=edge_tol)
    # print("Side 1")
    parametric_location1 = interface_parametric_location(
        problem.splines[mapping_list[i][1]], problem.mortar_meshes[i], 
        interface_phy_coords_proj[i][1], max_iter=max_iter, rtol=rtol, 
        print_res=print_res, interp_phy_loc=interp_phy_loc, r=r, 
        edge_tol=edge_tol)
    mortar_meshes_locations_newton += [[parametric_location0, 
                                        parametric_location1],]

print("Setting up mortar meshes...")
problem.mortar_meshes_setup(mapping_list, mortar_meshes_locations_newton,
                            penalty_coefficient)

print("Setting up splines...")
# Distributive downward load
loads = [f1]*num_srfs
source_terms = []
residuals = []
for i in range(len(splines)):
    source_terms += [inner(loads[i], problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]
problem.set_residuals(residuals)

print("Solving linear non-matching system...")
problem.solve_linear_nonmatching_system()

print("Saving results...")
for i in range(len(splines)):
    save_results(splines[i], problem.spline_funcs[i], i, 
                 save_path=SAVE_PATH, folder="results_temp0/", 
                 save_cpfuncs=True, comm=problem.comm)

right_srf_ind = 0
xi = np.array([1, 1])
QoI = -problem.spline_funcs[right_srf_ind](xi)[2]\
    /problem.splines[right_srf_ind].cpFuncs[3](xi)
print("Quantity of interest: {:10.8f}.".format(QoI))