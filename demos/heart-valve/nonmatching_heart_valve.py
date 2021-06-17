from os import path

from CouDALFISh import *
from tIGAr.BSplines import *
from VarMINT import *
from ShNAPr.SVK import *
from ShNAPr.contact import *

from PENGoLINS.occ_utils import *
from PENGoLINS.nonmatching_coupling import *

from create_nonmatching_leaflets import nonmatching_occ_bs, \
                                        nonmatching_nurbs_srfs

parameters["std_out_all_processes"] = False

SAVE_PATH = "./"
SAVE_PATH = "/home/han/Documents/test_results/"
RESTART_PATH = SAVE_PATH+"restarts/"
viz = True
out_skip = 1

def zero_bc(spline_generator, direction=0, side=0, n_layers=2):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, 
                                              nLayers=n_layers)
        spline_generator.addZeroDofs(field, side_dofs)

def ikNURBS2tIGArspline(ikNURBS, num_field=3, quad_deg_const=2, 
                        zero_bcs=None, direction=0, side=0,
                        zero_domain=None, fields=[0,1,2], index=0):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)
    quad_deg = ikNURBS.degree[0]*quad_deg_const
    if path.exists(DIR):
        spline = ExtractedSpline(DIR, quad_deg)
    else:
        spline_mesh = NURBSControlMesh(ikNURBS, useRect=False)
        spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)
        if zero_bcs is not None:
            zero_bcs(spline_generator, direction, side)
        if zero_domain is not None:
            for i in fields:
                spline_generator.addZeroDofsByLocation(zero_domain(), i)
        spline_generator.writeExtraction(DIR)
        spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

# Check if restarting
restarting = path.exists(RESTART_PATH+"step.dat")
if restarting:
    step_file = open(RESTART_PATH+"step.dat", "r")
    fs = step_file.read()
    step_file.close()
    tokens = fs.split()
    start_step = int(tokens[0])
    t = float(tokens[1])
    if mpirank == 0:
        print("Restarting ... \nt = "+str(t))
else:
    start_step = 0
    t = 0.

# Create heart valve patches
h_th = Constant(0.04)  # Thickness
E = Constant(1e7)
nu = Constant(0.4)
penalty_coefficient = 1.0e3

# Total numbers of control points for all patches
total_cp = 0
for i in range(len(nonmatching_nurbs_srfs)):
    for j in range(len(nonmatching_nurbs_srfs[i])):
        total_cp += nonmatching_nurbs_srfs[i][j].control.shape[0]\
                   *nonmatching_nurbs_srfs[i][j].control.shape[1]

if mpirank == 0:
    print("Total DoFs:", total_cp*3)
    print("Creating splines....")

ikNURBS_srfs = [ik_nurbs for ik_nurbs_i in nonmatching_nurbs_srfs \
                for ik_nurbs in ik_nurbs_i]
num_srfs = len(ikNURBS_srfs)
if mpirank == 0:
    print("Number of non-matching shell patches:", num_srfs)
splines = []
bcs_funcs = [zero_bc, zero_bc, None, zero_bc]*3
bcs = [[0,0], [0,1], [None, None], [1,1]]*3
for i in range(num_srfs):
    splines += [ikNURBS2tIGArspline(ikNURBS_srfs[i], zero_bcs=bcs_funcs[i], 
                                   direction=bcs[i][0], side=bcs[i][1], 
                                   index=i),]

# Compute physical locations of non-matching interfaces
if mpirank == 0:
    print("Computing non-matching interfaces...")
nonmatching_occ_bs_flat = [occ_bs for occ_bs_i in nonmatching_occ_bs \
                           for occ_bs in occ_bs_i]
mapping_list = []
intersection_curves = []
interface_phy_coords = []
interface_phy_coords_proj = []
for i in range(len(nonmatching_occ_bs_flat)):
    for j in range(i+1, len(nonmatching_occ_bs_flat) ):
        bs_intersect = BSplineSurfacesIntersections(
                       nonmatching_occ_bs_flat[i], 
                       nonmatching_occ_bs_flat[j], rtol=1e-4)
        if bs_intersect.num_intersections > 0:
            mapping_list += [[i, j],]*bs_intersect.num_intersections
            intersection_curves += bs_intersect.intersections
            intersection_coords = bs_intersect.intersection_coords(30)
            intersection_coords_proj = []
            for k in range(len(intersection_coords)):
                intersection_coords[k] = intersection_coords[k]
                intersection_coords_proj += [[project_locations_on_surface(
                                              intersection_coords[k],
                                              nonmatching_occ_bs_flat[i]), 
                                              project_locations_on_surface(
                                              intersection_coords[k],
                                              nonmatching_occ_bs_flat[j])],]
            interface_phy_coords += intersection_coords
            interface_phy_coords_proj += intersection_coords_proj
num_interfaces = len(mapping_list)

if mpirank == 0:
    print("Number of non-matching interfaces:", num_interfaces)

# Define non-matching problem
nonmatching_problem = NonMatchingCoupling(splines, E, h_th, nu, 
                                          comm=selfcomm)

mortar_nels = []
mortar_pts = []
for i in range(num_interfaces):
    mortar_nels += [np.max([np.max(ikNURBS_srfs[
                    mapping_list[i][0]].control.shape),
                    np.max(ikNURBS_srfs[
                    mapping_list[i][1]].control.shape)])*2,]
    mortar0_pts = np.array([[0.,0.],[0.,1.]])
    mortar_pts += [mortar0_pts,]
nonmatching_problem.create_mortar_meshes(mortar_nels, mortar_pts)
nonmatching_problem.create_mortar_funcs('CG',1)
nonmatching_problem.create_mortar_funcs_derivative('CG',1)

if mpirank == 0:
    print("Computing non-matching interface parametric locations...")
max_iter = 100
rtol = 1e-9
print_res = False
interp_phy_loc = False
r = 0.7
edge_tol = 1e-3
mortar_meshes_locations_newton = []
for i in range(num_interfaces):
    # print("interface index:", i)
    parametric_location0 = interface_parametric_location(
        nonmatching_problem.splines[mapping_list[i][0]], 
        nonmatching_problem.mortar_meshes[i], 
        interface_phy_coords_proj[i][0], max_iter=max_iter, rtol=rtol, 
        print_res=print_res, interp_phy_loc=interp_phy_loc, r=r, 
        edge_tol=edge_tol)
    parametric_location1 = interface_parametric_location(
        nonmatching_problem.splines[mapping_list[i][1]], 
        nonmatching_problem.mortar_meshes[i], 
        interface_phy_coords_proj[i][1], max_iter=max_iter, rtol=rtol, 
        print_res=print_res, interp_phy_loc=interp_phy_loc, r=r, 
        edge_tol=edge_tol)
    mortar_meshes_locations_newton += [[parametric_location0, 
                                        parametric_location1],]

nonmatching_problem.mortar_meshes_setup(mapping_list, 
    mortar_meshes_locations_newton, penalty_coefficient)

# Define contact context:
R_self = 0.045
r_max = 0.035
k_contact = 1e11

def phiPrime(r):
    if(r>r_max):
        return 0.0
    return -k_contact*(r_max-r)
def phiDoublePrime(r):
    if(r>r_max):
        return 0.0
    return k_contact
contactContext_sh = ShellContactContext(splines, R_self, r_max, 
                                        phiPrime, phiDoublePrime)

# # Create background fluid mesh
# from mshr import *
resolution = 70
CYLINDER_RAD = 1.1
BOTTOM = -0.5
TOP = 2.0
# tube = Cylinder(Point(0,0,BOTTOM),
#                 Point(0,0,TOP),CYLINDER_RAD,CYLINDER_RAD)

# SINUS_CENTER_FAC = 0.5
# SINUS_RAD_FAC = 0.8
# SINUS_Z_SHIFT = -0.2
# sinusRad = SINUS_RAD_FAC*CYLINDER_RAD
# sinusZ = sinusRad + SINUS_Z_SHIFT
# for i in range(0,3):
#     sinusTheta = math.pi/3.0 + i*2.0*math.pi/3.0
#     sinusCenterRad = SINUS_CENTER_FAC*CYLINDER_RAD
#     sinusCenter = Point(sinusCenterRad*math.cos(sinusTheta),
#                         sinusCenterRad*math.sin(sinusTheta),sinusZ)
#     tube += Sphere(sinusCenter,sinusRad)

# Cannot re-generate fluid mesh when restarting
if restarting or path.exists("mesh.h5"):  
    mesh = Mesh()
    f = HDF5File(worldcomm,"mesh.h5","r")
    f.read(mesh,"/mesh",True)
    f.close()
else:
    mesh = generate_mesh(tube, resolution)
    f = HDF5File(worldcomm,"mesh.h5","r")
    f.read(mesh,"/mesh",True)
    f.close()

# Print mesh statistics:
nel_f = mesh.num_entities_global(mesh.topology().dim())
nvert_f = mesh.num_entities_global(0)
if mpirank == 0:
    print("======= Fluid mesh information =======")
    print("  Number of elements in fluid mesh: "+str(nel_f))
    print("  Number of nodes in fluid mesh: "+str(nvert_f))

########### Time integration ############
# Time integrators for shells
rho_inf = Constant(0.0)
delta_t = Constant(1e-4)
total_time = 0.08
n_steps = int((total_time/float(delta_t)))

####### Fluid Formulation #######
# Set up VMS fluid problem using VarMINT:
VE = VectorElement("Lagrange",mesh.ufl_cell(),1)
QE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
VQE = MixedElement([VE,QE])
V_f = equalOrderSpace(mesh)
Vscalar = FunctionSpace(mesh,"Lagrange",1)
up = Function(V_f)
up_old = Function(V_f)
updot_old = Function(V_f)
vq = TestFunction(V_f)
time_int_f = GeneralizedAlphaIntegrator(rho_inf,delta_t,up,
                                        (up_old,updot_old),t)

# Define traction boundary condition at inflow:
xSpatial = SpatialCoordinate(mesh)
PRESSURE = Expression("((t<0.05)? 2e4 : -1e5)",t=0.0,degree=1)
inflowChar = conditional(lt(xSpatial[2],BOTTOM+1e-3),1.0,Constant(0.0))
# It's actually prefereable here to use a characteristic function instead
# of marking the inflow facets, because the "zero traction" BC at the outflow
# still requires stabilization terms.  As such, we can kill two birds with
# on stone by defining a spatially-varying traction like this and applying
# it on the full boundary with VarMINT's stable Neumann BC formulation.
inflowTraction = as_vector((0.0,0.0,PRESSURE))*inflowChar

def uPart(up):
    return as_vector([up[0],up[1],up[2]])

quadDeg = 2
dx_f = dx(metadata={"quadrature_degree":quadDeg})
ds_f = ds(metadata={"quadrature_degree":quadDeg})
rho = Constant(1.0)
mu = Constant(3e-2)
up_alpha = time_int_f.x_alpha()
u_alpha = uPart(up_alpha)
p = time_int_f.x[3]
v,q = split(vq)
up_t = time_int_f.xdot_alpha()
u_t = uPart(up_t)
cutFunc = Function(Vscalar)
stabEps = 1e-3
res_f = interiorResidual(u_alpha,p,v,q,rho,mu,mesh,u_t=u_t,Dt=delta_t,dx=dx_f,
                         stabScale=stabScale(cutFunc,stabEps))
n = FacetNormal(mesh)
res_f += stableNeumannBC(inflowTraction,rho,u_alpha,v,n,
                         ds=ds_f,gamma=Constant(1.0))
# The scaled radius criterion is somewhat sloppy, and is essentially to
# compensate for the fact that the Cylinder CSG primitive in mshr is faceted,
# so some vertices on the curved boundary of the cylinder will be at radii
# significantly (relative to machine precision) less than the nominal cylinder
# radius.
bcs_f = [DirichletBC(V_f.sub(0), Constant(d*(0.0,)),
                     (lambda x, on_boundary :
                      on_boundary and
                      math.sqrt(x[0]*x[0]+x[1]*x[1])>0.98*CYLINDER_RAD)),]

# Form to evaluate net inflow:
u = uPart(up)
netInflow = -inflowChar*dot(u,n)*ds_f

############ Shell formulation ##############
dens = Constant(1.)
y_old_hom_list = []
ydot_old_hom_list = []
yddot_old_hom_list = []
time_int_shs = []
y_alpha_list = []
ydot_alpha_list = []
yddot_alpha_list = []

for i in range(nonmatching_problem.num_splines):
    y_old_hom_list += [Function(nonmatching_problem.splines[i].V)]
    ydot_old_hom_list += [Function(nonmatching_problem.splines[i].V)]
    yddot_old_hom_list += [Function(nonmatching_problem.splines[i].V)]
    time_int_shs += [GeneralizedAlphaIntegrator(rho_inf, delta_t, 
        nonmatching_problem.spline_funcs[i], (y_old_hom_list[i], 
        ydot_old_hom_list[i], yddot_old_hom_list[i]), 
        t=t, useFirstOrderAlphaM=True),]
    y_alpha_list += [nonmatching_problem.splines[i].rationalize(
                     time_int_shs[i].x_alpha())]
    ydot_alpha_list += [nonmatching_problem.splines[i].rationalize(
                        time_int_shs[i].xdot_alpha())]
    yddot_alpha_list += [nonmatching_problem.splines[i].rationalize(
                        time_int_shs[i].xddot_alpha())]

res_shs = []
for i in range(num_srfs):
    dW = Constant(1./time_int_shs[i].ALPHA_F)\
         *SVK_residual(nonmatching_problem.splines[i],
         nonmatching_problem.spline_funcs[i], 
         nonmatching_problem.spline_test_funcs[i], E, nu, h_th, 0)
    dMass = dens*h_th*inner(yddot_alpha_list[i], 
            nonmatching_problem.spline_test_funcs[i])\
            *nonmatching_problem.splines[i].dx
    res_shs += [dW+dMass]

# Linear solver for fluid sub-problem
# Linear solver settings for the fluid: The nonlinear solver typically
# converges quite well, even if the fluid linear solver is not converging.
# This is the main "trick" needed to make scaling of SUPG/LSIC parameters
# for mass conservation tractable in 3D problems.  
fluidLinearSolver = PETScKrylovSolver("gmres","jacobi")
fluidLinearSolver.parameters["error_on_nonconvergence"] = False
fluidLinearSolver.ksp().setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
fluidKSPrtol = 1e-2
maxKSPIt = 300
fluidLinearSolver.ksp().setTolerances(rtol=fluidKSPrtol,max_it=maxKSPIt)
fluidLinearSolver.ksp().setGMRESRestart(maxKSPIt)

DAL_penalty = 5e2
DAL_r = 1e-5
blockItTol = 1e-2
# Define FSI problem with CouDALFISn
fsi_problem = CouDALFISh(mesh,res_f,time_int_f,
                         splines,res_shs,time_int_shs,
                         DAL_penalty,r=DAL_r,
                         bcs_f=bcs_f,
                         blockItTol=blockItTol,
                         contactContext_sh=contactContext_sh,
                         nonmatching_sh=nonmatching_problem,
                         fluidLinearSolver=fluidLinearSolver,
                         cutFunc=cutFunc)

# Creat files to store computed data
if mpirank == 0:
    u_file_names = []
    u_files = []
    F_file_names = []
    F_files = []
    for i in range(num_srfs):
        u_file_names += [[],]
        u_files += [[],]
        F_file_names += [[],]
        F_files += [[],]
        for j in range(3):
            # For shell patches' displacement
            u_file_names[i] += [SAVE_PATH+"results/"+"u"+str(i)\
                                +"_"+str(j)+"_file.pvd",]
            u_files[i] += [File(nonmatching_problem.comm, 
                                u_file_names[i][j]),]
            # For shell patches' initial configuration
            F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)\
                                +"_"+str(j)+"_file.pvd",]
            F_files[i] += [File(nonmatching_problem.comm, 
                                F_file_names[i][j]),]
            if j == 2:
                # For shell patches' weights
                F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)\
                                    +"_3_file.pvd",]
                F_files[i] += [File(nonmatching_problem.comm, 
                                    F_file_names[i][3]),]
# For fluid velocity and pressure
v_file_name = SAVE_PATH+"results/"+"v"+"_file.pvd"
v_file = File(v_file_name)
p_file_name = SAVE_PATH+"results/"+"p"+"_file.pvd"
p_file = File(p_file_name)
output_file_name = "flow-rate"

# Initial conditions
if restarting:
    fsi_problem.readRestarts(RESTART_PATH, start_step)

# Time stepping loop
for time_step in range(start_step, n_steps):

    PRESSURE.t = time_int_f.t-(1.0-float(time_int_f.ALPHA_M))*float(delta_t)
    if mpirank == 0:
        print("------- Time step "+str(time_step+1)
              +" , t = "+str(time_int_f.t)+" -------")
        
    if time_step%out_skip == 0:

        # Restart data
        fsi_problem.writeRestarts(RESTART_PATH, time_step)

        if mpirank == 0:
            step_file = open(RESTART_PATH+"step.dat", "w")
            step_file.write(str(time_step)+" "\
                            +str(time_int_f.t-float(delta_t)))
            step_file.close()

        if viz:
            # Save initial zero solution
            if time_step%out_skip == 0:
                if mpirank == 0:
                    # Structure
                    for i in range(num_srfs):
                        soln_split = nonmatching_problem.\
                                     spline_funcs[i].split()
                        for j in range(3):
                            soln_split[j].rename("u"+str(i)+"_"+str(j), 
                                                 "u"+str(i)+"_"+str(j))
                            u_files[i][j] << soln_split[j]
                            nonmatching_problem.splines[i].cpFuncs[j].rename(
                                "F"+str(i)+"_"+str(j), "F"+str(i)+"_"+str(j))
                            F_files[i][j] << nonmatching_problem.splines[i].\
                                             cpFuncs[j]
                            if j == 2:
                                nonmatching_problem.splines[i].cpFuncs[3].\
                                    rename("F"+str(i)+"_3", "F"+str(i)+"_3")
                                F_files[i][3] << nonmatching_problem.\
                                                 splines[i].cpFuncs[3]
                # Fluid
                (vout, pout) = up.split()
                vout.rename("v", "v")
                pout.rename("p", "p")
                v_file << vout
                p_file << pout

    fsi_problem.takeStep()

    flow_rate = assemble(netInflow)
    if mpirank == 0:
        mode = "a"
        if time_step == 0:
            mode = "w"
        outFile = open(SAVE_PATH+output_file_name,mode)
        outFile.write(str(time_int_f.t)+" "+str(flow_rate)+"\n")
        outFile.close()