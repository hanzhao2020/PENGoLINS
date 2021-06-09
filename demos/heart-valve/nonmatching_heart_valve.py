from os import path
import sys
sys.path.append("./")

from tIGAr.BSplines import *
from CouDALFISh import *
from VarMINT import *
from ShNAPr.SVK import *

from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.occ_utils import *
from PENGoLINS.contact import *

from create_nonmatching_leaflets import nonmatching_occ_bs, \
                                        nonmatching_nurbs_srfs

# from CouDALFISh_shells import *
from PENGoLINS.FSI_coupling import *

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

def ikNURBS2tIGArspline(ikNURBS, num_field=3, quad_deg_const=4, 
                        zero_bcs=None, direction=0, side=0,
                        zero_domain=None, fields=[0,1,2], index=0,):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)
    quad_deg = ikNURBS.degree[0]*quad_deg_const

    spline_mesh = NURBSControlMesh(ikNURBS, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)
    if zero_bcs is not None:
        zero_bcs(spline_generator, direction, side)
    if zero_domain is not None:
        for i in fields:
            spline_generator.addZeroDofsByLocation(zero_domain(), i)
    spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)

    # spline = ExtractedSpline(DIR, quad_deg)
    return spline

# def main():

#### Leaflet shell patches ####
h_th = Constant(0.04)  # Thickness
E = Constant(1e7)
nu = Constant(0.4)
penalty_coefficient = 1.0e3

total_cp = 0
for i in range(len(nonmatching_nurbs_srfs)):
    for j in range(len(nonmatching_nurbs_srfs[i])):
        total_cp += nonmatching_nurbs_srfs[i][j].control.shape[0]\
                   *nonmatching_nurbs_srfs[i][j].control.shape[1]
print("Total DoFs:", total_cp*3)

nonmatching_occ_bs_flat = [occ_bs for occ_bs_i in nonmatching_occ_bs \
                           for occ_bs in occ_bs_i]
print("Computing non-matching interfaces...")
mapping_list = []
intersection_curves = []
interface_phy_coords = []
interface_phy_coords_proj = []
for i in range(len(nonmatching_occ_bs_flat)  ):
    for j in range(i+1, len(nonmatching_occ_bs_flat) ):
        bs_intersect = BSplineSurfacesIntersections(nonmatching_occ_bs_flat[i], 
                                                    nonmatching_occ_bs_flat[j], 
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
                                              nonmatching_occ_bs_flat[i]), 
                                              project_locations_on_surface(
                                              intersection_coords[k],
                                              nonmatching_occ_bs_flat[j])],]
            interface_phy_coords += intersection_coords
            interface_phy_coords_proj += intersection_coords_proj

num_interfaces = len(mapping_list)
print("Number of non-matching interfaces:", num_interfaces)

print("Creating splines....")
ikNURBS_srfs = [ik_nurbs for ik_nurbs_i in nonmatching_nurbs_srfs \
                for ik_nurbs in ik_nurbs_i]
num_srfs = len(ikNURBS_srfs) 
splines = []
# bc_indices = [0, 1, 3]
bcs_funcs = [zero_bc, zero_bc, None, zero_bc]*3
bcs = [[0,0], [0,1], [None, None], [1,1]]*3
for i in range(num_srfs):
    splines += [ikNURBS2tIGArspline(ikNURBS_srfs[i], zero_bcs=bcs_funcs[i], 
                                   direction=bcs[i][0], side=bcs[i][1], 
                                   index=i),]

#### Import background fluid mesh #####
mesh = Mesh()
f = HDF5File(worldcomm,"mesh.h5","r")
f.read(mesh,"/mesh",True)
f.close()
# Print mesh statistics:
Nel_f = mesh.num_entities_global(mesh.topology().dim())
Nvert_f = mesh.num_entities_global(0)
print("======= Fluid mesh information =======")
print("  Number of elements in fluid mesh: "+str(Nel_f))
print("  Number of nodes in fluid mesh: "+str(Nvert_f))

R_self = 0.045
r_max = 0.035
k_contact = 1e11
# Define contact context:
def phiPrime(r):
    if(r>r_max):
        return 0.0
    return -k_contact*(r_max-r)
def phiDoublePrime(r):
    if(r>r_max):
        return 0.0
    return k_contact
contactContext_sh = ShellsContactContext(splines,
                                         R_self, r_max, phiPrime, 
                                         phiDoublePrime)

nonmatching_problem = NonMatchingCoupling(splines, E, h_th, 
                                          nu, comm=selfcomm)

mortar_nels = []
mortar_pts = []

for i in range(num_interfaces):
    mortar_nels += [np.max([np.max(ikNURBS_srfs[mapping_list[i][0]].control.shape),
                    np.max(ikNURBS_srfs[mapping_list[i][1]].control.shape)])*2,]
    mortar0_pts = np.array([[0.,0.],[0.,1.]])
    mortar_pts += [mortar0_pts,]
nonmatching_problem.create_mortar_meshes(mortar_nels, mortar_pts)
nonmatching_problem.create_mortar_funcs('CG',1)
nonmatching_problem.create_mortar_funcs_derivative('CG',1)

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
            u_file_names[i] += [SAVE_PATH+"results/"+"u"+str(i)+"_"+str(j)+"_file.pvd",]
            u_files[i] += [File(nonmatching_problem.comm, u_file_names[i][j]),]
            F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)+"_"+str(j)+"_file.pvd",]
            F_files[i] += [File(nonmatching_problem.comm, F_file_names[i][j]),]
            if j == 2:
                F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)+"_3_file.pvd",]
                F_files[i] += [File(nonmatching_problem.comm, F_file_names[i][3]),]

########### Time integration ############
# Time integrators for shells
rho_inf = Constant(0.0)
delta_t = Constant(1e-4)
total_time = 0.07
n_steps = int((total_time/float(delta_t)))

####### Fluid Formulation #######
# Set up VMS fluid problem using VarMINT:
CYLINDER_RAD = 1.1
BOTTOM = -0.5
TOP = 2.0

VE = VectorElement("Lagrange",mesh.ufl_cell(),1)
QE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
VQE = MixedElement([VE,QE])
V_f = equalOrderSpace(mesh)
Vscalar = FunctionSpace(mesh,"Lagrange",1)
up = Function(V_f)
up_old = Function(V_f)
updot_old = Function(V_f)
vq = TestFunction(V_f)
timeInt_f = GeneralizedAlphaIntegrator(rho_inf,delta_t,up,(up_old,updot_old))

# Define traction boundary condition at inflow:
xSpatial = SpatialCoordinate(mesh)
PRESSURE = Expression("((t<0.04)? 2e4 : -1e5)",t=0.0,degree=1)
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
dx0 = dx(metadata={"quadrature_degree":quadDeg})
ds0 = ds(metadata={"quadrature_degree":quadDeg})
rho = Constant(1.0)
mu = Constant(3e-2)
up_alpha = timeInt_f.x_alpha()
u_alpha = uPart(up_alpha)
p = timeInt_f.x[3]
v,q = split(vq)
up_t = timeInt_f.xdot_alpha()
u_t = uPart(up_t)
cutFunc = Function(Vscalar)
stabEps = 1e-3
res_f = interiorResidual(u_alpha,p,v,q,rho,mu,mesh,u_t=u_t,Dt=delta_t,dx=dx0,
                         stabScale=stabScale(cutFunc,stabEps))
n = FacetNormal(mesh)
res_f += stableNeumannBC(inflowTraction,rho,u_alpha,v,n,
                         ds=ds0,gamma=Constant(1.0))
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
netInflow = -inflowChar*dot(u,n)*ds0

v_file_name = SAVE_PATH+"results/"+"v"+"_file.pvd"
v_file = File(v_file_name)
p_file_name = SAVE_PATH+"results/"+"p"+"_file.pvd"
p_file = File(p_file_name)
#########################################

############ Shell formulation ##############
dens = Constant(1.)
y_old_hom_list = []
ydot_old_hom_list = []
yddot_old_hom_list = []
time_int_sh_list = []
y_alpha_list = []
ydot_alpha_list = []
yddot_alpha_list = []

for i in range(nonmatching_problem.num_splines):
    y_old_hom_list += [Function(nonmatching_problem.splines[i].V)]
    ydot_old_hom_list += [Function(nonmatching_problem.splines[i].V)]
    yddot_old_hom_list += [Function(nonmatching_problem.splines[i].V)]
    time_int_sh_list += [GeneralizedAlphaIntegrator(rho_inf, delta_t, 
        nonmatching_problem.spline_funcs[i], (y_old_hom_list[i], 
        ydot_old_hom_list[i], yddot_old_hom_list[i]), 
        useFirstOrderAlphaM=True),]
    y_alpha_list += [nonmatching_problem.splines[i].rationalize(
                     time_int_sh_list[i].x_alpha())]
    ydot_alpha_list += [nonmatching_problem.splines[i].rationalize(
                        time_int_sh_list[i].xdot_alpha())]
    yddot_alpha_list += [nonmatching_problem.splines[i].rationalize(
                        time_int_sh_list[i].xddot_alpha())]

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

res_sh_list = []
for i in range(num_srfs):
    dW = Constant(1./time_int_sh_list[i].ALPHA_F)\
         *SVK_residual(nonmatching_problem.splines[i], nonmatching_problem.spline_funcs[i],\
         nonmatching_problem.spline_test_funcs[i], E, nu, h_th, 0)
    dMass = dens*h_th*inner(yddot_alpha_list[i], 
            nonmatching_problem.spline_test_funcs[i])*nonmatching_problem.splines[i].dx
    res_sh_list += [dW+dMass]

blockItTol = 1e-2
DAL_r = 1e-5
# Couple with CouDALFISh:
fsi_problem = FSI_coupling(mesh,res_f,timeInt_f,
                            splines,res_sh_list,time_int_sh_list,
                            DAL_penalty,r=DAL_r,
                            bcs_f=bcs_f,
                            block_it_tol=blockItTol,
                            contact_shs=None,#contactContext_sh,
                            nonmatching_shs=nonmatching_problem,
                            fluid_linear_solver=fluidLinearSolver,
                            cut_func=cutFunc)

for time_iter in range(n_steps):
    PRESSURE.t = timeInt_f.t-(1.0-float(timeInt_f.ALPHA_M))*float(delta_t)
    print("------- Time step "+str(time_iter+1)+" , t = "+str(timeInt_f.t)+" -------")
    # Save initial zero solution
    if time_iter == 0:
        if mpirank == 0:
            for i in range(num_srfs):
                soln_split = nonmatching_problem.spline_funcs[i].split()
                for j in range(3):
                    soln_split[j].rename("u"+str(i)+"_"+str(j), "u"+str(i)+"_"+str(j))
                    u_files[i][j] << soln_split[j]
                    nonmatching_problem.splines[i].cpFuncs[j].rename("F"+str(i)+"_"+str(j),
                                                         "F"+str(i)+"_"+str(j))
                    F_files[i][j] << nonmatching_problem.splines[i].cpFuncs[j]
                    if j == 2:
                        nonmatching_problem.splines[i].cpFuncs[3].rename("F"+str(i)+"_3",
                                                             "F"+str(i)+"_3")
                        F_files[i][3] << nonmatching_problem.splines[i].cpFuncs[3]

        (vout, pout) = up.split()
        vout.rename("v", "v")
        pout.rename("p", "p")
        v_file << vout
        p_file << pout

    fsi_problem.take_step()

    if mpirank == 0:
        for i in range(num_srfs):
            soln_split = nonmatching_problem.spline_funcs[i].split()
            for j in range(3):
                soln_split[j].rename("u"+str(i)+"_"+str(j), "u"+str(i)+"_"+str(j))
                u_files[i][j] << soln_split[j]
                nonmatching_problem.splines[i].cpFuncs[j].rename("F"+str(i)+"_"+str(j),
                                                     "F"+str(i)+"_"+str(j))
                F_files[i][j] << nonmatching_problem.splines[i].cpFuncs[j]
                if j == 2:
                    nonmatching_problem.splines[i].cpFuncs[3].rename("F"+str(i)+"_3",
                                                         "F"+str(i)+"_3")
                    F_files[i][3] << nonmatching_problem.splines[i].cpFuncs[3]
    (vout, pout) = up.split()
    vout.rename("v", "v")
    pout.rename("p", "p")
    v_file << vout
    p_file << pout
    outputFileName = "flow-rate"
    flowRate = assemble(netInflow)
    if mpirank == 0:
        mode = "a"
        if time_iter == 0:
            mode = "w"
        outFile = open(outputFileName,mode)
        outFile.write(str(timeInt_f.t)+" "+str(flowRate)+"\n")
        outFile.close()