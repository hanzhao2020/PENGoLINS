from os import path
from tIGAr.timeIntegration import *
from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *

parameters["std_out_all_processes"] = False

SAVE_PATH = "./"

class SplineBC(object):
    """
    Setting Dirichlet boundary condition to tIGAr spline generator.
    """
    def __init__(self, directions=[0,1], sides=[[0,1],[0,1]], 
                 fields=[[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]],
                 n_layers=[[1,1],[1,1]]):
        self.fields = fields
        self.directions = directions
        self.sides = sides
        self.n_layers = n_layers

    def set_bc(self, spline_generator):
        for direction in self.directions:
            for side in self.sides[direction]:
                for field in self.fields[direction][side]:
                    scalar_spline = spline_generator.getScalarSpline(field)
                    side_dofs = scalar_spline.getSideDofs(direction,
                                side, nLayers=self.n_layers[direction][side])
                    spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                           spline_bc=None, zero_domain=None, 
                           fields=[0,1,2], index=0):
    """
    Convert igakit NURBS to ExtractedBSpline.
    """
    # DIR = SAVE_PATH+"spline_data/extraction_"+str(index)
    quad_deg = surface.UDegree()*quad_deg_const
    spline_mesh = NURBSControlMesh4OCC(surface, useRect=False)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if spline_bc is not None:
        spline_bc.set_bc(spline_generator)
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
preprocessor.compute_intersections(rtol=1e-1, mortar_refine=3)
num_interfaces = preprocessor.num_intersections_all

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", num_interfaces)

# display, _, _, _ = init_display()
# preprocessor.display_intersections(display, show_triedron=True)
# preprocessor.display_surfaces(display)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances
splines = []
spline_bc0 = SplineBC(directions=[0,1], sides=[[0,1], [0]],
                      n_layers=[[1,1],[2]])
bcs_list = [spline_bc0,]*3 + [None,]*4

for i in range(num_surfs):
    splines += [OCCBSpline2tIGArSpline(bicuspid_leaflets[i], 
                                       spline_bc=bcs_list[i], index=i),]

# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th, nu, comm=worldcomm)
problem.create_mortar_meshes(preprocessor.mortar_nels)

if mpirank == 0:
    print("Setting up mortar meshes...")

problem.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

# Create time integrators
rho_inf = Constant(0.5)
delta_t = Constant(1e-4)
n_steps = 100
dens = Constant(1.)

y_old_hom_list = []
ydot_old_hom_list = []
yddot_old_hom_list = []
time_integrator_list = []
y_alpha_list = []
ydot_alpha_list = []
yddot_alpha_list = []
for i in range(num_surfs):
    y_old_hom_list += [Function(problem.splines[i].V)]
    ydot_old_hom_list += [Function(problem.splines[i].V)]
    yddot_old_hom_list += [Function(problem.splines[i].V)]
    time_integrator_list += [GeneralizedAlphaIntegrator(rho_inf, delta_t, 
        problem.spline_funcs[i], (y_old_hom_list[i], ydot_old_hom_list[i], 
        yddot_old_hom_list[i])),]
    y_alpha_list += [problem.splines[i].rationalize(time_integrator_list[i].x_alpha())]
    ydot_alpha_list += [problem.splines[i].rationalize(time_integrator_list[i].xdot_alpha())]
    yddot_alpha_list += [problem.splines[i].rationalize(time_integrator_list[i].xddot_alpha())]
    time_integrator_list[i].xdot_old.interpolate(Expression(("-1.0","0.0","0.0"),degree=1))

# Create SVK residuals
pressure = Constant(5e3)
source_terms = []
res_list = []
dMass_list = []
residuals = []

# Follower pressure
for i in range(len(splines)):
    A0,A1,A2,dA2,A,B = surfaceGeometry(problem.splines[i], 
                                       problem.splines[i].F)
    a0,a1,a2,da2,a,b = surfaceGeometry(problem.splines[i], 
                       problem.splines[i].F+problem.spline_funcs[i])
    source_terms += [(pressure)*sqrt(det(a)/det(A))\
                 *inner(a2,problem.splines[i].rationalize(
                  problem.spline_test_funcs[i]))*problem.splines[i].dx,]
    res_list += [Constant(1./time_integrator_list[i].ALPHA_F)\
                 *SVK_residual(problem.splines[i], problem.spline_funcs[i],
                 problem.spline_test_funcs[i], E, nu, h_th, source_terms[i])]
    dMass_list += [dens*h_th*inner(yddot_alpha_list[i],
                   problem.spline_test_funcs[i])*problem.splines[i].dx,]
    residuals += [res_list[i]+dMass_list[i]]

problem.set_residuals(residuals)

# Create pvd files
FILE_FOLDER = "results/"
u_file_names = []
u_files = []
F_file_names = []
F_files = []
for i in range(num_surfs):
    u_file_names += [[],]
    u_files += [[],]
    F_file_names += [[],]
    F_files += [[],]
    for j in range(3):
        # print("J:", j)
        u_file_names[i] += [SAVE_PATH+FILE_FOLDER+"u"
                            +str(i)+"_"+str(j)+"_file.pvd",]
        u_files[i] += [File(u_file_names[i][j]),]
        F_file_names[i] += [SAVE_PATH+FILE_FOLDER+"F"
                            +str(i)+"_"+str(j)+"_file.pvd",]
        F_files[i] += [File(F_file_names[i][j]),]
        if j == 2:
            F_file_names[i] += [SAVE_PATH+FILE_FOLDER+"F"
                                +str(i)+"_3_file.pvd",]
            F_files[i] += [File(F_file_names[i][3]),]

for time_iter in range(n_steps):
    # Save initial zero solution
    if time_iter == 0:
        for i in range(num_surfs):
            soln_split = problem.spline_funcs[i].split()
            for j in range(3):
                soln_split[j].rename("u"+str(i)+"_"+str(j), 
                                     "u"+str(i)+"_"+str(j))
                u_files[i][j] << soln_split[j]
                problem.splines[i].cpFuncs[j].rename("F"+str(i)+"_"+str(j),
                                                     "F"+str(i)+"_"+str(j))
                F_files[i][j] << problem.splines[i].cpFuncs[j]
                if j == 2:
                    problem.splines[i].cpFuncs[3].rename("F"+str(i)+"_3",
                                                         "F"+str(i)+"_3")
                    F_files[i][3] << problem.splines[i].cpFuncs[3]

    # Solve nonlinear problem
    print("--- Step:", time_iter, ", time:", time_integrator_list[i].t, "---")
    soln = problem.solve_nonlinear_nonmatching_problem(rtol=1e-4, max_it=30, 
                                                       zero_mortar_funcs=False)

    # Save results and move forward
    for i in range(problem.num_splines):
        soln_split = problem.spline_funcs[i].split()
        for j in range(3):
            soln_split[j].rename("u"+str(i)+"_"+str(j), "u"+str(i)+"_"+str(j))
            u_files[i][j] << soln_split[j]
            problem.splines[i].cpFuncs[j].rename("F"+str(i)+"_"+str(j),
                                                 "F"+str(i)+"_"+str(j))
            F_files[i][j] << problem.splines[i].cpFuncs[j]
            if j == 2:
                problem.splines[i].cpFuncs[3].rename("F"+str(i)+"_3",
                                                     "F"+str(i)+"_3")
                F_files[i][3] << problem.splines[i].cpFuncs[3]
        time_integrator_list[i].advance()