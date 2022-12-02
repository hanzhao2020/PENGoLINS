from tIGAr.timeIntegration import *
from tIGAr.NURBS import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.igakit_utils import *

SAVE_PATH = "./"

def create_surf(pts, num_el0, num_el1,p):
    knots0 = np.linspace(0,1,num_el0+1)[1:-1]
    knots1 = np.linspace(0,1,num_el1+1)[1:-1]
    L1 = line(pts[0],pts[1])
    L2 = line(pts[2],pts[3])
    srf = ruled(L1,L2)
    deg0, deg1 = srf.degree 
    srf.elevate(0,p-deg0)
    srf.elevate(1,p-deg1)
    srf.refine(0,knots0)
    srf.refine(1,knots1)
    return srf

def create_spline(srf, num_field=3, BCs=[]):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)

    for field in range(num_field):
        for BC in BCs:
            direction = BC[0]
            side = BC[1]
            scalar_spline = spline_generator.getScalarSpline(field)
            side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=2)
            spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 2*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

nu = Constant(0.5) # incompressible material
mu = Constant(1e4)
E = mu*2*(1+nu)
h_th = Constant(0.03)
L = 2.

pts0 = [[0., 0., 0.], [L/2, 0., 0.],
        [0., L/2, 0.], [L/2, L/2, 0.]]
pts1 = [[L/2, 0., 0.], [L, 0., 0.],
        [L/2, L/2, 0.], [L, L/2, 0.]]
pts2 = [[0., L/2, 0.], [L/2, L/2, 0.],
        [0., L, 0.], [L/2, L, 0.]]
pts3 = [[L/2, L/2, 0.], [L, L/2, 0.],
        [L/2, L, 0.], [L, L, 0.]]
pts_list = [pts0, pts1, pts2, pts3]
num_srfs = len(pts_list)

num_el_list = [8,]
QoI_list = []
QoI_normal_list = []

num_el = num_el_list[0]
penalty_coefficient = 1.0e3
print("Penalty coefficient:", penalty_coefficient)

print("Creating geometry...")
p = 3
total_dofs = 0
nurbs_srfs = []
splines = []
BCs = [[[0,0],[1,0]],
       [[0,1],[1,0]],
       [[0,0],[1,1]],
       [[0,1],[1,1]]]
for i in range(num_srfs):
    nurbs_srfs += [create_surf(pts_list[i], num_el+i, num_el+i, p),]
    splines += [create_spline(nurbs_srfs[i], BCs=BCs[i]),]
    total_dofs += nurbs_srfs[i].control.shape[0]\
               *nurbs_srfs[i].control.shape[1]*3
print("Total DoFs:", total_dofs)

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
        # print("J:", j)
        u_file_names[i] += [SAVE_PATH+"results/"+"u"+str(i)
                            +"_"+str(j)+"_file.pvd",]
        u_files[i] += [File(selfcomm, u_file_names[i][j]),]
        F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)
                            +"_"+str(j)+"_file.pvd",]
        F_files[i] += [File(selfcomm, F_file_names[i][j]),]
        if j == 2:
            F_file_names[i] += [SAVE_PATH+"results/"+"F"
                                +str(i)+"_3_file.pvd",]
            F_files[i] += [File(selfcomm, F_file_names[i][3]),]

problem = NonMatchingCoupling(splines, E, h_th, nu, comm=selfcomm)

mapping_list = [[0,1], [2,3], [0,2], [1,3]]
num_mortar_mesh = len(mapping_list)

mortar_nels = []
mortar_mesh_locations = []
v_mortar_locs = [np.array([[1., 0.], [1., 1.]]),
                 np.array([[0., 0.], [0., 1.]])]
h_mortar_locs = [np.array([[0., 1.], [1., 1.]]),
                 np.array([[0., 0.], [1., 0.]])]

for i in range(num_mortar_mesh):
    mortar_nels += [(num_el+i+2)*2,]
    if i < 2:
        mortar_mesh_locations += [v_mortar_locs,]
    else:
        mortar_mesh_locations += [h_mortar_locs,]

problem.create_mortar_meshes(mortar_nels)
problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                            penalty_coefficient)

# Time integration part
rho_inf = Constant(0.5)
delta_t = Constant(0.005)
dens = Constant(10.)
y_old_hom_list = []
ydot_old_hom_list = []
yddot_old_hom_list = []
time_integrator_list = []
y_alpha_list = []
ydot_alpha_list = []
yddot_alpha_list = []
for i in range(problem.num_splines):
    y_old_hom_list += [Function(problem.splines[i].V)]
    ydot_old_hom_list += [Function(problem.splines[i].V)]
    yddot_old_hom_list += [Function(problem.splines[i].V)]
    time_integrator_list += [GeneralizedAlphaIntegrator(rho_inf, delta_t, 
        problem.spline_funcs[i], (y_old_hom_list[i], ydot_old_hom_list[i], 
        yddot_old_hom_list[i])),]
    y_alpha_list += [problem.splines[i].rationalize(
                     time_integrator_list[i].x_alpha())]
    ydot_alpha_list += [problem.splines[i].rationalize(
                        time_integrator_list[i].xdot_alpha())]
    yddot_alpha_list += [problem.splines[i].rationalize(
                         time_integrator_list[i].xddot_alpha())]
    time_integrator_list[i].xdot_old.interpolate(
                            Expression(("0.0","0.0","2.0"),degree=1))

source_terms = []
res_list = []
dMass_list = []
residuals = []
load_mag = 1e3
for i in range(problem.num_splines):
    # SVK model
    # X = problem.splines[i].spatialCoordinates()
    # load = load_mag*sin(X[0]*pi/L)*sin(X[1]*pi/L)  # sin load
    load = Constant(load_mag)
    f = as_vector([Constant(0.), Constant(0.), load])
    source_terms += [inner(f, problem.splines[i].rationalize(\
        problem.spline_test_funcs[i]))*problem.splines[i].dx,]
    res_list += [Constant(1./time_integrator_list[i].ALPHA_F)\
                  *SVK_residual(problem.splines[i], 
                                problem.spline_funcs[i], 
                                problem.spline_test_funcs[i], 
                                E, nu, h_th, source_terms[i]),]
    dMass_list += [dens*h_th*inner(yddot_alpha_list[i], 
        problem.spline_test_funcs[i])*problem.splines[i].dx,]
    residuals += [res_list[i]+dMass_list[i]]

problem.set_residuals(residuals)

total_steps = 100

for time_iter in range(total_steps):
    print("--- Step:", time_iter, ", time:", time_integrator_list[i].t, "---")
    # Save initial zero solution
    if time_iter == 0:
        for i in range(num_srfs):
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

    print("Solving nonlinear non-matching problem...")
    problem.solve_nonlinear_nonmatching_problem(rtol=1e-3, max_it=100,
                                                zero_mortar_funcs=False)

    for i in range(problem.num_splines):
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

        time_integrator_list[i].advance()

xi = [np.array([1.,1.]), np.array([0.,1.]), 
      np.array([1.,0.]), np.array([0.,0.])]
for i in range(problem.num_splines):
    QoI = problem.spline_funcs[i](xi[i])[2]\
         /splines[0].cpFuncs[3](xi[i])
    QoI_list += [QoI,]
    print("Vertical displacement at center for patch {}: {:10.8f}."\
          .format(i, QoI))