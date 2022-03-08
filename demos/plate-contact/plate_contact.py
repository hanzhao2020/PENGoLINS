from tIGAr.timeIntegration import *
from tIGAr.NURBS import *
from ShNAPr.contact import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.igakit_utils import *

import os
import psutil
import matplotlib.pyplot as plt

SAVE_PATH = "./"

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0]/float(1024**2)
    return mem

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

def create_spline(srf, num_field=3, BCs=[0,1]):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)

    for field in range(num_field):
        scalar_spline = spline_generator.getScalarSpline(field)
        for para_direction in range(2):
            if BCs[para_direction] == 1:
                side = 0 # only consider fixing the 0 side
                side_dofs = scalar_spline.getSideDofs(para_direction, 
                                                      side, nLayers=2)
                spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 3*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

E = Constant(21e6)
nu = Constant(0.0)
h_th = Constant(0.05)
load_mag = Constant(1e3) # N/m
f = as_vector([Constant(0.), Constant(0.), -load_mag])
f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])

L = 1.
w = 1.
h0 = -0.1
h1 = -0.2
p = 3

pts0 = [[0., 0., 0.], [w/2, 0., 0.],\
        [0., L, 0.], [w/2, L, 0.]]

pts1 = [[w/2, 0., 0.], [w, 0., 0.],\
        [w/2, L, 0.], [w, L, 0.]]

pts2 = [[0., 0., h0], [w/2, 0., h0],\
        [0., L, h0], [w/2, L, h0]]

pts3 = [[w/2, 0., h0], [w, 0., h0],\
        [w/2, L, h0], [w, L, h0]]

pts4 = [[0., 0., h1], [w/2, 0., h1],\
        [0., L, h1], [w/2, L, h1]]

pts5 = [[w/2, 0., h1], [w, 0., h1],\
        [w/2, L, h1], [w, L, h1]]

pts_list = [pts0, pts1, pts2, pts3, pts4, pts5]
num_srfs = len(pts_list)
num_el_list = [i+6 for i in range(num_srfs)]

srfs = []
splines = []
BCs = [0,1]
for i in range(num_srfs):
    srfs += [create_surf(pts_list[i], num_el_list[i], num_el_list[i], p)]
    splines += [create_spline(srfs[i], BCs=BCs)]

penalty_coefficient = 1e3

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
                                + str(i)+"_3_file.pvd",]
            F_files[i] += [File(selfcomm, F_file_names[i][3]),]

R_self = 0.05
r_max = 0.015
k_contact = 1e11

def phi_prime(r):
    res = 0
    if r < r_max:
        res = -k_contact*(r_max - r)
    return res

def phi_double_prime(r):
    res = 0
    if r < r_max:
        res = k_contact
    return res

contact = ShellContactContext(splines, R_self, r_max, 
                              phi_prime, phi_double_prime)

problem = NonMatchingCoupling(splines, E, h_th, nu, 
                              contact=contact, comm=selfcomm)

mapping_list = [[0,1], [2,3], [4,5]]
num_mortar_mesh = len(mapping_list)

mortar_nels = []
mortar_mesh_locations = []
v_mortar_locs = [np.array([[1., 0.], [1., 1.]]),
                 np.array([[0., 0.], [0., 1.]])]
h_mortar_locs = [np.array([[0., 1.], [1., 1.]]),
                 np.array([[0., 0.], [1., 0.]])]

for i in range(num_mortar_mesh):
    mortar_nels += [(num_el_list[i*2+1])*2,]
    mortar_mesh_locations += [v_mortar_locs,]

problem.create_mortar_meshes(mortar_nels)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)
problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                            penalty_coefficient)

# Time integration part
rho_inf = Constant(0.5)
delta_t = Constant(0.0002)
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
    if i == 0 or i == 1:
        time_integrator_list[i].xdot_old.interpolate(
            Expression(("0.0","0.0","2.0"),degree=1))
    else:
        time_integrator_list[i].xdot_old.interpolate(
            Expression(("0.0","0.0","0.0"),degree=1))

total_steps = 500
memory_profile = []
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
    source_terms = []
    res_list = []
    dMass_list = []
    residuals = []
    load_mag = 1e3
    f_list = [f, f, f0, f0, f0, f0]
    for i in range(problem.num_splines):
        # # SVK model
        source_terms += [inner(f_list[i], problem.splines[i].rationalize(\
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
    print("Inspection: Memory usage: {:8.2f} MB.\n"\
          .format(memory_usage_psutil()))
    memory_profile += [memory_usage_psutil(),]

print("Memory usage increases: {:10.4f} MB"\
      .format(memory_profile[-1]-memory_profile[0]))

plt.figure()
plt.plot(np.arange(len(memory_profile)), memory_profile, '-o')
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Memory usage (MB)")
# plt.show()

for i in range(0, problem.num_splines, 2):
    xi = np.array([1.,1.])
    QoI = -problem.spline_funcs[i](xi)[2]\
         /splines[0].cpFuncs[3](xi)
    print("Vertical displacement at corner: {:10.8f} (patch {})."\
          .format(QoI, i))