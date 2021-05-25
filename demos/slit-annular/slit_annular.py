from PENGoLINS.nonmatching_coupling import *

import os
import psutil

import matplotlib.pyplot as plt

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0]/float(1024**2)
    return mem

SAVE_PATH = "./"

def create_srf(num_el_r, num_el_alpha, p=3, Ri=6, Ro=10, angle_lim=[0,90]):
    angle = (math.radians(angle_lim[0]), math.radians(angle_lim[1]))
    Ci = circle(center=[0,0,0], radius=Ri, angle=angle)
    Co = circle(center=[0,0,0], radius=Ro, angle=angle)
    S = ruled(Ci,Co)
    deg1, deg2 = S.degree
    S.elevate(0,p-deg1)
    S.elevate(1,p-deg2)
    new_knots_r = np.linspace(0,1,num_el_r+1)[1:-1]
    new_knots_alpha = np.linspace(0,1,num_el_alpha+1)[1:-1]
    S.refine(0,new_knots_alpha)
    S.refine(1,new_knots_r)
    return S

def create_spline(srf, num_field=3, BCs=[0,0], fix_z_node=False):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(num_field, spline_mesh)

    for field in range(3):
        scalar_spline = spline_generator.getScalarSpline(field)
        parametric_direction = 0
        for side in [0,1]:
            side_dofs = scalar_spline.getSideDofs(parametric_direction, side,
                                                  nLayers=2)
            if BCs[side] == 1:
                spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 2*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

p = 3
num_el = 8
num_srfs = 4

E = Constant(21e6)
nu = Constant(0.0)
h_th = Constant(0.03)
line_force = Constant(0.8) # N/m

# Use the load stepping to improve convergence
load_step = 50
line_force_ratio = np.linspace(1/load_step,1,load_step)
penalty_coefficient = 1.0e3

nurbs_srfs = []
splines = []
BCs_list = [None, None, None, [0,1]]
mortar_nels = []

for i in range(num_srfs):
    nurbs_srfs += [create_srf(num_el+i, num_el*2+i, 
                              angle_lim=[360/num_srfs*i, 360/num_srfs*(i+1)])]
    if BCs_list[i] is not None:
        splines += [create_spline(nurbs_srfs[-1], BCs=BCs_list[i])]
    else:
        splines += [create_spline(nurbs_srfs[-1])]

total_dofs = 0
for i in range(num_srfs):
    total_dofs += nurbs_srfs[i].control.shape[0]*nurbs_srfs[i].control.shape[1]*3
print("Total DoFs:", total_dofs)
print("Penalty coefficient:", penalty_coefficient)
print("Starting analysis...")
problem = NonMatchingCoupling(splines, E, h_th, nu)

mapping_list = [[0,1], [1,2], [2,3]]
num_mortar_mesh = len(mapping_list)

mortar_nels = []
mortar_pts = []
mortar_mesh_locations = []
v_mortar_locs = [np.array([[1., 0.], [1., 1.]]),
                 np.array([[0., 0.], [0., 1.]])]

for i in range(num_mortar_mesh):
    mortar_nels += [(num_el+i+1)*2,]
    mortar_pts += [np.array([[0., 0.], [0., 1.]]),]
    mortar_mesh_locations += [v_mortar_locs,]

problem.create_mortar_meshes(mortar_nels, mortar_pts)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)

problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                            penalty_coefficient)

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
        u_files[i] += [File(u_file_names[i][j]),]
        F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)+"_"+str(j)+"_file.pvd",]
        F_files[i] += [File(F_file_names[i][j]),]
        if j == 2:
            F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)+"_3_file.pvd",]
            F_files[i] += [File(F_file_names[i][3]),]

WA_list = []
WB_list = []

memory_profile = []


print("Memory usage: {:8.2f} MB.\n".format(memory_usage_psutil()))
print("---------------- Starting loop ------------------")

for nonlinear_test_iter in range(load_step):

    print("-------------- Iteration:", nonlinear_test_iter, "--------------")
    print("Line force density ratio:", line_force_ratio[nonlinear_test_iter])

    print("Inspection 1: Memory usage: {:8.2f} MB.\n".format(memory_usage_psutil()))

    for i in range(len(problem.transfer_matrices_list)):
        for j in range(len(problem.transfer_matrices_list[i])):
            for k in range(len(problem.transfer_matrices_list[i][j])):
                    problem.mortar_vars[i][j][k].interpolate(Constant((0,0,0)))

    if nonlinear_test_iter > 0:
        for i in range(num_srfs):
            problem.spline_funcs[i].assign(soln[i])

    if nonlinear_test_iter == 0:
        for i in range(num_srfs):
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

    f = as_vector([Constant(0.), Constant(0.), 
        line_force_ratio[nonlinear_test_iter]*line_force])
    f0 = as_vector([Constant(0.), Constant(0.), Constant(0.)])
    source_terms = []
    for i in range(len(splines)):
        source_terms += [inner(f0, problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*problem.splines[i].dx]

    # Apply load to the right end boundary
    class loadBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    load_srf_ind = 0
    left = loadBoundary()
    spline_boundaries1 = MeshFunction("size_t", 
        problem.splines[load_srf_ind].mesh, 1)
    spline_boundaries1.set_all(0)
    left.mark(spline_boundaries1, 1)
    problem.splines[load_srf_ind].ds.setMarkers(markers=spline_boundaries1)
    source_terms[load_srf_ind] += inner(f, 
        problem.splines[load_srf_ind].rationalize(
        problem.spline_test_funcs[load_srf_ind]))\
        *problem.splines[load_srf_ind].ds(1)

    residuals = []
    for i in range(problem.num_splines):
        residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
            problem.spline_test_funcs[i], E, nu, h_th, source_terms[i]),]
    
    print("Inspection 2: Memory usage: {:8.2f} MB.\n".format(memory_usage_psutil()))

    problem.assemble_residuals(residuals)

    print("Inspection 3: Memory usage: {:8.2f} MB.\n".format(memory_usage_psutil()))

    print("Solving nonlinear non-matching system...")

    soln = problem.solve_nonlinear_nonmatching_system(rel_tol=1e-3, max_iter=100)

    print("Inspection 4: Memory usage: {:8.2f} MB.\n".format(memory_usage_psutil()))

    for i in range(num_srfs):
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

    print("Inspection 5: Memory usage: {:8.2f} MB.\n".format(memory_usage_psutil()))

    xi_list = [array([0.0, 0.]), array([0.0, 1.])]
    spline_ind = 0
    for j in range(len(xi_list)):
        xi = xi_list[j]
        QoI_temp = problem.spline_funcs[spline_ind](xi)[2]\
                 /splines[spline_ind].cpFuncs[3](xi)
        if j == 0:
            print("Vertical displacement for point A = {:8.6f}".format(QoI_temp))
            WA_list += [QoI_temp,]
        else:
            print("Vertical displacement for point B = {:8.6f}".format(QoI_temp))
            WB_list += [QoI_temp,]

    memory_profile += [memory_usage_psutil(),]


plt.figure()
plt.plot(np.arange(load_step), memory_profile, '-o')
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Memory usage (MB)")
plt.show()