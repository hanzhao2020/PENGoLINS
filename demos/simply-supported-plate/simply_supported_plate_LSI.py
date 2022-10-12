from tIGAr.NURBS import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.igakit_utils import *

parameters["form_compiler"]["representation"] = "tsfc"

def create_surf(pts,num_el0, num_el1,p):
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
            side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=1)
            spline_generator.addZeroDofs(field, side_dofs)

    quad_deg = 2*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

E = Constant(4.8e5)
L = 12.
load_mag = 1.

D = float(E)*float(h_th)**3/(12*(1-float(nu)**2))
u_max = load_mag*L**4/(4*pi**4*D)

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

num_el = 2
penalty_coefficient = 1.0e3
print("Penalty coefficient:", penalty_coefficient)

print("Creating geometry...")
p = 3
nurbs_srfs = []
splines = []
BCs = [[[0,0],[1,0]],
       [[0,1],[1,0]],
       [[0,0],[1,1]],
       [[0,1],[1,1]]]

total_dofs = 0
for i in range(num_srfs):
    nurbs_srfs += [create_surf(pts_list[i], num_el+i, num_el+i, p),]
    splines += [create_spline(nurbs_srfs[i], BCs=BCs[i]),]
    total_dofs += nurbs_srfs[i].shape[0]*nurbs_srfs[i].shape[1]*3
print("Total DoFs:", total_dofs)

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
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)
problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                            penalty_coefficient)

# Isotropic Lee-Sacks material
def psi_el(E_, c0=67.6080e3, c1=13.2848e3, c2=38.1878e3):
    C = 2.0*E_ + Identity(3)
    I1 = tr(C)
    return 0.5*c0*(I1-3.0) \
           + 0.5*c1*(exp(c2*(I1-3.0)**2)-1)

source_terms = []
residuals = []
for i in range(len(splines)):
    X = problem.splines[i].spatialCoordinates()
    load = -load_mag*sin(X[0]*pi/L)*sin(X[1]*pi/L)
    f = as_vector([Constant(0.), Constant(0.), load])
    source_terms += [inner(f, problem.splines[i].rationalize(\
        problem.spline_test_funcs[i]))*problem.splines[i].dx,]
    residuals += [hyperelastic_residual(problem.splines[i], 
                                        problem.spline_funcs[i], 
                                        problem.spline_test_funcs[i], 
                                        h_th, psi_el, source_terms[i]),]
problem.set_residuals(residuals)
problem.solve_nonlinear_nonmatching_problem(rtol=1e-5, max_it=100,
                                            zero_mortar_funcs=False)

xi = np.array([1.,1.])
QoI = -problem.spline_funcs[0](xi)[2]\
     /splines[0].cpFuncs[3](xi)

print("Vertical displacement at center: {:10.8f}.".format(QoI))

SAVE_PATH = "./"
for i in range(len(splines)):
    save_results(splines[i], problem.spline_funcs[i], i, 
        save_cpfuncs=True, save_path=SAVE_PATH, comm=problem.comm)

"""
Visualization with Paraview:
 
1. Load output files for one extracted spline and apply an AppendAttributes 
   filter to combine them.
2. Apply Calculator filter to the AppendAttributes with the formula

    (F0_0/F0_3-coordsX)*iHat + (F0_1/F0_3-coordsY)*jHat + (F0_2/F0_3-coordsZ)*kHat

   for index=0 to get the undeformed configuration, then apply WarpByVector
   filter to the Calculator with scale 1.
3. Apply another Calculator filter to WarpByVector with the formula

    (u0_0/F0_3)*iHat + (u0_1/F0_3)*jHat + (u0_2/F0_3)*kHat

   for index=0 to get the displacement, then apply WarpByVector filter to the 
   Calculator. 

Note: for spline patches with index other than 0, replace ``u0`` and ``F0``
by ``ui`` and ``Fi`` with corresponding index i. 
"""