from tIGAr.common import *
from tIGAr.NURBS import *

from ShNAPr.kinematics import *
from ShNAPr.hyperelastic import *

from igakit.cad import *

parameters["form_compiler"]["representation"] = "tsfc"

def create_surf(pts, num_el0, num_el1, p):
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

h_th = Constant(0.375)
L = 12.
load_mag = 1.

p = 3
pts = [[0., 0., 0.], [L, 0., 0.],
       [0., L, 0.], [L, L, 0.]]
num_el = 64
num_field = 3
nurbs_surf = create_surf(pts, num_el, num_el, p)

print('DoFs:', nurbs_surf.shape[0]*nurbs_surf.shape[1]*3)

spline_mesh = NURBSControlMesh(nurbs_surf, useRect=False)
spline_generator = EqualOrderSpline(selfcomm, num_field, spline_mesh)

for field in range(num_field):
    for direction in [0,1]:
        for side in [0,1]:
            scalar_spline = spline_generator.getScalarSpline(field)
            side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=1)
            spline_generator.addZeroDofs(field, side_dofs)

quad_deg = 2*nurbs_surf.degree[0]
spline = ExtractedSpline(spline_generator, quad_deg)

u_hom = Function(spline.V)
u = spline.rationalize(u_hom)
X = spline.F
x = X + u

# Isotropic Lee-Sacks material
def psi_el(E_, c0=67.6080e3, c1=13.2848e3, c2=38.1878e3):
    C = 2.0*E_ + Identity(3)
    I1 = tr(C)
    return 0.5*c0*(I1-3.0) \
           + 0.5*c1*(exp(c2*(I1-3.0)**2)-1)

# Obtain a through-thickness integration measure:
N_QUAD_PTS = 4
dxi2 = throughThicknessMeasure(N_QUAD_PTS,h_th)
psi = incompressiblePotentialKL(spline,X,x,psi_el)
Wint = psi*dxi2*spline.dx

# Take the Gateaux derivative of Wint in test function direction z_hom.
v_hom = TestFunction(spline.V)
v = spline.rationalize(v_hom)
dWint = derivative(Wint,u_hom,v_hom)

X_coord = spline.spatialCoordinates()
load = -load_mag*sin(X_coord[0]*pi/L)*sin(X_coord[1]*pi/L)
f = as_vector([Constant(0.), Constant(0.), load])
dWext = inner(f, v)*spline.dx
res = dWint - dWext
Dres = derivative(res, u_hom)

spline.maxIters = 100
spline.relativeTolerance = 1e-6
spline.solveNonlinearVariationalProblem(res,Dres,u_hom)

xi = np.array([.5, .5])
QoI = -u_hom(xi)[2]/spline.cpFuncs[3](xi)
print("Vertical displacement at center: {:10.8f}.".format(QoI))

# Output of control mesh:
for i in range(0,3+1):
    name = "F"+str(i)
    spline.cpFuncs[i].rename(name,name)
    File("results/"+name+"-file.pvd") << spline.cpFuncs[i]

# Output of homogeneous displacement components:
u_hom_split = u_hom.split()
for i in range(0,3):
    name = "u"+str(i)
    u_hom_split[i].rename(name,name)
    File("results/"+name+"-file.pvd") << u_hom_split[i]