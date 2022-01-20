import numpy as np
from scipy.optimize import fsolve, newton_krylov
from PENGoLINS.OCC_utils import *
from PENGoLINS.igakit_utils import *

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

######## 1. Create surface 1 and surface 2 ########
num_pts1 = 8  # Number of knots on one side for surface 1
num_pts2 = 8  # Number of knots on one side for surface 2
num_pts_int = 16  # Number of points to evaluate for intersection curve

# Add random disturbance in (0,1)*`x_init_disturb` to the 
# initial guess, which is the exact solution in this example.
# If `x_init_disturb` is 0, initial guess is exact solution.
x_init_disturb = 1/(num_pts_int*10)
p = 3  # Spline degree
L = 10  # Length of spline surface

#### Surface 1 ####
pts1 = [[-L,0,-L/2], [0,0,-L/2],
        [-L,0,L/2], [0,0,L/2]]
# Create igakit NURBS instance
ik_surf1 = create_surf(pts1, num_pts1, num_pts1, p)
# Convert igakit NURBS instance to OCC BSplineSurface
surf1 = ikNURBS2BSpline_surface(ik_surf1)

#### Surface 2 ####
pts2 = [[-L/2,-L/2,0], [L/2,-L/2,0],
        [-L/2,L/2,0], [L/2,L/2,0]]

ik_surf2 = create_surf(pts2, num_pts2, num_pts2, p)
surf2 = ikNURBS2BSpline_surface(ik_surf2)


######## 2. Compute intersections by PythonOCC ########
surf_int = GeomAPI_IntSS(surf1, surf2, 1e-4)
int_curve = surf_int.Line(1)

# # Display surfaces and intersection
# display, start_display, add_menu, add_function_to_menu = init_display()
# display.DisplayShape(make_face(surf1, 1e-6))
# display.DisplayShape(make_face(surf2, 1e-6))
# display.DisplayShape(int_curve, color='RED')
# exit()

######## 3. Get points on two ends of intersection ########
# Parametric coordinate of left (first) end for intersection curve
s0 = int_curve.FirstParameter()
int_s0_pt = gp_Pnt()
int_curve.D0(s0, int_s0_pt)
# Physical location of first curve parameter
int_s0 = np.array(int_s0_pt.Coord())
# Parametric coordinates of first end for surface 1
first_pt = gp_Pnt(int_s0[0], int_s0[1], int_s0[2])
first_pt_proj1 = GeomAPI_ProjectPointOnSurf(first_pt, surf1, 1e-9)
uv0_surf1 = np.array(first_pt_proj1.LowerDistanceParameters())
# Parametric coordinates of first end for surface 2
first_pt_proj2 = GeomAPI_ProjectPointOnSurf(first_pt, surf2, 1e-9)
uv0_surf2 = np.array(first_pt_proj2.LowerDistanceParameters())

# Parametric coordinate of right (last) end for intersection curve
s1 = int_curve.LastParameter()
int_s1_pt = gp_Pnt()
int_curve.D0(s1, int_s1_pt)
# Physical location of last curve parameter
int_s1 = np.array(int_s1_pt.Coord())
# Parametric coordinates of last end for surface 1
last_pt = gp_Pnt(int_s1[0], int_s1[1], int_s1[2])
last_pt_proj1 = GeomAPI_ProjectPointOnSurf(last_pt, surf1, 1e-9)
uvn_surf1 = np.array(last_pt_proj1.LowerDistanceParameters())
# Parametric coordinates of last end for surface 2
last_pt_proj2 = GeomAPI_ProjectPointOnSurf(last_pt, surf2, 1e-9)
uvn_surf2 = np.array(last_pt_proj2.LowerDistanceParameters())

######## 4. Solve coupled system for interior points ########
int_length = curve_length(int_curve)
# Exact element length for intersection curve
int_el_length = int_length/(num_pts_int-1)

def check_angle(coords):
    """
    Check the angle between two vectors, which are defined
    by three points.
    pt1 -- vec1 --> pt2 -- vec2 --> pt3
    If the angle magnitude between two vectors is larger than 
    90 degree, change the last point to the symmetric 
    point of the second point. This function is intended
    to avoid two parametric points are the same, which 
    corresponds to the angle is 180 degree.

    Parameters
    ----------
    coords : ndarray, size: 3 by 2

    Returns
    -------
    new_coord : ndarray, size: 1 by 2
    """
    vec1 = coords[1] - coords[0]
    vec2 = coords[2] - coords[1]
    cos_anlge = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if cos_anlge < 0:
        new_coord = 2*coords[1] - coords[2]
    else:
        new_coord = coords[2]
    return new_coord

def int_uv_coords(x, surf1=surf1, surf2=surf2, 
                  uv0_surf1=uv0_surf1, uvn_surf1=uvn_surf1):
    """
    Returns residuals of coupled system when solving interior
    parametric coordinates for intersection curve

    Parameters
    ----------
    x : ndarray, size: (num_pts_int-2)*4
        x = [u_1^1, v_1^1, u_2^1, v_2^1, ..., u_{n-1}^1, v_{n-1}^1,
             u_1^2, v_1^2, u_2^2, v_2^2, ..., u_{n-1}^2, v_{n-1}^2]
        Subscript indicates interior index, {1, n-1}
        Superscript means surface index, {1, 2}
    surf1 : OCC BSplineSurface
    surf2 : OCC BSplineSurface
    uv0_surf1 : ndarray
        Parametric coordinate of point corresponds to the 
        first parameter of intersection curve on surface 1
    uvn_surf1 : ndarray
        Parametric coordinate of point corresponds to the 
        last parameter of intersection curve on surface 1

    Returns
    -------
    res : ndarray: size: (num_pts_int-2)*4
    """
    x_coords = x.reshape(-1,2)
    num_pts_interior = int(x.size/4)
    res = np.zeros(x.size)
    pt_temp1 = gp_Pnt()
    pt_temp2 = gp_Pnt()

    # Enforce each pair of parametric points from two surfaces
    # have the same physical location.
    ind_off1 = num_pts_interior
    for i in range(num_pts_interior):
        surf1.D0(x_coords[i,0], x_coords[i,1], pt_temp1)
        surf2.D0(x_coords[i+ind_off1,0], x_coords[i+ind_off1,1], pt_temp2)
        res[i*3:(i+1)*3] = np.array(pt_temp1.Coord()) \
                         - np.array(pt_temp2.Coord())

    # Enforce two adjacent elements has the same magnitude 
    # in physical space for surface 1.
    pt_temp1 = gp_Pnt()
    pt_temp2 = gp_Pnt()
    pt_temp3 = gp_Pnt()
    ind_off2 = num_pts_interior*3

    for i in range(num_pts_interior*3, num_pts_interior*4):
        # For the first two elements
        if i == num_pts_interior*3:
            x_coords[i+1-ind_off2] = check_angle(np.array([uv0_surf1, 
                                                x_coords[i-ind_off2], 
                                                x_coords[i+1-ind_off2]]))

            surf1.D0(uv0_surf1[0], uv0_surf1[1], pt_temp1)
            surf1.D0(x_coords[i-ind_off2,0],
                     x_coords[i-ind_off2,1], pt_temp2)
            surf1.D0(x_coords[i+1-ind_off2,0], 
                     x_coords[i+1-ind_off2,1], pt_temp3)
        # For the last two elements
        elif i == num_pts_interior*4-1:
            x_coords[i-1-ind_off2] = check_angle(np.array([uvn_surf1, 
                                                x_coords[i-ind_off2], 
                                                x_coords[i-1-ind_off2]]))
            surf1.D0(x_coords[i-1-ind_off2,0], 
                     x_coords[i-1-ind_off2,1], pt_temp1)
            surf1.D0(x_coords[i-ind_off2,0], 
                     x_coords[i-ind_off2,1], pt_temp2)
            surf1.D0(uvn_surf1[0], uvn_surf1[1], pt_temp3)
        # For interior elements
        else:
            x_coords[i+1-ind_off2] = check_angle(np.array([
                                                 x_coords[i-1-ind_off2], 
                                                 x_coords[i-ind_off2], 
                                                 x_coords[i+1-ind_off2]]))

            surf1.D0(x_coords[i-1-ind_off2,0], 
                     x_coords[i-1-ind_off2,1], pt_temp1)
            surf1.D0(x_coords[i-ind_off2,0],
                     x_coords[i-ind_off2,1], pt_temp2)
            surf1.D0(x_coords[i+1-ind_off2,0], 
                     x_coords[i+1-ind_off2,1], pt_temp3)

        res[i] = np.linalg.norm(np.array(pt_temp1.Coord())
                              - np.array(pt_temp2.Coord())) \
               - np.linalg.norm(np.array(pt_temp2.Coord())
                              - np.array(pt_temp3.Coord()))
    return res

# Create initial guess.
# In this example, the points of intersection curve on the
# two surfaces are also equally spaced in respective 
# parametric space.
x0 = np.ones((num_pts_int-2)*4)
x0[0:(num_pts_int-2)*2] = np.linspace(uv0_surf1, uvn_surf1, 
                          num_pts_int)[1:-1].reshape(-1,1)[:,0]
x0[(num_pts_int-2)*2:(num_pts_int-2)*4] = np.linspace(uv0_surf2, 
                                          uvn_surf2, num_pts_int)[1:-1].\
                                          reshape(-1,1)[:,0]
# Add random with magnitude from 0 to 1 times `x_init_disturb`
# to the initial guess
x0_disturb = x0 + np.random.random((num_pts_int-2)*4)*x_init_disturb

print("Solving parametric coordinates ...")
# uv_root = fsolve(int_uv_coords, x0=x0_disturb)
uv_root = newton_krylov(int_uv_coords, xin=x0_disturb, 
                        maxiter=1000, f_tol=1e-12)

######## 5. Check solution ########
# Check function residual after solve
uv_res = int_uv_coords(uv_root)
print("Residual after solve:", np.linalg.norm(uv_res))

# Check element length after solve
# Element length on surface 1
int_el_length_list1 = np.zeros(num_pts_int-1)
pt_temp1 = gp_Pnt()
pt_temp2 = gp_Pnt()
uv_root_coor1 = uv_root.reshape(-1,2)[0:num_pts_int-2]
uv_root_coor1 = np.concatenate([np.array([uv0_surf1]), uv_root_coor1], axis=0)
uv_root_coor1 = np.concatenate([uv_root_coor1, np.array([uvn_surf1])], axis=0)
for i in range(num_pts_int-1):
    surf1.D0(uv_root_coor1[i,0], uv_root_coor1[i,1], pt_temp1)
    surf1.D0(uv_root_coor1[i+1,0], uv_root_coor1[i+1,1], pt_temp2)
    int_el_length_list1[i] = np.linalg.norm(np.array(pt_temp1.Coord()) 
                          - np.array(pt_temp2.Coord()))
# Element length on surface 2
int_el_length_list2 = np.zeros(num_pts_int-1)
uv_root_coor2 = uv_root.reshape(-1,2)[num_pts_int-2:]
uv_root_coor2 = np.concatenate([np.array([uv0_surf2]), uv_root_coor2], axis=0)
uv_root_coor2 = np.concatenate([uv_root_coor2, np.array([uvn_surf2])], axis=0)
for i in range(num_pts_int-1):
    surf2.D0(uv_root_coor2[i,0], uv_root_coor2[i,1], pt_temp1)
    surf2.D0(uv_root_coor2[i+1,0], uv_root_coor2[i+1,1], pt_temp2)
    int_el_length_list2[i] = np.linalg.norm(np.array(pt_temp1.Coord()) 
                          - np.array(pt_temp2.Coord()))

print("Exact element length:", int_el_length)
print("Element length on surface 1 after solve:\n", int_el_length_list1)
print("Element length on surface 2 after solve:\n", int_el_length_list2)