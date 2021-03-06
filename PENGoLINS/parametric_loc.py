"""
The "parametric_loc" module
---------------------------
contains functions that computes of the parametric locations of 
non-matching interfaces based on its physical location using
Newton's method.
"""

from PENGoLINS.nonmatching_utils import *


def geometric_mapping_finite_difference(mesh, f, xi, u_lim, v_lim, 
                                        EPS=1.0e-11, order=2):
    """
    Return the derivatives of function "f" on location "xi", using 
    finite difference method.

    Parameters
    ----------
    f : function, function is used to compute the derivative
    xi : ndarray, location of the derivative
    u_lim : list of floats
        Contains two elements, the limits of ``xi`` in x direction.
    v_lim : list of floats
        Contains two elements, the limits of ``xi`` in y direction.
    EPS : float, optional
        Mesh size of the finite difference method. 
        Default is 1.0e-11.
    order : int, {1, 2}, optional
        Order of accuracy of the finite difference method. 
        Default is 2.

    Returns
    -------
    FD_result : ndarray
    """
    lims = [u_lim, v_lim]
    num_rows = len(f)
    num_cols = len(xi)
    FD_result = np.zeros((num_rows, num_cols))

    if order == 1:
        # First order accurate    
        for i in range(num_rows):
            for j in range(num_cols):
                xi_temp = xi.copy()
                if lims[j][0] <= xi_temp[j] <= lims[j][1]/2.:
                    xi_temp[j] += EPS
                    # Forward difference
                    # FD_result[i,j] = (f[i](xi_temp)-f[i](xi))/EPS
                    fi_xi_temp = eval_func(mesh, f[i], xi_temp)
                    fi_xi = eval_func(mesh, f[i], xi)
                    FD_result[i,j] = (fi_xi_temp - fi_xi)/EPS
                elif lims[j][1]/2. < xi_temp[j] <= lims[j][1]:
                    xi_temp[j] -= EPS
                    # Backward difference
                    # FD_result[i,j] = (f[i](xi)-f[i](xi_temp))/EPS
                    fi_xi_temp = eval_func(mesh, f[i], xi_temp)
                    fi_xi = eval_func(mesh, f[i], xi)
                    FD_result[i,j] = (fi_xi - fi_xi_temp)/EPS
                else:
                    if mpirank == 0:
                        raise ValueError("The point is not inside "
                                         "the domain.")
    elif order == 2:
        # Second order accurate
        for i in range(num_rows):
            for j in range(num_cols):
                xi_temp0 = xi.copy()
                xi_temp1 = xi.copy()

                if xi[0] < lims[0][0] or xi[0] > lims[0][1] or \
                    xi[1] < lims[1][0] or xi[1] > lims[1][1]:
                    if mpirank == 0:
                        raise ValueError("The point is not inside "
                                         "the domain.")
                elif xi[j] < lims[j][0]+EPS:
                    # Forward difference in second order accuracy
                    xi_temp0[j] += EPS
                    xi_temp1[j] += 2*EPS
                    # FD_result[i][j] = (-f[i](xi_temp1) + 4*f[i](xi_temp0) \
                    #                 - 3*f[i](xi))/(2*EPS)
                    fi_xi_temp0 = eval_func(mesh, f[i], xi_temp0)
                    fi_xi_temp1 = eval_func(mesh, f[i], xi_temp1)
                    fi_xi = eval_func(mesh, f[i], xi)
                    FD_result[i][j] = (-fi_xi_temp1 + 4*fi_xi_temp0 \
                                    - 3*fi_xi)/(2*EPS)
                elif xi[j] > lims[j][1]-EPS:
                    # Backward difference in second order accuracy
                    xi_temp0[j] -= EPS
                    xi_temp1[j] -= 2*EPS
                    # FD_result[i][j] = (f[i](xi_temp1) - 4*f[i](xi_temp0) \
                    #                 + 3*f[i](xi))/(2*EPS)
                    fi_xi_temp0 = eval_func(mesh, f[i], xi_temp0)
                    fi_xi_temp1 = eval_func(mesh, f[i], xi_temp1)
                    fi_xi = eval_func(mesh, f[i], xi)
                    FD_result[i][j] = (fi_xi_temp1 - 4*fi_xi_temp0 \
                                    + 3*fi_xi)/(2*EPS)
                else:
                    # Central difference
                    xi_temp0[j] -= EPS
                    xi_temp1[j] += EPS
                    fi_xi_temp0 = eval_func(mesh, f[i], xi_temp0)
                    fi_xi_temp1 = eval_func(mesh, f[i], xi_temp1)
                    # FD_result[i][j] = (f[i](xi_temp1) - f[i](xi_temp0))\
                    #                   /(2*EPS)
                    FD_result[i][j] = (fi_xi_temp1 - fi_xi_temp0)/(2*EPS)

    else:
        if mpirank == 0:
            raise ValueError("{}-th order accuracy is not supported yet.")
    return FD_result

def point_physical_location(spline, xi):
    """
    Return the physical location of a parametric point ``xi``.

    Parameters
    ----------
    spline : ExtractedSpline
    xi : ndarray, parametric location

    Returns
    -------
    phy_loc : ndarray
    """
    phy_loc = np.zeros(spline.nsd)
    for i in range(spline.nsd):
        # phy_loc[i] = spline.F[i](xi)
        phy_loc[i] = eval_func(spline.mesh, spline.F[i], xi)
    return phy_loc

def solve_nonsquare(A,b, perturbation=1e-12):
    """
    Solve non-square system of equation.

    Parameters:
    ----------
    A : ndarray
    b : ndarray
    perturbation : float

    Returns
    -------
    res : ndarray
    """
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)
    detATA = np.linalg.det(ATA)
    if abs(detATA) < 1e-16:
        if mpirank == 0:
            print("Matrix near sigular, adding perturbation to "
                  "diagonal entries.")
            print("Determint of the Matrix: {:10.6e}".format(detATA))
        ATA = ATA + np.eye(ATA.shape[0])*perturbation
    return np.linalg.solve(ATA, ATb)

def physical_location_residual(spline, xi, X):
    """
    Return the residual of the physical location of a parametric 
    point ``xi``.

    Parameters
    ----------
    spline : ExtractedSpline
    xi : ndarray, parametric point
    X : numpu.ndarray, physical location

    Returns
    -------
    res : ndarray
    """
    res = (point_physical_location(spline, xi) - X)
    return res

def check_parametric_location(xi, u_lim, v_lim):
    """
    Check if the parametric point ``xi`` is located inside the 
    domain between ``u_lim`` and ``v_lim``. If not, move this 
    point to the boundary of the domain.

    Parameters
    ----------
    xi : ndarray
    u_lim : list of floats, 2 elements
    v_lim : list of floats, 2 elements
    """
    lims = [u_lim, v_lim]
    for i in range(len(lims)):
        if xi[i] < lims[i][0]:
            xi[i] = lims[i][0] + 1e-16
        elif xi[i] > lims[i][1]:
            xi[i] = lims[i][1] - 1e-16

def edge_detection(parametric_location, r=0.8, tol=1e-4, 
                   u_lim=[0.,1.], v_lim=[0.,1.]):
    """
    Dectect if the ``parametric_location`` is located on the edge.

    Parameters
    ----------
    parametric_location : ndarray
    r : float
        The ratio between the number of points that were thought on 
        edge and the total number of points. Default is 0.8.
    tol : float
        The tolerance that treats a point is on the edge. Default is 1e-4.
    u_lim : list of floats
        The list that contains the limits in u-direction. Default 
        is [0., 1.].
    v_lim : list of floats
        The list that contains the limits in v-direction. Default
        is [0., 1.].

    Returns 
    -------
    parametric_location : ndarray
    """
    num_pts = parametric_location.shape[0]
    pts_start = int(num_pts*0.05)
    pts_end = int(num_pts*0.95)
    num_pts_test = parametric_location[pts_start:pts_end, :].shape[0]
    u0_count, u1_count, v0_count, v1_count = 0, 0, 0, 0

    for i in range(num_pts_test):
        u_coord, v_coord = parametric_location[pts_start:pts_end,:][i]
        if abs(u_coord - u_lim[0]) < tol:
            u0_count += 1
        elif abs(u_coord - u_lim[1]) < tol:
            u1_count += 1
        if abs(v_coord - v_lim[0]) < tol:
            v0_count += 1
        elif abs(v_coord - v_lim[1]) < tol:
            v1_count += 1

    if u0_count/num_pts_test > r:
        parametric_location[:, 0] = np.ones(num_pts)*u_lim[0]
    if u1_count/num_pts_test > r:
        parametric_location[:, 0] = np.ones(num_pts)*u_lim[1]
    if v0_count/num_pts_test > r:
        parametric_location[:, 1] = np.ones(num_pts)*v_lim[0]
    if v1_count/num_pts_test > r:
        parametric_location[:, 1] = np.ones(num_pts)*v_lim[1]

    return parametric_location

def point_parametric_location(spline, X, u_lim=[0.,1.], v_lim=[0.,1.], 
                              max_iter=50, rtol=1e-6, increase_rtol=True,
                              max_rtol=1e-1, print_res=False):
    """
    Compute the parametric location of a physical point ``X`` 
    using Newton's method inside the domain ``u_lim`` and 
    ``v_lim``. 

    Parameters
    ----------
    spline : ExtractedSpline
    X : ndarray, physical location of a point
    u_lim : list of floats, optional
        Limit of the domain in the x-direction. Default is [0., 1.].
    v_lim : list of floats, optional
        Limit of the domain in the y-direction. Default is [0., 1.].
    max_iter : int, optional
        Maximum number of Newton's iteration. Default is 20.
    rtol : float, optional
        Convergence criteria of Newton's method. Default is 1e-6.
    increase_rtol : bool, optional
        If True, the ``rtol`` will increase and continue the iteration
        until the ``max_rtol`` is reached. Default is True.
    max_rtol : float, optional
        Default is 1e-1. 
    print_res : bool, optional
        Print residual information each iteration. Default is False.

    Returns
    -------
    xi : ndarray
    """
    # print("Computing parametric location...")
    xi = np.array([0.5,0.5])
    i = 0

    while True:
        res = physical_location_residual(spline, xi, X)
        res_norm = np.linalg.norm(res)
        
        if res_norm < rtol:
            break

        if i >= max_iter:
            if increase_rtol:
                if rtol <= max_rtol:
                    if mpirank == 0:
                        print("The value of residual: {:10.8f}."
                              .format(res_norm))
                        print("The maximum number of iterations {} has been "
                              "exceeded with tolerance {}.".format(max_iter, 
                              rtol))
                    i = 0
                    rtol = rtol*ceil(res_norm/rtol)
                    if rtol > max_rtol:
                        if mpirank == 0:
                            raise StopIteration("The maximum tolerance {} "
                            "cannot be reached.".format(max_rtol))
                    if mpirank == 0:
                        print("Now using larger tolerance {}.".format(rtol))
                else:
                    if mpirank == 0:
                        print("The value of the residual: {:10.8f}."
                              .fotmat(res_norm))
                        raise StopIteration("The maximum tolerance {} "
                              "cannot be reached.".format(max_rtol))
            else:
                if mpirank == 0:
                    print("The value of residual: {:10.8f}."
                          .format(res_norm))
                    raise StopIteration("Maximum number of iterations "
                          "{} has been exceeded with tolerance {}."
                          .format(max_iter, rtol))

        Dphy_loc = geometric_mapping_finite_difference(spline.mesh, spline.F, 
                                                       xi, u_lim, v_lim)
        dxi = solve_nonsquare(Dphy_loc, -res)
        xi = xi+dxi
        check_parametric_location(xi, u_lim, v_lim)

        if print_res:
            if mpirank == 0:
                print("Iteration: {}, residual norm: {}."
                      .format(i, res_norm))
                print("\tParametric location:", xi)

        i += 1
        res_norm_old = res_norm

    if print_res:
        if mpirank == 0:
            print("Final residual norm:", res_norm)

    return xi

def interface_parametric_location(spline, mortar_mesh, physical_location, 
                                  u_lim=[0.,1.], v_lim=[0.,1.], max_iter=20, 
                                  rtol=1e-9, increase_rtol=True, 
                                  max_rtol=1e-1, print_res=False, 
                                  interp_phy_loc=True, r=0.8, edge_tol=1e-4):
    """
    Compute the parametric locations of the points of the non-matching 
    interface based on their physical locations and geometric mapping 
    of the extracted spline using Newton's method.

    Parameters
    ----------
    spline : ExtractedSpline
    mortar_mesh : dolfin Mesh
    physical_location : ndarray
    u_lim : list of floats, optional
        Limit of the domain in the x-direction. Default is [0., 1.].
    v_lim : list of floats, optional
        Limit of the domain in the y-direction. Default is [0., 1.].
    max_iter : int, optional
        Maximum number of Newton's iteration. Default is 20.
    rtol : float, optional
        Convergence criteria of Newton's method. Default is 1e-9.
    increase_rtol : bool, optional
        If True, the rtol will increase and continue the iteration
        until the max_rtol is reached. Default is True.
    max_rtol : float, optional
        Default is 1e-1.
    print_res : bool, optional
        Print residual information each iteration. Default is False.
    interp_phy_loc : bool, optional
        If True, interpolate the ``physical_location`` to desired
        number of points and find their parametric locations. This 
        will extend the computation time. If False, find the 
        parametric locations of ``physical_location`` first and 
        interpolate the parametric locations later. Default is False.
    r : float, optional
        The ratio that is used to determine if the parametric points
        are located on edge. Default is 0.8.
    edge_tol : float, optional
        The tolerance that is used to determine if the parametric points
        are located on edge. Default is 1e-4.

    Returns
    -------
    parametric_location_data : ndarray
    """
    num_pts = int(mortar_mesh.num_entities_global(0))

    if interp_phy_loc:
        # Interpolate physical locations and compute all 
        # parametric locations
        physical_location_data = generate_interpolated_data(
                                 physical_location, num_pts)
        parametric_location_data = np.zeros((num_pts, spline.mesh.\
                                             geometric_dimension()))
        for i in range(num_pts):
            if print_res:
                if mpirank == 0:
                    print("Physical location point index:", i)
            parametric_location_data[i] = point_parametric_location(spline, 
                                          physical_location_data[i], 
                                          u_lim=u_lim, v_lim=v_lim, 
                                          max_iter=max_iter, rtol=rtol, 
                                          increase_rtol=True, max_rtol=1e-1, 
                                          print_res=print_res)
    else:
        # Compute parametric locations of given physical 
        # locations and do interpolation
        parametric_location = np.zeros((physical_location.shape[0], 
                                        spline.mesh.geometric_dimension()))
        for i in range(physical_location.shape[0]):
            if print_res:
                if mpirank == 0:
                    print("Physical location point index:", i)
            parametric_location[i] = point_parametric_location(spline, 
                                     physical_location[i], u_lim=u_lim, 
                                     v_lim=v_lim, max_iter=max_iter, 
                                     rtol=rtol, increase_rtol=True, 
                                     max_rtol=1e-1, print_res=print_res)
        parametric_location_data = generate_interpolated_data(
                                   parametric_location, num_pts)

    parametric_location_data = edge_detection(parametric_location_data,
                                              r=r, tol=edge_tol,
                                              u_lim=u_lim, v_lim=v_lim)

    return parametric_location_data

def interface_physical_location(spline, parametric_location):
    """
    Compute the physical location of the interface or mortar mesh 
    based on its parametric location.

    Parameters
    ----------
    spline : ExtractedSpline
    parametric_location : ndarray

    Returns
    -------
    physical_location : ndarray
    """
    physical_location = np.zeros((parametric_location.shape[0],spline.nsd))
    for i in range(parametric_location.shape[0]):
        # xi_temp = parametric_location[i,:]
        for j in range(spline.nsd):
            # physical_location[i,j] = spline.cpFuncs[j](xi_temp)\
            #                         /spline.cpFuncs[-1](xi_temp)
            # physical_location[i,j] = spline.F[j](parametric_location[i,:])
            physical_location[i,j] = eval_func(spline.mesh, spline.F[j], 
                                               parametric_location[i,:])
    return physical_location

if __name__ == "__main__":
    pass