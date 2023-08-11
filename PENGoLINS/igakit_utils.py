"""
The "igakit_utils" module
---------------------------
provides functions to covert types between igakit NURBS
and PythonOCC BSplineSurface. This module requires install
of igakit.
"""

from igakit.cad import *
from igakit.io import VTK
from PENGoLINS.occ_utils import *

def BSpline_surface2ikNURBS(occ_bs_surf, p=3, u_num_insert=0, 
                            v_num_insert=0, refine=False):
    """
    Convert OCC BSplineSurface to igakit NURBS and refine the 
    surface via knot insertion and order elevation as need.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    p : int, optional, default is 3
    u_num_insert : int, optional, default is 0
    v_num_insert : int, optional, default is 0
    refine : bool, default is True

    Returns
    -------
    ikNURBS : igakit NURBS
    """
    bs_data = BSplineSurfaceData(occ_bs_surf)
    ikNURBS = NURBS(bs_data.knots, bs_data.control)

    if refine:
        # Order elevation
        ikNURBS.elevate(0, p-ikNURBS.degree[0])
        ikNURBS.elevate(1, p-ikNURBS.degree[1])

        # Knot insertion
        u_multiplicity, v_multiplicity = \
            BSpline_surface_interior_multiplicity(occ_bs_surf)
        u_knots, v_knots = bs_data.UKnots, bs_data.VKnots

        if u_num_insert > 0:
            u_knots_insert_single = np.linspace(0,1,u_num_insert+2)[1:-1]
            for i in bs_data.UKnots:
                for k in u_knots_insert_single:
                    if abs(i-k) < (1/u_num_insert/2):
                        u_knots_insert_single = np.delete(
                            u_knots_insert_single,
                            np.argwhere(u_knots_insert_single==k))
            u_knots_insert = []
            for i in range(len(u_knots_insert_single)):
                u_knots_insert += [u_knots_insert_single[i]]*u_multiplicity
            ikNURBS.refine(0, u_knots_insert)
            # print("u_knots_insert:", u_knots_insert)

        if v_num_insert > 0:
            v_knots_insert_single = np.linspace(0,1,v_num_insert+2)[1:-1]
            for i in bs_data.VKnots:
                for k in v_knots_insert_single:
                    if abs(i-k) < (1/v_num_insert/2):
                        v_knots_insert_single = np.delete(
                            v_knots_insert_single,
                            np.argwhere(v_knots_insert_single==k))
            v_knots_insert = []
            for i in range(len(v_knots_insert_single)):
                v_knots_insert += [v_knots_insert_single[i]]*v_multiplicity
            ikNURBS.refine(1, v_knots_insert)
            # print("v_knots_insert:", v_knots_insert)

    return ikNURBS

def ikNURBS2BSpline_surface(ik_nurbs):
    """
    Convert igakit NNURBS to OCC Geom_BSplineSurface.

    Parameters
    ----------
    ik_nurbs : igakit NURBS

    Returns
    -------
    BSpline_surface : OCC BSplineSurface
    """
    num_control_u = ik_nurbs.control.shape[0]
    num_control_v = ik_nurbs.control.shape[1]
    p_u, p_v = ik_nurbs.degree

    poles = TColgp_Array2OfPnt(1, num_control_u, 1, num_control_v)
    weights = TColStd_Array2OfReal(1, num_control_u, 1, num_control_v)
    for i in range(num_control_u):
        for j in range(num_control_v):
            pt_temp = gp_Pnt(ik_nurbs.control[i,j,0], 
                             ik_nurbs.control[i,j,1], 
                             ik_nurbs.control[i,j,2])
            poles.SetValue(i+1, j+1, pt_temp)
            weights.SetValue(i+1, j+1, ik_nurbs.control[i,j,3])

    u_knots_unique = np.unique(ik_nurbs.knots[0])
    u_knots_len = len(u_knots_unique)
    u_knots = TColStd_Array1OfReal(1, u_knots_len)
    u_mults_array = count_knots_multiplicity(ik_nurbs.knots[0])
    u_mults = TColStd_Array1OfInteger(1, u_knots_len)
    for i in range(u_knots_len):
        u_knots.SetValue(i+1, u_knots_unique[i])
        u_mults.SetValue(i+1, int(u_mults_array[i]))

    v_knots_unique = np.unique(ik_nurbs.knots[1])
    v_knots_len = len(v_knots_unique)
    v_knots = TColStd_Array1OfReal(1, v_knots_len)
    v_mults_array = count_knots_multiplicity(ik_nurbs.knots[1])
    v_mults = TColStd_Array1OfInteger(1, v_knots_len)
    for i in range(v_knots_len):
        v_knots.SetValue(i+1, v_knots_unique[i])
        v_mults.SetValue(i+1, int(v_mults_array[i]))

    # print("u_mults_array:", u_mults_array)
    # print("v_mults_array:", v_mults_array)

    BSpline_surface = Geom_BSplineSurface(poles, weights, u_knots, v_knots, 
                                          u_mults, v_mults, p_u, p_v)

    return BSpline_surface