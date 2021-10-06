"""
The "occ_utils" module
---------------------------
contains pythonocc functions that can be used to preprocess
CAD geometry.
"""

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCC.Core.Geom import Geom_Curve, Geom_Surface
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve, GeomAdaptor_Surface
from OCC.Core.GeomAPI import GeomAPI_IntSS, GeomAPI_IntCS
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.TColStd import TColStd_Array2OfReal, TColStd_Array2OfInteger
from OCC.Extend.DataExchange import read_step_file, read_iges_file
from OCC.Extend.ShapeFactory import point_list_to_TColgp_Array1OfPnt
from OCC.Extend.ShapeFactory import make_face
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Display.SimpleGui import init_display

from OCC.Core.Approx import Approx_ParametrizationType

from igakit.cad import *
from PENGoLINS.calculus_utils import *

def read_igs_file(filename, as_compound=False):
    """
    Read iga file into python.

    Parameters
    ----------
    filename : str
    as_compound : bool, optional, default is True

    Returns
    -------
    igs_shapes : TopoDS_Compound or list of TopoDS_Faces
    """
    igs_shapes = read_iges_file(filename, return_as_shapes=False)
    if not as_compound:
        if not isinstance(igs_shapes, list):
            igs_te = TopologyExplorer(igs_shapes)
            igs_shapes_temp = []
            for face in igs_te.faces():
                igs_shapes_temp += [face,]
        igs_shapes = igs_shapes_temp
    return igs_shapes

def read_stp_file(filename, as_compound=False):
    """
    Read stp file into python.

    Parameters
    ----------
    filename : str
    as_compound : bool, optional, default is True

    Returns
    -------
    stp_shapes : TopoDS_Compound or list of TopoDS_Faces
    """
    if as_compound:
        stp_shapes = read_step_file(filename, as_compound=as_compound)
    else:
        stp_shapes = read_step_file(filename, as_compound=as_compound)
        if not isinstance(stp_shapes, list):
            stp_te = TopologyExplorer(stp_shapes)
            stp_shapes_temp = []
            for face in stp_te.faces():
                stp_shapes_temp += [face,]
        stp_shapes = stp_shapes_temp
    return stp_shapes

def BSpline_surface_interior_multiplicity(occ_bs_surf):
    """
    Return the multiplicities of the inner elements of knot vectors.

    Parameters
    ----------
    occ_bs_surf : OCC B-spline surface

    Returns
    -------
    u_multiplicity : int
    v_multiplicity : int
    """
    bs_data = BSplineSurfaceData(occ_bs_surf)
    if bs_data.UMultiplicities.size > 2:
        u_multiplicity = bs_data.UMultiplicities[1]
    else:
        u_multiplicity = 1
    if bs_data.VMultiplicities.size > 2:
        v_multiplicity = bs_data.VMultiplicities[1]
    else:
        v_multiplicity = 1
    return u_multiplicity, v_multiplicity

def TColStdArray1OfReal2Array(TColStdArray, dtype="float64"):
    """
    Convert occ 1D real TColStdArray to ndarray.

    Parameters
    ----------
    TColStdArray : TColStd_Array1OfReal
    dtype : str, optional

    Returns
    -------
    np_array : ndarray
    """
    np_array = np.zeros(TColStdArray.Length(), dtype=dtype)
    for i in range(TColStdArray.Length()):
        np_array[i] = TColStdArray.Value(i+1)
    return np_array

def array2TColStdArray1OfReal(np_array):
    """
    Convert ndarray to 1D real TColStdArray.

    Parameters
    ----------
    np_array : ndarray

    Returns
    -------
    TColStdArray : TColStd_Array1OfReal
    """
    TColStdArray = TColStd_Array1OfReal(1,np_array.shape[0])
    for i in range(np_array.shape[0]):
        TColStdArray.SetValue(i+1, np_array[i])
    return TColStdArray

def array2TColStdArray1OfInteger(np_array):
    """
    Convert ndarray to 1D integer TColStdArray.

    Parameters
    ----------
    np_array : ndarray

    Returns
    -------
    TColStdArray : TColStd_Array1OfInteger
    """
    TColStdArray = TColStd_Array1OfInteger(1,np_array.shape[0])
    for i in range(np_array.shape[0]):
        TColStdArray.SetValue(i+1, int(np_array[i]))
    return TColStdArray


def topoedge2curve(topo_edge, BSpline=False):
    """
    Convert OCC topo edge to OCC Geom Curve.

    Parameters
    ----------
    topo_edge : OCC edge
    BSpline : bool, optional

    Returns
    -------
    curve : OCC curve
    """
    edge_adaptor = BRepAdaptor_Curve(topo_edge)
    if BSpline:
        curve = edge_adaptor.BSpline()
    else:
        curve = edge_adaptor.Curve().Curve()
    return curve

def surface2topoface(surface, tol=1e-6):
    """
    Convert OCC Geom Surface to OCC topo face.

    Parameters
    ----------
    surface : OCC Geom surface
    tol : float, optional, default is 1e-6.

    Returns
    -------
    face : OCC topo face
    """
    return make_face(surface, tol)

def topoface2surface(topo_face, BSpline=False):
    """
    Convert OCC topo face to OCC Geom Surface.

    Parameters
    ----------
    topo_edge : OCC face
    BSpline : bool, optional, default is False.

    Returns
    -------
    surface : OCC surface
    """
    face_adaptor = BRepAdaptor_Surface(topo_face)
    if BSpline:
        surface = face_adaptor.BSpline()
    else:
        surface = face_adaptor.Surface().Surface()
    return surface

def copy_surface(surface, BSpline=False):
    """
    Duplicate OCC Geom_Surface

    Parameters
    ----------
    surface : OCC Geom Surface or OCC Geom BSplineSurface
    BSpline : bool, optional

    Returns
    -------
    surface_copy : OCC Geom Surface or OCC Geom BSplineSurface
    """
    surface_shape = surface2topoface(surface)
    surface_shape_copy = BRepBuilderAPI_Copy(surface_shape).Shape()
    surface_copy = topoface2surface(surface_shape_copy, BSpline=BSpline)
    return surface_copy

def get_curve_coord(curve, num_pts=20, sort_axis=None, flip=False):
    """
    Return the coordinates of a OCC curve.

    Parameters
    ----------
    curve : OCC Curve
    num_pts : int, optional
        The number of points evaluated on the curve. 
        Default is 20.
    sort_axis : int, {0, 1, 2} or None, optional
        Sort the coordinates based on the x, y or z axis.
    flip : bool, optional.
        Flip the coordinates. Default is False.
    
    Returns
    -------
    curve_coord : ndarray
    """
    curve_coord = np.zeros((num_pts, 3))
    u_para = np.linspace(curve.FirstParameter(),
                         curve.LastParameter(),num_pts)
    p_temp = gp_Pnt()
    for i in range(num_pts):
        curve.D0(u_para[i], p_temp)
        for j in range(3):
            curve_coord[i][j] = p_temp.Coord()[j]
    if sort_axis is not None:
        curve_coord = sort_coord(curve_coord,sort_axis)
    elif flip:
        curve_coord = curve_coord[::-1]
    return curve_coord

def get_face_edges(face, BSpline=False):
    """
    Return a list of edges of a OCC face.

    Parameters
    ----------
    face : OCC face
    BSpline : bool, optional, default is false.

    Returns
    -------
    edges : list of OCC edges
    """
    face_te = TopologyExplorer(face)
    edges = []
    for edge in face_te.edges():
        edge_test = topoedge2curve(edge, BSpline=False)
        if edge_test != None:
            edge_curve = topoedge2curve(edge, BSpline) 
            edges += [edge_curve,]
    return edges

def project_locations_on_surface(locations, surf):
    """
    Project a list of points onto surface.

    Parameters
    ----------
    locations : ndarray
    surf : OCC surface

    Returns
    -------
    res : ndarray
    """
    num_rows, num_cols = locations.shape
    res = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        pt = gp_Pnt(locations[i,0], locations[i,1], locations[i,2])
        proj = GeomAPI_ProjectPointOnSurf(pt, surf)
        proj_pt = proj.NearestPoint()
        res[i,:] = proj_pt.Coord()
    return res

def parametric_coord(locations, surf):
    """
    Return the parametric locations of a list of physical coordinates
    on surface ``surf``.

    Parameters
    ----------
    locations : ndarray
    surf : OCC Surface

    Returns
    -------
    uv_coords : ndarray
    """
    uv_coords = []
    for i in range(locations.shape[0]):
        pt = gp_Pnt(locations[i,0], locations[i,1], locations[i,2])
        pt_proj = GeomAPI_ProjectPointOnSurf(pt, surf)#, 1e-12)
        uv_coords += [pt_proj.LowerDistanceParameters(),]
    return np.array(uv_coords)

def decrease_knot_multiplicity(occ_bs_surf, rtol=1e-2):
    """
    Decrease the multiplicity of interior knots of OCC B-Spline Surface
    to get maximal continuity by removing knots.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    rtol : float, default is 1e-2

    Returns
    -------
    occ_bs_surf : OCC BSplineSurface
    """
    pt00 = gp_Pnt()
    pt11 = gp_Pnt()
    occ_bs_surf.D0(occ_bs_surf.Bounds()[0], occ_bs_surf.Bounds()[2], pt00)
    occ_bs_surf.D0(occ_bs_surf.Bounds()[1], occ_bs_surf.Bounds()[3], pt11)
    diag_len = np.linalg.norm(np.array(pt00.Coord())-np.array(pt11.Coord()))
    geom_tol = rtol*diag_len  # Geometric tolerance

    u_knots = occ_bs_surf.UKnots()
    u_mults = occ_bs_surf.UMultiplicities()
    u_excess = []
    rem_u_knots = []

    for i in range(2, u_knots.Size()):
        u_excess += [u_mults.Value(i) > 1]
        if u_excess[-1]:
            rem_u_knots += [occ_bs_surf.RemoveUKnot(i, 1, geom_tol)]
    if True in u_excess:
        print("*** Warning: u knots have interior multiplicity ", 
              "greater than 1 ***")
    if False in rem_u_knots:
        print("Excessive u knots are not removed!")
    elif True in rem_u_knots:
        print("Excessive u knots are removed with tolerance:", geom_tol)

    v_knots = occ_bs_surf.VKnots()
    v_mults = occ_bs_surf.VMultiplicities()
    v_excess = []
    rem_v_knots = []

    for i in range(2, v_knots.Size()):
        v_excess += [v_mults.Value(i) > 1]
        if v_excess[-1]:
            rem_v_knots += [occ_bs_surf.RemoveVKnot(i, 1, geom_tol)]
    if True in v_excess:
        print("*** Warning: v knots have interior multiplicity ", 
              "greater than 1 ***")
    if False in rem_v_knots:
        print("Excessive v knots are not removed!")
    elif True in rem_v_knots:
        print("Excessive v knots are removed with tolerance: ", geom_tol)

    return occ_bs_surf

def remove_dense_knots(occ_bs_surf, dist_ratio=0.4, rtol=1e-2):
    """
    Check distance between B-Spline surface knots, remove one if 
    distance between two adjacent knots is smaller than a ratio of
    average distance under relative tolerance ``rtol``. A larger
    ratio indicates more knots shoule be removed if it is allowed 
    by the tolerance.

    This function is currently experimental and not stable.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    dist_ratio : float, default is 0.4
    rtol : float, default is 1e-2

    Returns
    -------
    occ_bs_surf : OCC BSplineSurface
    """
    # print("Removing knots")

    for i in range(3):
        bs_data = BSplineSurfaceData(occ_bs_surf)

        # Compute the approximated diagonal length of B-Spline surface
        pt00 = gp_Pnt()
        pt11 = gp_Pnt()
        occ_bs_surf.D0(occ_bs_surf.Bounds()[0], occ_bs_surf.Bounds()[2], pt00)
        occ_bs_surf.D0(occ_bs_surf.Bounds()[1], occ_bs_surf.Bounds()[3], pt11)
        diag_len = np.linalg.norm(np.array(pt00.Coord())
                                  -np.array(pt11.Coord()))
        geom_tol = rtol*diag_len  # Geometric tolerance

        # Check u direction
        u_knots_diff = np.diff(bs_data.UKnots)
        u_knots_diff_avg = np.average(u_knots_diff)
        u_ind_off = 2
        for u_diff_ind, u_knot_dist in enumerate(u_knots_diff):
            if u_knot_dist < u_knots_diff_avg*dist_ratio:
                rem_u_ind = u_diff_ind + u_ind_off
                if rem_u_ind == occ_bs_surf.NbUKnots():
                    rem_u_ind -= 1
                # print("Remove u ind:", rem_u_ind)
                rem_u_knot = False
                geom_tol_u = geom_tol
                while rem_u_knot is not True:
                    rem_u_knot = occ_bs_surf.RemoveUKnot(rem_u_ind, 
                                                         0, geom_tol_u)
                    geom_tol_u = geom_tol_u*(1+1e-3)
                    if geom_tol_u > geom_tol*20:
                        break
                # 0 indicates knot multiplicity.
                # rem_u_knot is boolean, True is the knot is removed.
                if rem_u_knot:
                    u_ind_off -= 1
                    # print("u knots is removed")
                # else:
                    # print("u knots is not removed")

        # Check v direction
        v_knots_diff = np.diff(bs_data.VKnots)
        v_knots_diff_avg = np.average(v_knots_diff)
        v_ind_off = 2
        for v_diff_ind, v_knot_dist in enumerate(v_knots_diff):
            if v_knot_dist < v_knots_diff_avg*dist_ratio:
                rem_v_ind = v_diff_ind + v_ind_off
                if rem_v_ind == occ_bs_surf.NbVKnots():
                    rem_v_ind -= 1
                # print("Remove v ind:", rem_v_ind)
                rem_v_knot = False
                geom_tol_v = geom_tol
                while rem_v_knot is not True:
                    rem_v_knot = occ_bs_surf.RemoveVKnot(rem_v_ind, 
                                                         0, geom_tol_v)
                    geom_tol_v = geom_tol_v*(1+1e-3)
                    if geom_tol_v > geom_tol*20:
                        break
                # 0 indicates knot multiplicity.
                # rem_v_knot is boolean, True is the knot is removed.
                if rem_v_knot:
                    v_ind_off -= 1
                    # print("v knots is removed")
                # else:
                    # print("v knots is not removed")

    return occ_bs_surf

def reconstruct_BSpline_surface(occ_bs_surf, u_num_eval=30, v_num_eval=30, 
                                bs_degree=3, bs_continuity=3, 
                                tol3D=1e-3, geom_scale=1., 
                                correct_knots=False, dist_ratio=0.5,
                                rtol=1e-2):
    """
    Return reconstructed B-spline surface by evaluating the positions
    of the original surface.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    u_num_eval : int, optional
        The number of points evaluated in the u-direction. Default is 30.
    v_num_eval : int, optional
        The number of points evaluated in the v-direction. Default is 30.
    bs_degree : int, optional, default is 3.
    bs_continuity : int, optional, default is 3.
    tol3D : float, optional, default is 1e-3.
    geom_scale : float, optional, default is 1.0.

    Returns
    -------
    occ_bs_res : occ Geom BSplineSurface
    """
    u_num_eval_max = u_num_eval
    v_num_eval_max = v_num_eval

    for num_iter in range(int(u_num_eval)):
        occ_bs_res_pts = TColgp_Array2OfPnt(1, u_num_eval, 1, v_num_eval)
        para_u = np.linspace(occ_bs_surf.Bounds()[0], occ_bs_surf.Bounds()[1], 
                             u_num_eval)
        para_v = np.linspace(occ_bs_surf.Bounds()[2], occ_bs_surf.Bounds()[3], 
                             v_num_eval)
        pt_temp = gp_Pnt()
        for i in range(u_num_eval):
            for j in range(v_num_eval):
                occ_bs_surf.D0(para_u[i], para_v[j], pt_temp)
                pt_temp0 = gp_Pnt(pt_temp.Coord()[0]*geom_scale, 
                                  pt_temp.Coord()[1]*geom_scale, 
                                  pt_temp.Coord()[2]*geom_scale)
                occ_bs_res_pts.SetValue(i+1, j+1, pt_temp0)
        occ_bs_res = GeomAPI_PointsToBSplineSurface(occ_bs_res_pts, 
                        Approx_ParametrizationType(0), bs_degree, bs_degree, 
                        bs_continuity, tol3D).Surface()

        # Check element shape, reduce number of evaluations if needed
        # bs_res_data = BSplineSurfaceData(occ_bs_res)
        # phy_coords = knots_geom_mapping(occ_bs_res)
        # bs_AR = BSpline_element_AR(occ_bs_res)
        # bs_AR_avg = [np.zeros(bs_AR.shape[0]), np.zeros(bs_AR.shape[1])]
        # for i in range(bs_AR_avg[0].shape[0]):
        #     bs_AR_avg[0][i] = np.average(bs_AR[i,:])
        #     if bs_AR_avg[0][i] > 3:
        #         phy_coords_list = phy_coords[i:i+2,:,:]
        #         element_AR_avg_u = np.average(compute_list_element_AR(
        #                            phy_coords_list))
        #         if (element_AR_avg_u < 0.1 and i !=0 
        #             and i!= bs_AR_avg[0].shape[0]-1):
        #             if u_num_eval > round(u_num_eval_max/5):
        #                 u_num_eval -= 1

        # for i in range(bs_AR_avg[1].shape[0]):
        #     bs_AR_avg[1][i] = np.average(bs_AR[:,i])
        #     if bs_AR_avg[1][i] > 3:
        #         phy_coords_list = phy_coords[:,i:i+2,:]
        #         element_AR_avg_v = np.average(compute_list_element_AR(
        #                                       phy_coords_list))
        #         if (element_AR_avg_v < 0.1 and i !=0 
        #             and i!= bs_AR_avg[1].shape[0]-1):
        #             if v_num_eval > round(v_num_eval_max/5):
        #                 v_num_eval -= 1

        # phy_coords = knots_geom_mapping(occ_bs_res)
        # bs_el_AR_avg = [np.zeros(phy_coords.shape[0]-1), 
        #                 np.zeros(phy_coords.shape[1]-1)]
        # for i in range(1,bs_el_AR_avg[0].shape[0]-1):
        #     phy_coords_list = phy_coords[i:i+2,:,:]
        #     element_AR_avg_u = np.average(compute_list_element_AR(
        #                        phy_coords_list))
        #     bs_el_AR_avg[0][i] = element_AR_avg_u
        # if len(bs_el_AR_avg[0][1:-1]) > 0:
        #     if np.min(bs_el_AR_avg[0][1:-1]) < 0.1:
        #         if u_num_eval > round(u_num_eval_max/5):
        #             u_num_eval -= 1

        # for i in range(1,bs_el_AR_avg[1].shape[0]-1):
        #     phy_coords_list = phy_coords[:,i:i+2,:]
        #     element_AR_avg_v = np.average(compute_list_element_AR(
        #                        phy_coords_list))
        #     bs_el_AR_avg[1][i] = element_AR_avg_v
        # if len(bs_el_AR_avg[1][1:-1]) > 0:
        #     if np.min(bs_el_AR_avg[1][1:-1]) < 0.1:
        #         if v_num_eval > round(v_num_eval_max/5):
        #             v_num_eval -= 1

        # if np.min(bs_el_AR_avg[0]) > 0.1 and np.min(bs_el_AR_avg[1]) > 0.1:
        #     break

    # # Check if surface has excessive interior u and v knots
    # decrease_knot_multiplicity(occ_bs_res, rtol)

    # Remove densely distributed knots
    if correct_knots:
        remove_dense_knots(occ_bs_res, dist_ratio, rtol)

    return occ_bs_res

def knots_geom_mapping(occ_bs_surf, u_knots=None, v_knots=None):
    """
    Get physical locations of ``u_knots`` and ``v_knots``.

    Parameters
    ----------
    occ_bs_surf: OCC B-Spline surface
    u_knots : ndarray
    v_knots : ndarray
    """
    surf_data = BSplineSurfaceData(occ_bs_surf)
    if u_knots is None:
        u_knots = surf_data.UKnots
    if v_knots is None:
        v_knots = surf_data.VKnots
    num_u_knots = u_knots.shape[0]
    num_v_knots = v_knots.shape[0]
    phy_pts = np.zeros((num_u_knots, num_v_knots, 3))
    for i in range(num_u_knots):
        for j in range(num_v_knots):
            pts_temp = gp_Pnt()
            occ_bs_surf.D0(u_knots[i], v_knots[j], pts_temp)
            phy_pts[i,j,:] = pts_temp.Coord()
    return phy_pts

def form_rectangle(quad_coords, mode=0):
    """
    From rectange to compute quad element aspect ratio.

    Parameters
    ----------
    quad_coords : ndarray or list of ndarray
        Coordinates of four corners of quadrilateral element, 
        shape of (2,2,dim), [[A,B],[C,D]].

        Coordinates order:
        C----D
        |    |
        |    |
        A----B
    mode : int, {0, 1}, default is 0

    Returns
    -------
    rect_coords : ndarray
        Coordinates of four corners of formed rectangle, 
        shape of (2,2,dim).
    """
    # Compute midpoints E,F,G,H of quadrilateral element.
    # C--F--D
    # |     |
    # G     H 
    # |     |
    # A--E--B
    A, B = quad_coords[0,0], quad_coords[0,1]
    C, D = quad_coords[1,0], quad_coords[1,1]
    E = (A + B)/2
    F = (C + D)/2
    G = (A + C)/2
    H = (B + D)/2
    # print(E, "\n", F, "\n", G, "\n", H)
    if mode == 0:
        EF = F-E
        # project E onto two lines that parallel with EF and
        # pass through G and H
        E_proj_G = G + np.dot(E-G,EF)/np.dot(EF,EF)*EF
        E_proj_H = H + np.dot(E-H,EF)/np.dot(EF,EF)*EF
        # project F onto two lines that parallel with EF and
        # pass through G and H
        F_proj_G = G + np.dot(F-G,EF)/np.dot(EF,EF)*EF
        F_proj_H = H + np.dot(F-H,EF)/np.dot(EF,EF)*EF
        rect_coords = np.array([[E_proj_G, E_proj_H],
                                [F_proj_G, F_proj_H]])
    elif mode == 1:
        GH = H-G
        # project G onto two lines that parallel with GH and
        # pass through E and F
        G_proj_E = E + np.dot(G-E,GH)/np.dot(GH,GH)*GH
        G_proj_F = F + np.dot(G-F,GH)/np.dot(GH,GH)*GH
        # project H onto two lines that parallel with GH and
        # pass through E and F
        H_proj_E = E + np.dot(H-E,GH)/np.dot(GH,GH)*GH
        H_proj_F = F + np.dot(H-F,GH)/np.dot(GH,GH)*GH
        rect_coords = np.array([[G_proj_E, G_proj_F],
                                [H_proj_E, H_proj_F]])
    else:
        raise ValueError('Unknown mode '+str(mode))
    return rect_coords

def quad_element_AR(quad_coords):
    """
    Compute the aspect ratio of a quadrilateral element.

    Parameters
    ----------
    quad_coords : ndarray, coordinates of quad element

    Returns
    -------
    quad_AR : float
    """
    has_singularity = False
    quad_edge_len = np.ones((2,2))
    for i in range(quad_coords.shape[0]):
        length_0i = np.linalg.norm(quad_coords[0,i]- quad_coords[1,i])
        length_1i = np.linalg.norm(quad_coords[i,0]- quad_coords[i,1])
        if length_0i < 1e-15 or length_1i < 1e-15:
            has_singularity = True

    rect0 = form_rectangle(quad_coords, mode=0)
    rect1 = form_rectangle(quad_coords, mode=1)

    rect0_min_side = np.min([np.linalg.norm(rect0[0,0]-rect0[0,1]), 
                            np.linalg.norm(rect0[0,0]-rect0[1,0])])
    rect0_max_side = np.max([np.linalg.norm(rect0[0,0]-rect0[0,1]), 
                            np.linalg.norm(rect0[0,0]-rect0[1,0])])
    AR0 = rect0_max_side/rect0_min_side

    rect1_min_side = np.min([np.linalg.norm(rect1[0,0]-rect1[0,1]), 
                            np.linalg.norm(rect1[0,0]-rect1[1,0])])
    rect1_max_side = np.max([np.linalg.norm(rect1[0,0]-rect1[0,1]), 
                            np.linalg.norm(rect1[0,0]-rect1[1,0])])
    AR1 = rect1_max_side/rect1_min_side

    quad_AR = np.max([AR0, AR1])
    if has_singularity:
        quad_AR = quad_AR/np.sqrt(3)
    return quad_AR

def BSpline_element_AR(occ_bs_surf, u_knots=None, v_knots=None):
    """
    Compute the aspect ratio for all element of B-Spline ``occ_bs_surf``.

    Parameters
    ----------
    occ_bs_surf : OCC B-Spline surface

    Returns
    -------
    bs_AR : ndarray
    """
    surf_data = BSplineSurfaceData(occ_bs_surf)
    if u_knots is None:
        u_knots = surf_data.UKnots
    if v_knots is None:
        v_knots = surf_data.VKnots

    surf_coords = knots_geom_mapping(occ_bs_surf, u_knots, v_knots)
    surf_AR = np.zeros((surf_coords.shape[0]-1, surf_coords.shape[1]-1))

    for i in range(surf_AR.shape[0]):
        for j in range(surf_AR.shape[1]):
            quad_coords = np.array([[surf_coords[i,j], 
                                     surf_coords[i,j+1]],
                                    [surf_coords[i+1,j], 
                                     surf_coords[i+1,j+1]]])
            surf_AR[i,j] = quad_element_AR(quad_coords)
    return surf_AR

def compute_list_element_AR(phy_coords):
    """
    Compute averaged aspect ratio for one line of elements.

    Parameters
    ----------
    phy_coords : ndarray, shape (2,n,3) or (n,2,3)

    Returns:
    --------
    element_AR : ndarray, shape (1,n)
    """
    if phy_coords.shape[0] != 2:
        phy_coords = phy_coords.transpose(1,0,2)

    phy_coords_diff0 = np.diff(phy_coords, axis=0)
    element_len0 = np.linalg.norm(phy_coords_diff0, axis=-1)
    phy_coords_diff1 = np.diff(phy_coords, axis=1)
    element_len1 = np.linalg.norm(phy_coords_diff1, axis=-1)

    element_AR = np.zeros(phy_coords.shape[1]-1)
    for i in range(len(element_AR)):
        AR_fac = 1.
        sin_el1_angle = 1
        if element_len0[0,i] < 1e-13 or element_len0[0,i+1] < 1e-13:
            AR_fac = 2.
        elif element_len1[0,i] < 1e-13 or element_len1[1,i] < 1e-13:
            sin_el1_angle = np.linalg.norm(np.cross(phy_coords[0,i], 
                            phy_coords[1,i]))/(np.linalg.norm(
                            phy_coords[0,i])*np.linalg.norm(phy_coords[1,i]))
            AR_fac = 0.5
        element_AR[i] = (element_len0[0,i]+element_len0[0,i+1])\
                       /(element_len1[0,i]+element_len1[1,i])\
                       *AR_fac*sin_el1_angle
    return element_AR

def correct_BSpline_surface_element_shape(occ_bs_surf, 
                                          u_knots_insert, v_knots_insert):
    """
    Automatic element shape correction to get reasonable aspect ratio.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    u_knots_insert : int
    v_knots_insert : int

    Returns
    -------
    u_knots_insert : ndarray
        Knots vector for insertion in u direction after shape correction
    v_knots_insert : ndarray
        Knots vector for insertion in v direction after shape correction
    """
    bs_data = BSplineSurfaceData(occ_bs_surf)
    u_knots_full = np.sort(np.concatenate([bs_data.UKnots, u_knots_insert]))
    v_knots_full = np.sort(np.concatenate([bs_data.VKnots, v_knots_insert]))

    phy_coords = knots_geom_mapping(occ_bs_surf, u_knots_full, v_knots_full)
    bs_AR = BSpline_element_AR(occ_bs_surf, 
                               u_knots_full, v_knots_full)
    bs_AR_avg = [np.zeros(bs_AR.shape[0]), np.zeros(bs_AR.shape[1])]

    u_knots_add_insert = []
    v_knots_add_insert = []

    for i in range(bs_AR_avg[0].shape[0]):
        bs_AR_avg[0][i] = np.average(bs_AR[i,:])
        if bs_AR_avg[0][i] > 3:
            phy_coords_list = phy_coords[i:i+2,:,:]
            element_AR_avg_u = np.average(compute_list_element_AR(
                               phy_coords_list))
            # print("element AR average u:", element_AR_avg_u)
            if element_AR_avg_u > 3:
                u_knots_add_insert += [np.linspace(u_knots_full[i], 
                                       u_knots_full[i+1], round(
                                       (element_AR_avg_u+1)*0.8))[1:-1]]

    for i in range(bs_AR_avg[1].shape[0]):
        bs_AR_avg[1][i] = np.average(bs_AR[:,i])
        if bs_AR_avg[1][i] > 3:
            phy_coords_list = phy_coords[:,i:i+2,:]
            element_AR_avg_v = np.average(compute_list_element_AR(
                                          phy_coords_list))
            # print("element AR average v:", element_AR_avg_v)
            if element_AR_avg_v > 3:
                v_knots_add_insert += [np.linspace(v_knots_full[i],
                                       v_knots_full[i+1], round(
                                       (element_AR_avg_v+1)*0.8))[1:-1]]
    u_knots_insert_res = []
    v_knots_insert_res = []
    if len(u_knots_add_insert) == 0 and len(v_knots_add_insert) == 0:
        u_knots_insert_res = u_knots_insert
        v_knots_insert_res = v_knots_insert
    else:
        if len(u_knots_add_insert) > 0:
            u_knots_add_insert = np.concatenate(u_knots_add_insert)
        else:
            u_knots_add_insert = np.array([])

        if len(v_knots_add_insert) > 0:
            v_knots_add_insert = np.concatenate(v_knots_add_insert)
        else:
            v_knots_add_insert = np.array([])

        u_knots_corret_init = np.sort(np.concatenate([u_knots_full[1:-1], 
                                      u_knots_add_insert]))
        v_knots_corret_init = np.sort(np.concatenate([v_knots_full[1:-1], 
                                      v_knots_add_insert]))

        u_knots_init_ind = [u_ind for u_ind, u_knot in \
                            enumerate(u_knots_corret_init) \
                            if u_knot in bs_data.UKnots]
        v_knots_init_ind = [v_ind for v_ind, v_knot in \
                            enumerate(v_knots_corret_init) \
                            if v_knot in bs_data.VKnots]

        num_insert_max = int(np.max([len(u_knots_insert), 
                                     len(v_knots_insert)])*1.2)
        num_u_knots = len(u_knots_corret_init)
        num_v_knots = len(v_knots_corret_init)
        num_knots_max = np.max([num_u_knots, num_v_knots])

        if num_knots_max > num_insert_max:
            num_insert_ratio = num_insert_max/num_knots_max
            num_u_insert = round(num_u_knots*num_insert_ratio)
            num_v_insert = round(num_v_knots*num_insert_ratio)

            u_knots_ind_temp = np.array(np.linspace(0, num_u_knots+1, 
                               num_u_insert+2)[1:-1], dtype=int)
            v_knots_ind_temp = np.array(np.linspace(0, num_v_knots+1, 
                               num_v_insert+2)[1:-1], dtype=int)

            u_insert_ind = []
            if len(u_knots_init_ind) > 0:
                for u_ind in u_knots_ind_temp:
                    if np.min(np.abs(u_ind-u_knots_init_ind)) > \
                        np.average(np.diff(u_knots_ind_temp))*0.4:
                        u_insert_ind += [u_ind,]
            else:
                u_insert_ind = u_knots_ind_temp

            v_insert_ind = []
            if len(v_knots_init_ind) > 0:
                for v_ind in v_knots_ind_temp:
                    if np.min(np.abs(v_ind-v_knots_init_ind)) > \
                        np.average(np.diff(v_knots_ind_temp))*0.4:
                        v_insert_ind += [v_ind,]
            else:
                v_insert_ind = v_knots_ind_temp

            u_knots_insert_temp = u_knots_corret_init[u_insert_ind]
            v_knots_insert_temp = v_knots_corret_init[v_insert_ind]

            for u_knot in u_knots_insert_temp:
                if u_knot not in bs_data.UKnots:
                    u_knots_insert_res += [u_knot,]
            for v_knot in v_knots_insert_temp:
                if v_knot not in bs_data.VKnots:
                    v_knots_insert_res += [v_knot,]
        else:
            u_knots_insert_res = np.sort(np.concatenate([u_knots_insert, 
                                         u_knots_add_insert]))
            v_knots_insert_res = np.sort(np.concatenate([v_knots_insert, 
                                         v_knots_add_insert]))

    u_knots_insert_res = np.array(u_knots_insert_res)
    v_knots_insert_res = np.array(v_knots_insert_res)

    return (u_knots_insert_res, v_knots_insert_res)


def refine_BSpline_surface(occ_bs_surf, u_degree=3, v_degree=3,
                           u_num_insert=0, v_num_insert=0,
                           correct_element_shape=True):
    """
    Increase B-Spline surface order, insert knots and 
    correct element shape if needed

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    u_degree : int, default is 3
    v_degree : int, default is 3
    u_num_insert : int, default is 0
    v_num_insert : int, default is 0
    correct_element_shape : bool, default is True

    Returns
    -------
    occ_bs_surf : OCC BSplineSurface
    """
    # Increase B-Spline surface order
    if occ_bs_surf.UDegree() > u_degree:
        print("Current u degree is greater than input u degree.")
    if occ_bs_surf.VDegree() > v_degree:
        print("Current v degree is greater than input v degree.")
    occ_bs_surf.IncreaseDegree(u_degree, v_degree)

    bs_data = BSplineSurfaceData(occ_bs_surf) 
    u_knots, v_knots = bs_data.UKnots, bs_data.VKnots

    if u_num_insert > 0:
        u_knots_pop = []
        u_knots_insert_temp = np.linspace(u_knots[0], u_knots[-1], 
                                          u_num_insert+2)[1:-1]
        u_dist_avg = np.average(np.diff(u_knots_insert_temp))
        for i in u_knots:
            for j in u_knots_insert_temp:
                if abs(i-j) < 0.6*u_dist_avg:
                    u_knots_pop += [j]
        u_knots_insert = [u_knot for u_knot in u_knots_insert_temp 
                          if u_knot not in u_knots_pop]
        u_knots_insert = np.array(u_knots_insert)
    else:
        u_knots_insert = np.array([])

    if v_num_insert > 0:
        v_knots_pop = []
        v_knots_insert_temp = np.linspace(v_knots[0], v_knots[-1], 
                                          v_num_insert+2)[1:-1]
        v_dist_avg = np.average(np.diff(v_knots_insert_temp))
        for i in v_knots:
            for j in v_knots_insert_temp:
                if abs(i-j) < 0.6*v_dist_avg:
                    v_knots_pop += [j]
        v_knots_insert = [v_knot for v_knot in v_knots_insert_temp 
                          if v_knot not in v_knots_pop]
        v_knots_insert = np.array(v_knots_insert)
    else:
        v_knots_insert = np.array([])

    # Correct element shape
    if correct_element_shape:
        u_knots_insert, v_knots_insert = \
            correct_BSpline_surface_element_shape(occ_bs_surf, 
            u_knots_insert, v_knots_insert)

    # TColStdArray1OfReal_u_knots = TColStd_Array1OfReal(1, 
    #                               u_knots_insert.shape[0])
    # TColStdArray1OfInteger_u_mults = TColStd_Array1OfInteger(1, 
    #                             u_knots_insert.shape[0])
    # for i in range(u_knots_insert.shape[0]):
    #     TColStdArray1OfReal_u_knots.SetValue(i+1, u_knots_insert[i])
    #     TColStdArray1OfInteger_u_mults.SetValue(i+1, 1)

    # TColStd1OfReal_v_knots = TColStd_Array1OfReal(1, 
    #                          v_knots_insert.shape[0])
    # TColStd1OfInteger_v_mults = TColStd_Array1OfInteger(1, 
    #                             v_knots_insert.shape[0])
    # for i in range(v_knots_insert.shape[0]):
    #     TColStd1OfReal_v_knots.SetValue(i+1, v_knots_insert[i])
    #     TColStd1OfInteger_v_mults.SetValue(i+1, 1)

    if u_knots_insert.shape[0] > 0:
        TColStdArray1OfReal_u_knots = array2TColStdArray1OfReal(
                                      u_knots_insert)
        TColStdArray1OfInteger_u_mults = array2TColStdArray1OfInteger(
                                         np.ones(u_knots_insert.shape[0], 
                                         dtype=int))
        occ_bs_surf.InsertUKnots(TColStdArray1OfReal_u_knots,
                                 TColStdArray1OfInteger_u_mults)

    if v_knots_insert.shape[0] > 0:
        TColStdArray1OfReal_v_knots = array2TColStdArray1OfReal(
                                      v_knots_insert)
        TColStdArray1OfInteger_v_mults = array2TColStdArray1OfInteger(
                                         np.ones(v_knots_insert.shape[0], 
                                         dtype=int))
        occ_bs_surf.InsertVKnots(TColStdArray1OfReal_v_knots,
                                 TColStdArray1OfInteger_v_mults)

    return occ_bs_surf


def BSpline_surface2ikNURBS(occ_bs_surf, p=3, u_num_insert=0, 
                            v_num_insert=0, refine=False):
    """
    Convert OCC BSplineSurface to igakit NURBS and refine the 
    surface via knot insertion and order elevation as need.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    p : int, optional, default is 3.
    u_num_insert : int, optional, default is 0.
    v_num_insert : int, optional, default is 0.
    refine : bool, default is True.

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

def get_int_cs_coords(int_cs, unique_coord=True):
    """
    Return the coordinates of the intersections between a curve and 
    a surface.

    Parameters
    ----------
    int_cs : OCC GeomAPI_IntCS
    unique_coord : bool, optional, default is True

    Return
    ------
    points_coord_res : ndarray
    """
    if int_cs.NbPoints() > 0:
        points_coord = np.zeros((int_cs.NbPoints(), 3))
        for i in range(int_cs.NbPoints()):
            points_coord[i,:] = int_cs.Point(i+1).Coord()

        points_coord = np.unique(points_coord, axis=0)

        if unique_coord:
            points_coord_res = [points_coord[0]]
            for i in range(len(points_coord)-1):
                if not np.allclose(points_coord[i], points_coord[i+1]):
                    points_coord_res += [points_coord[i+1],]
            points_coord_res = np.array(points_coord_res)
        else:
            points_coord_res = points_coord

    else:
        points_coord_res = np.array([])
    return points_coord_res

def count_knots_multiplicity(knots):
    """
    Count the knots vector multiplicity of each elements.

    Parameters
    ----------
    knots : ndarray

    Returns
    -------
    knot_mult : ndarray
    """
    knot_old = knots[0]
    knot_mult = []
    mult_count = int(1)
    for i in range(1, len(knots)):
        if knots[i] == knot_old:
            mult_count += 1
        else:
            knot_mult += [mult_count,]
            mult_count = 1
        if i == len(knots)-1:
            knot_mult += [mult_count,]
        knot_old = knots[i]
    knot_mult = np.array(knot_mult, dtype="int")
    return knot_mult

def ikNURBS2BSpline_surface(ik_nurbs):
    """
    Convert igakit NNURBS to OCC Geom_BSplineSurface.

    Parameters
    ----------
    ik_nurbs : igakit NURBS

    Returns
    -------
    BSpline_surface : OCC Geom_BSplineSurface
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

    BSpline_surface = Geom_BSplineSurface(poles, weights, u_knots, v_knots, 
                                          u_mults, v_mults, p_u, p_v)

    return BSpline_surface

def BSpline_surface_section(BSpline, para_loc, u_degree, v_degree, 
                            continuity=None, tol3D=1e-4):
    """
    Returns section of the OCC BSplineSurface based on parametric 
    locations. 

    Parameters
    ----------
    BSpline : OCC BSplineSurface
    para_loc : ndarray, parametric locations for new surface
    u_degree : int
    v_degree : int
    continuity : int, optional, default is None
    tol3D : float, optional, default is 1e-4

    Returns
    -------
    bs_sec : OCC BSplineSurface
    """
    if continuity is None:
        continuity = max(u_degree, v_degree)

    bs_sec_pts = TColgp_Array2OfPnt(1, para_loc.shape[0], 
                                    1, para_loc.shape[1])
    pt_temp = gp_Pnt()
    for i in range(para_loc.shape[0]):
        for j in range(para_loc.shape[1]):
            BSpline.D0(para_loc[i,j,0], para_loc[i,j,1], pt_temp)
            bs_sec_pts.SetValue(i+1, j+1, pt_temp)

    bs_sec = GeomAPI_PointsToBSplineSurface(bs_sec_pts, u_degree, v_degree, 
                                            continuity, tol3D).Surface()
    return bs_sec


class BSplineCurveData(object):
    """
    Class contains informations of an OCC B-spline curve.
    """
    def __init__(self, curve, normalize=True):
        """
        Gets the properties of a OCC curve.

        Parameters
        ----------
        curve : OCC B-spline curve
        normalize : bool, optional, default is True.
        """
        self.curve = curve
        self.normalize = normalize

        self.Knots = TColStdArray1OfReal2Array(self.curve.Knots())
        self.Multiplicities = TColStdArray1OfReal2Array(
                              self.curve.Multiplicities())
        if normalize:
            self.Knots = self.Knots/self.Knots[-1]
        self.degree = self.curve.Degree()

    @property
    def control(self):
        """
        Return the control points of the B-spline curve.

        Returns
        -------
        control_points : ndarray
        """
        control_points = np.zeros((self.curve.NbPoles(), 4))
        curve_poles = self.curve.Poles()
        for i in range(self.curve.NbPoles()):
            control_points[i, :-1] = curve_poles.Value(i+1).Coord()
            control_points[i, -1] = self.curve.Weight(i+1)
        return control_points

    @property
    def knots(self):
        """
        Return the knots vector of the B-spline curve.

        Returns
        -------
        knots : list of floats
        """
        knots = []
        curve_knots = self.curve.Knots()
        curve_mult = self.curve.Multiplicities()
        for i in range(self.curve.NbKnots()):
            knots += [curve_knots.Value(i+1)]*curve_mult.Value(i+1)
        knots = np.array(knots)
        if self.normalize:
            knots = knots/np.max(knots)
        return knots

    @property
    def weights(self):
        """
        Return the weights of the B-spline curve.

        Returns
        -------
        res : ndarray
        """
        return self.control[:,-1]

class BSplineSurfaceData(object):
    """
    Class contains informations of an OCC B-spline curve.
    """
    def __init__(self, surface, normalize=True):
        """
        Get the properties of a OCC surface.

        Parameters
        ----------
        surface : OCC B-spline surface
        normalize : bool, optional. Default is True.
        """
        self.surface = surface
        self.normalize = normalize

        self.UKnots = TColStdArray1OfReal2Array(self.surface.UKnots())
        self.VKnots = TColStdArray1OfReal2Array(self.surface.VKnots())
        self.UMultiplicities = TColStdArray1OfReal2Array(
            self.surface.UMultiplicities(), dtype="int32")
        self.VMultiplicities = TColStdArray1OfReal2Array(
            self.surface.VMultiplicities(), dtype="int32")
        if normalize:
            self.UKnots = self.UKnots/self.UKnots[-1]
            self.VKnots = self.VKnots/self.VKnots[-1]

        self.degree = (self.surface.UDegree(), self.surface.VDegree())
    
    @property
    def control(self):
        """
        Return the control points of the B-spline surface.

        Returns
        -------
        control_points : ndarray
        """
        control_points = np.zeros((self.surface.NbUPoles(),
                                   self.surface.NbVPoles(), 4))
        surface_poles = self.surface.Poles()
        for i in range(self.surface.NbUPoles()):
            for j in range(self.surface.NbVPoles()):
                control_points[i,j,:-1] = surface_poles.Value(i+1,j+1).Coord()
                control_points[i,j,-1] = self.surface.Weight(i+1, j+1)
        return control_points

    @property
    def knots(self):
        """
        Return the knots vector of the B-spline surface.

        Returns
        -------
        knots : list of floats
        """
        ukonts, vkonts = [], []
        surface_ukonts = self.surface.UKnots()
        surface_umult = self.surface.UMultiplicities()
        surface_vkonts = self.surface.VKnots()
        surface_vmult = self.surface.VMultiplicities()
        for i in range(self.surface.NbUKnots()):
            ukonts += [self.UKnots[i]]*self.UMultiplicities[i]
        for i in range(self.surface.NbVKnots()):
            vkonts += [self.VKnots[i]]*self.VMultiplicities[i]
        ukonts = np.array(ukonts)
        vkonts = np.array(vkonts)
        if self.normalize:
            ukonts = ukonts/np.max(ukonts)
            vkonts = vkonts/np.max(vkonts)
        return (ukonts, vkonts)

    @property
    def weights(self):
        """
        Return the weights of the B-spline surface.

        Returns
        -------
        res : ndarray
        """
        return self.control[:,:,-1]

class BSplineSurfacesConnectedEdges(object):
    """
    Class computes the connected edges between two OCC B-spline 
    surfaces based on their control points.
    """
    def __init__(self, surf1, surf2):
        """
        Parameters
        ----------
        surf1 : OCC B-spline surface
        surf2 : OCC B-spline surface
        """
        self.surf1 = surf1
        self.surf2 = surf2

    @property
    def connected_edges(self):
        """
        Compute the OCC B-spline curve of the connected edges.

        Returns
        -------
        connected_edges : list of OCC Geom_Curves
        """
        face0 = make_face(self.surf1, 1e-6)
        edges0 = get_face_edges(face0)
        self.connected_edges0 = []
        for i in range(len(edges0)):
            int_cs = GeomAPI_IntCS(edges0[i], self.surf2)
            int_cs_coord = get_int_cs_coords(int_cs, unique_coord=True)
            if len(int_cs_coord) > 2:
                self.connected_edges0 += [edges0[i],]

        face1 = make_face(self.surf2, 1e-6)
        edges1 = get_face_edges(face1)
        self.connected_edges1 = []
        for i in range(len(edges1)):
            int_cs = GeomAPI_IntCS(edges1[i], self.surf1)
            int_cs_coord = get_int_cs_coords(int_cs, unique_coord=True)
            if len(int_cs_coord) > 2:
                self.connected_edges1 += [edges1[i],]

        if len(self.connected_edges0) == len(self.connected_edges1):
            connected_edges = self.connected_edges0
        else:
            connected_edges = []

        return connected_edges

    @property
    def num_connected_edges(self):
        """
        Return the number of connected edges between two OCC B-spline
        surfaces.

        Returns
        -------
        res : int
        """
        return len(self.connected_edges)

    def get_coordinate(self, ind, num_pts=20, sort_axis=None):
        """
        Return the physical coordinates of ``ind``-th connected edge.

        Parameters
        ----------
        ind : int
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        connected_edge_coords : ndarray
        """
        assert ind < self.num_connected_edges and ind >= 0
        connected_edge_coords = get_curve_coord(self.intersections[ind],
                                              num_pts, sort_axis)
        return connected_edge_coords
    
    def get_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the coordinates of the connected edges.

        Parameters
        ----------
        num_pts : int
        sort_axis : int, {1, 2, 3}, optional 

        Returns
        -------
        connected_edges_coords : list of ndarray
        """
        connected_edges_coords = []
        if self.num_connected_edges > 0:
            for i in range(len(self.connected_edges)):
                connected_edges_coords += [get_curve_coord(
                                           self.connected_edges[i], 
                                           num_pts, sort_axis)]
        else:
            print("Surface-surface connected edges are not detected, returns",
                  " empty list.")
        return connected_edges_coords

    def get_parametric_coordinate(self, ind, num_pts=20, sort_axis=None):
        """
        Return the parametric coordinates of ``ind``-th connected edge.

        Parameters
        ----------
        ind : int
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        connected_edge_para_coords : list of two ndarray
        """
        assert ind < self.num_connected_edges and ind >= 0

        edge_phy_coords = self.get_coordinate(ind, num_pts, sort_axis)
        connected_edge_para_coords = [parametric_coord(edge_phy_coords, 
                                                       self.surf1), 
                                      parametric_coord(edge_phy_coords, 
                                                       self.surf2)]
        return connected_edge_para_coords

    def get_parametric_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the parametric coordinates of the connected edges.

        Parameters
        ---------- 
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        connected_edges_para_coords : list of lists that contains two ndarray
        """
        connected_edges_para_coords = []
        if self.num_connected_edges > 0:
            for i in range(self.num_connected_edges):
                connected_edges_para_coords += \
                    [self.get_parametric_coordinate(i, num_pts, sort_axis)]
        else:
            print("Surface-surface connected edges are not detected, returns",
                  " empty list.")
        return connected_edges_para_coords


class BSplineSurfacesIntersections(BSplineSurfacesConnectedEdges):
    """
    Class computes intersections between two B-spline surfaces.
    """
    def __init__(self, surf1, surf2, rtol=1e-6):
        """
        Parameters
        ----------
        surf1 : OCC B-spline surface
        surf2 : OCC B-spline surface
        rtol : float, optional. Default is 1e-6.
        """
        super().__init__(surf1, surf2)
        self.int_ss = GeomAPI_IntSS(surf1, surf2, rtol)

    @property
    def num_intersections(self):
        """
        Return the number of intersections between two surfaces. If these
        two surfaces have connected surfaces, the number of intersections
        is equal to the number of connected edges. Assume the connected 
        edges and the surface-surface intersection don't exist at the same
        time. Check the connected edges first.

        Returns
        -------
        num_intersections : int
        """
        if self.num_connected_edges > 0:
            num_intersections = self.num_connected_edges
        else:
            if self.int_ss.NbLines() > 1:
                # Check if this is surface-edge intersection but computed 
                # as multiple intersections by ``GeomAPI_IntSS``.
                if (len(self.connected_edges0) > 0 and 
                    len(self.connected_edges1) == 0):
                    num_intersections = len(self.connected_edges0)
                elif (len(self.connected_edges0) == 0 and 
                      len(self.connected_edges1) > 0):
                    num_intersections = len(self.connected_edges1)
                else:
                    num_intersections = self.int_ss.NbLines()
            else:   
                num_intersections = self.int_ss.NbLines()
        return num_intersections

    @property
    def intersections(self):
        """
        Return the intersection curves.

        Returns
        -------
        intersections : list of OCC Geom_Curves
        """
        intersections = []
        if self.num_connected_edges > 0:
            intersections = self.connected_edges
        else:
            if self.int_ss.NbLines() > 1:
                if (len(self.connected_edges0) > 0 and 
                    len(self.connected_edges1) == 0):
                    intersections = self.connected_edges0
                elif (len(self.connected_edges0) == 0 and 
                      len(self.connected_edges1) > 0):
                    intersections = self.connected_edges1
                else:
                    for i in range(self.int_ss.NbLines()):
                        intersections += [self.int_ss.Line(i+1)]
            else:   
                for i in range(self.int_ss.NbLines()):
                    intersections += [self.int_ss.Line(i+1)]
        return intersections

    def get_coordinate(self, ind, num_pts=20, sort_axis=None):
        """
        Return the physical coordinates of ``ind``-th intersection curve.

        Parameters
        ----------
        ind : int
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        int_coords : ndarray
        """
        assert ind < self.num_intersections and ind >= 0
        int_coords = get_curve_coord(self.intersections[ind],
                                     num_pts, sort_axis)
        return int_coords

    def get_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the physical coordinates of the intersection curves.

        Parameters
        ---------- 
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        ints_coords : list of ndarray
        """
        ints_coords = []
        if self.num_intersections > 0:
            for i in range(self.num_intersections):
                ints_coords += [get_curve_coord(self.intersections[i], 
                                                num_pts, sort_axis)]
        else:
            print("Surface-surface intersections are not detected, returns ",
                  "empty list.")
        return ints_coords

    def get_parametric_coordinate(self, ind, num_pts=20, sort_axis=None):
        """
        Return the parametric coordinates of ``ind``-th intersection curve.

        Parameters
        ----------
        ind : int
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        int_para_coords : list of two ndarray
        """
        assert ind < self.num_intersections and ind >= 0
        int_phy_coords = self.get_coordinate(ind, num_pts, sort_axis)
        int_para_coords = [parametric_coord(int_phy_coords, self.surf1), 
                           parametric_coord(int_phy_coords, self.surf2)]
        return int_para_coords

    def get_parametric_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the parametric coordinates of the intersection curves.

        Parameters
        ---------- 
        num_pts : int, optional, default is 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        ints_para_coords : list of lists that contains two ndarray
        """
        ints_para_coords = []
        if self.num_intersections > 0:
            for i in range(self.num_intersections):
                ints_para_coords += [self.get_parametric_coordinate(i, 
                                     num_pts, sort_axis)]
        else:
            print("Surface-surface intersections are not detected, returns ",
                  "empty list.")
        return ints_para_coords


if __name__ == "__main__":
    pass