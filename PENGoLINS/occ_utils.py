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
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face
from OCC.Core.Approx import Approx_ParametrizationType
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import (brepgprop_LinearProperties,
                                brepgprop_SurfaceProperties,
                                brepgprop_VolumeProperties)
from OCC.Extend.DataExchange import (read_step_file, read_iges_file,
                                     write_step_file, write_iges_file,
                                     TopoDS_Shape, TopoDS_Compound,
                                     BRep_Builder)
from OCC.Extend.ShapeFactory import point_list_to_TColgp_Array1OfPnt
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Display.SimpleGui import init_display

from PENGoLINS.math_utils import *

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

def write_geom_file(geom, filename):
    """
    Write OCC Geom surfaces or TopoDS Faces to igs file. If 
    a list of the geometries is given, write them to a 
    compound file.

    Parameters
    ----------
    geom : OCC geometry (i.e., TopoDS_Face, TopoDS_Edge, 
           TopoDS_Shape, Geom_Surface, Geom_BSplineSurface,
           Geom_Curve, Geom_BSplineCurve) or a list of geometry
    filename : str
    """
    if isinstance(geom, list):
        brep_builder = BRep_Builder()
        topods_compound = TopoDS_Compound()
        brep_builder.MakeCompound(topods_compound)

        for i in range(len(geom)):
            if isinstance(geom[i], Geom_Surface) or \
                isinstance(geom[i], Geom_BSplineSurface):
                brep_builder.Add(topods_compound, 
                                 make_face(geom[i], 1e-9))
            elif isinstance(geom[i], Geom_Curve) or \
                isinstance(geom[i], Geom_BSplineCurve):
                brep_builder.Add(topods_compound, 
                                 make_edge(geom[i], 1e-9))
            else:
                brep_builder.Add(topods_compound, geom[i])
        to_save = topods_compound
    else:
        if isinstance(geom, Geom_Surface) or \
            isinstance(geom, Geom_BSplineSurface):
            to_save = make_face(geom, 1e-9)
        elif isinstance(geom, Geom_Curve) or \
            isinstance(geom, Geom_BSplineCurve):
            to_save = make_edge(geom, 1e-9)
        else:
            to_save = geom

    file_format = filename.split(".")[-1]
    if file_format.lower() in ["igs", "iges"]:
        write_iges_file(to_save, filename)
    elif file_format.lower() in ["stp", "step"]:
        write_step_file(to_save, filename)
    else:
        raise ValueError("{} format is not supported.".format(file_format))

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
    TColStdArray = TColStd_Array1OfReal(1, np_array.shape[0])
    for i in range(np_array.shape[0]):
        TColStdArray.SetValue(i+1, np_array[i])
    return TColStdArray

def array2TColStdArray2OfReal(np_array):
    """
    Convert ndarray to 2D real TColStdArray.

    Parameters
    ----------
    np_array : ndarray

    Returns
    -------
    TColStdArray : TColStd_Array2OfReal
    """
    TColStdArray = TColStd_Array2OfReal(1, np_array.shape[0], 
                                        1, np_array.shape[1])
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            TColStdArray.SetValue(i+1, j+1, np_array[i,j])
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
    TColStdArray = TColStd_Array1OfInteger(1, np_array.shape[0])
    for i in range(np_array.shape[0]):
        TColStdArray.SetValue(i+1, int(np_array[i]))
    return TColStdArray

def array2TColStdArray2OfInteger(np_array):
    """
    Convert ndarray to 2D integer TColStdArray.

    Parameters
    ----------
    np_array : ndarray

    Returns
    -------
    TColStdArray : TColStd_Array2OfInteger
    """
    TColStdArray = TColStd_Array2OfInteger(1, np_array.shape[0], 
                                           1, np_array.shape[1])
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            TColStdArray.SetValue(i+1, j+1, int(np_array[i,j]))
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

def surface2topoface(surface, tol=1e-9):
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

def copy_BSpline_surface(bs):
    """
    Create a new BSpline surface same with ``bs``.

    Parameters
    ----------
    bs : OCC BSpline Surface

    Returns
    -------
    bs_new : OCC BSpline Surface
    """
    n_rows = bs.Poles().NbRows()
    n_cols = bs.Poles().NbColumns()
    weights_array = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            weights_array[i,j] = bs.Weight(i+1,j+1)
    weights = array2TColStdArray2OfReal(weights_array)
    bs_new = Geom_BSplineSurface(bs.Poles(), weights, 
                                 bs.UKnots(), bs.VKnots(), 
                                 bs.UMultiplicities(), bs.VMultiplicities(), 
                                 bs.UDegree(), bs.VDegree(),
                                 bs.IsUPeriodic(), bs.IsVPeriodic())
    return bs_new

def get_curve_coord(curve, num_pts=20, sort_axis=None, 
                    flip=False, cut_side=None, cut_ratio=0):
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
    cut_side: None or str {"left", "right", "both"}, optional
        The side of the curve that will be ignored when
        measuring physical coordinates
    cut_ratio : float, optional
        The ratio of total parametric length to cut on the
        ``cut_side`` side, this ratio is designed for the 
        intersection between surfaces that have singularity. 
        Cut the part of the end of the intersection can help 
        reducing the stress concentration near the surface 
        singularity.
    
    Returns
    -------
    curve_coord : ndarray
    """
    curve_coord = np.zeros((num_pts, 3))
    para_range = curve.LastParameter() - curve.FirstParameter()
    if cut_side is None:
        u_para = np.linspace(curve.FirstParameter(),
                             curve.LastParameter(), num_pts)
    elif cut_side == "left":
        u_para = np.linspace(curve.FirstParameter()+cut_ratio*para_range,
                             curve.LastParameter(), num_pts)
    elif cut_side == "right":
        u_para = np.linspace(curve.FirstParameter(),
                             curve.LastParameter()-cut_ratio*para_range, 
                             num_pts)
    elif cut_side == "both":
        u_para = np.linspace(curve.FirstParameter()+cut_ratio*para_range,
                             curve.LastParameter()-cut_ratio*para_range, 
                             num_pts)
    else:
        if mpirank == 0:
            raise ValueError("Undefined ``cut_side`` name ", cut_side)

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

def curve_length(curve):
    """
    Measure the length of a given ``curve``.

    Parameters
    ----------
    curve : OCC Geom_Curve or Geom_BSplineCurve

    Returns
    -------
    length : float
    """
    edge = make_edge(curve)
    prop = GProp_GProps()
    brepgprop_LinearProperties(edge, prop)
    length = prop.Mass()
    return length

def surface_area(surface, tol=1e-9):
    """
    Returns the area of a given ``surface``.

    Parameters
    ----------
    surface : OCC Geom_Surface or Geom_BSplineSurface

    Returns
    -------
    area : float
    """
    face = make_face(surface, tol)
    prop = GProp_GProps()
    brepgprop_LinearProperties(face, prop)
    area = prop.Mass()
    return area

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
        pt_proj = GeomAPI_ProjectPointOnSurf(pt, surf, 1e-9)
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
        if mpirank == 0:
            print("*** Warning: u knots have interior multiplicity " 
                  "greater than 1 ***")
    if False in rem_u_knots:
        if mpirank == 0:
            print("Excessive u knots are not removed!")
    elif True in rem_u_knots:
        if mpirank == 0:
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
        if mpirank == 0:
            print("*** Warning: v knots have interior multiplicity "
                  "greater than 1 ***")
    if False in rem_v_knots:
        if mpirank == 0:
            print("Excessive v knots are not removed!")
    elif True in rem_v_knots:
        if mpirank == 0:
            print("Excessive v knots are removed with tolerance: ", geom_tol)

    return occ_bs_surf

def remove_surf_dense_knots(occ_bs_surf, dist_ratio_remove=0.5, 
                            rtol=1e-2, max_rtol_ratio=10):
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
    dist_ratio_remove : float, default is 0.5
        The ratio of average knots distance, if two the distance
        between two knows is smaller the this ratio times average
        distance, then one of the knots will be removed with 
        the following ``rtol``
    rtol : float, default is 1e-2
    max_rtol_ratio : int, default is 10
        The maximum ratio for the relative tolerance when removing
        knots

    Returns
    -------
    occ_bs_surf : OCC BSplineSurface
    """
    # print("Removing knots")

    for i in range(3):  # Use three iterations to remove dense knots
        bs_data = BSplineSurfaceData(occ_bs_surf)
        geom_tol = rtol*surface_area(occ_bs_surf)

        # Check u direction
        u_knots_diff = np.diff(bs_data.UKnots)
        u_knots_diff_avg = np.average(u_knots_diff)
        u_ind_off = 2
        for u_diff_ind, u_knot_dist in enumerate(u_knots_diff):
            if u_knot_dist < u_knots_diff_avg*dist_ratio_remove:
                rem_u_ind = u_diff_ind + u_ind_off
                if rem_u_ind == occ_bs_surf.NbUKnots():
                    rem_u_ind -= 1
                rem_u_knot = False
                geom_tol_u = geom_tol
                while rem_u_knot is not True:
                    rem_u_knot = occ_bs_surf.RemoveUKnot(rem_u_ind, 
                                                         0, geom_tol_u)
                    geom_tol_u = geom_tol_u*(1+1e-3)
                    if geom_tol_u > geom_tol*max_rtol_ratio:
                        break
                # 0 indicates knot multiplicity.
                # rem_u_knot is boolean, True means the knot is removed.
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
            if v_knot_dist < v_knots_diff_avg*dist_ratio_remove:
                rem_v_ind = v_diff_ind + v_ind_off
                if rem_v_ind == occ_bs_surf.NbVKnots():
                    rem_v_ind -= 1
                rem_v_knot = False
                geom_tol_v = geom_tol
                while rem_v_knot is not True:
                    rem_v_knot = occ_bs_surf.RemoveVKnot(rem_v_ind, 
                                                         0, geom_tol_v)
                    geom_tol_v = geom_tol_v*(1+1e-3)
                    if geom_tol_v > geom_tol*max_rtol_ratio:
                        break
                # 0 indicates knot multiplicity.
                # rem_v_knot is boolean, True means the knot is removed.
                if rem_v_knot:
                    v_ind_off -= 1
                    # print("v knots is removed")
                # else:
                    # print("v knots is not removed")

    return occ_bs_surf

def reparametrize_BSpline_surface(occ_bs_surf, u_num_eval=30, v_num_eval=30, 
                                bs_degree=3, bs_continuity=3, 
                                tol3D=1e-3, geom_scale=1., 
                                remove_dense_knots=True, 
                                dist_ratio_remove=0.5, rtol=1e-2):
    """
    Return reparametrized B-spline surface by evaluating the positions
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

    # for num_iter in range(int(u_num_eval)):
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
    occ_bs_res_temp = GeomAPI_PointsToBSplineSurface(occ_bs_res_pts, 
                    Approx_ParametrizationType(0), bs_degree, bs_degree, 
                    bs_continuity, tol3D)

    # print("reparametrizaion Is Done for BSpline surface", 
    #       occ_bs_res_temp.IsDone())

    occ_bs_res = occ_bs_res_temp.Surface()

    # Check if surface has excessive interior u and v knots
    decrease_knot_multiplicity(occ_bs_res, rtol)
    # Remove densely distributed knots
    if remove_dense_knots:
        remove_surf_dense_knots(occ_bs_res, dist_ratio_remove, rtol)

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
    if u_knots is None:
        surf_data = BSplineSurfaceData(occ_bs_surf)
        u_knots = surf_data.UKnots
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
        if length_0i < 1e-12 or length_1i < 1e-12:
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

    eps = 1e-12
    element_AR = np.zeros(phy_coords.shape[1]-1)
    for i in range(len(element_AR)):
        AR_fac = 1.
        sin_el_angle = 1
        if element_len0[0,i] < eps or element_len0[0,i+1] < eps:
            AR_fac = 2.
        elif element_len1[0,i] < eps or element_len1[1,i] < eps:
            # sin value of the singularity angle
            vec1 = phy_coords[0,i] - phy_coords[1,i]
            vec2 = phy_coords[0,i+1] - phy_coords[1,i+1]
            sin_el_angle = np.linalg.norm(np.cross(vec1,vec2))\
                           /(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            AR_fac = 0.5
        element_AR[i] = (element_len0[0,i]+element_len0[0,i+1])\
                       /(element_len1[0,i]+element_len1[1,i])\
                       *AR_fac*sin_el_angle
    return element_AR

def correct_BSpline_surface_element_shape(occ_bs_surf, 
                                          u_knots_insert, v_knots_insert, 
                                          aspect_ratio_lim=4, dist_ratio=0.7):
    """
    Automatic element shape correction to get reasonable aspect ratio.

    Parameters
    ----------
    occ_bs_surf : OCC BSplineSurface
    u_knots_insert : int
    v_knots_insert : int
    aspect_ratio_lim : float, optional, default is 4
    dist_ratio : float, optional, default is 0.7
        The ratio for averge distance between knots, when the
        difference between an inserting knot and an existing
        knot is larger than the ratio times average distance,
        then this knot can be inserted (same for operation 
        on indices)

    Returns
    -------
    u_knots_insert_res : ndarray
        Knots vector for insertion in u direction after shape correction
    v_knots_insert_res : ndarray
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
        if bs_AR_avg[0][i] > aspect_ratio_lim:
            phy_coords_list = phy_coords[i:i+2,:,:]
            element_AR_avg_u = np.average(compute_list_element_AR(
                                          phy_coords_list))
            if element_AR_avg_u > aspect_ratio_lim:
                u_knots_add_insert += [np.linspace(u_knots_full[i], 
                                       u_knots_full[i+1], round(
                                       (element_AR_avg_u)))[1:-1]]

    for i in range(bs_AR_avg[1].shape[0]):
        bs_AR_avg[1][i] = np.average(bs_AR[:,i])
        if bs_AR_avg[1][i] > aspect_ratio_lim:
            phy_coords_list = phy_coords[:,i:i+2,:]
            element_AR_avg_v = np.average(compute_list_element_AR(
                                          phy_coords_list))
            if element_AR_avg_v > aspect_ratio_lim:
                v_knots_add_insert += [np.linspace(v_knots_full[i],
                                       v_knots_full[i+1], round(
                                       (element_AR_avg_v)))[1:-1]]
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

        num_insert_max = np.max([len(u_knots_insert)+len(bs_data.UKnots), 
                                 len(v_knots_insert)+len(bs_data.VKnots)])
        num_u_knots = len(u_knots_corret_init)
        num_v_knots = len(v_knots_corret_init)
        num_knots_max = np.max([num_u_knots, num_v_knots])

        if num_knots_max > num_insert_max:
            # print("Removing over refined knots")
            # print("num_insert_max:", num_insert_max)
            num_insert_ratio = num_insert_max/num_knots_max
            # num_u_insert = ceil(num_u_knots*num_insert_ratio)
            # num_v_insert = ceil(num_v_knots*num_insert_ratio)
            num_u_insert = floor(num_u_knots*num_insert_ratio)
            num_v_insert = floor(num_v_knots*num_insert_ratio)
            # print("num_u_insert:", num_u_insert)
            # print("num_v_insert:", num_v_insert)

            u_knots_ind_temp_all = np.array(np.linspace(0, num_u_knots+1, 
                                   num_u_insert+2), dtype=int)
            u_knots_ind_temp = u_knots_ind_temp_all[1:-1]
            v_knots_ind_temp_all = np.array(np.linspace(0, num_v_knots+1, 
                                   num_v_insert+2), dtype=int)
            v_knots_ind_temp = v_knots_ind_temp_all[1:-1]

            u_insert_ind = []
            if len(u_knots_init_ind) > 0:
                for u_ind in u_knots_ind_temp:
                    if np.min(np.abs(u_ind-u_knots_init_ind)) > \
                        np.average(np.diff(u_knots_ind_temp_all))*dist_ratio:
                        u_insert_ind += [u_ind,]
            else:
                u_insert_ind = u_knots_ind_temp

            v_insert_ind = []
            if len(v_knots_init_ind) > 0:
                for v_ind in v_knots_ind_temp:
                    if np.min(np.abs(v_ind-v_knots_init_ind)) > \
                        np.average(np.diff(v_knots_ind_temp_all))*dist_ratio:
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

def remove_knots_near_singularity(occ_bs_surf, u_knots_insert, 
                                  v_knots_insert, max_remove_percent=0.1):
    """
    This function is used to remove knots near the singulartiy to
    prevent the very tiny elements.

    Parameters
    ----------
    occ_bs_surf : OCC Geom BSplienSurface
    u_knots_insert : ndarray
    v_knots_insert : ndarray
    max_remove_percent : float, optional, default is 0.1
        Maximum percent of number of knots that can be removed
    """
    bs_data = BSplineSurfaceData(occ_bs_surf)
    u_knots = bs_data.UKnots
    v_knots = bs_data.VKnots
    num_u_insert = len(u_knots_insert)
    num_v_insert = len(v_knots_insert)
    phy_coords = knots_geom_mapping(occ_bs_surf, u_knots, v_knots)

    eps = 1e-12  # tolerance for singularity

    # Check singularities along u direction
    if np.linalg.norm(phy_coords[0,0] - phy_coords[0,-1]) < eps:
        vec1 = phy_coords[1,0]- phy_coords[0,0]
        vec2 = phy_coords[1,-1]- phy_coords[0,-1]
        cos_angle_u0 = np.linalg.norm(np.dot(vec1,vec2))\
                       /(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        num_remove_u0 = round(num_u_insert*max_remove_percent*cos_angle_u0**2)
        start_ind_u = num_remove_u0
    else:
        start_ind_u = 0

    if np.linalg.norm(phy_coords[-1,0] - phy_coords[-1,-1]) < eps:
        vec1 = phy_coords[-2,0] - phy_coords[-1,0]
        vec2 = phy_coords[-2,-1] - phy_coords[-1,-1]
        cos_angle_u1 = np.linalg.norm(np.dot(vec1,vec2))\
                       /(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        num_remove_u1 = round(num_u_insert*max_remove_percent*cos_angle_u1**2)
        end_ind_u = num_u_insert - num_remove_u1
    else:
        end_ind_u = num_u_insert

    # Check singularities along v direction
    if np.linalg.norm(phy_coords[0,0] - phy_coords[-1,0]) < eps:
        vec1 = phy_coords[0,1]- phy_coords[0,0]
        vec2 = phy_coords[-1,1]- phy_coords[-1,0]
        cos_angle_v0 = np.linalg.norm(np.dot(vec1,vec2))\
                       /(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        num_remove_v0 = round(num_v_insert*max_remove_percent*cos_angle_v0**2)
        start_ind_v = num_remove_v0
    else:
        start_ind_v = 0

    if np.linalg.norm(phy_coords[0,-1] - phy_coords[-1,-1]) < eps:
        vec1 = phy_coords[0,-2] - phy_coords[0,-1]
        vec2 = phy_coords[-1,-2] - phy_coords[-1,-1]
        cos_angle_v1 = np.linalg.norm(np.dot(vec1,vec2))\
                       /(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        num_remove_v1 = round(num_v_insert*max_remove_percent*cos_angle_v1**2)
        end_ind_v = num_v_insert - num_remove_v1
    else:
        end_ind_v = num_v_insert

    u_knots_insert_res = u_knots_insert[start_ind_u:end_ind_u]
    v_knots_insert_res = v_knots_insert[start_ind_v:end_ind_v]

    return (u_knots_insert_res, v_knots_insert_res)

def refine_BSpline_surface(occ_bs_surf, u_degree=3, v_degree=3,
                           u_num_insert=0, v_num_insert=0,
                           correct_element_shape=True,
                           aspect_ratio_lim=1.8, dist_ratio=0.7,
                           copy_surf=True):
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
    dist_ratio : float, optional, default is 0.7
        The ratio for averge distance between knots, when the
        difference between an inserting knot and an existing
        knot is larger than the ratio times average distance,
        then this knot can be inserted (same for operation 
        on indices)
    correct_element_shape : bool, default is True
    copy_surf : bool, optional, default is True
        If True, perform refinement for copy of the input
        BSpline surface

    Returns
    -------
    occ_bs_surf : OCC BSplineSurface
    """
    if copy_surf:
        occ_bs_surf = copy_BSpline_surface(occ_bs_surf)
    # Increase B-Spline surface order
    if occ_bs_surf.UDegree() > u_degree:
        if mpirank == 0:
            print("*** Warning: Current u degree is greater "
                  "than input u degree.")
    if occ_bs_surf.VDegree() > v_degree:
        if mpirank == 0:
            print("*** Warning: Current v degree is greater "
                  "than input v degree.")

    occ_bs_surf.IncreaseDegree(u_degree, v_degree)

    bs_data = BSplineSurfaceData(occ_bs_surf) 
    u_knots, v_knots = bs_data.UKnots, bs_data.VKnots

    # Remove inserting knots than are too close to existing knots
    # based on aspect ratio
    if u_num_insert > 0:
        u_knots_pop = []
        u_knots_insert_temp = np.linspace(u_knots[0], u_knots[-1], 
                                          u_num_insert+2)[1:-1]
        u_dist_avg = (u_knots[-1] - u_knots[0])/(u_num_insert+1)
        for i in u_knots:
            for j in u_knots_insert_temp:
                if abs(i-j) < dist_ratio*u_dist_avg:
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
        v_dist_avg = (v_knots[-1] - v_knots[0])/(v_num_insert+1)
        for i in v_knots:
            for j in v_knots_insert_temp:
                if abs(i-j) < dist_ratio*v_dist_avg:
                    v_knots_pop += [j]
        v_knots_insert = [v_knot for v_knot in v_knots_insert_temp 
                          if v_knot not in v_knots_pop]
        v_knots_insert = np.array(v_knots_insert)
    else:
        v_knots_insert = np.array([])

    # Remove a few knows near singularity depends on the angle
    u_knots_insert, v_knots_insert = remove_knots_near_singularity(
                                     occ_bs_surf, u_knots_insert, 
                                     v_knots_insert)

    # Correct element shape based on limit aspect ratio
    if correct_element_shape:
        u_knots_insert, v_knots_insert = \
            correct_BSpline_surface_element_shape(occ_bs_surf, 
            u_knots_insert, v_knots_insert, 
            aspect_ratio_lim, dist_ratio)

    # Knot insertion
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

def BSpline_mesh_size(BSpline_surf_data):
    """
    Compute averge mesh size of given BSpline surface data.

    Parameters
    ----------
    BSpline_surf_data : instance of BSplineSurfaceData

    Returns
    -------
    mesh_size_array : ndarray
    """
    surface = BSpline_surf_data.surface
    u_knots = BSpline_surf_data.UKnots
    v_knots = BSpline_surf_data.VKnots
    num_u_knots = len(u_knots)
    num_v_knots = len(v_knots)
    phy_coords = knots_geom_mapping(surface, u_knots, v_knots)
    mesh_size_array = np.zeros((num_u_knots-1, num_v_knots-1))

    for i in range(num_u_knots-1):
        for j in range(num_v_knots-1):
            mesh_size_array[i,j] = np.linalg.norm(phy_coords[i+1,j+1] 
                                                - phy_coords[i,j])

    return mesh_size_array

def point_surface_distance(point, surf):
    """
    Compute the nearest distance between a point and surface.

    Parameters
    ----------
    point : OCC gp_Pnt
    surf : OCC Geom BSplineSurface

    Returns
    -------
    lower_dist : float
    """
    pt_proj = GeomAPI_ProjectPointOnSurf(point, surf)
    lower_dist = pt_proj.LowerDistance()
    return lower_dist


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
        Get the properties of an OCC BSplineSurface.

        Parameters
        ----------
        surface : OCC BSplineSurface
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
        self.check_singularity()
    
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

    def check_singularity(self):
        """
        Check if B-spline surface has singularity. If true, return the 
        singularity positions in an array.
        """
        phy_coords = knots_geom_mapping(self.surface, self.UKnots, 
                                        self.VKnots).reshape(1,-1,3)[0]
        phy_coords_tree = cKDTree(phy_coords)
        singu_pairs = phy_coords_tree.query_pairs(1e-12)
        if len(singu_pairs) > 0:
            self.singularity = True
            # Indices of singularities in ``phy_coords``
            singu_inds = np.unique(np.array(list(singu_pairs)))
            # Coordinates of singularities
            singu_coords_dup = phy_coords[singu_inds]
            # Unique coordinates of singularities, change data type
            # to "float32" to make the coordinates less accurate
            # and get the desired unique coordinates.
            _, singu_ind_temp = np.unique(np.array(singu_coords_dup, 
                                dtype='float32'), True, axis=0)
            singu_coords = singu_coords_dup[singu_ind_temp]
            self.num_singularity = len(singu_coords)
            self.singularity_coords = singu_coords
        else:
            self.singularity = False
            self.num_singularity = 0
            self.singularity_coords = []


if __name__ == "__main__":
    pass