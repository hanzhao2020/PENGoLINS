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

def BSpline_surface_interior_multiplicity(occ_bs_surface):
    """
    Return the multiplicities of the inner elements of knot vectors.

    Parameters
    ----------
    occ_bs_surface : OCC B-spline surface

    Returns
    -------
    u_multiplicity : int
    v_multiplicity : int
    """
    bs_data = BSplineSurfaceData(occ_bs_surface)
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

def reconstruct_BSpline_surface(occ_bs, u_num_eval=30, v_num_eval=30, 
                                bs_degree=3, bs_continuity=3, 
                                tol3D=1e-3, geom_scale=1.):
    """
    Return reconstructed B-spline surface by evaluating the positions
    of the original surface.

    Parameters
    ----------
    occ_bs : OCC BSplineSurface
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
    occ_bs_res_pts = TColgp_Array2OfPnt(1, u_num_eval, 1, v_num_eval)
    para_u = np.linspace(occ_bs.Bounds()[0], occ_bs.Bounds()[1], u_num_eval)
    para_v = np.linspace(occ_bs.Bounds()[2], occ_bs.Bounds()[3], v_num_eval)
    pt_temp = gp_Pnt()
    for i in range(u_num_eval):
        for j in range(v_num_eval):
            occ_bs.D0(para_u[i], para_v[j], pt_temp)
            pt_temp0 = gp_Pnt(pt_temp.Coord()[0]*geom_scale, 
                              pt_temp.Coord()[1]*geom_scale, 
                              pt_temp.Coord()[2]*geom_scale)
            occ_bs_res_pts.SetValue(i+1, j+1, pt_temp0)
    occ_bs_res = GeomAPI_PointsToBSplineSurface(occ_bs_res_pts, bs_degree, 
                                                bs_degree, bs_continuity, 
                                                tol3D).Surface()
    return occ_bs_res


def BSpline_surface2ikNURBS(occ_bs_surface, p=3, u_num_insert=0, 
                            v_num_insert=0, refine=True):
    """
    Convert OCC BSplineSurface to igakit NURBS and refine the 
    surface via knot insertion and order elevation as need.

    Parameters
    ----------
    occ_bs_surface : OCC BSplineSurface
    p : int, optional, default is 3.
    u_num_insert : int, optional, default is 0.
    v_num_insert : int, optional, default is 0.
    refine : bool, default is True.

    Returns
    -------
    ikNURBS : igakit NURBS
    """
    bs_data = BSplineSurfaceData(occ_bs_surface)
    ikNURBS = NURBS(bs_data.knots, bs_data.control)

    if refine:
        u_multiplicity, v_multiplicity = \
            BSpline_surface_interior_multiplicity(occ_bs_surface)
        ikNURBS.elevate(0, p-ikNURBS.degree[0])
        ikNURBS.elevate(1, p-ikNURBS.degree[1])
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