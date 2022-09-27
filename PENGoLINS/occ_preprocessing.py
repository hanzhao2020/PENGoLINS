"""
The "OCC_preprocessing" module
------------------------------
provides class for geometry preprocessing and computation
of surface-surface intersections.
"""

from PENGoLINS.occ_utils import *
from PENGoLINS.nurbs4occ import *

class BSplineSurfacesConnectedEdges(object):
    """
    Class computes the connected edges between two OCC B-spline 
    surfaces based on their control points.
    """
    def __init__(self, surf1, surf2, 
                 check_singularity=True, cut_ratio=0.03):
        """
        Parameters
        ----------
        surf1 : OCC B-spline surface
        surf2 : OCC B-spline surface
        check_singularity : bool, optional
            If True, the algorithm will check if two ends of the
            connected edges coincide withe the surface singularity.
            If so, a portion of the curve,
            ``cut_ratio``*parametric_length, on this end will be 
            ignored when getting physical and parametric coordinates.
            This will help reducing the stress concentration near the 
            surface singularity. Default is True
        cut_ratio : float, optional, default is 0.03
        """
        self.surf1 = surf1
        self.surf2 = surf2
        self.surf1_data = BSplineSurfaceData(self.surf1)
        self.surf2_data = BSplineSurfaceData(self.surf2)
        self.check_singularity = check_singularity
        self.cut_ratio = cut_ratio

    @property
    def connected_edges(self):
        """
        Compute the OCC B-spline curve of the connected edges.

        Returns
        -------
        connected_edges : list of OCC Geom_Curves
        """
        num_pts_lim_on_edge = 2
        num_pts_check_on_edge = 8
        max_pt_surf_dist_ratio = 2e-2

        face0 = make_face(self.surf1, 1e-9)
        edges0 = get_face_edges(face0)
        self.connected_edges0 = []
        for i in range(len(edges0)):
            int_cs = GeomAPI_IntCS(edges0[i], self.surf2)
            int_cs_coord = get_int_cs_coords(int_cs, unique_coord=True)
            if len(int_cs_coord) >= num_pts_lim_on_edge:
                # There are more than 2 intersecting points between
                # this edge and surface, then check ``num_pts_check_on_edge``
                # number of points on the edge and get the max distance
                # between these points and the surface, if the the max 
                # distance is larger than a limit, this edge will not be 
                # treated as intersection edge with the surface.
                first_para = edges0[i].FirstParameter()
                last_para = edges0[i].LastParameter()
                para_check = np.linspace(first_para, last_para, 
                                         num_pts_check_on_edge)
                pts_surf_distances = np.zeros(num_pts_check_on_edge)
                for j in range(num_pts_check_on_edge):
                    pt_temp = gp_Pnt()
                    pt_coord = edges0[i].D0(para_check[j], pt_temp)
                    pts_surf_distances[j] = point_surface_distance(
                                            pt_temp, self.surf2)
                max_dist = np.max(pts_surf_distances)
                edge_length = curve_length(edges0[i])
                if max_dist < max_pt_surf_dist_ratio*edge_length:
                    self.connected_edges0 += [edges0[i],]

        face1 = make_face(self.surf2, 1e-9)
        edges1 = get_face_edges(face1)
        self.connected_edges1 = []
        for i in range(len(edges1)):
            int_cs = GeomAPI_IntCS(edges1[i], self.surf1)
            int_cs_coord = get_int_cs_coords(int_cs, unique_coord=True)
            if len(int_cs_coord) >= num_pts_lim_on_edge:
                first_para = edges1[i].FirstParameter()
                last_para = edges1[i].LastParameter()
                para_check = np.linspace(first_para, last_para, 
                                         num_pts_check_on_edge)
                pts_surf_distances = np.zeros(num_pts_check_on_edge)
                for j in range(num_pts_check_on_edge):
                    pt_temp = gp_Pnt()
                    pt_coord = edges1[i].D0(para_check[j], pt_temp)
                    pts_surf_distances[j] = point_surface_distance(
                                            pt_temp, self.surf1)
                max_dist = np.max(pts_surf_distances)
                edge_length = curve_length(edges1[i])
                if max_dist < max_pt_surf_dist_ratio*edge_length:
                    self.connected_edges1 += [edges1[i],]

        # if len(self.connected_edges0) == len(self.connected_edges1):
        #     # If the number of intersecting edges between edges of surface 1
        #     # and surface 2 and the number of intersecting edges between
        #     # edges of surface 2 and surface 1 are the same, then the 
        #     # intersecting edges are considered as connected edges.
        #     # Else, there will be no connected edges.
        #     connected_edges = self.connected_edges0
        # else:
        #     connected_edges = []

        if len(self.connected_edges0) >= len(self.connected_edges1):
            connected_edges = self.connected_edges0
        else:
            connected_edges = self.connected_edges1

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

    def check_curve_near_singularity(self, curve):
        """
        """
        dim = 3
        first_para = curve.FirstParameter()
        last_para = curve.LastParameter()
        first_coord = np.zeros(dim)
        last_coord = np.zeros(dim)
        first_pnt = gp_Pnt()
        last_pnt = gp_Pnt()
        curve.D0(first_para, first_pnt)
        curve.D0(last_para, last_pnt)
        for i in range(dim):
            first_coord[i] = first_pnt.Coord()[i]
            last_coord[i] = last_pnt.Coord()[i]
        left_side = False
        right_side = False
        # Check if first_coord and last coord near singularity coordinates
        # For surface 1
        if self.surf1_data.singularity:
            for i in range(self.surf1_data.num_singularity):
                if np.linalg.norm(first_coord-self.surf1_data.\
                    singularity_coords[i]) < 1e-3:
                    left_side = True
                if np.linalg.norm(last_coord-self.surf1_data.\
                    singularity_coords[i]) < 1e-3:
                    right_side = True
        # For surface 2
        if self.surf2_data.singularity:
            for i in range(self.surf2_data.num_singularity):
                if np.linalg.norm(first_coord-self.surf2_data.\
                    singularity_coords[i]) < 1e-3:
                    left_side = True
                if np.linalg.norm(last_coord-self.surf2_data.\
                    singularity_coords[i]) < 1e-3:
                    right_side = True

        if left_side is True and right_side is True:
            cut_side = "both"
        elif left_side is True and right_side is False:
            cut_side = "left"
        elif left_side is False and right_side is True:
            cut_side = "right"
        else:
            cut_side = None

        return cut_side

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
        assert ind < self.num_connected_edges and ind >= 0, \
            "``ind`` is out of range (0, "+str(self.num_connected_edges)+")"

        cut_side = None
        if self.check_singularity:
            cut_side = self.check_curve_near_singularity(
                       self.connected_edges[ind])

        connected_edge_coords = get_curve_coord(self.connected_edges[ind],
                                                num_pts, sort_axis,
                                                cut_side=cut_side, 
                                                cut_ratio=self.cut_ratio)
        return connected_edge_coords
    
    def get_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the coordinates of the connected edges.

        Parameters
        ----------
        num_pts : int or list of ints, optional, default is list of 20
        sort_axis : int, {0, 1, 2} or None, optional 

        Returns
        -------
        connected_edges_coords : list of ndarray
        """
        if isinstance(num_pts, int):
            num_pts = [num_pts]*self.num_connected_edges
        elif isinstance(num_pts, list):
            assert len(num_pts) == self.num_connected_edges, \
                "List ``num_pts`` has incompatible size"
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(num_pts), " is not "
                                "supported for ``num_pts``")

        connected_edges_coords = []
        if self.num_connected_edges > 0:
            for i in range(len(self.connected_edges)):
                # connected_edges_coords += [get_curve_coord(
                #                            self.connected_edges[i], 
                #                            num_pts[i], sort_axis)]
                connected_edges_coords += [self.get_coordinate(i,
                                           num_pts[i], sort_axis)]
        else:
            if mpirank == 0:
                print("Surface-surface connected edges are not detected, "
                      "returns empty list.")
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
        assert ind < self.num_connected_edges and ind >= 0, \
            "``ind`` is out of range (0, "+str(self.num_connected_edges)+")"
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
        num_pts : int or list of ints, optional, default is list of 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        connected_edges_para_coords : list of lists that contains two ndarray
        """
        if isinstance(num_pts, int):
            num_pts = [num_pts]*self.num_connected_edges
        elif isinstance(num_pts, list):
            assert len(num_pts) == self.num_connected_edges, \
                "List ``num_pts`` has incompatible size"
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(num_pts), " is not "
                                "supported for ``num_pts``")

        connected_edges_para_coords = []
        if self.num_connected_edges > 0:
            for i in range(self.num_connected_edges):
                connected_edges_para_coords += \
                    [self.get_parametric_coordinate(i, num_pts[i], sort_axis)]
        else:
            if mpirank == 0:
                print("Surface-surface connected edges are not detected, "
                      "returns empty list.")
        return connected_edges_para_coords


class BSplineSurfacesIntersections(BSplineSurfacesConnectedEdges):
    """
    Class computes intersections between two B-spline surfaces.
    """
    def __init__(self, surf1, surf2, rtol=1e-6, 
                 check_singularity=True, cut_ratio=0.03):
        """
        Parameters
        ----------
        surf1 : OCC B-spline surface
        surf2 : OCC B-spline surface
        rtol : float, optional. Default is 1e-6.
        check_singularity : bool, optional
            If True, the algorithm will check if two ends of the
            intersections coincide withe the surface singularity.
            If so, a portion of the curve, 
            ``cut_ratio``*parametric_length, on this end will be 
            ignored when getting physical and parametric coordinates. 
            This will help reducing the stress concentration near the 
            surface singularity. Default is True
        cut_ratio : float, optional, default is 0.03
        """
        super().__init__(surf1, surf2, check_singularity, cut_ratio)
        self.int_ss = GeomAPI_IntSS(surf1, surf2, rtol)

    @property
    def num_intersections(self):
        """
        Return the number of intersections between two surfaces. If these
        two surfaces have connected edges, the number of intersections
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
                # as multiple intersections by ``GeomAPI_IntSS``. Using
                # the edge-surface intersection information from 
                # the parent class to check.
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
        assert ind < self.num_intersections and ind >= 0, \
            "``ind`` is out of range (0, "+str(self.num_intersections)+")"

        cut_side = None
        if self.check_singularity:
            cut_side = self.check_curve_near_singularity(
                       self.intersections[ind])

        int_coords = get_curve_coord(self.intersections[ind],
                                          num_pts, sort_axis,
                                          cut_side=cut_side, 
                                          cut_ratio=self.cut_ratio)
        return int_coords

    def get_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the physical coordinates of the intersection curves.

        Parameters
        ---------- 
        num_pts : int or list of ints, optional, default is list of 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        ints_coords : list of ndarray
        """
        if isinstance(num_pts, int):
            num_pts = [num_pts]*self.num_intersections
        elif isinstance(num_pts, list):
            assert len(num_pts) == self.num_intersections, \
                "List ``num_pts`` has incompatible size"
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(num_pts), " is not "
                                "supported for ``num_pts``")

        ints_coords = []
        if self.num_intersections > 0:
            for i in range(self.num_intersections):
                ints_coords += [self.get_coordinate(i, num_pts[i],
                                                    sort_axis),]
        else:
            if mpirank == 0:
                print("Surface-surface intersections are not detected, "
                      "returns empty list.")
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
        assert ind < self.num_intersections and ind >= 0, \
            "``ind`` is out of range (0, "+str(self.num_intersections)+")"
        int_phy_coords = self.get_coordinate(ind, num_pts, sort_axis)
        int_para_coords = [parametric_coord(int_phy_coords, self.surf1), 
                           parametric_coord(int_phy_coords, self.surf2)]
        return int_para_coords

    def get_parametric_coordinates(self, num_pts=20, sort_axis=None):
        """
        Return the parametric coordinates of the intersection curves.

        Parameters
        ---------- 
        num_pts : int or list of ints, optional, default is list of 20
        sort_axis : int, {0, 1, 2}, optional, default is None

        Returns
        -------
        ints_para_coords : list of lists that contains two ndarray
        """
        if isinstance(num_pts, int):
            num_pts = [num_pts]*self.num_intersections
        elif isinstance(num_pts, list):
            assert len(num_pts) == self.num_intersections, \
                "List ``num_pts`` has incompatible size"
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(num_pts), " is not "
                                "supported for ``num_pts``")

        ints_para_coords = []
        if self.num_intersections > 0:
            for i in range(self.num_intersections):
                ints_para_coords += [self.get_parametric_coordinate(i, 
                                     num_pts[i], sort_axis)]
        else:
            if mpirank == 0:
                print("Surface-surface intersections are not detected, "
                      "returns empty list.")
        return ints_para_coords


class OCCPreprocessing(object):
    """
    The preprocessing class for imported OCC BSpline surfaces geometry,
    which includes reparametrization of BSpline surfaces, BSpline 
    surfaces refinement and surface-surface intersections computation.
    """
    def __init__(self, BSpline_surfs, reparametrize=False, refine=False):
        """
        Parameters
        ----------
        BSpline_surfs: list of OCC BSpline Surfaces
        reparametrize : bool, optional, default is False
            If True, the input BSpline_surfs will be reparametrized 
            with maximal continuity and stored in 
            ``self.BSpline_surfs_repara``. If False, the 
            ``self.BSpline_surfs_repara`` will be the same with input 
            BSpline_surfs
        refine : bool, optional, default is False
            If True, the reparametrized B-spline surfaces will be refined
            using order elevation and knot insertion depends on user's
            input and stored in ``self.BSpline_surfs_refine``.
            If False, ``self.BSpline_surfs_refine`` is the same with
            ``self.BSpline_surfs_repara``
        """
        if not isinstance(BSpline_surfs, list):
            if mpirank == 0:
                raise TypeError("Only list of OCC BSplineSurface "
                                "is supported.")
        self.BSpline_surfs = BSpline_surfs
        self.BSpline_surfs_data = [BSplineSurfaceData(surf)
                                   for surf in self.BSpline_surfs]
        self.num_surfs = len(self.BSpline_surfs)

        self.reparametrize = reparametrize
        self.refine = refine
        self.reparametrize_is_done = False
        self.refine_is_done = False

        # self.BSpline_surfs_repara = []
        # self.BSpline_surfs_repara_data = []

        # self.BSpline_surfs_refine = []
        # self.BSpline_surfs_refine_data = []

        # if self.reparametrize:
        #     self.BSpline_surfs_repara = []
        #     self.BSpline_surfs_repara_data = []
        # else:
        #     self.BSpline_surfs_repara = self.BSpline_surfs
        #     self.BSpline_surfs_repara_data = [BSplineSurfaceData(surf) 
        #         for surf in self.BSpline_surfs_repara]

        # if self.refine:
        #     self.BSpline_surfs_refine = []
        #     self.BSpline_surfs_refine_data = []
        # else:
        #     self.BSpline_surfs_refine = self.BSpline_surfs_repara
        #     self.BSpline_surfs_refine_data = [BSplineSurfaceData(surf)
        #         for surf in self.BSpline_surfs_refine]

        self.compute_int_is_done = False
        self.num_intersections_all = 0  # Number of intersections
        self.mapping_list = []
        self.mortar_nels = []
        self.intersection_curves = []
        self.intersections_length = []
        self.intersections_phy_coords = []
        self.intersections_para_coords = []

    def reparametrize_BSpline_surfaces(self, u_num_evals=30, v_num_evals=30, 
                                       bs_degrees=3, bs_continuities=3, 
                                       tol3D=1e-3, geom_scale=1., 
                                       remove_dense_knots=True, 
                                       dist_ratio_remove=0.5, rtol=1e-2):
        """
        Reparametrize inputting OCC BSplineSurfaces with specified 
        degree and continuity with tolerance ``tol3D``. ``geom_scale``
        is used to convert geometry units.

        Parameters
        ----------
        u_num_evals : int or list of ints, optional
            If not given, default is list of 30
        v_num_evals : int or list of ints
            If not given, default is list of 30
        bs_degrees : int or list of ints, optional
            If not given, default is list of 3
        bs_continuities : int or list of ints, optional
            If not given, default is list of 3
        tol3D : float, optional, default is 1e-3
        geom_scale : float, optional, default is 1.0
        remove_dense_knots : bool, optional, default is True
        dist_ratio_remove : float, optional, default is 0.5
        rtol : float, optional, default is 1e-2
            Relative tolerance of removing densely distributed knots
        """
        if not self.reparametrize:
            if mpirank == 0:
                raise RuntimeError("Argument ``reparametrize`` is passed "
                    "as False, BSpline surfaces cannot be reparametrized.")
        self.BSpline_surfs_repara = []
        self.BSpline_surfs_repara_data = []

        if isinstance(u_num_evals, int):
            u_num_evals = [u_num_evals]*self.num_surfs
        elif isinstance(u_num_evals, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(u_num_evals), " is not "
                                "supported for ``u_num_evals``")

        if isinstance(v_num_evals, int):
            v_num_evals = [v_num_evals]*self.num_surfs
        elif isinstance(v_num_evals, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(v_num_evals), " is not "
                                "supported for ``v_num_evals``")

        if isinstance(bs_degrees, int):
            bs_degrees = [bs_degrees]*self.num_surfs
        elif isinstance(bs_degrees, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(bs_degrees), " is not "
                                "supported for ``bs_degrees``")

        if isinstance(bs_continuities, int):
            bs_continuities = [bs_continuities]*self.num_surfs
        elif isinstance(bs_continuities, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(bs_continuities), " is not "
                                "supported for ``bs_continuities``")

        for i in range(self.num_surfs):
            self.BSpline_surfs_repara += [reparametrize_BSpline_surface(
                                            self.BSpline_surfs[i],
                                            u_num_evals[i], v_num_evals[i], 
                                            bs_degrees[i], bs_continuities[i], 
                                            tol3D, geom_scale, 
                                            remove_dense_knots, 
                                            dist_ratio_remove, rtol),]
            self.BSpline_surfs_repara_data += [BSplineSurfaceData(
                                            self.BSpline_surfs_repara[-1]),]

        self.reparametrize_is_done = True

    def refine_BSpline_surfaces(self, u_degrees=3, v_degrees=3,
                                u_num_inserts=0, v_num_inserts=0,
                                correct_element_shape=True,
                                aspect_ratio_lim=4, dist_ratio=0.7,
                                copy_surf=True):
        """
        Refine reparametrized OCC BSplineSurfaces by order elevation and
        knot insertion. (If ``reparametrize`` is False, reparametrized OCC
        BSpline surfaces are the same with input BSpline surfaces) 
        By default, the shape of elements will be corrected to better 
        aspect ratio.

        Parameters
        ----------
        u_degrees : int or list of ints, optional
            If not given, default is list of 3
        v_degrees : int or list of ints, optional
            If not given, default is list of 3
        u_num_inserts : int or list of ints, optional
            If not given, default is list of 0
        v_num_inserts : int or list of ints
            If not given, default is list of 0
        correct_element_shape: bool, optional, default is True
        aspect_ratio_lim : float, optional, default is 4
        dist_ratio : float, optional, default is 0.7
        copy_surf : bool, optional, default is True
        """
        if not self.refine:
            if mpirank == 0:
                raise RuntimeError("Argument ``refine`` is passed "
                    "as False, BSpline surfaces cannot be refined.")

        if self.reparametrize_is_done:
            BSpline_surfs_temp = self.BSpline_surfs_repara
        else:
            if self.reparametrize is False:
                BSpline_surfs_temp = self.BSpline_surfs
            else:
                if mpirank == 0:
                    raise RuntimeError("Argument ``reparametrize`` is "
                          "passed as True, but reparametrization has not "
                          "been performed yet.")

        self.BSpline_surfs_refine = []
        self.BSpline_surfs_refine_data = []
        # self.BSpline_avg_mesh_sizes = []

        if isinstance(u_degrees, int):
            u_degrees = [u_degrees]*self.num_surfs
        elif isinstance(u_degrees, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(u_degrees), " is not "
                                "supported for ``u_degrees``")

        if isinstance(v_degrees, int):
            v_degrees = [v_degrees]*self.num_surfs
        elif isinstance(v_degrees, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(v_degrees), " is not "
                                "supported for ``v_degrees``")

        if isinstance(u_num_inserts, int):
            u_num_inserts = [u_num_inserts]*self.num_surfs
        elif isinstance(u_num_inserts, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(u_num_inserts), " is not "
                                "supported for ``u_num_inserts``")

        if isinstance(v_num_inserts, int):
            v_num_inserts = [v_num_inserts]*self.num_surfs
        elif isinstance(v_num_inserts, list):
            pass
        else:
            if mpirank == 0:
                raise TypeError("Type ", type(v_num_inserts), " is not "
                                "supported for ``v_num_inserts``")

        for i in range(self.num_surfs):
            self.BSpline_surfs_refine += [refine_BSpline_surface(
                BSpline_surfs_temp[i], u_degrees[i], v_degrees[i],
                u_num_inserts[i], v_num_inserts[i], correct_element_shape,
                aspect_ratio_lim, dist_ratio, copy_surf),]
            self.BSpline_surfs_refine_data += [BSplineSurfaceData(
                                    self.BSpline_surfs_refine[-1]),]

        self.refine_is_done = True

    @property
    def total_DoFs(self):
        """
        Returns total number degrees of freedom for refined BSpline
        surfaces.
        """
        if self.refine_is_done:
            BSpline_surfs_data_temp = self.BSpline_surfs_refine_data
        else:
            if self.reparametrize_is_done:
                BSpline_surfs_data_temp = self.BSpline_surfs_repara_data
            else:
                BSpline_surfs_data_temp = self.BSpline_surfs_data

        total_DoFs = 0
        for i in range(self.num_surfs):
            total_DoFs += \
                BSpline_surfs_data_temp[i].control.shape[0]\
                *BSpline_surfs_data_temp[i].control.shape[1]*3
        return total_DoFs

    def compute_intersections(self, rtol=1e-4, mortar_refine=1, 
                              mortar_nels=None, min_mortar_nel=8, 
                              sort_axis=None, check_singularity=True,
                              cut_ratio=0.03):
        """
        Compute intersections between all input BSpline surfaces.

        Parameters
        ----------
        rtol : float, optional, default is 1e-4
            Tolerance for surface-surface intersection computation
        mortar_refine : int, optional, default is 1
            Level of refinement for number of elements for mortar meshes
        mortar_nels : int or list of ints, optional, default is None
            Number of elements for mortar meshes. If an int instance is 
            given, all mortar meshes have the same number of elements. 
            If not given, the number of elements will be computed by 
            intersection curve length over average mesh size
        min_mortar_nel : int, optional
            Minimum number of elements of mortar mesh, default is 8
        sort_axis : int, {0, 1, 2} or None, optional, default is None
        check_singularity : bool, optional
            If True, the algorithm will check if two ends of the
            intersections coincide withe the surface singularity.
            If so, a portion of the curve, 
            ``cut_ratio``*parametric_length, on this end will be 
            ignored when getting physical and parametric coordinates. 
            This will help reducing the stress concentration near the 
            surface singularity. Default is True
        cut_ratio : float, optional, default is 0.03
        """
        if self.refine_is_done:
            self.avg_mesh_sizes = [np.average(BSpline_mesh_size(surf_data))
                for surf_data in self.BSpline_surfs_refine_data]
            if self.reparametrize_is_done:
                BSpline_surfs_temp = self.BSpline_surfs_repara
            else:
                if self.reparametrize is False:
                    BSpline_surfs_temp = self.BSpline_surfs
                else:
                    if mpirank == 0:
                        raise RuntimeError("Argument ``reparametrize`` is "
                              "passed as True, but reparametrization has "
                              "not been performed yet.")
        else:
            if self.refine is False:
                if self.reparametrize_is_done:
                    self.avg_mesh_sizes = [np.average(BSpline_mesh_size(
                        surf_data)) for surf_data in 
                        self.BSpline_surfs_repara_data]
                    BSpline_surfs_temp = self.BSpline_surfs_repara
                else:
                    if self.reparametrize is False:
                        self.avg_mesh_sizes = [np.average(BSpline_mesh_size(
                        surf_data)) for surf_data in self.BSpline_surfs_data]
                        BSpline_surfs_temp = self.BSpline_surfs
                    else:
                        if mpirank == 0:
                            raise RuntimeError("Argument ``reparametrize`` "
                                  "is passed as True, but reparametrization "
                                  "has not been performed yet.")
            else:
                if mpirank == 0:
                    raise RuntimeError("Argument ``refine`` is passed as "
                          "True, but surface refinement has not been "
                          "performed yet.")

        self.mapping_list = []
        self.intersection_curves = []
        self.intersections_phy_coords = []
        self.intersections_para_coords = []

        if mortar_nels is not None:
            if isinstance(mortar_nels, list):
                self.mortar_nels = [nel*mortar_refine for nel in mortar_nels]
                mortar_nel_count = 0
            elif isinstance(mortar_nels, int):
                mortar_nel = mortar_nels*mortar_refine
                self.mortar_nels = []
            else:
                if mpirank == 0:
                    raise TypeError("Type ", type(mortar_nels), " is not "
                                    "supported for ``mortar_nels``")
        else:
            self.intersections_length = []
            self.mortar_nels = []

        for i in range(self.num_surfs):
            for j in range(i+1, self.num_surfs):
                # print("i:", i, ", j:", j, " ---------------------")
                bs_intersection = BSplineSurfacesIntersections(
                                    BSpline_surfs_temp[i], 
                                    BSpline_surfs_temp[j], rtol=rtol, 
                                    check_singularity=check_singularity,
                                    cut_ratio=cut_ratio)
                if bs_intersection.num_intersections > 0:
                    num_int = bs_intersection.num_intersections
                    self.mapping_list += [[i, j],]*num_int
                    self.intersection_curves += \
                        bs_intersection.intersections

                    if mortar_nels is None:
                        for k in range(num_int):
                            self.intersections_length += [curve_length(
                                bs_intersection.intersections[k])]
                            self.mortar_nels += [np.max([
                                int(min_mortar_nel*mortar_refine),
                                ceil(self.intersections_length[-1]/np.min(
                                [self.avg_mesh_sizes[i], 
                                self.avg_mesh_sizes[j]])*mortar_refine)]),]
                        mortar_nels_temp = self.mortar_nels[-num_int:]
                    else:
                        if isinstance(mortar_nels, list):
                            mortar_nels_temp = self.mortar_nels[
                                mortar_nel_count:mortar_nel_count+num_int]
                            mortar_nel_count += num_int
                        elif isinstance(mortar_nels, int):
                            mortar_nels_temp = [morter_nel]*num_int
                            self.mortar_nels += mortar_nels_temp

                    # print("mortar_nels_temp:", mortar_nels_temp)
                    self.intersections_phy_coords += \
                        bs_intersection.get_coordinates(
                            mortar_nels_temp, sort_axis)
                    self.intersections_para_coords += \
                        bs_intersection.get_parametric_coordinates(
                            mortar_nels_temp, sort_axis)

        self.num_intersections_all = len(self.intersection_curves)
        self.compute_int_is_done = True

    def start_display(self, display, show_triedron=False):
        """
        Initialize PythonOCC build-in display function

        Parameters
        ----------
        display : OCC Viewer3d
            An OCC 3D viewer object, which can be initialized:
            display, start_display, add_menu, add_function_to_menu = \
                init_display()
        show_triedron : bool, optional, default is False
        """
        if not show_triedron:
            display.hide_triedron()
        display.set_bg_gradient_color([255,255,255], [255,255,255])

    def display_surfaces(self, display, show_triedron=False, 
                         show_bdry=True, color=None, transparency=0.0, 
                         zoom_factor=1.0, save_fig=False, 
                         filename="surfaces.png"):
        """
        Display all BSpline surfaces using PythonOCC build-in display
        function.

        Parameters
        ----------
        display : OCC Viewer3d
        show_triedron : bool, optional, default is False
        show_bdry : bool, optional, default is True
        color : str, optional, default is None
        transparency : float or list of floats, optional
            Default is list of 0.0
        zoom_factor : float, optional, default is 1.0
        save_fig : bool, optional, default is False
        filename : str, optional, default is "surfaces.png"
        """
        if self.refine_is_done:
            BSpline_surfs_temp = self.BSpline_surfs_refine
        else:
            if self.reparametrize_is_done:
                BSpline_surfs_temp = self.BSpline_surfs_repara
            else:
                BSpline_surfs_temp = self.BSpline_surfs

        self.start_display(display, show_triedron)

        if not show_bdry:
            display.default_drawer.SetFaceBoundaryDraw(False)

        if isinstance(transparency, float) or isinstance(transparency, int):
            transparency_list = [transparency]*self.num_surfs
        elif isinstance(transparency, list):
            transparency_list = transparency

        for i in range(self.num_surfs):
            display.DisplayShape(BSpline_surfs_temp[i],
                                 color=color, 
                                 transparency=transparency_list[i])
            
        display.View_Iso()
        display.FitAll()
        display.ZoomFactor(zoom_factor)
        if save_fig:
            display.View.Dump('./'+filename)
        return display

    def display_intersections(self, display, show_triedron=False, 
                              color="BLUE", zoom_factor=1.0, 
                              save_fig=False, filename="intersections.png"):
        """
        Display all intersection curves using PythonOCC build-in display
        function.

        Parameters
        ----------
        display : OCC Viewer3d
        show_triedron : bool, optional, default is False
        color : str, optional, default is "BLUE"
        zoom_factor : float, optional, default is 1.0
        save_fig : bool, optional, default is False
        filename : str, optional, default is "intersections.png"
        """
        self.start_display(display, show_triedron)

        for i in range(self.num_intersections_all):
            display.DisplayShape(self.intersection_curves[i],
                                 color=color)

        display.View_Iso()
        display.FitAll()
        display.ZoomFactor(zoom_factor)
        if save_fig:
            display.View.Dump('./'+filename)

    def display_surfaces_intersections(self, display, show_triedron=False, 
                                       show_surf_bdry=True, 
                                       surf_color=None, int_color="BLUE", 
                                       surf_transparency=0.0, zoom_factor=1.0, 
                                       save_fig=False, 
                                       filename="surfaces_intersections.png"):
        """
        Display all BSpline surfaces and intersection curves using 
        PythonOCC build-in display function.

        Parameters
        ----------
        display : OCC Viewer3d
        show_triedron : bool, optional, default is False
        show_surf_bdry : bool, optional, default is True
        surf_color : str, optional, default is None
        int_color : str, optional, default is "BLUE"
        surf_transparency : float or list of floats, optional
            Default is list of 0.0
        zoom_factor : float, optional, default is 1.0
        save_fig : bool, optional, default is False
        filename : str, optional, default is "surfaces_intersections.png"
        """
        if self.refine_is_done:
            BSpline_surfs_temp = self.BSpline_surfs_refine
        else:
            if self.reparametrize_is_done:
                BSpline_surfs_temp = self.BSpline_surfs_repara
            else:
                BSpline_surfs_temp = self.BSpline_surfs

        self.start_display(display, show_triedron)

        if not show_surf_bdry:
            display.default_drawer.SetFaceBoundaryDraw(False)

        if (isinstance(surf_transparency, float) or 
            isinstance(surf_transparency, int)):
            surf_transparency_list = [surf_transparency]*self.num_surfs
        elif isinstance(surf_transparency, list):
            surf_transparency_list = surf_transparency

        for i in range(self.num_surfs):
            display.DisplayShape(BSpline_surfs_temp[i],
                                 color=surf_color, 
                                 transparency=surf_transparency_list[i])

        for i in range(self.num_intersections_all):
            display.DisplayShape(self.intersection_curves[i],
                                 color=int_color)

        display.View_Iso()
        display.FitAll()
        display.ZoomFactor(zoom_factor)
        if save_fig:
            display.View.Dump('./'+filename)

    def save_intersections_data(self, filename, data_path='./'):
        """
        Save intersections related data to a numpy file.

        Parameters
        ----------
        filename: str, with .npz extension
        data_path : str, default is './'
        """
        if not self.compute_int_is_done:
            raise RuntimeError("Surface-surface intersections have not "
                               "been computed yet, no data can be saved.")

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        np.savez(data_path+filename,
                 name1=self.num_intersections_all,
                 name2=self.mapping_list,
                 name3=self.intersections_phy_coords,
                 name4=self.intersections_para_coords,
                 name5=self.intersections_length,
                 name6=self.mortar_nels)

    def load_intersections_data(self, filename, data_path='./'):
        """
        Load intersections related data to preprocessor.

        Parameters
        ----------
        filename : str
        data_path : str, default is './'
        """
        if self.compute_int_is_done:
            raise RuntimeError("Surface-surface intersections have been "
                               "computed, cannot load data again.")

        intersections_data = np.load(data_path+filename, allow_pickle=True)
        self.compute_int_is_done = True
        self.num_intersections_all = int(intersections_data['name1'])
        self.mapping_list = intersections_data['name2']
        self.intersections_phy_coords = intersections_data['name3']
        self.intersections_para_coords = intersections_data['name4']
        self.intersections_length = intersections_data['name5']
        self.mortar_nels = intersections_data['name6']

if __name__ == "__main__":
    pass