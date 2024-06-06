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

    def compute_connected_edges(self):
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
        self.edges_length0 = []
        for i in range(len(edges0)):
            int_cs = GeomAPI_IntCS(edges0[i], self.surf2)
            int_cs_coord = get_int_cs_coords(int_cs, unique_coord=True)
            edge_length = curve_length(edges0[i])
            self.edges_length0 += [edge_length,]
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
                if max_dist < max_pt_surf_dist_ratio*self.edges_length0[i]:
                    self.connected_edges0 += [edges0[i],]

        face1 = make_face(self.surf2, 1e-9)
        edges1 = get_face_edges(face1)
        self.connected_edges1 = []
        self.edges_length1 = []
        for i in range(len(edges1)):
            edge_length = curve_length(edges1[i])
            self.edges_length1 += [edge_length,]
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
                if max_dist < max_pt_surf_dist_ratio*self.edges_length1[i]:
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
            self.connected_edges = self.connected_edges0
        else:
            self.connected_edges = self.connected_edges1

        return self.connected_edges

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
    def __init__(self, surf1, surf2, rtol=1e-6, edge_rel_ratio=1e-3,
                 check_singularity=True, cut_ratio=0.03):
        """
        Parameters
        ----------
        surf1 : OCC B-spline surface
        surf2 : OCC B-spline surface
        rtol : float, optional. Default is 1e-6.
        edge_rel_ratio : float, optional. Default is 1e-4
            Compute the relative ratio between the length of an intersection
            and the minimum length of two surfaces' edges, if the ratio
            is smaller than this value, then this intersection will be 
            negelected.
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
        self.rtol = rtol
        self.edge_rel_ratio = edge_rel_ratio

    def compute_intersections(self):
        """
        Return the intersection curves.

        Returns
        -------
        intersections : list of OCC Geom_Curves
        """
        self.compute_connected_edges()
        self.nz_edges_length = [elen for elen in self.edges_length0
                                if elen > 1e-9]
        self.nz_edges_length += [elen for elen in self.edges_length1
                                 if elen > 1e-9]
        self.min_edge_length = np.min(self.nz_edges_length)

        self.intersections = []
        if self.num_connected_edges > 0:
            self.intersections = self.connected_edges
        else:
            self.int_ss = GeomAPI_IntSS(self.surf1, self.surf2, self.rtol)
            if self.int_ss.NbLines() > 0:
                int_lines = [self.int_ss.Line(i) for i in 
                             range(1, self.int_ss.NbLines()+1)]
                if len(int_lines) > 1:
                    # Check if this is surface-edge intersection but computed 
                    # as multiple intersections by ``GeomAPI_IntSS``. Using
                    # the edge-surface intersection information from 
                    # the parent class to check.
                    if (len(self.connected_edges0) > 0 and 
                        len(self.connected_edges1) == 0):
                        self.intersections = self.connected_edges0
                    elif (len(self.connected_edges0) == 0 and 
                          len(self.connected_edges1) > 0):
                        self.intersections = self.connected_edges1
                    else:
                        for i in range(len(int_lines)):
                            int_length = curve_length(int_lines[i])
                            # Negelect lines that are too short
                            if (int_length/self.min_edge_length 
                                > self.edge_rel_ratio):
                                self.intersections += [int_lines[i]]
                else:
                    int_length = curve_length(int_lines[0])
                    # Negelect lines that are too short
                    if (int_length/self.min_edge_length 
                        > self.edge_rel_ratio):
                        self.intersections += [int_lines[0]]
        return self.intersections

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
        return len(self.intersections)

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
                              edge_rel_ratio=1e-3, cut_ratio=0):
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
        edge_rel_ratio : float, optional. Default is 1e-4
            Compute the relative ratio between the length of an intersection
            and the minimum length of two surfaces' edges, if the ratio
            is smaller than this value, then this intersection will be 
            negelected.
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
            self.intersections_type = []

        for i in range(self.num_surfs):
            for j in range(i+1, self.num_surfs):
                bs_intersection = BSplineSurfacesIntersections(
                                    BSpline_surfs_temp[i], 
                                    BSpline_surfs_temp[j], rtol=rtol, 
                                    edge_rel_ratio=edge_rel_ratio,
                                    check_singularity=check_singularity,
                                    cut_ratio=cut_ratio)
                bs_intersection.compute_intersections()
                if bs_intersection.num_intersections > 0:
                    num_int = bs_intersection.num_intersections
                    self.mapping_list += [[i, j],]*num_int
                    self.intersection_curves += \
                        bs_intersection.intersections
                    # Determine number of elements for mortar meshes
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
                            mortar_nels_temp = [mortar_nels]*num_int
                            self.mortar_nels += mortar_nels_temp

                    # print("mortar_nels_temp:", mortar_nels_temp)
                    num_mortar_pts = [nel+1 for nel in mortar_nels_temp]
                    self.intersections_phy_coords += \
                        bs_intersection.get_coordinates(
                            num_mortar_pts, sort_axis)
                    self.intersections_para_coords += \
                        bs_intersection.get_parametric_coordinates(
                            num_mortar_pts, sort_axis)

        self.num_intersections_all = len(self.intersection_curves)
        self.compute_int_is_done = True

    ######################################################################
    #### Shape optimization with moving intersections related methods ####
    ######################################################################

    def check_intersections_type(self, edge_tol=1e-4):
        """
        This function is used to check the surface--surface intersections'
        type. Four types of intersections are considered:
            1. surf-surf: the intersection is at the middle of two surfaces
            2. surf-edge: the intersection is at the middle of surface A, and 
                          is an edge of surface B
            3. edge-surf: the counterpart of the previous case
            4. edge-edge: the intersection is the edge of both surfaces
        After the type for each intersection, an addtional string is used
        to determine the parametric location for intersections that are 
        also surface edges, e.g.
            'na-xi0.1': this string corresponds to type 'surf-edge'. The
                        parametric coordinates of the mortar mesh w.r.t. 
                        surface B are all 1s in the 0 direction, i.e.,
                        xi = [[1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0],
                              [0.0, 0.1, 0.2, 0.3, 0.4, ..., 1.0]]^T
        Possible combinations for parametric coordinate for edge 
        intersection: {'xi0.0', 'xi0.1', 'xi1.0', 'xi1.1'}

        Parameters
        ----------
        edge_tol : the tolerance to treat an intersection as surface edge.

        Returns
        -------
        self.intersections_type : list of lists
        """
        self.intersections_type = [[None, None] for i in 
                                    range(self.num_intersections_all)]
        num_surf_side = 2
        num_para_dir = 2
        for int_ind in range(self.num_intersections_all):
            type_indicator = [None, None]
            edge_indicator = [None, None]
            for surf_side in range(num_surf_side):
                for para_dir in range(num_para_dir):
                    # Get intersections' parametric coordinates at 
                    # side `surf_side` with direction `para_dir`
                    xi_coord = self.intersections_para_coords\
                               [int_ind][surf_side][:,para_dir]
                    xi_coord_size = xi_coord.size
                    val0 = np.sum((xi_coord)**2)
                    val1 = np.sum((xi_coord-np.ones(xi_coord_size))**2)
                    edge_pre = 'xi'+str(para_dir)+'.'
                    if np.sum((xi_coord)**2) < edge_tol:
                        type_indicator[surf_side] = 'edge'
                        edge_indicator[surf_side] = edge_pre+'0'
                        break
                    elif np.sum((xi_coord-np.ones(xi_coord_size))**2) < edge_tol:
                        type_indicator[surf_side] = 'edge'
                        edge_indicator[surf_side] = edge_pre+'1'
                        break
                    else:
                        type_indicator[surf_side] = 'surf'
                        edge_indicator[surf_side] = 'na'
            int_type = str(type_indicator[0])+'-'+str(type_indicator[1])
            edge_side = str(edge_indicator[0])+'-'+str(edge_indicator[1])
            self.intersections_type[int_ind][0] = int_type
            self.intersections_type[int_ind][1] = edge_side
        return self.intersections_type

    def get_diff_intersections(self):
        """
        Based on intersections type, determine the intersections' indices
        that can be differentiated. Except for type 'edge-edge', all the 
        other three types of intersections can be differentiated. For 
        types 'surf-edge' and 'edge-surf', determine the edge intersections'
        parametric coordinates constraint to maintain the type, e.g., for
        'surf-edge' with indicator 'na-xi0.1', all the first (0-th) rows of 
        intersection's parametric coordinates w.r.t. surface B are 1.

        Indicator for edge constraint, e.g., 'surf-xi0.1', the indicator is
        '1-0.1', where the first '1' stands for side, '0' stands for first 
        parametric location, and the last `1` stands for value.
        """
        self.diff_int_inds = []
        self.diff_int_edge_cons = []

        for int_ind  in range(self.num_intersections_all):
            int_type = self.intersections_type[int_ind][0]
            int_edge_side = self.intersections_type[int_ind][1]
            if int_type == 'surf-surf':
                self.diff_int_inds += [int_ind]
                self.diff_int_edge_cons += ['na']
            elif int_type == 'surf-edge':
                self.diff_int_inds += [int_ind]
                cons_indicator = '1-'
                cons_indicator += int_edge_side[int_edge_side.index('.')-1]
                cons_indicator += '.'
                cons_indicator += int_edge_side[int_edge_side.index('.')+1]
                self.diff_int_edge_cons += [cons_indicator]
            elif int_type == 'edge-surf':
                self.diff_int_inds += [int_ind]
                cons_indicator = '0-'
                cons_indicator += int_edge_side[int_edge_side.index('.')-1]
                cons_indicator += '.'
                cons_indicator += int_edge_side[int_edge_side.index('.')+1]
                self.diff_int_edge_cons += [cons_indicator]
        return self.diff_int_inds, self.diff_int_edge_cons

    def set_diff_intersections_indices_by_mapping(self, surf_mappling_list):
        """
        Give a list of mapping indices ``surf_mappling_list``, 
        e.g., [[0,1], [0,2], [1,3]], return associated intersections' 
        indices.
        Note: the the entry in the mappling list element should be 
        smaller than the second entry.

        Parameters
        ----------
        surf_mappling_list : list of lists

        Returns
        -------
        self.diff_int_inds : list of inds
        """
        self.diff_int_inds = []
        for surf_mapping in surf_mappling_list:
            inds = [int_ind for int_ind, mapping 
                    in enumerate(self.mapping_list) 
                    if surf_mapping == mapping]
            self.diff_int_inds += inds
        return self.diff_int_inds

    def set_diff_intersections_indices(self, diff_int_inds):
        """
        Set differentiable intersections' indices manually.

        Parameters
        ----------
        diff_int_inds : list of inds

        Returns
        -------
        self.diff_int_inds : list of inds
        """
        self.diff_int_inds = diff_int_inds
        return self.diff_int_inds

    def set_diff_intersections_edge_cons(self, diff_int_edge_cons):
        """
        Set the values in ``self.diff_int_edge_cons`` if it is not 
        determined automatically, and after calling the method
        ``set_diff_intersections_indices`` or 
        ``set_diff_intersections_indices_by_mapping`` (so that the 
        differentiable intersections are determinted).

        Parameters
        ----------
        diff_int_edge_cons : list of strs

        Returns
        -------
        self.diff_int_edge_cons : list of inds
        """
        self.diff_int_edge_cons = diff_int_edge_cons
        return self.diff_int_edge_cons


    ######################################################################
    ######################################################################

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
            # print('intersection: {}'.format(i))
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

        # np.savez(data_path+filename,
        #          name1=self.num_intersections_all,
        #          name2=self.mapping_list,
        #          name3=self.intersections_phy_coords,
        #          name4=self.intersections_para_coords,
        #          name5=self.intersections_length,
        #          name6=self.mortar_nels)

        np.savez(data_path+filename,
                 name1=self.num_intersections_all,
                 name2=self.mapping_list,
                 name3=np.asarray(self.intersections_phy_coords, dtype=object),
                 name4=np.asarray(self.intersections_para_coords, dtype=object),
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