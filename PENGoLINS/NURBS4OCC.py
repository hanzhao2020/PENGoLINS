"""
The ``NURBS4OCC`` module
------------------------
reads in NURBS data in PythonOCC Geom_BSplineSurface format 
to create NURBS control mesh.
"""

from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.OCC_utils import *

class NURBSControlMesh4OCC(AbstractControlMesh):
    """
    This class represents a control mesh with NURBS geometry.
    """
    def __init__(self,occ_bs_surf,useRect=USE_RECT_ELEM_DEFAULT,
                 overRefine=0):
        """
        Generates a NURBS control mesh from PythonOCC B-spline surface 
        input data .
        The optional parameter ``overRefine``
        indicates how many levels of refinement to apply beyond what is
        needed to represent the spline functions; choosing a value greater
        than the default of zero may be useful for
        integrating functions with fine-scale features.
        overRefine > 0 only works for useRect=False.
        """

        bs_data = BSplineSurfaceData(occ_bs_surf)

        # create a BSpline scalar space given the knot vector(s)
        self.scalarSpline = BSpline(bs_data.degree,bs_data.knots,
                                    useRect,overRefine)
        
        # get the control net; already in homogeneous form
        nvar = len(bs_data.degree)
        if(nvar==1):
            self.bnet = bs_data.control
        elif(nvar==2):
            M = bs_data.control.shape[0]
            N = bs_data.control.shape[1]
            dim = bs_data.control.shape[2]
            self.bnet = zeros((M*N,dim))
            for j in range(0,N):
                for i in range(0,M):
                    self.bnet[ij2dof(i,j,M),:]\
                        = bs_data.control[i,j,:]
        else:
            M = bs_data.control.shape[0]
            N = bs_data.control.shape[1]
            O = bs_data.control.shape[2]
            dim = bs_data.control.shape[3]
            self.bnet = zeros((M*N*O,dim))
            for k in range(0,O):
                for j in range(0,N):
                    for i in range(0,M):
                        self.bnet[ijk2dof(i,j,k,M,N),:]\
                            = bs_data.control[i,j,k,:]
            
    def getScalarSpline(self):
        return self.scalarSpline

    def getHomogeneousCoordinate(self,node,direction):
        return self.bnet[node,direction]

    def getNsd(self):
        return self.bnet.shape[1]-1