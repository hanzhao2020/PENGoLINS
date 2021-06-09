"""
The ``contact`` module:
-----------------------
This module provides functionality for general frictionless self-contact 
of shell structures, using the nonlocal formulation given in

  https://doi.org/10.1016/j.cma.2017.11.007

Note that an objective, angular-momentum-conserving frictional extension 
for shell structures is outlined in Remark 7 of

  https://doi.org/10.1007%2Fs42102-019-00012-y

although it has never been implemented.  Alternatively, it would be quite
easy to implement the "naive" fricitional model in the cited reference.
"""

from tIGAr import *
from scipy.spatial import cKDTree
import numpy as np
from numpy import zeros
from numpy.linalg import norm as npNorm
from numpy import outer as npOuter
from numpy import identity as npIdentity

ADD_MODE = PETSc.InsertMode.ADD

# Shells only exist in three space dimensions.
d = 3

# Overkill preallocation for the contact tangent matrix; if a node interacts
# with too many other nodes, this could be exceeded, and the contact force
# assembly will slow down drastically.
PREALLOC = 500

# For rapid prototyping, one can numerically take the derivative of
# $\phi'$ needed for the LHS of the formulation.  This is the hard-coded
# epsilon used for a finite difference.
PHI_EPS = 1e-10

# Potentially-fragile assumption on the ordering of DoFs in mixed-element
# function spaces:
def nodeToDof(node,direction):
    return d*node + direction

class ShellContactContext:
    """
    Because there is some one-time initialization associated with the contact
    method, it makes sense to hold a persistent state in an object.
    """
    def __init__(self,spline,R_self,r_max,
                 phiPrime,phiDoublePrime=None):
        """
        ``spline`` is the ``tIGAr`` ``ExtractedSpline`` object defining the
        shell structure.  ``R_self`` is the radius defining the reference-
        configuration neighborhood with which each point does not interact
        through contact forces.  ``r_max`` is the maximum range used to
        identify potentially-contacting points in the current configuration;
        this should ideally be the radius of the support of ``phiPrime``,
        which defines the magnitude of central contact forces as a function
        of distance between points.  It's derivative, ``phiDoublePrime``
        is ideally passed as well, but, if it is omitted, a finite-difference
        approximation is taken.  The functions ``phiPrime`` and
        ``phiDoublePrime`` should be Python functions, each taking a single
        real-valued argument.  
        """
        self.spline = spline
        if isinstance(self.spline, list):
            self.splines = self.spline
            self.multiPatch = True
        elif isinstance(self.spline, ExtractedSpline):
            self.splines = [self.spline]
            self.multiPatch = False
        else:
            raise TypeError("Spline type "+Type(self.spline)\
                            +" is not supported.")

        self.nSplines = len(self.splines)
        self.phiPrime = phiPrime
        if(phiDoublePrime==None):
            # Do not use centered or backward differences, because
            # arguments to $\phi$ are assumed positive.
            phiDoublePrime = lambda r : (phiPrime(r+PHI_EPS)
                                         - phiPrime(r))/PHI_EPS
        self.phiDoublePrime = phiDoublePrime
        self.R_self = R_self
        self.r_max = r_max

        # Potentially-fragile assumption: that there is a correspondence
        # in DoF order between the scalar space used for each component of
        # the control mapping and the mixed space used for the displacement.  
        self.nNodes = [spline.cpFuncs[0].vector().get_local().size \
                       for spline in self.splines]
        self.nodeXs = [np.zeros((num_nodes, d)) for num_nodes in self.nNodes]
        # (Could be optimized for numba, but not worthwhile for one-time
        # initialization step.)
        self.weights = []
        for s_ind, spline in enumerate(self.splines):
            self.weights += [spline.cpFuncs[d].vector().get_local()]
            for i in range(0,self.nNodes[s_ind]):
                wi = self.weights[s_ind][i]
                for j in range(0,d):
                    Xj_hom = spline.cpFuncs[j].vector().get_local()[i]
                    self.nodeXs[s_ind][i,j] = Xj_hom/wi

        # Using quadrature points coincident with the FE nodes of the
        # extracted representation of the spline significantly simplifies
        # the assembly process.
        self.quadWeights = [np.zeros(nNodes) for nNodes in self.nNodes]
        for s_ind, spline in enumerate(self.splines):
            W = assemble(inner(Constant(d*(1.0,)),
                               TestFunction(spline.V))*spline.dx)

            # Unfortunately, mass lumping with super-linear Lagrange FEs on
            # simplices is not robust, and leads to some negative weights.
            # The following mass-conservative smoothing procedure improves
            # performance.
            if(spline.mesh.ufl_cell()==triangle):
                u = TrialFunction(spline.V)
                v = TestFunction(spline.V)
                w = Function(spline.V)
                w.vector()[:] = W
                smoothL = Constant(2.0)*CellDiameter(spline.mesh)
                smoothRes = inner(u-w,v)*dx + (smoothL**2)*inner(grad(u),
                                                                 grad(v))*dx
                w_smoothed = Function(spline.V)
                solve(lhs(smoothRes)==rhs(smoothRes),w_smoothed)
                W = w_smoothed.vector()

            quadWeightsTemp = W.get_local()
            for i in range(0,self.nNodes[s_ind]):
                self.quadWeights[s_ind][i] = quadWeightsTemp[nodeToDof(i,0)]

    def evalFunction(self,vFunc,s_ind):
        """
        Obtain a ``self.nNodes``-by-``d`` array containing physical values of
        the homogeneous function ``vFunc`` from space 
        ``self.splines[s_ind].V`` evaluated at FE nodes.
        """
        vFlat = vFunc.vector().get_local()
        v = vFlat.reshape((-1,d))
        # Divide nodal velocity/displacement through by FE nodal weights.
        for i in range(0,self.nNodes[s_ind]):
            wi = self.splines[s_ind].cpFuncs[d].vector().get_local()[i]
            for j in range(0,d):
                v[i,j] /= wi
        return v

    def GlobalToLocalNode(self,globalNode):
        """
        Convert the ``globalNode`` index to the local node index and the 
        spline index.
        """
        tempNode = globalNode
        for i in range(self.nSplines):
            tempNode = tempNode - self.nNodes[i]
            if tempNode < 0:
                s_ind = i
                localNode = self.nNodes[i] + tempNode
                break
        return s_ind, localNode
    
    def assembleContact(self,dispFunc,output_PETSc=False):
        """
        Return FE stiffness matrix/matrices and load vector/vectors 
        contributions associated with contact forces, based on an 
        FE displacement ``dispFunc``.  
        """
        if not isinstance(dispFunc, list):
            dispFunc = [dispFunc]
        
        # First build the lists of contact contributions 
        # with elements Nones. If contact force is detected, this 
        # method can return the contributions of matrices/vectors. 
        # If contact is not found, then this method just returns 
        # list of Nones and we don't have to create new matrices/vectors.       
        Fvs = [None for i in range(self.nSplines)]
        Kms = [[None for i in range(self.nSplines)] \
              for j in range(self.nSplines)]

        # Compute deformed positions of nodes in physical space.
        nodexs = []
        for i in range(self.nSplines):
            nodexs += [self.nodeXs[i] + self.evalFunction(dispFunc[i], i)]
        nodexsGlobal = np.concatenate(nodexs, axis=0)

        tree = cKDTree(nodexsGlobal)
        pairs = tree.query_pairs(self.r_max, output_type='ndarray')


        # Because the ndarray output from the scipy cKDTree maps onto a C++
        # type, this loop could likely be optimized further by placing it in
        # a JIT-compiled C++ extension module.  (For this, string
        # representations of phiPrime and phiDoublePrime would be needed.)
        for pair in pairs:
            node1 = pair[0]
            node2 = pair[1]
            s_ind1, node1_local = self.GlobalToLocalNode(node1)
            s_ind2, node2_local = self.GlobalToLocalNode(node2)
            s_inds = [s_ind1, s_ind2]

            # Positions of the two nodal quadrature points in the reference
            # configuration:
            X1 = self.nodeXs[s_ind1][node1_local, :]
            X2 = self.nodeXs[s_ind2][node2_local, :]
            R12 = npNorm(X2-X1)

            # Do not add contact forces between points that are close in the
            # reference configuration.  (Otherwise, the entire structure 
            # would expand, trying to get away from itself.)
            if R12 > self.R_self:
                # Contact force detected, create new PETSc matrices and 
                # vetors to store the contributions if they are Nones.
                for i in range(len(s_inds)):
                    if Fvs[s_inds[i]] is None:
                        Fv = PETSc.Vec(self.splines[s_inds[i]].comm)
                        Fv.createSeq(d*self.nNodes[s_inds[i]], 
                                     comm=self.splines[s_inds[i]].comm)
                        Fv.setUp()
                        Fvs[s_inds[i]] = Fv
                    for j in range(len(s_inds)):
                        if Kms[s_inds[i]][s_inds[j]] is None:
                            Km = PETSc.Mat(self.splines[s_inds[i]].comm)
                            Km.createAIJ([[d*self.nNodes[s_inds[i]], None], 
                                          [None, d*self.nNodes[s_inds[j]]]], 
                                         comm=self.splines[s_inds[i]].comm)
                            Km.setPreallocationNNZ([PREALLOC,PREALLOC])
                            Km.setUp()
                            Km.setOption(
                                PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,
                                False)
                            Kms[s_inds[i]][s_inds[j]] = Km

                # Positions of nodes in the current configuration:
                x1 = nodexs[s_ind1][node1_local,:]
                x2 = nodexs[s_ind2][node2_local,:]

                # Force computation: see (24) from original reference.
                r12vec = x2-x1
                r12 = npNorm(r12vec)
                r12hat = r12vec/r12
                r_otimes_r = npOuter(r12hat,r12hat)
                I = npIdentity(d)
                C = self.quadWeights[s_ind1][node1_local]\
                  * self.quadWeights[s_ind2][node2_local]
                f12 = C*self.phiPrime(r12)*r12hat

                # Nodal FE spline (not quadrature) weights:
                w1 = self.weights[s_ind1][node1_local]
                w2 = self.weights[s_ind2][node2_local]

                # Add equal-and-opposite forces to the RHS vector.
                for direction in range(0,d):
                    dof1 = nodeToDof(node1_local,direction)
                    dof2 = nodeToDof(node2_local,direction)
                    
                    # (Weights are involved here because the FE test function
                    # is in homogeneous representation.)
                    Fvs[s_ind1].setValue(dof1, -f12[direction]/w1, 
                                         addv=ADD_MODE)
                    Fvs[s_ind2].setValue(dof2, f12[direction]/w2, 

                                         addv=ADD_MODE)
                # Tangent computation: see (25)--(26) from original
                # reference.  
                k12_tensor = C*(self.phiDoublePrime(r12)*r_otimes_r 
                           + (self.phiPrime(r12)/r12)*(I-r_otimes_r))

                # Add tangent contributions to the LHS matrix.
                for d1 in range(0,d):
                    for d2 in range(0,d):
                        n1dof1 = nodeToDof(node1_local, d1)
                        n1dof2 = nodeToDof(node1_local, d2)
                        n2dof1 = nodeToDof(node2_local, d1)
                        n2dof2 = nodeToDof(node2_local, d2)
                        k12 = k12_tensor[d1, d2]
                        # 11 contribution:
                        Kms[s_ind1][s_ind1].setValue(n1dof1, n1dof2, 
                                            k12/(w1*w1), addv=ADD_MODE)
                        # 22 contribution:
                        Kms[s_ind2][s_ind2].setValue(n2dof1, n2dof2, 
                                            k12/(w2*w2), addv=ADD_MODE)
                        # Off-diagonal contributions:
                        Kms[s_ind1][s_ind2].setValue(n1dof1, n2dof2, 
                                            -k12/(w1*w2), addv=ADD_MODE)
                        Kms[s_ind2][s_ind1].setValue(n2dof1, n1dof2, 
                                            -k12/(w1*w2), addv=ADD_MODE)
                        # (Weights are involved here because FE test and 
                        # trial space basis functions are in homogeneous
                        # representation.)
        
        for i in range(self.nSplines):
            if Fvs[i] is not None:
                Fvs[i].assemble()
            for j in range(self.nSplines):
                if Kms[i][j] is not None:
                    Kms[i][j].assemble()

        if output_PETSc:
            if self.multiPatch:
                return Kms, Fvs
            else:
                return Kms[0][0], Fvs[0]
        else:
            Fs = [PETScVector(Fv) if Fv is not None else None for Fv in Fvs]
            Ks = [PETScMatrix(Kms[i][j]) if Kms[i][j] is not None else None \
                  for i in range(self.nSplines) for j in range(self.nSplines)]
            if self.multiPatch:
                return Ks, Fs
            else:
                return Ks[0][0], Fs[0]


class ShellContactNonlinearProblem(ExtractedNonlinearProblem):
    """
    Class encapsulating a nonlinear problem with an isogeometric shell 
    formulation and shell contact.  
    """
    def __init__(self,contactContext,residual,tangent,solution,**kwargs):
        self.contactContext = contactContext
        self.spline = self.contactContext.spline
        if isinstance(self.spline, list):
            raise TypeError("ShellContactNonlinearProblem only supports" 
                            "single ExtractedSpline.")
        super(ShellContactNonlinearProblem, self)\
            .__init__(self.spline,residual,tangent,solution,**kwargs)
    # Override methods from NonlinearProblem to perform extraction and
    # include contact forces:
    def form(self,A,P,B,x):
        self.solution.vector()[:] = self.spline.M*x
        self.Kc,self.Fc = self.contactContext.assembleContact(self.solution)
    def F(self,b,x):
        if self.Fc is not None:
            b[:] = self.spline.extractVector(assemble(self.residual)
                                             + self.Fc)
        else:
            b[:] = self.spline.extractVector(assemble(self.residual))
        return b
    def J(self,A,x):
        if self.Kc is not None:
            M = self.spline.extractMatrix(as_backend_type(
                                          assemble(self.tangent))+ self.Kc)
        else:
            M = self.spline.extractMatrix(as_backend_type(
                                          assemble(self.tangent)))
        Mm = as_backend_type(M).mat()
        A.mat().setSizes(Mm.getSizes())
        A.mat().setUp()
        A.mat().assemble()
        Mm.copy(result=A.mat())
        return A
