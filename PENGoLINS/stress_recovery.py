"""
The ``stress_recovery`` module:
-------------------------------
This module provides functionality to compute stresses 
and stress resultants for Kirchhoff--Love shell using 
St. Venant--Kirchhoff material module.
"""

import numpy as np 
from tIGAr import *
from ShNAPr.kinematics import *

def contravariantTensorToCartesian2D(T,a,a0,a1):
    """
    Convert 2D contravariant tensor from curvilinear basis 
    to local Cartesian basis.

    Parameters
    ----------
    T : ufl rank 1 or rank 2 tensor
    a : metric tensor
    a0, a1: curvilinear basis vectors

    Returns
    -------
    res : ufl tensor
    """
    e0,e1 = orthonormalize2D(a0,a1)
    e0c, e1c = e0, e1
    eca = as_matrix(((inner(e0c,a0),inner(e0c,a1)),
                     (inner(e1c,a0),inner(e1c,a1))))
    aec = eca.T

    rank = len(T.ufl_shape)
    if rank == 1:
        return eca*T
    elif rank == 2:
        return eca*T*aec
    else:
        raise ValueError("Rank "+str(rank)+" tensor is not supported.")


def contravariantTensorToCurvilinear2D(T,a,a0,a1):
    """
    Convert 2D contravariant tensor from local Cartesian basis 
    to curvilinear basis.

    Parameters
    ----------
    T : ufl rank 1 or rank 2 tensor
    a : metric tensor
    a0, a1: curvilinear basis vectors

    Returns
    -------
    res : ufl tensor
    """
    # raise indices on curvilinear basis
    ac = inv(a)
    a0c = ac[0,0]*a0 + ac[0,1]*a1
    a1c = ac[1,0]*a0 + ac[1,1]*a1
    e0,e1 = orthonormalize2D(a0,a1)
    ace = as_matrix(((inner(a0c,e0),inner(a0c,e1)),
                     (inner(a1c,e0),inner(a1c,e1))))
    eac = ace.T

    rank = len(T.ufl_shape)
    if rank == 1:
        return ace*T
    elif rank == 2:
        return ace*T*eac
    else:
        raise ValueError("Rank "+str(rank)+" tensor is not supported.")


def projectScalar(spline, toProject, V, lumpMass=False, linearSolver=None):
    """
    Project scalar ``toProject`` onto function space ``V``.

    Parameters
    ----------
    spline : tIGAr extracted spline
    toProject : ufl object
    V : dolfin function space
    lumpMass : bool, default is False
    linearSolver : default is None

    Returns
    -------
    u : dolfin function
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    if(lumpMass):
        lhsForm = inner(Constant(1.0),v)*spline.dx.meas
    else:
        lhsForm = inner(u,v)*spline.dx.meas #lhs(res)
    rhsForm = inner(toProject,v)*spline.dx.meas #rhs(res)
    A = assemble(lhsForm)
    b = assemble(rhsForm)
    u = Function(V)
    if(lumpMass):
        as_backend_type(u.vector())\
            .vec().pointwiseDivide(as_backend_type(b).vec(),
                                   as_backend_type(A).vec())
    else:
        if(linearSolver == None):
            solve(A,u.vector(),b)
        else:
            linearSolver.solve(A,u.vector(),b)
    return u


class ShellForceSVK:
    """
    Class to compute Kirchhoff--Love shell's stress resultants using
    St. Venant--Kirchhoff (SVK) material model.
    """
    def __init__(self, spline, u_hom, E, nu, h_th, linearize=False):
        """
        Parameters
        ----------
        spline : tIGAr extracted spline
        u_hom : dolfin Function. Numerical solution of problem.
        E : dolfin constant. Material's Young's modulus
        nu : dolfin constant. Material's Poisson's ratio
        h_th : dolfin constant. Shell's thickness
        linearize : bool, default is False.
            If ``linearize`` is True, using linearized membrane strains
            and curvature changes to compute stress resultants. For 
            linear problem, set this argument as True.
        """
        self.spline = spline
        self.u_hom = u_hom
        self.u_hom_soln = Function(self.spline.V)
        self.u_hom_soln.assign(self.u_hom)
        self.E = E 
        self.nu = nu
        self.h_th = h_th
        self.linearize = linearize

        # Material matrix of isotropic material
        self.D = (self.E/(1.0 - self.nu*self.nu))\
               *as_matrix([[1.0,     self.nu, 0.0],
                           [self.nu, 1.0,     0.0],
                           [0.0,     0.0,     0.5*(1.0-self.nu)]])

        # Surface geometry of midsurface in reference and 
        # deformed configurations
        self.A0, self.A1, self.A2, self.deriv_A2, self.A, self.B \
            = surfaceGeometry(self.spline, self.spline.F)
        self.a0, self.a1, self.a2, self.deriv_a2, self.a, self.b \
            = surfaceGeometry(self.spline, self.spline.F + 
                              spline.rationalize(self.u_hom_soln))

    def membraneStrain(self):
        """
        Returns the membrane strains of Kirchhoff--Love shell in local
        Cartesian basis.
        """
        epsilon = 0.5*(self.a - self.A)
        epsilonBar = covariantRank2TensorToCartesian2D(epsilon, self.A, 
                                                       self.A0, self.A1)
        if self.linearize:
            self.u_hom_soln.interpolate(Constant((0.,0.,0.)))
            epsilonBarLin = derivative(epsilonBar, self.u_hom_soln, 
                                       self.u_hom)
            return epsilonBarLin
        else:
            return epsilonBar

    def curvatureChange(self):
        """
        Returns the curvature changes of Kirchhoff--Love shell in local
        Cartesian basis.
        """
        kappa = self.B - self.b
        kappaBar = covariantRank2TensorToCartesian2D(kappa, self.A, 
                                                     self.A0, self.A1)
        if self.linearize:
            self.u_hom_soln.interpolate(Constant((0.,0.,0.)))
            kappaBarLin = derivative(kappaBar, self.u_hom_soln, self.u_hom)
            return kappaBarLin
        else:
            return kappaBar

    def normalForce(self):
        """
        Returns the normal forces of Kirchhoff--Love shell in actual
        configuration.
        """
        epsilonBar = self.membraneStrain()
        epsilonBar2D = voigt2D(epsilonBar)
        nBar2D = self.h_th*self.D*epsilonBar2D
        nBar = invVoigt2D(nBar2D, strain=False)
        n = contravariantTensorToCurvilinear2D(nBar, self.A, self.A0, self.A1)
        nHat = contravariantTensorToCartesian2D(n, self.a, self.a0, self.a1)
        return nHat

    def bendingMoments(self):
        """
        Returns the bending moments of Kirchhoff--Love shell in actual
        configuration.
        """
        kappaBar = self.curvatureChange()
        kappaBar2D = voigt2D(kappaBar)
        mBar2D = (self.h_th**3)*self.D*kappaBar2D/12.0
        mBar = invVoigt2D(mBar2D, strain=False)
        m = contravariantTensorToCurvilinear2D(mBar, self.A, self.A0, self.A1)
        mHat = contravariantTensorToCartesian2D(m, self.a, self.a0, self.a1)
        return mHat

    def shearForce(self):
        """
        Returns the shear forces of Kirchhoff--Love shell in actual
        configuration. Transverse strains are neglected in the 
        Kirchhoff--Love theory, shear forces can be obtained from 
        bending moments by taking equilibrium into consideration.
        """
        mBar = self.bendingMoments()

        # # Shear force components
        # qBar1 = grad(mBar[0,0])[0]/sqrt(self.a[0,0]) \
        #       + grad(mBar[0,1])[1]/sqrt(self.a[1,1])
        # qBar2 = grad(mBar[1,1])[1]/sqrt(self.a[1,1]) \
        #       + grad(mBar[1,0])[0]/sqrt(self.a[0,0])

        e0, e1 = orthonormalize2D(self.a0, self.a1)
        # Shear force components

        # # Dividing $(a_{\alpha\alpha})^{1/2}$ gives a much smaller
        # # shear force value compared to refernece value (7 compared to 278)
        # qBar1 = dot(self.spline.grad(mBar[0,0]), e0)/sqrt(self.a[0,0]) \
        #       + dot(self.spline.grad(mBar[0,1]), e1)/sqrt(self.a[1,1])
        # qBar2 = dot(self.spline.grad(mBar[1,1]), e1)/sqrt(self.a[1,1]) \
        #       + dot(self.spline.grad(mBar[1,0]), e0)/sqrt(self.a[0,0])

        qBar1 = dot(self.spline.grad(mBar[0,0]), e0) \
              + dot(self.spline.grad(mBar[0,1]), e1)
        qBar2 = dot(self.spline.grad(mBar[1,1]), e1) \
              + dot(self.spline.grad(mBar[1,0]), e0)

        qBar = as_vector([qBar1, qBar2])
        q = contravariantTensorToCurvilinear2D(qBar, self.A, self.A0, self.A1)
        qHat = contravariantTensorToCartesian2D(q, self.a, self.a0, self.a1)
        return qHat

class ShellStressSVK(ShellForceSVK):
    """
    Class to compute Kirchhoff--Love shell's stresses using
    St. Venant--Kirchhoff (SVK) material model.
    """
    def __init__(self, spline, u_hom, E, nu, h_th, linearize=False):
        """
        Parameters
        ----------
        spline : tIGAr extracted spline
        u_hom : dolfin Function. Numerical solution of problem.
        E : dolfin constant. Material's Young's modulus
        nu : dolfin constant. Material's Poisson's ratio
        h_th : dolfin constant. Shell's thickness
        linearize : bool, default is False.
            If ``linearize`` is True, using linearized membrane strains
            and curvature changes to compute stresses. For linear 
            problem, set this argument as True.
        """
        super().__init__(spline, u_hom, E, nu, h_th, linearize)

    def secondPiolaKirchhoffStress(self, xi2):
        """
        Returns 2nd Piola--Kirchhoff stresses in curvilinear basis at the
        through thickness coordinate ``xi2`` (-h_th/2 <= xi2 <= h_th/2). 
        """
        epsilonBar = self.membraneStrain()
        kappaBar = self.curvatureChange()
        # Green-Lagrange strain tensor
        self.EBar = epsilonBar + xi2*kappaBar

        # Kirchhoff--Love shell geometry at through thickness 
        # coordinate ``xi2``
        self.G = metricKL(self.A, self.B, xi2)
        self.G0, self.G1 = curvilinearBasisKL(self.A0, self.A1, 
                                              self.deriv_A2, xi2)
        self.g = metricKL(self.a, self.b, xi2)
        self.g0, self.g1 = curvilinearBasisKL(self.a0, self.a1, 
                                              self.deriv_a2, xi2)

        self.EBar2D = voigt2D(self.EBar, strain=True)
        PK2StressBar2D = self.D*self.EBar2D
        PK2StressBar = invVoigt2D(PK2StressBar2D, strain=False)
        PK2Stress = contravariantTensorToCurvilinear2D(PK2StressBar, self.G, 
                                                       self.G0, self.G1)
        return PK2Stress

    def cauchyStress(self, xi2):
        """
        Returns Cauchy stress tensor in local Cartesian basis at the 
        through thickness coordinate ``xi2`` (-h_th/2 <= xi2 <= h_th/2). 
        """
        PK2Stress = self.secondPiolaKirchhoffStress(xi2)

        # In-plane Jacobian determinant
        J0 = sqrt(det(self.g)/det(self.G))
        # Compute 33 component of Green-Lagrange strain tensor using
        # plane stress condition
        E22Bar = -self.nu*(self.EBar[0,0]+self.EBar[1,1])
        # 33 component of right Cauchy-Green deformation tensor
        # E = 1/2*(C-I)
        C22Bar = 2*E22Bar + 1
        # Determinant of deformation gradient/Jacobian determinant
        J = J0*sqrt(C22Bar)

        # Cauchy stress
        sigma = 1/J*PK2Stress
        sigmaHat = contravariantTensorToCartesian2D(sigma, self.g, 
                                                    self.g0, self.g1)
        return sigmaHat

    def vonMisesStress(self, xi2):
        """
        Returns Kirchhoff--Love shell's von Mises stress at the through 
        thickness coordinate ``xi2`` (-h_th/2 <= xi2 <= h_th/2).
        """
        sigmaHat = self.cauchyStress(xi2)
        # von Mises stress formula with plane stress
        vonMises = sqrt(sigmaHat[0,0]**2 - sigmaHat[0,0]*sigmaHat[1,1] 
                        + sigmaHat[1,1]**2 + 3*sigmaHat[0,1]**2)
        return vonMises