---
layout: default
title: Get started
parent: Tutorial
nav_order: 2
---

# Get started

This section provides an overview of PENGoLINS code structure, analysis workflow, and usage of major functionality, so that users can have a basic understanding of how PENGoLINS works and what it can do. More detailed demos are discussed in the [Benchmarks]({{ site.baseurl }}/docs/benchmarks) and [Applications]({{ site.baseurl }}/docs/applications) sections.

## Preprocessing
Two major functionalities are developed for PENGoLINS in order to perform IGA for non-matching shell structures. The first one is geometry preprocessing, which is able to refine imported B-spline surfaces, reparametrize them to generate IGA-friendly surfaces if needed, and compute locations of surface-surface intersections in both physical and parametric spaces. The preprocessing step is demonstrated in the following code snippet
```python
from PENGoLINS.OCC_preprocessing import *
# Import a CAD geometry "example.igs" consisting of multiple B-spline 
# surfaces into a python list and convert them to the type of OCC 
# Geom B-spline surfaces.
igs_shapes = read_igs_file("example.igs", as_compound=False)
geom_surfaces = [topoface2surface(face, BSpline=True) for face in igs_shapes]

# Initialize class ``OCCPreprocessing`` by giving the list ``geom_surfaces``
# into it, two additional arguments, ``reparametrize`` and ``refine``, are 
# set as False by default. For conciseness and demonstration purposess, we 
# only use the default values in this section.
preprocessor = OCCPreprocessing(geom_surfaces)

# Compute locations of intersections between all surfaces and generate 
# related information that is necessary for analysis.
preprocessor.compute_intersections()
```

After these steps, users will have access to the attributes, which are required by the IGA for non-matching structures, about surface-surface intersections and B-spline surfaces of instance `preprocessor`. For example:
```python
# List of reparametrized surfaces if reparametrization is performed.
preprocessor.BSpline_surfs_repara
# List of refined surfaces if refinement is performed.
preprocessor.BSpline_surfs_refine
# List of indices of pairs of surfaces that have one or more intersections.
preprocessor.mapping_list
# List of numbers of elements for each mortar mesh, which can also 
# be defined by users.
preprocessor.mortar_nels
# List of parametric coordinates for each mortar mesh.
preprocessor.intersections_para_coords
```

With the above information, we are ready to run IGA for the imported non-matching shell structures, which is illustrated in the following section.


## IGA for non-matching shells

In order to analyze imported geometry using the open-source IGA Python library, tIGAr, we first need to create a list of ``splines`` of instances of type ``tIGAr.common.ExtractedSpline`` and define the boundary condition properly, then initialize the non-matching problem with it and related material properties. Examples of tIGAr extracted spline objects creation are demonstrated in the [benchmarks]({{ site.baseurl }}/docs/benchmarks) and [applications]({{ site.baseurl }}/docs/applications) sections.
```python
# Create a list contains tIGAr extracted splines: ``splines``.
# Define shell structures' material properties, Young's modulus: E, 
# shell thickness: h_th, Poisson's ratio: nu.
problem = NonMatchingCoupling(splines, E, h_th, nu)
```
Next, we want to create the mortar meshes, solution functions and their derivatives on the mortar meshes to integrate the differences of displacement and rotation between spline patches.
```python
# Create mortar meshes by giving a list contains number of elements for 
# each mortar mesh.
problem.create_mortar_meshes(preprocessor.mortar_nels)
# Create Dolfin functions represent solutions and their derivatives
# with finite element family ``CG1``.
problem.create_mortar_funcs('CG', 1)
problem.create_mortar_funcs_derivative('CG', 1)
```
Once mortar meshes and their functions are available, the transfer matrices between function spaces of spline patches and mortar meshes can be created, as well as the penalty parameters based on the information from ``preprocessor``. These routines are encapsulated into the following method:
```python
# Define an int ``penalty_coefficient``, the recommended value is 1000.
problem.mortar_meshes_setup(preprocessor.mapping_list,
                            preprocessor.intersections_para_coords,
                            penalty_coefficient)
```
The ``int`` object ``penalty_coefficient`` is the problem-independent, dimensionless penalty coefficient, which is used to calculate the penalty parameters.

Now, we can define the nonlinear residual for each spline patch using the St. Venant--Krichhoff constitutive model and pass them to the instance ``problem``. For shell structure under self weight as an example (assume self weight is along with z-direction):
```python
areal_force_density = Constant(90.0)
f = as_vector([Constant(0.0), Constant(0.0), -areal_force_density])
residuals = []
for i in range(problem.num_splines):
    source_term = inner(f, problem.splines[i].rationalize(
                  problem.spline_test_funcs[i]))*problem.splines[i].dx
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
                  problem.spline_test_funcs[i], E, nu, h_th, source_term)]
# Pass list of residuals to method ``set_residuals`` in ``problem``. 
# User can also pass optional derivatives of residuals to this method. 
# Derivatives will be computed symbolically using function ``derivative`` 
# in UFL if they are not provided.
problem.set_residuals(residuals)
```
Now, the ``problem`` object has collected all the required information to perform IGA for non-matching shell structures. We can solve the linearized problem:
```python
problem.solve_linear_nonmatching_problem()
```
or solve the nonlinear problem using Newton's iteration:
```python
problem.solve_nonlinear_nonmatching_problem()
```
Direct solver is employed to solve the linear system by default, users can also choose the PETSc Krylov solver or provide a customized solver. The resulting solution in homogenous coordinate can be accessed in ``problem.spline_funcs[i]``, where ``i`` is the index of spline patch.