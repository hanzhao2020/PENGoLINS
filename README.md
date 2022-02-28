# PENGoLINS

A Python module for **PEN**alty-based **GL**uing of **I**sogeometric **N**on-matching **S**hells (where the lower-case "o" is added for pronunciation; pronounced like "pangolins", mammals that are covered in large, protective scales). This framework performs isogeometric analysis (IGA) for collections of non-matching Kirchhoff--Love shells using the penalty method. 

Detailed tutorial and examples are demonstrated in [PENGoLINS documentation](https://hanzhao2020.github.io/PENGoLINS/).

Couping between non-matching patches uses a slightly modification of penalty formulation proposed in Section 2 of

https://doi.org/10.1016/j.cma.2018.08.038

## Dependencies

1. [Numpy](https://numpy.org/) and [SciPy](https://scipy.org/) can be installed through ``pip3``.

2. Installation of [FEniCS](https://fenicsproject.org/) can be found [here](https://fenicsproject.org/download/archive/).

3. IGA Python library [tIGAr](https://github.com/david-kamensky/tIGAr) can be installed based on [here](https://github.com/david-kamensky/tIGAr/blob/master/README.md).

4. The Python module for isogeometric Kirchhoff--Love shell [ShNAPr](https://github.com/david-kamensky/ShNAPr) is required.

5. Leveraging geometry kernel requires [pythonOCC](https://github.com/tpaviot/pythonocc-core), whose installation instruction can be found [here](https://github.com/tpaviot/pythonocc-core/blob/master/INSTALL.md).

6. Geometry in benchmark problems are created using [igakit](https://bitbucket.org/dalcinl/igakit/src/master/).

7. Running FSI analysis requires the variational multiscale incompressible Navier--Stokes toolkit [VarMINT](https://github.com/david-kamensky/VarMINT) and Python module for coupling of fluids with immersed shells [CouDALFISh](https://github.com/david-kamensky/CouDALFISh).

## Installation of PENGoLINS
First clone the GitHub repository and install it using `pip3` on the top-level directory of the repository
```
git clone https://github.com/hanzhao2020/PENGoLINS.git
pip3 install -e.
```