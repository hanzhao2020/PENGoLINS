---
layout: default
title: Installation
parent: Tutorial
nav_order: 1
---

# Installation instructions

## Dependencies
To install PENGoLINS, a list of dependencies have to be installed beforehand.

1. [Numpy](https://numpy.org/) and [SciPy](https://scipy.org/) can be installed through `pip3`.

2. Installation of [FEniCS](https://fenicsproject.org/) can be found [here](https://fenicsproject.org/download/archive/).

3. IGA Python library [tIGAr](https://github.com/david-kamensky/tIGAr) can be installed based on [here](https://github.com/david-kamensky/tIGAr/blob/master/README.md).

4. The Python module for isogeometric Kirchhoff--Love shell [ShNAPr](https://github.com/david-kamensky/ShNAPr) is required.

5. Leveraging geometry kernel requires [pythonOCC](https://github.com/tpaviot/pythonocc-core), whose installation instruction can be found [here](https://github.com/tpaviot/pythonocc-core/blob/master/INSTALL.md).

6. Geometry in benchmark problems are created using [igakit](https://bitbucket.org/dalcinl/igakit/src/master/), which can be installed using the following command
```bash
pip3 install https://bitbucket.org/dalcinl/igakit/get/master.tar.gz
```

7. Running FSI analysis requires the variational multiscale incompressible Navier--Stokes toolkit [VarMINT](https://github.com/david-kamensky/VarMINT) and Python module for coupling of fluids with immersed shells [CouDALFISh](https://github.com/david-kamensky/CouDALFISh).

## Install PENGoLINS
First clone the GitHub repository and install it using `pip3` on the top-level directory of the repository
```bash
git clone https://github.com/hanzhao2020/PENGoLINS.git
pip3 install -e.
```