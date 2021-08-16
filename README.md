# PENGoLINS

A Python module for **PEN**alty-based **GL**uing of **I**sogeometric **N**on-matching **S**hells (where the lower-case "o" is added for pronunciation; pronounced like "pangolins", mammals that are covered in large, protective scales). This framework performs isogeometric analysis (IGA) for collections of non-matching Kirchhoff--Love shells using the penalty method. 

Details of penalty formulation for non-matching shells are given in Section 2 of

https://doi.org/10.1016/j.cma.2018.08.038

## Dependencies
* [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/) are required.
* This module is developed based on [FEniCS](https://fenicsproject.org/).
* Performing IGA relies on [tIGAr](https://github.com/david-kamensky/tIGAr).
* Geometry preprocessing requires [igakit](https://bitbucket.org/dalcinl/igakit/src/master/) and [pythonocc](https://github.com/tpaviot/pythonocc-core).
* Kirchhoff--Love shell models are obtained from [ShNAPr](https://github.com/david-kamensky/ShNAPr).
* Fluid--thin structure interaction analysis requires [VarMINT](https://github.com/david-kamensky/VarMINT) and [CouDALFISh](https://github.com/david-kamensky/CouDALFISh).


## Installation
It is recommended that install all dependencies in an environment of [Anaconda](https://www.anaconda.com/), which is the most convenient way to install geometry tool [pythonocc](https://github.com/tpaviot/pythonocc-core). Then clone the repository and install it by running `pip3 install -e.` in the top-level directory of the repository.

API documentation can be created by changing the directory to `docs` and running `make html`. The main documentation is located in `docs/_build/html/index.html`.