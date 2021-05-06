# PENGoLINS

A Python module for **PEN**alty-based **GL**uing of **I**sogeometric **N**on-matching **S**hells (where the lower-case "o" is added for pronunciation; pronounced like "pangolins", mammals that covered in large, protective scales). This module couples non-matching shell patches (patches share common boundaries or have intersections but have different parametric discretizations) to maintain their displacement and rotational continuity during isogeometric analysis (IGA). The isogeometric analysis relies on [tIGAr](https://github.com/david-kamensky/tIGAr) (A Python library for IGA using [FEniCS](https://fenicsproject.org/)) ,and the isogeometric discretization of thin shells uses [ShNAPr](https://github.com/david-kamensky/ShNAPr). Usage of NURBS requires [igakit](https://bitbucket.org/dalcinl/igakit/src/master/) and computation of non-matching interface locations requires [pythonocc](https://github.com/tpaviot/pythonocc-core). \
A significant portion of the penalty formulation for non-matching patches is taken from Section 2 of 

https://doi.org/10.1016/j.cma.2018.08.038



