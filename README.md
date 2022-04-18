# PENGoLINS

A Python module for **PEN**alty-based **GL**uing of **I**sogeometric **N**on-matching **S**hells (where the lower-case "o" is added for pronunciation; pronounced like "pangolins", mammals that are covered in large, protective scales). This framework performs isogeometric analysis (IGA) for collections of non-matching Kirchhoff--Love shells using the penalty method. Penalty formulation and design of PENGoLINS are discussed in the following article:
```
@article{Zhao2022,
title = "An open-source framework for coupling non-matching isogeometric shells with application to aerospace structures",
journal = "Computers \& Mathematics with Applications",
volume = "111",
pages = "109--123",
year = "2022",
issn = "0898-1221",
doi = "https://doi.org/10.1016/j.camwa.2022.02.007",
author = "H. Zhao and X. Liu and A. H. Fletcher and R. Xiang and J. T. Hwang and D. Kamensky"
}
```
A preprint of the above article can be found [here](https://github.com/LSDOlab/lsdo_bib/blob/main/pdf/zhao2022open.pdf). We found that the penalty formulation in the published article may result in the large deformation residual stagnating for patches with an angle close to $180^\circ$. Thus, the code repository is updated using the penalty formulation from [Herrema et al.](https://doi.org/10.1016/j.cma.2018.08.038), where the intersection's tangent vector is approximated numerically.

Detailed tutorial and examples are demonstrated in [PENGoLINS documentation](https://hanzhao2020.github.io/PENGoLINS/).

## Dependencies

1. [Numpy](https://numpy.org/) and [SciPy](https://scipy.org/) can be installed through ``pip3``.

2. Installation of [FEniCS](https://fenicsproject.org/) can be found [here](https://fenicsproject.org/download/archive/).

3. IGA Python library [tIGAr](https://github.com/david-kamensky/tIGAr) can be installed based on [here](https://github.com/david-kamensky/tIGAr/blob/master/README.md).

4. The Python module for isogeometric Kirchhoff--Love shell [ShNAPr](https://github.com/david-kamensky/ShNAPr) is required.

5. Leveraging geometry kernel requires [pythonOCC](https://github.com/tpaviot/pythonocc-core), whose installation instruction can be found [here](https://github.com/tpaviot/pythonocc-core/blob/master/INSTALL.md).

6. Geometry in benchmark problems are created using [igakit](https://github.com/dalcinl/igakit).

7. Running FSI analysis requires the variational multiscale incompressible Navier--Stokes toolkit [VarMINT](https://github.com/david-kamensky/VarMINT) and Python module for coupling of fluids with immersed shells [CouDALFISh](https://github.com/david-kamensky/CouDALFISh).

## Installation of PENGoLINS
First clone the GitHub repository and install it using `pip3` on the top-level directory of the repository
```
git clone https://github.com/hanzhao2020/PENGoLINS.git
pip3 install -e.
```