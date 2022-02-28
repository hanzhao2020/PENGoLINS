---
layout: default
title: Introduction
nav_order: 1
description: "The introduction page for PENGoLINS."
permalink: /
---

# PENGoLINS introduction

[PENGoLINS](https://github.com/hanzhao2020/PENGoLINS) (PENalty-based GLuing of Isogeometric Non-matching Shells) is an open-source Python library for penalty of non-matching Kirchhoff--Love shells using a [FEniCS](https://fenicsproject.org/)-based implementation of isogeometric analysis (IGA) called [tIGAr](https://github.com/david-kamensky/tIGAr). Users can directly perform shell structure analysis on CAD models, in STEP or IGES format, consisting of multiple B-spline/NURBS patches with non-matching parametrizations at their intersections. The [pythonOCC](https://github.com/tpaviot/pythonocc-core), a Python interface of [OpenCASCADE](https://www.opencascade.com/), is leveraged for computation of surface-surface intersections, where displacement and rotational continuities are maintained using penalty method. Furthermore, this library can be extended to fluid-structure interaction (FSI) and nonlocal contact analysis by integrating with existing open-source Python frameworks, viz., [VarMINT](https://github.com/david-kamensky/VarMINT), [ShNAPr](https://github.com/david-kamensky/ShNAPr) and [CouDALFISh](https://github.com/david-kamensky/CouDALFISh). Coupling between separately-parametrized patches uses a modification of the penalty formulation proposed by [Herrema et al.](https://doi.org/10.1016/j.cma.2018.08.038), which is verified using several [benchmark problems]({{ site.baseurl }}/docs/benchmarks). [Applications]({{ site.baseurl}}/docs/applications) to aerospace structures and prosthetic heart values are demonstrated in this documentation.

<!-- Insert figure -->
<!-- <p align="center">
  <img src="./figures/eVTOL_wing_geometry.png" title="eVTOL wing geometry" width="450">
</p> -->

<!-- Embed pdf into page -->
<!-- <object data="./figures/eVTOL_wing_convergence.pdf" type="application/pdf" width="500px" >
    <embed src="./figures/eVTOL_wing_convergence.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="./figures/eVTOL_wing_convergence.pdf">Download PDF</a>.</p>
    </embed>
</object> -->