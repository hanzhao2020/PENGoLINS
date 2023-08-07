"""
Download compressed bicuspid valve geometry from the following link:
    https://drive.google.com/file/d/1fBrtRn_ufo5SQiJQTH7XFIQE5ptLuJdv/view?usp=sharing

Then run this script to generate igs file.
"""

from PENGoLINS.occ_utils import *
from PENGoLINS.igakit_utils import *

file_path = "./BAV_scaled_fusion_90/"
write_dir = "./"
num_surfs = 7
ik_nurbs_surfs = []

for i in range(num_surfs):
    # fname = "smesh."+str(i)+".1.txt"
    fname = "smesh."+str(i+1)+".dat"
    f = open(file_path+fname, 'r')
    fs = f.read()
    f.close()
    lines = fs.split("\n")

    nvar = -1
    if nvar == -1:
        nsd = int(lines[0])
        nvar = len(lines[1].split())

    # Load spline degree, number of control points
    degrees = []
    deg_strs = lines[1].split()
    ncps = []
    ncp_strs = lines[2].split()
    for d in range(nvar):
        degrees += [int(deg_strs[d]),]
        ncps += [int(ncp_strs[d])]

    # Load knot vectors
    kvecs = []
    for d in range(nvar):
        kvecs_strs = lines[3+d].split()
        kvec = []
        for s in kvecs_strs:
            kvec += [float(s),]
        kvecs += [np.array(kvec)/np.array(kvec)[-1],]

    # Load control points
    ncp = 1
    for d in range(nvar):
        ncp *= ncps[d]

    bnet = []
    for pt in range(ncp):
        bnet_row = []
        coord_strs = lines[3+nvar+pt].split()
        w = float(coord_strs[nsd])
        for d in range(nsd):
            bnet_row += [float(coord_strs[d])*w,]
        bnet_row += [w,]
        bnet += [bnet_row,]

    control = np.array(bnet)
    control_new = control.reshape(kvecs[1].size-degrees[1]-1,
                                  kvecs[0].size-degrees[0]-1,4)

    s = NURBS(kvecs, np.transpose(control_new, (1,0,2)))
    ik_nurbs_surfs += [s]

    # # Save to vtk files (for geometry visualization in ParaView)
    # write_filename = "leaflet_"+str(i)+".vtk"
    # VTK().write(write_dir+write_filename, ik_nurbs_surfs[i])

occ_surfs = [ikNURBS2BSpline_surface(s) for s in ik_nurbs_surfs]
write_geom_file(occ_surfs, write_dir+"nonmatching_bicuspid.igs")