from PENGoLINS.occ_utils import *
from igakit.cad import *
from igakit.io import VTK

def coeff_vec(n, p, peak_ind=None):
    vec = np.ones(n)
    if peak_ind is None:
        peak_ind = int(n/2)
    vec[:peak_ind] = np.linspace(1, p, peak_ind)
    vec[peak_ind:] = np.linspace(p, 1, n-peak_ind)
    return vec

def seperate_leaflet(BSpline_surf, p_u=3, p_v=3):
    # Seperate the original leaflet to 4 surfaces to avoid singularity
    # For surface 0
    t0, t00, t01 = 0., 0.15, 0.4
    t1, t10, t11 = 0., 0.45, 1.
    n0, n1 = 10, 12
    vec0 = np.linspace(t0, t01, n1)
    vec1 = np.linspace(t00, t01, n1)
    vec2 = np.linspace(t1, t10, n1)

    para_loc0 = np.zeros((n0, n1, 2))
    for i in range(n1):
        if i == 0:
            para_loc0[:,i,0] = np.linspace(vec0[i], vec1[i],n0)
            para_loc0[:,i,1] = np.linspace(t1, vec2[i], n0)
        else:
            para_loc0[:,i,0] = np.linspace(vec0[i], vec1[i],n0)
            para_loc0[:,i,1] = np.linspace(t11, vec2[i], n0)

    coeff_vec0 = coeff_vec(n1, 0.7, 4)
    coeff_vec1 = coeff_vec(n1, 0.9, 4)
    for i in range(n0):
        para_loc0[i][1][1] = para_loc0[i][1][1]*coeff_vec0[i]
        para_loc0[i][2][1] = para_loc0[i][2][1]*coeff_vec1[i]

    # For surface 1
    t0, t00, t01 = 1., 0.85, 0.6
    t1, t10, t11 = 0., 0.45, 1.
    vec0 = np.linspace(t0, t01, n1)
    vec1 = np.linspace(t00, t01, n1)
    vec2 = np.linspace(t1, t10, n1)

    para_loc1 = np.zeros((n0, n1, 2))
    for i in range(n1):
        if i == 0:
            para_loc1[:,i,0] = np.linspace(vec0[i], vec1[i],n0)
            para_loc1[:,i,1] = np.linspace(t1, vec2[i], n0)
        else:
            para_loc1[:,i,0] = np.linspace(vec0[i], vec1[i],n0)
            para_loc1[:,i,1] = np.linspace(t11, vec2[i], n0)
    for i in range(n0):
        para_loc1[i][1][1] = para_loc1[i][1][1]*coeff_vec0[i]
        para_loc1[i][2][1] = para_loc1[i][2][1]*coeff_vec1[i]
    para_loc1 = np.flip(para_loc1, axis=0)

    # For surface 2
    n0, n1 = 11, 11
    para_loc2 = np.zeros((n0, n1, 2))
    t00, t01, t02, t03 = 0.15, 0.4, 0.6, 0.85
    t10, t11 = 0., 0.45
    vec0 = np.linspace(t00, t01, n1)
    vec1 = np.linspace(t03, t02, n1)
    vec2 = np.linspace(t10, t11, n1)
    for i in range(n1):
        para_loc2[:,i,0] = np.linspace(vec0[i], vec1[i], n0)
        para_loc2[:,i,1] = np.linspace(vec2[i], vec2[i], n0)

    # For surface 3
    n0, n1 = 8, 8
    para_loc3 = np.zeros((n0, n1, 2))
    t00, t01 = 0.4, 0.6
    t10, t11 = 0.45, 1.
    vec0 = np.linspace(t00, t01, n0)
    vec1 = np.linspace(t10, t11, n1)
    for i in range(n1):
        para_loc3[:,i,0] = vec0
        para_loc3[:,i,1] = np.linspace(vec1[i], vec1[i], n0)

    bs_sec0 = BSpline_surface_section(BSpline_surf, para_loc0, p_u, p_v)
    bs_sec1 = BSpline_surface_section(BSpline_surf, para_loc1, p_u, p_v)
    bs_sec2 = BSpline_surface_section(BSpline_surf, para_loc2, p_u, p_v)
    bs_sec3 = BSpline_surface_section(BSpline_surf, para_loc3, p_u, p_v)

    bs_sec_list = [bs_sec0, bs_sec1, bs_sec2, bs_sec3]
    return bs_sec_list

file_path = "./leaflet-geometries/"

num_srfs = 3
nurbs_srfs = []

for i in range(num_srfs):
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
    nurbs_srfs += [s]


occ_bspline_surfs = [ikNURBS2BSpline_surface(s) for s in nurbs_srfs]
nonmatching_occ_bs = []
for i in range(num_srfs):
    nonmatching_occ_bs += [seperate_leaflet(occ_bspline_surfs[i],
                           p_u=nurbs_srfs[i].degree[0],
                           p_v=nurbs_srfs[i].degree[1]),]

nonmatching_nurbs_srfs = []
for i in range(len(nonmatching_occ_bs)):
    nonmatching_nurbs_srfs += [[],]
    for j in range(len(nonmatching_occ_bs[i])):
        nonmatching_nurbs_srfs[i] += [BSpline_surface2ikNURBS(
            nonmatching_occ_bs[i][j], 3, 0, 0),]
        # # Write non-matching heart valves into VTK files
        # write_dir = "./nonmatching_leaflets/"
        # file_name = "leaflet_" + str(i) + "_" + str(j) + ".vtk"
        # VTK().write(write_dir+file_name, nonmatching_nurbs_srfs[i][j])