import math
from dolfin import *
from mshr import *
from tIGAr import *

resolution = 70
CYLINDER_RAD = 1.1
BOTTOM = -0.5
TOP = 2.0
tube = Cylinder(Point(0,0,BOTTOM),
                Point(0,0,TOP),CYLINDER_RAD,CYLINDER_RAD)

SINUS_CENTER_FAC = 0.5
SINUS_RAD_FAC = 0.8
SINUS_Z_SHIFT = -0.2
sinusRad = SINUS_RAD_FAC*CYLINDER_RAD
sinusZ = sinusRad + SINUS_Z_SHIFT
for i in range(0,3):
    sinusTheta = math.pi/3.0 + i*2.0*math.pi/3.0
    sinusCenterRad = SINUS_CENTER_FAC*CYLINDER_RAD
    sinusCenter = Point(sinusCenterRad*math.cos(sinusTheta),
                        sinusCenterRad*math.sin(sinusTheta),sinusZ)
    tube += Sphere(sinusCenter,sinusRad)

mesh = generate_mesh(tube,resolution)
f = HDF5File(worldcomm, "mesh.h5","w")
f.write(mesh,"/mesh")
f.close()