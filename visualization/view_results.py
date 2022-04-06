# trace generated using paraview version 5.9.0

import os
import argparse

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', dest='file_path', default='./',
                    help="File path for saved results.")
parser.add_argument('--show_geom', dest='show_geom', default=False, 
                    help="Show undeformed geometry.")
parser.add_argument('--show_geom_edge', dest='show_geom_edge', default=False, 
                    help="Show edges on undeformed geometry.")
parser.add_argument('--show_disp', dest='show_disp', default=True, 
                    help="Show deformed geometry.")
parser.add_argument('--show_disp_edge', dest='show_disp_edge', default=False,
                    help="Show edges on deformed geometry.")
parser.add_argument('--show_stress', dest='show_stress', default=False,
                    help="Show von Mises stress on deformed geometry.")
parser.add_argument('--geom_opacity', dest='geom_opacity', default=1.0,
                    help="Opacity for undeformed geometry.")
parser.add_argument('--disp_opacity', dest='disp_opacity', default=1.0,
                    help="Opacity for deformed geometry.")
parser.add_argument('--disp_scale', dest='disp_scale', default=1,
                    help="Scale factor for displacement.")
parser.add_argument('--start_ind', dest='start_ind', default=0,
                    help="Index for first spline.")
parser.add_argument('--end_ind', dest='end_ind', default=1,
                    help="Index for last spline.")
parser.add_argument('--save_state', dest='save_state', default=True,
                    help="Save current view to a Paraview state file.")
parser.add_argument('--filename_prefix', dest='filename_prefix', default='state',
                    help="Finename prerix for saved state file.")
parser.add_argument('--interact', dest='interact', default=False,
                    help="Launch interactive window.")

args = parser.parse_args()

### Arguments
file_path = args.file_path

show_geom = bool(args.show_geom)
show_geom_edge = bool(args.show_geom_edge)

show_disp = bool(args.show_disp)
show_disp_edge = bool(args.show_disp_edge)

show_stress = bool(args.show_stress)

geom_opacity = float(args.geom_opacity)
disp_opacity = float(args.disp_opacity)
disp_scale = int(args.disp_scale)

start_ind = int(args.start_ind)
end_ind = int(args.end_ind)
num_surf = end_ind - start_ind

save_state = bool(args.save_state)
filename = args.filename_prefix
interact = bool(args.interact)

if save_state:
    filename += "_patch_"+str(start_ind)+"_"+str(end_ind)
    if show_geom:
        filename += "_geom"
    if show_disp:
        filename += "_disp"+str(disp_scale)
    if show_stress:
        filename += "_stress"
    filename += ".pvsm"

renderView1 = GetActiveViewOrCreate('RenderView')

for surf_ind in range(start_ind, end_ind):

    print("--- Displaying surface", surf_ind)

    # create a new 'PVD Reader'
    f0_0_filepvd = PVDReader(registrationName='F'+str(surf_ind)+'_0_file.pvd', 
                             FileName=file_path+'F'+str(surf_ind)+'_0_file.pvd')
    f0_0_filepvd.PointArrays = ['F'+str(surf_ind)+'_0']

    # create a new 'PVD Reader'
    f0_1_filepvd = PVDReader(registrationName='F'+str(surf_ind)+'_1_file.pvd', 
                             FileName=file_path+'F'+str(surf_ind)+'_1_file.pvd')
    f0_1_filepvd.PointArrays = ['F'+str(surf_ind)+'_1']

    # create a new 'PVD Reader'
    f0_2_filepvd = PVDReader(registrationName='F'+str(surf_ind)+'_2_file.pvd', 
                             FileName=file_path+'F'+str(surf_ind)+'_2_file.pvd')
    f0_2_filepvd.PointArrays = ['F'+str(surf_ind)+'_2']

    # create a new 'PVD Reader'
    f0_3_filepvd = PVDReader(registrationName='F'+str(surf_ind)+'_3_file.pvd', 
                             FileName=file_path+'F'+str(surf_ind)+'_3_file.pvd')
    f0_3_filepvd.PointArrays = ['F'+str(surf_ind)+'_3']

    # create a new 'PVD Reader'
    u0_0_filepvd = PVDReader(registrationName='u'+str(surf_ind)+'_0_file.pvd', 
                             FileName=file_path+'u'+str(surf_ind)+'_0_file.pvd')
    u0_0_filepvd.PointArrays = ['u'+str(surf_ind)+'_0']

    # create a new 'PVD Reader'
    u0_1_filepvd = PVDReader(registrationName='u'+str(surf_ind)+'_1_file.pvd', 
                             FileName=file_path+'u'+str(surf_ind)+'_1_file.pvd')
    u0_1_filepvd.PointArrays = ['u'+str(surf_ind)+'_1']

    # create a new 'PVD Reader'
    u0_2_filepvd = PVDReader(registrationName='u'+str(surf_ind)+'_2_file.pvd', 
                             FileName=file_path+'u'+str(surf_ind)+'_2_file.pvd')
    u0_2_filepvd.PointArrays = ['u'+str(surf_ind)+'_2']

    if show_stress:
        # create a new 'PVD Reader'
        von_Mises_top_0pvd = PVDReader(registrationName='von_Mises_top_'+str(surf_ind)+'.pvd', 
                                       FileName=file_path+'von_Mises_top_'+str(surf_ind)+'.pvd')
        von_Mises_top_0pvd.PointArrays = ['von_Mises_top_'+str(surf_ind)+'']

    # set active source
    SetActiveSource(f0_0_filepvd)

    # # set active source
    # SetActiveSource(von_Mises_top_0pvd)

    if show_stress:
        # create a new 'Append Attributes'
        appendAttributes1 = AppendAttributes(registrationName='AppendAttributes'+str(surf_ind), 
                                             Input=[f0_0_filepvd, f0_1_filepvd, f0_2_filepvd, 
                                                    f0_3_filepvd, u0_0_filepvd, u0_1_filepvd, 
                                                    u0_2_filepvd, von_Mises_top_0pvd])
    else:
        appendAttributes1 = AppendAttributes(registrationName='AppendAttributes'+str(surf_ind), 
                                             Input=[f0_0_filepvd, f0_1_filepvd, f0_2_filepvd, 
                                                    f0_3_filepvd, u0_0_filepvd, u0_1_filepvd, 
                                                    u0_2_filepvd])

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator'+str(surf_ind)+'_1', 
                             Input=appendAttributes1)

    # Properties modified on calculator1
    calculator1.Function = '(F'+str(surf_ind)+'_0/F'+str(surf_ind)\
                         +'_3-coordsX)*iHat + (F'+str(surf_ind)+'_1/F'\
                         +str(surf_ind)+'_3-coordsY)*jHat + (F'+str(surf_ind)\
                         +'_2/F'+str(surf_ind)+'_3-coordsZ)*kHat'

    # create a new 'Warp By Vector'
    warpByVector1 = WarpByVector(registrationName='WarpByVector'+str(surf_ind)+'_1', 
                                 Input=calculator1)
    warpByVector1.Vectors = ['POINTS', 'Result']

    if show_geom:
        warpByVector1Display = Show(warpByVector1, renderView1, 
                                    'UnstructuredGridRepresentation')
        ColorBy(warpByVector1Display, None)
        warpByVector1Display.Opacity = geom_opacity
        if show_geom_edge:
            warpByVector1Display.SetRepresentationType('Surface With Edges')

    # create a new 'Calculator'
    calculator2 = Calculator(registrationName='Calculator'+str(surf_ind)+'_2', 
                             Input=warpByVector1)

    # Properties modified on calculator2
    calculator2.Function = '(u'+str(surf_ind)+'_0/F'+str(surf_ind)\
                         +'_3)*iHat + (u'+str(surf_ind)+'_1/F'\
                         +str(surf_ind)+'_3)*jHat + (u'+str(surf_ind)\
                         +'_2/F'+str(surf_ind)+'_3)*kHat'

    # create a new 'Warp By Vector'
    warpByVector2 = WarpByVector(registrationName='WarpByVector'+str(surf_ind)+'_2', 
                                 Input=calculator2)
    warpByVector2.Vectors = ['POINTS', 'Result']
    warpByVector2.ScaleFactor = disp_scale
    
    if show_disp:
        warpByVector2Display = Show(warpByVector2, renderView1, 
                                    'UnstructuredGridRepresentation')
        ColorBy(warpByVector2Display, ('POINTS', 'Result', 'Magnitude'))
        warpByVector2Display.Opacity = disp_opacity
        if show_disp_edge:
            warpByVector2Display.SetRepresentationType('Surface With Edges')

    # # show color bar/color legend
    # warpByVector2Display.SetScalarBarVisibility(renderView1, True)
    
    # UpdatePipeline(time=0.0, proxy=warpByVector2)

    if show_stress:
        # create a new 'Calculator'
        calculator3 = Calculator(registrationName='Calculator'+str(surf_ind)+'_3', 
                                 Input=warpByVector2)

        # Properties modified on calculator3
        calculator3.Function = 'von_Mises_top_'+str(surf_ind)+''

        # UpdatePipeline(time=0.0, proxy=calculator3)
        calculator3Display = Show(calculator3, renderView1, 
                                  'UnstructuredGridRepresentation')

renderView1.InteractionMode = '3D'
renderView1.ResetCamera()
renderView1.Update()

if save_state:
    if not os.path.exists("./states/"):
        os.mkdir("states")
    servermanager.SaveState("./states/"+filename)

if interact:
    RenderAllViews()
    Interact()