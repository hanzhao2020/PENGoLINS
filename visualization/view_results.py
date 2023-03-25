# trace generated using paraview version 5.9.0
import os
import argparse
import distutils

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', dest='file_path', default='./results/',
                    help="File path for saved results.")
# For geometry
parser.add_argument('--show_geom', dest='show_geom', default=False, 
                    help="Show undeformed geometry.")
parser.add_argument('--show_geom_edge', dest='show_geom_edge', default=False, 
                    help="Show edges on undeformed geometry.")
parser.add_argument('--geom_opacity', dest='geom_opacity', default=1.0,
                    help="Opacity for undeformed geometry.")
# For displacements
parser.add_argument('--show_disp', dest='show_disp', default=False, 
                    help="Show deformed geometry.")
parser.add_argument('--show_disp_edge', dest='show_disp_edge', default=False,
                    help="Show edges on deformed geometry.")
parser.add_argument('--disp_opacity', dest='disp_opacity', default=1.0,
                    help="Opacity for deformed geometry.")
parser.add_argument('--disp_scale', dest='disp_scale', default=1.0,
                    help="Scale factor for displacement.")
# For von Mises stress
parser.add_argument('--show_stress', dest='show_stress', default=True,
                    help="Show von Mises stress on deformed geometry.")
parser.add_argument('--show_stress_edge', dest='show_stress_edge', default=False,
                    help="Show edges on von Mises stress.")
parser.add_argument('--stress_opacity', dest='stress_opacity', default=1.0,
                    help="Opacity for stress.")
# For thickness
parser.add_argument('--show_thickness', dest='show_thickness', default=False,
                    help="Show shell thickness on deformed geometry")
parser.add_argument('--show_thickness_edge', dest='show_thickness_edge', default=False,
                    help="Show edges on thickness")
parser.add_argument('--thickness_opacity', dest='thickness_opacity', default=1.0,
                    help="Opacity for thickness.")
# For surface indices
parser.add_argument('--start_ind', dest='start_ind', default=0,
                    help="Index for first spline.")
parser.add_argument('--end_ind', dest='end_ind', default=2,
                    help="Index for last spline.")
parser.add_argument('--ind_step', dest='ind_step', default=1,
                    help="Step for surface indices.")
parser.add_argument('--save_state', dest='save_state', default=True,
                    help="Save current view to a Paraview state file.")
parser.add_argument('--filename_prefix', dest='filename_prefix', default='state',
                    help="Finename prerix for saved state file.")
parser.add_argument('--interact', dest='interact', default=False,
                    help="Launch interactive window.")

args = parser.parse_args()
true_list = ['true', 'yes', 'y', '1']

### Arguments
file_path = args.file_path
# For geometry
show_geom = str(args.show_geom).lower() in true_list
show_geom_edge = str(args.show_geom_edge).lower() in true_list
geom_opacity = float(args.geom_opacity)
# For displacements
show_disp = str(args.show_disp).lower() in true_list
show_disp_edge = str(args.show_disp_edge).lower() in true_list
disp_opacity = float(args.disp_opacity)
disp_scale = int(args.disp_scale)
# For stress
show_stress = str(args.show_stress).lower() in true_list
show_stress_edge = str(args.show_stress_edge).lower() in true_list
stress_opacity = float(args.stress_opacity)
# For thickness
show_thickness = str(args.show_thickness).lower() in true_list
show_thickness_edge = str(args.show_thickness_edge).lower() in true_list
thickness_opacity = float(args.thickness_opacity)
# For surface indices
start_ind = int(args.start_ind)
end_ind = int(args.end_ind)
ind_step = int(args.ind_step)
# # To use custom indices
# custom_inds = [0,1,2,3,4]

save_state = str(args.save_state).lower() in true_list
filename = args.filename_prefix
interact = str(args.interact).lower() in true_list

if save_state:
    filename += "_patch_"+str(start_ind)+"_"+str(end_ind)
    if show_geom:
        filename += "_geom"
    if show_disp:
        filename += "_disp"
    if show_stress:
        filename += "_stress"
    if show_thickness:
        filename += "_thickness"
    filename += "_dispscale"+str(disp_scale)
    filename += ".pvsm"

renderView1 = GetActiveViewOrCreate('RenderView')

geom_fname_pre = "F"
disp_fname_pre = "u"
stress_fname_pre = "von_Mises_top_"
thickness_fname_pre = "t"

for surf_ind in range(start_ind, end_ind, ind_step):
# for surf_ind in custom_inds:
    print("--- Displaying surface", surf_ind)

    # create a new 'PVD Reader'
    f0_0_filepvd = PVDReader(registrationName=geom_fname_pre+str(surf_ind)+'_0_file.pvd', 
                             FileName=file_path+geom_fname_pre+str(surf_ind)+'_0_file.pvd')
    f0_0_filepvd.PointArrays = [geom_fname_pre+str(surf_ind)+'_0']

    # create a new 'PVD Reader'
    f0_1_filepvd = PVDReader(registrationName=geom_fname_pre+str(surf_ind)+'_1_file.pvd', 
                             FileName=file_path+geom_fname_pre+str(surf_ind)+'_1_file.pvd')
    f0_1_filepvd.PointArrays = [geom_fname_pre+str(surf_ind)+'_1']

    # create a new 'PVD Reader'
    f0_2_filepvd = PVDReader(registrationName=geom_fname_pre+str(surf_ind)+'_2_file.pvd', 
                             FileName=file_path+geom_fname_pre+str(surf_ind)+'_2_file.pvd')
    f0_2_filepvd.PointArrays = [geom_fname_pre+str(surf_ind)+'_2']

    # create a new 'PVD Reader'
    f0_3_filepvd = PVDReader(registrationName=geom_fname_pre+str(surf_ind)+'_3_file.pvd', 
                             FileName=file_path+geom_fname_pre+str(surf_ind)+'_3_file.pvd')
    f0_3_filepvd.PointArrays = [geom_fname_pre+str(surf_ind)+'_3']

    # create a new 'PVD Reader'
    u0_0_filepvd = PVDReader(registrationName=disp_fname_pre+str(surf_ind)+'_0_file.pvd', 
                             FileName=file_path+disp_fname_pre+str(surf_ind)+'_0_file.pvd')
    u0_0_filepvd.PointArrays = [disp_fname_pre+str(surf_ind)+'_0']

    # create a new 'PVD Reader'
    u0_1_filepvd = PVDReader(registrationName=disp_fname_pre+str(surf_ind)+'_1_file.pvd', 
                             FileName=file_path+disp_fname_pre+str(surf_ind)+'_1_file.pvd')
    u0_1_filepvd.PointArrays = [disp_fname_pre+str(surf_ind)+'_1']

    # create a new 'PVD Reader'
    u0_2_filepvd = PVDReader(registrationName=disp_fname_pre+str(surf_ind)+'_2_file.pvd', 
                             FileName=file_path+disp_fname_pre+str(surf_ind)+'_2_file.pvd')
    u0_2_filepvd.PointArrays = [disp_fname_pre+str(surf_ind)+'_2']

    if show_stress:
        # create a new 'PVD Reader'
        von_Mises_0pvd = PVDReader(registrationName=stress_fname_pre+str(surf_ind)+'.pvd', 
                                       FileName=file_path+stress_fname_pre+str(surf_ind)+'.pvd')
        von_Mises_0pvd.PointArrays = [stress_fname_pre+str(surf_ind)]

    if show_thickness:
        t0_filepvd = PVDReader(registrationName=thickness_fname_pre+str(surf_ind)+'_file.pvd',
                               FileName=file_path+thickness_fname_pre+str(surf_ind)+'_file.pvd')
        t0_filepvd.PointArrays = [thickness_fname_pre+str(surf_ind)]

    # set active source
    SetActiveSource(f0_0_filepvd)

    # # set active source
    # SetActiveSource(von_Mises_0pvd)

    appendAttributes_input = [f0_0_filepvd, f0_1_filepvd, f0_2_filepvd, f0_3_filepvd, 
                              u0_0_filepvd, u0_1_filepvd, u0_2_filepvd]

    if show_stress:
        appendAttributes_input += [von_Mises_0pvd]
    if show_thickness:
        appendAttributes_input += [t0_filepvd]

    appendAttributes1 = AppendAttributes(registrationName='AppendAttributes'+str(surf_ind), 
                                         Input = appendAttributes_input)

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='CalculatorGeom'+str(surf_ind), 
                             Input=appendAttributes1)

    # Properties modified on calculator1
    calculator1.Function = '('+geom_fname_pre+str(surf_ind)+'_0/'\
                         +geom_fname_pre+str(surf_ind)+'_3-coordsX)*iHat '\
                        +'+ ('+geom_fname_pre+str(surf_ind)+'_1/'\
                         +geom_fname_pre+str(surf_ind)+'_3-coordsY)*jHat '\
                        +'+ ('+geom_fname_pre+str(surf_ind)+'_2/'\
                         +geom_fname_pre+str(surf_ind)+'_3-coordsZ)*kHat'

    # create a new 'Warp By Vector'
    warpByVector1 = WarpByVector(registrationName='WarpByVectorGeom'+str(surf_ind), 
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
    calculator2 = Calculator(registrationName='CalculatorDisp'+str(surf_ind), 
                             Input=warpByVector1)

    # Properties modified on calculator2
    calculator2.Function = '('+disp_fname_pre+str(surf_ind)+'_0/'\
                         +geom_fname_pre+str(surf_ind)+'_3)*iHat '\
                        +'+ ('+disp_fname_pre+str(surf_ind)+'_1/'\
                         +geom_fname_pre+str(surf_ind)+'_3)*jHat '\
                        +'+ ('+disp_fname_pre+str(surf_ind)+'_2/'\
                         +geom_fname_pre+str(surf_ind)+'_3)*kHat'

    # create a new 'Warp By Vector'
    warpByVector2 = WarpByVector(registrationName='WarpByVectorDisp'+str(surf_ind), 
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
        calculator3 = Calculator(registrationName='CalculatorvM'+str(surf_ind), 
                                 Input=warpByVector2)

        # Properties modified on calculator3
        calculator3.Function = stress_fname_pre+str(surf_ind)

        # UpdatePipeline(time=0.0, proxy=calculator3)
        calculator3Display = Show(calculator3, renderView1, 
                                  'UnstructuredGridRepresentation')
        calculator3Display.Opacity = stress_opacity
        if show_stress_edge:
            calculator3Display.SetRepresentationType('Surface With Edges')

    if show_thickness:
        # create a new 'Calculator'
        calculator4 = Calculator(registrationName='CalculatorTh'+str(surf_ind), 
                                 Input=warpByVector2)

        # Properties modified on calculator4
        calculator4.Function = thickness_fname_pre+str(surf_ind)

        # UpdatePipeline(time=0.0, proxy=calculator4)
        calculator4Display = Show(calculator4, renderView1, 
                                  'UnstructuredGridRepresentation')
        calculator4Display.Opacity = thickness_opacity
        if show_thickness_edge:
            calculator4Display.SetRepresentationType('Surface With Edges')


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
exit()