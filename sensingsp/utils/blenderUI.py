import sensingsp as ssp
import matplotlib.pyplot as plt
import numpy as np
import bpy
import os
import tempfile

def define_settings():
    if not("Simulation Settings" in bpy.data.objects):
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(.01, .01, .01))
        sim_axes = bpy.context.object
        sim_axes.name = f'Simulation Settings'
        sim_axes["RF Frequency (GHz)"] = 70
        sim_axes["Velocity Calc. Jump"] = 1
        # sim_axes["Velocity Calc. Jump"] = bpy.props.IntProperty(name="Velocity Calc. Jump", default=1, min=1)
        sim_axes["Bounce Number"] = 2
        sim_axes["Render Blender Frames"] = True#False
        sim_axes["Open Output Folder"] = True
        sim_axes["do RayTracing LOS"] = True
        sim_axes["do RayTracing Simple"] = False
        sim_axes["CUDA SignalGeneration Enabled"] = False
        sim_axes["Debug_BypassCPITiming"] = False
        
        
        # bpy.data.objects["Simulation Settings"]["do RayTracing LOS"]=False
       
        
        # bpy.types.Object.velocity_calc_jump = bpy.props.IntProperty(name="Velocity Calc. Jump", default=1, min=1)
        # bpy.types.Object.bounce_number = bpy.props.IntProperty(name="Bounce Number", default=2, min=2)
        # bpy.types.Object.render_blender_frames = bpy.props.BoolProperty(name="Render Blender Frames", default=True)
        # bpy.types.Object.open_output_folder = bpy.props.BoolProperty(name="Open Output Folder", default=True)

        
        temp_dir = tempfile.gettempdir()
        radarsim_dir = os.path.join(temp_dir, 'SensingSP')
        os.makedirs(radarsim_dir, exist_ok=True)
        outputFiles = os.path.join(radarsim_dir, 'RadarOutputsFolder')
        os.makedirs(outputFiles, exist_ok=True)
        sim_axes["Radar Outputs Folder"]=outputFiles
        sim_axes["Video Directory"]=radarsim_dir
        sim_axes["Add Ris"] = f"0,8,8,{os.path.join(radarsim_dir, 'Ris.mat')},2"
        sim_axes["Add Probe"] = f"1,.1,20,20,db1,range_effect1,add_color1,add_open3d1,colormap1,0.05"


def blender_buttons_run_2():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.define_axes(2)
    ssp.config.setDopplerProcessingMethod_FFT_Winv(1)
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        ssp.integratedSensorSuite.SensorsSignalProccessing_Angle_frame(Signals)
        ssp.utils.increaseCurrentFrame()