import sensingsp as ssp
import numpy as np
from mathutils import Vector
# import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import cv2
import os
import bpy
from scipy.signal import lfilter
from scipy.io import savemat
def make_simple_scenario():
    predefine_movingcube_6843()
def run_simple_chain():
    processing_1()
def add_scenario(st):
    if st=='Target RCS Simulation' or st=='Target RCS Simulation Plane':
        ssp.utils.initialize_environment()
        rangeTarget = 20
        radar = ssp.radar.utils.addRadar(
            radarSensor=ssp.radar.utils.RadarSensorsCategory.SISO_mmWave76GHz,
            location_xyz=[-rangeTarget, 0, 0])
        if st=='Target RCS Simulation':
            ssp.radar.utils.addTarget(
                refRadar=radar,
                range=rangeTarget-0.5)
        else:
            ssp.radar.utils.addTarget(
                refRadar=radar,
                range=rangeTarget,shape="plane")
        NF = 3600
        radar.parent.rotation_euler  = (0,0,0)
        radar.parent.keyframe_insert(data_path="rotation_euler", frame=1)
        radar.parent.rotation_euler  = (0,0,2*np.pi)
        radar.parent.keyframe_insert(data_path="rotation_euler", frame=NF)
        for fcurve in radar.parent.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

        ssp.utils.initialize_simulation(startframe=1,endframe=NF,RunonGPU=False,rayTracing=3)

    if st=='2 Cubes + 6843':
        predefine_movingcube_6843()
    if st == 'Pattern SISO':
        predefine_Pattern_SISO()
        
    if st=='Hand Gesture + 1642':
        predefine_Hand_Gesture_1642()
        
    if st == "Hand Gesture + 3 Xethru Nature paper":
        predefine_Hand_Gesture_3Xethru_Nature_paper()
        
    if st == 'Ray Tracing 1':
        ssp.utils.delete_all_objects()
        ssp.utils.define_settings()
        cube = ssp.environment.add_cube(location=Vector((1.5, -.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((2.5, 1.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((4.5, 0.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, 1.5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((2.5, -1.5, 0)), direction=Vector((0, 1, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.name.startswith('Probe_')==False:
                    if "RCS0" not in obj:
                        obj["RCS0"] = 1.0
                    RCS0 = obj["RCS0"]
                    if "Backscatter N" not in obj:
                        obj["Backscatter N"] = 1
                    Backscatter_N = obj["Backscatter N"]
                    if "Backscatter Dev (deg)" not in obj:
                        obj["Backscatter Dev (deg)"] = 0.0
        
        
        ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
        
        obj = bpy.data.objects.get("RX_0_0_1_0_00001")
        if obj:
            obj.location = (1, 0, -3)  # Replace x, y, z with the desired coordinates
            obj.scale = (1, 1, 1)
    if st == 'Ray Tracing 2':
        ssp.utils.delete_all_objects()
        ssp.utils.define_settings()
        cube = ssp.environment.add_cube(location=Vector((1.5, -.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((2.5, 1.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((4.5, 0.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, 1.5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((2.5, -1.5, 0)), direction=Vector((0, 1, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.name.startswith('Probe_')==False:
                    if "RCS0" not in obj:
                        obj["RCS0"] = 1.0
                    RCS0 = obj["RCS0"]
                    if "Backscatter N" not in obj:
                        obj["Backscatter N"] = 1
                    Backscatter_N = obj["Backscatter N"]
                    if "Backscatter Dev (deg)" not in obj:
                        obj["Backscatter Dev (deg)"] = 0.0
        
        
        ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    if st == 'Ray Tracing 3':
        ssp.utils.delete_all_objects()
        ssp.utils.define_settings()
        cube = ssp.environment.add_cube(location=Vector((1.5, -.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((2.5, 1.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((4.5, 0.5, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, 1.5, .05)), subdivision=0)
        cube = ssp.environment.add_cube(location=Vector((2.5, -1.5, 0)), direction=Vector((0, 1, 0)), scale=Vector((.5, .5, .05)), subdivision=0)
        
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.name.startswith('Probe_')==False:
                    if "RCS0" not in obj:
                        obj["RCS0"] = 1.0
                    RCS0 = obj["RCS0"]
                    if "Backscatter N" not in obj:
                        obj["Backscatter N"] = 1
                    Backscatter_N = obj["Backscatter N"]
                    if "Backscatter Dev (deg)" not in obj:
                        obj["Backscatter Dev (deg)"] = 0.0
        
        
        ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        radar = ssp.radar.utils.predefined_array_configs_TI_AWR1642(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    if st == '2 Slot Example' or st == '2 Slot as RIS':
        TX_theta0 = -20.
        RX_theta0 = 15.
        RX_theta1 = 85.
        RX_thetaN = 100
        
        if "Extra Settings" in bpy.data.objects:
            if "2 Slots, TX" in bpy.data.objects["Extra Settings"]:
                TX_theta0 = float(bpy.data.objects["Extra Settings"]["2 Slots, TX"])
            if "2 Slots, RX" in bpy.data.objects["Extra Settings"]:
                if len(bpy.data.objects["Extra Settings"]["2 Slots, RX"].split(","))==3:
                    RX_theta0,RX_theta1,RX_thetaN = [float(x) for x in bpy.data.objects["Extra Settings"]["2 Slots, RX"].split(",")]
                    RX_thetaN = int(RX_thetaN)
        ssp.utils.delete_all_objects()
        if "Extra Settings" not in bpy.data.objects:
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(.01, .01, .01))
            sim_axes = bpy.context.object
            sim_axes.name = f'Extra Settings'
            sim_axes["2 Slots, TX"] = f'{TX_theta0}'
            sim_axes["2 Slots, RX"] = f'{RX_theta0},{RX_theta1},{RX_thetaN}' 
        
        ssp.utils.define_settings()
        
        HL = ssp.LightSpeed/62e9/2
        L = 13e-2 
        Sec = HL/1
        N2 = int(L/Sec)
        Lx = 7.5e-3
        N2x = int(Lx/Sec)
        D0 = 25e-3
        option = 2
        if st == '2 Slot Example':
            ssp.utils.set_RayTracing_advanced_intense()
            if option==1:
                cube = ssp.environment.add_cube(location=Vector((0, -D0/2, 0)), direction=Vector((1, 0, 0)), scale=Vector((L/2, 5e-3/2, 1e-3/2)), subdivision=0)
                cube = ssp.environment.add_cube(location=Vector((0, D0/2, 0)), direction=Vector((1, 0, 0)), scale=Vector((L/2, 5e-3/2, 1e-3/2)), subdivision=0)
            elif option==0:
                for i in range(N2):
                    ssp.environment.add_cube(location=Vector((0, -D0/2, i*HL-L/2)), direction=Vector((1, 0, 0)), scale=Vector((HL/2, 5e-3/2, 1e-3/2)), subdivision=0)
                    ssp.environment.add_cube(location=Vector((0, D0/2, i*HL-L/2)), direction=Vector((1, 0, 0)), scale=Vector((HL/2, 5e-3/2, 1e-3/2)), subdivision=0)
            elif option==2:
                ssp.utils.set_RayTracing_light()
                for i in range(N2):
                    for j in range(N2x):
                        bpy.ops.mesh.primitive_plane_add(size=Sec, enter_editmode=False, align='WORLD', location=(0, (j-N2x/2.0)*Sec-D0/2, i*Sec-L/2), scale=(1, 1, 1))
                        bpy.context.object.rotation_euler[2] = 1.5708
                        bpy.context.object.rotation_euler[0] = 1.5708
                        bpy.ops.mesh.primitive_plane_add(size=Sec, enter_editmode=False, align='WORLD', location=(0, (j-N2x/2.0)*Sec+D0/2, i*Sec-L/2), scale=(1, 1, 1))
                        bpy.context.object.rotation_euler[2] = 1.5708
                        bpy.context.object.rotation_euler[0] = 1.5708

            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    if obj.name.startswith('Probe_')==False:
                        if "RCS0" not in obj:
                            obj["RCS0"] = .1
                        RCS0 = obj["RCS0"]
                        if "Backscatter N" not in obj:
                            obj["Backscatter N"] = 1
                        Backscatter_N = obj["Backscatter N"]
                        if "Backscatter Dev (deg)" not in obj:
                            obj["Backscatter Dev (deg)"] = 0.0
                        if "SpecularDiffusionFactor" not in obj:
                            obj["SpecularDiffusionFactor"] = 2.
                            
            
        
        ssp.integratedSensorSuite.define_suite(0, location=Vector((-1, 0, 0)), rotation=Vector((0, 0, 0)))
        radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=62e9)
        
        if st == '2 Slot as RIS':
            ssp.utils.set_RayTracing_advanced_intense()
            N1,N2=2,158
            N2 = int(L/HL)
            
            ssp.ris.utils.add_ris(isuite=0, iris=0, location=Vector((1, 0,0)), rotation=Vector((np.pi/2,0, np.pi/2)), f0=62e9,N1=N1,N2=N2)
                
            for i1 in range(N1):
                for i2 in range(N2):
                    NF = 80
                    name = f'RIS_Element_{0}_{0}_{i1}_{i2}_{NF}'
                    ris = bpy.data.objects.get(name)
                    ris['amplitude']=np.sqrt(1/HL)  # 1/HL for test
                    ris['phase']=0
                    if i1 == 0:
                        ris.location = (-D0/2, (i2-N2/2.0)*HL, 0)
                    if i1 == 1:
                        ris.location = ( D0/2, (i2-N2/2.0)*HL, 0)
        

        radar['Transmit_Power_dBm']=30
        radar["Transmit_Antenna_Element_Pattern"] = "NotOmni"
        radar["Transmit_Antenna_Element_Gain_db"] = 32
        radar["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = 2.5
        radar["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = 2.6
        radar["Receive_Antenna_Element_Gain_db"] = 32
        radar["Receive_Antenna_Element_Pattern"] = "NotOmni"
        radar["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = 2.5
        radar["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = 2.6
            
        name = f'TX_{0}_{0}_{1}_{0}_{0+1:05}'
        tx = bpy.data.objects.get(name)
        theta = TX_theta0
        tx.rotation_euler = (0, np.deg2rad(theta), 0)
        x = np.cos(np.deg2rad(theta))
        y = np.sin(np.deg2rad(theta))
        tx.location = (y, 0, x-1)
        
        name = f'RX_{0}_{0}_{1}_{0}_{0+1:05}'
        rx = bpy.data.objects.get(name)
        
        theta_v = np.linspace(RX_theta0,RX_theta1,RX_thetaN)
        for iframe,theta in enumerate(theta_v):
            rx.rotation_euler = (0, np.deg2rad(theta), 0)
            x = np.cos(np.deg2rad(theta))
            y = np.sin(np.deg2rad(theta))
            rx.location = (y, 0, x-1)
            rx.keyframe_insert(data_path="location", frame=iframe+1)    
            rx.keyframe_insert(data_path="rotation_euler", frame=iframe + 1)
        
        tx.scale = (.01, .01, .01)
        rx.scale = (.1, .1, .1)
        ssp.utils.set_frame_start_end(start=1,end=theta_v.shape[0])     
        if option ==3:
            file="save_frompython.blend"
            folder = ssp.config.temp_folder
            ssp.environment.add_blenderfileobjects(os.path.join(folder, file),RCS0=1)
            
        ssp.utils.save_Blender()              

def sim_scenario(st):
    if st=='2 Cubes + 6843':
        processing_1()
    if st=='Hand Gesture + 1642':
        processing_HandGesture1642()
    if st == 'Pattern SISO':
        ssp.utils.trimUserInputs() 
        ssp.config.restart()
        ssp.config.setDopplerProcessingMethod_FFT_Winv(0)
        # # plt.ion() 
        # fig, FigsAxes = plt.subplots(2,3)
        # FigsAxes[1, 2] = fig.add_subplot(2, 3, 6, projection='3d')
        
        pat=[]
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            alld = []
            for itx in range(len(path_d_drate_amp[0][0][0][0][0])):
                for irx in range(len(path_d_drate_amp[0][0])):
                    for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                        alld.append(d_drate_amp[0:3])
            alld = np.array(alld)
            if len(alld)>0:
                pat.append([ssp.config.CurrentFrame,alld[0][2]])
            else:
                pat.append(0)
            ssp.utils.increaseCurrentFrame()
        f=np.array([x[0] for x in pat])
        p=np.array([x[1] for x in pat])
        theta = np.arange(360)
        p/=p.max()
        plt.figure()
        plt.plot(theta,p[:360])
        plt.plot(theta[:len(p[360:])],p[360:])
        plt.figure()
        plt.plot(theta,20*np.log10(p[:360]))
        plt.plot(theta[:len(p[360:])],20*np.log10(p[360:]))
        plt.show()
    
    if st == 'Ray Tracing 1' or st == 'Ray Tracing 2' or st == 'Ray Tracing 3' :
        ssp.environment.scenarios.raytracing_test()    
    
    if st == "Hand Gesture + 3 Xethru Nature paper":
        process_predefine_Hand_Gesture_3Xethru_Nature_paper()
    if st == '2 Slot Example' or st == '2 Slot as RIS':
        # TX_theta0 = -15.
        # RX_theta0 = 0.
        # RX_theta1 = 90.
        # RX_thetaN = 900
        
        if "Extra Settings" in bpy.data.objects:
            if "2 Slots, TX" in bpy.data.objects["Extra Settings"]:
                TX_theta0 = float(bpy.data.objects["Extra Settings"]["2 Slots, TX"])
            if "2 Slots, RX" in bpy.data.objects["Extra Settings"]:
                if len(bpy.data.objects["Extra Settings"]["2 Slots, RX"].split(","))==3:
                    RX_theta0,RX_theta1,RX_thetaN = [float(x) for x in bpy.data.objects["Extra Settings"]["2 Slots, RX"].split(",")]
                    RX_thetaN = int(RX_thetaN)
        
        theta_v = np.linspace(RX_theta0,RX_theta1,RX_thetaN)

        ssp.utils.trimUserInputs() 
        ssp.config.restart()
        wavelength = ssp.utils.research.simplest.wavelength(ssp.RadarSpecifications[0][0])
        o = []
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            itx,irx=0,0
            x = 0
            for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                d=d_drate_amp[0]
                amp=d_drate_amp[2]
                x+= amp*np.exp(-1j*2*np.pi*d/wavelength)
            o.append(np.abs(x))
            print(ssp.config.CurrentFrame)
            ssp.utils.increaseCurrentFrame()
        # return
        fig, axs = plt.subplots(1,2)
        axs[0].plot(-TX_theta0+theta_v[:len(o)],20*np.log10(o))
        data={"o":o,'theta_v':theta_v,'TX_theta0':TX_theta0}
        ssp.utils.savemat_in_tmpfolder('slots.mat',data)
        ssp.utils.open_temp_folder()
        # plt.plot(-TX_theta0+theta_v[:len(o)],(o))
        om = np.max(20*np.log10(o))
        axs[0].set_ylim([om-35,om+5])   
        axs[0].grid()
        axs[0].set_xlabel('Bistatic Angle (deg)')
        axs[0].set_ylabel('Received Signal Power (dB)')
        axs[1].plot(-TX_theta0+theta_v[:len(o)],20*np.log10(o)-om)
        # plt.plot(-TX_theta0+theta_v[:len(o)],(o))
        axs[1].set_ylim([-15,0])   
        axs[1].grid()
        axs[1].set_xlabel('Bistatic Angle (deg)')
        axs[1].set_ylabel('Normalized Received Signal Power (dB)')
        plt.show()
    
    if st=='Target RCS Simulation' or st=='Target RCS Simulation Plane':
        ssp.utils.trimUserInputs() 
        ssp.config.restart()
        wavelength = ssp.utils.research.simplest.wavelength(ssp.RadarSpecifications[0][0])
        o = []
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            itx,irx=0,0
            x = 0
            for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                d=d_drate_amp[0]
                amp=d_drate_amp[2]
                x+= amp*np.exp(-1j*2*np.pi*d/wavelength)
            o.append(np.abs(x))
            print(ssp.config.CurrentFrame)
            ssp.utils.increaseCurrentFrame()
        # return
        o=np.array(o)
        plt.figure()
        # plt.plot(20*np.log10(o/np.max(o)))
        plt.plot(360.0*np.arange(o.shape[0])/o.shape[0],20*np.log10(o))
        plt.grid()
        plt.xlabel('Angle (deg)')
        plt.ylabel('Received Signal Power (dB)')
        plt.show()
    
        
def predefine_movingcube_6843(interpolation_type='LINEAR'):
    #     Blender provides several interpolation types, and as of recent versions, there are 6 main interpolation options:

    # CONSTANT ('CONSTANT')
    # The value remains constant between keyframes, resulting in a stepped change at each keyframe.
    # LINEAR ('LINEAR')
    # The value changes linearly between keyframes, resulting in a straight transition between the keyframe points.
    # BEZIER ('BEZIER')
    # The value follows a bezier curve between keyframes, allowing for smooth transitions with adjustable handles for fine control over the curve.
    # SINE ('SINE')
    # The value follows a sine curve between keyframes, creating smooth, wave-like transitions.
    # QUAD ('QUAD')
    # The value follows a quadratic curve, which can create an accelerating or decelerating effect.
    # CUBIC ('CUBIC')
    # The value follows a cubic curve between keyframes, providing a slightly more complex smooth transition than quadratic.
    ssp.utils.delete_all_objects()
    ssp.utils.define_settings()
    cubex = ssp.environment.add_cube(location=Vector((2.7, np.sqrt(3**2-2.7**2), 0)), direction=Vector((1,0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
    cubex["RCS0"]=1
    cube = ssp.environment.add_cube(location=Vector((0, 0, 0)), direction=Vector((1, 0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
    cube["RCS0"]=1
    cube.location = (3,0,0)
    cube.keyframe_insert(data_path="location", frame=1)
    cube.location = (3, 3,0)
    cube.keyframe_insert(data_path="location", frame=30)
    cube.location = (3, -3,0)
    cube.keyframe_insert(data_path="location", frame=100)
    for fcurve in cube.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = interpolation_type
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    
    
    radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    # radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843_az(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    # radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9,LinearArray_TXPos=[0],LinearArray_RXPos=[i*3e8/70e9/2 for i in range(30)])
    
    
    
    # radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=0
    
    ssp.utils.set_frame_start_end(start=1,end=2)
    ssp.utils.useCUDA()
    # print(f"rangeResolution maxUnambigiousRange = {ssp.radar.utils.rangeResolution_and_maxUnambigiousRange(radar)}")

def predefine_TruckHuman_SameRG():
    ssp.utils.delete_all_objects()
    if 0:
        ssp.utils.open_Blend(os.path.join(ssp.config.temp_folder, "TruckHuman.blend"))
    # ssp.utils.open_Blend(os.path.join(ssp.config.temp_folder, "SimpleCube_y05.blend"))
    else:
        cube = ssp.environment.add_cube(location=Vector((20, 0, 0)), direction=Vector((1, 0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
        cube["RCS0"]=1e5
        t=np.deg2rad(2)
        cube = ssp.environment.add_cube(location=Vector((20*np.cos(t), 20*np.sin(t), 0)), direction=Vector((1, 0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
        cube["RCS0"]=1e5
        # cube.location = (20,0,0)
        # cube.keyframe_insert(data_path="location", frame=1)
        # cube.location = (20, 3,0)
        # cube.keyframe_insert(data_path="location", frame=50)
        # cube.location = (20, 0,0)
        # cube.keyframe_insert(data_path="location", frame=100)
        
    
    ssp.utils.define_settings()
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    radar = ssp.radar.utils.predefined_array_configs_DARWu(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    # radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    # radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), 
    #                                                              f0=70e9
    #                                                              ,LinearArray_TXPos =[(i)*4*3e8/70e9/2 for i in range(1)],
    #                                                             LinearArray_RXPos =[(i+10)*3e8/70e9/2 for i in range(4)])
    # radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    
    Res,MaxR =    ssp.radar.utils.rangeResolution_and_maxUnambigiousRange(radar)
    radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=2
    radar['DopplerProcessingMIMODemod']='Simple'
    radar['ADC_levels']=256*8
    radar['RangeFFT_OverNextP2']=1
    radar['DopplerFFT_OverNextP2']=0
    radar['AzFFT_OverNextP2']=5
    radar['NPulse']=6*64
    
# specifications['RangeFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['RangeFFT_OverNextP2']
# specifications['DopplerFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['']
# specifications['AzFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['AzFFT_OverNextP2']
# specifications['ElFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['ElFFT_OverNextP2']

    ssp.utils.set_frame_start_end(start=1,end=40)
    ssp.utils.useCUDA()
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.DopplerProcessingMethod_FFT_Winv = (True)

    ssp.config.directReceivefromTX =  False 
    ssp.config.RadarRX_only_fromscatters_itsTX = True
    ssp.config.RadarRX_only_fromitsTX = True
    ssp.config.Radar_TX_RX_isolation = True
    ssp.utils.save_Blender()
def predefine_TruckHuman_DiffRG():
    ssp.utils.delete_all_objects()
    if 1:
        ssp.utils.open_Blend(os.path.join(ssp.config.temp_folder, "TruckHumanModified.blend"))
    # ssp.utils.open_Blend(os.path.join(ssp.config.temp_folder, "SimpleCube_y05.blend"))
    else:
        cube = ssp.environment.add_cube(location=Vector((20, 20, 0)), direction=Vector((1, 0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
        cube["RCS0"]=1e6
        # cube.location = (20,0,0)
        # cube.keyframe_insert(data_path="location", frame=1)
        # cube.location = (20, 3,0)
        # cube.keyframe_insert(data_path="location", frame=50)
        # cube.location = (20, 0,0)
        # cube.keyframe_insert(data_path="location", frame=100)
        
    
    ssp.utils.define_settings()
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    radar = ssp.radar.utils.predefined_array_configs_DARWu(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    # radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    # radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), 
    #                                                              f0=70e9
    #                                                              ,LinearArray_TXPos =[(i)*4*3e8/70e9/2 for i in range(1)],
    #                                                             LinearArray_RXPos =[(i+10)*3e8/70e9/2 for i in range(4)])
    # radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    
    Res,MaxR =    ssp.radar.utils.rangeResolution_and_maxUnambigiousRange(radar)
    radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=20
    radar['DopplerProcessingMIMODemod']='Simple'
    radar['ADC_levels']=256*8
    radar['RangeFFT_OverNextP2']=1
    radar['DopplerFFT_OverNextP2']=0
    radar['AzFFT_OverNextP2']=5
    # radar['NPulse']=12
    
# specifications['RangeFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['RangeFFT_OverNextP2']
# specifications['DopplerFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['']
# specifications['AzFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['AzFFT_OverNextP2']
# specifications['ElFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['ElFFT_OverNextP2']

    ssp.utils.set_frame_start_end(start=1,end=40)
    ssp.utils.useCUDA()
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.DopplerProcessingMethod_FFT_Winv = (True)

    ssp.config.directReceivefromTX =  False 
    ssp.config.RadarRX_only_fromscatters_itsTX = True
    ssp.config.RadarRX_only_fromitsTX = True
    ssp.config.Radar_TX_RX_isolation = True
    ssp.utils.save_Blender()


def processing_1():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.setDopplerProcessingMethod_FFT_Winv(0)
    # plt.ion() 
    fig, FigsAxes = plt.subplots(2,3)
    FigsAxes[1, 2] = fig.add_subplot(2, 3, 6, projection='3d')
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        alld = []
        m = path_d_drate_amp[0][0][0][0][0][0][0][3]
        for itx in range(len(path_d_drate_amp[0][0][0][0][0])):
            for irx in range(len(path_d_drate_amp[0][0])):
                for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                    # if d_drate_amp[3]==m:
                    alld.append(d_drate_amp[0:3])
        alld = np.array(alld)
        fig.suptitle(f'Frame: {ssp.config.CurrentFrame}')
        FigsAxes[0,2].cla()
        # FigsAxes[1,2].cla()
        # FigsAxes[0,2].plot(alld[:,0],'.')
        # FigsAxes[1,2].plot(alld[:,1],'.')
        # FigsAxes[0,2].set_xlabel('scatter index')
        # FigsAxes[0,2].set_ylabel('d (m)')
        # FigsAxes[1,2].set_xlabel('scatter index')
        # FigsAxes[1,2].set_ylabel('dr (m/s)')
    
        # ssp.utils.force_zeroDoppler_4Simulation(path_d_drate_amp)
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        ssp.integratedSensorSuite.SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler(Signals,FigsAxes,fig)
        ssp.utils.increaseCurrentFrame()
    # plt.ioff() 
    plt.show()
    
def processing_4():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    fig, FigsAxes = plt.subplots(2,2)
    # FigsAxes[1] = fig.add_subplot(1, 2, 2, projection='3d')
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        fig.suptitle(f'Frame: {ssp.config.CurrentFrame}')
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        ssp.integratedSensorSuite.SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler_TruckHuman_Detection(Signals,FigsAxes,fig)
        ssp.utils.increaseCurrentFrame()
        print(f'Frame: {ssp.config.CurrentFrame}')
    # plt.ioff() 
    plt.show()

def processing_3():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    fig, FigsAxes = plt.subplots(1,2)
    # FigsAxes[1] = fig.add_subplot(1, 2, 2, projection='3d')
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        # fig2, FigsAxes2 = plt.subplots(len(ssp.lastSuite_Position[0]['Radar'][0]['TX-Position']),
        #                                len(ssp.lastSuite_Position[0]['Radar'][0]['RX-Position']))
        # for itx in range(len(ssp.lastSuite_Position[0]['Radar'][0]['TX-Position'])):
        #     for irx in range(len(ssp.lastSuite_Position[0]['Radar'][0]['RX-Position'])):
        #         print("Test",ssp.config.CurrentFrame,itx,irx,(path_d_drate_amp[0][0][irx][0][0][itx]))
        #         if itx+irx==0:
        #             FigsAxes2[itx,irx].plot([x[0] for x in path_d_drate_amp[0][0][irx][0][0][itx]],[x[2] for x in path_d_drate_amp[0][0][irx][0][0][itx]],'.')
        #         else:
        #             FigsAxes2[itx,irx].plot([x[0] for x in path_d_drate_amp[0][0][irx][0][0][itx]],[x[2] for x in path_d_drate_amp[0][0][irx][0][0][itx]],'.')
        # plt.show()
        fig.suptitle(f'Frame: {ssp.config.CurrentFrame}')
        # FigsAxes[0,2].cla()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        ssp.integratedSensorSuite.SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler_TruckHuman(Signals,FigsAxes,fig)
        ssp.utils.increaseCurrentFrame()
        print(f'Frame: {ssp.config.CurrentFrame}')
    # plt.ioff() 
    plt.show()
    
def processing_2():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.setDopplerProcessingMethod_FFT_Winv(1)
    fig, FigsAxes = plt.subplots(2,3)
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        # ssp.utils.force_zeroDoppler_4Simulation(path_d_drate_amp)
        # Channel_d_fd_amp = ssp.visualization.visualize_radar_path_d_drate_amp(path_d_drate_amp,1)
        radar_path_d_drate_amp(path_d_drate_amp,[fig,FigsAxes])
        ssp.utils.increaseCurrentFrame()
    # plt.show()
    

def radar_path_d_drate_amp(path_d_drate_amp,FigsAxes):
    alld = []
    m = path_d_drate_amp[0][0][0][0][0][0][0][3]
    for itx in range(len(path_d_drate_amp[0][0][0][0][0])):
        for irx in range(len(path_d_drate_amp[0][0])):
            for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                if d_drate_amp[3]==m:
                    alld.append(d_drate_amp[0:3])
    alld = np.array(alld)
    FigsAxes[1][0,0].plot(alld[:,0])
    FigsAxes[1][0,1].plot(alld[:,1])
    FigsAxes[1][1,0].plot(alld[:,2])
    FigsAxes[1][1,1].plot(np.diff(alld[:,0]))
    FigsAxes[1][1,2].plot(np.diff(np.diff(alld[:,0])))
    image=ssp.visualization.captureFig(FigsAxes[0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('d drate ', image)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        return

    
    # plt.draw() 
    # plt.pause(0.1)
def rcxchain(st):
    if st=="Hand Gestures":
        1

def raytracing_test():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.RayTracing_ReflectionPointEpsilon=1e-4
    empty_obj = bpy.data.objects.get("Plot Rays")
    if empty_obj:
        # Collect all children recursively
        def collect_children_recursive(obj):
            children = list(obj.children)
            for child in obj.children:
                children.extend(collect_children_recursive(child))
            return children

        # Get all descendants of the empty
        descendants = collect_children_recursive(empty_obj)

        # Deselect all objects first
        bpy.ops.object.select_all(action='DESELECT')

        # Select the empty and all its descendants
        empty_obj.select_set(True)
        for child in descendants:
            child.select_set(True)

        # Delete the selected objects
        bpy.ops.object.delete()
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
    allrays = bpy.context.object
    allrays.name = f'Plot Rays'
    while ssp.config.run():    
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
        framerays = bpy.context.object
        framerays.name = f'Plot Rays Frame {ssp.config.getCurrentFrame()}'
        framerays.parent=allrays 
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        path_index = 1
        for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
            for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
                for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
                    for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
                        for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
                            for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
                                for ip,p in enumerate(TX_d_drate_amp):
                                    d,dr,a,m=p
                                    print(path_index,len(m),m,d)
                                    tx = ssp.lastSuite_Position[istx]['Radar'][irtx]['TX-Position'][itx] 
                                    rx = ssp.lastSuite_Position[isrx]['Radar'][irrx]['RX-Position'][irx]
                                    v = [tx]
                                    for mi in m:
                                        if isinstance(mi, list) and len(mi) == 2 and isinstance(mi[1], Vector):
                                            v.append(mi[1])
                                        if isinstance(mi, str):
                                            _, is0, ir0, iris0 = mi.split('_')
                                            is0 = int(is0)
                                            ir0 = int(ir0)
                                            iris0 = int(iris0)
                                            v.append(ssp.lastSuite_Position[is0]['RIS'][ir0]['Position'][iris0])
                                    v.append(rx)
                                    ray = ssp.visualization.plot_continuous_curve(v,framerays,curve_name=f'path {path_index} Frame {ssp.config.CurrentFrame}')
                                    ray["Ray Tracing 20log Amp (dB)"]=float(20*np.log10(a))
                                    ray["Ray Tracing distance"]=float(d)
                                    ray["Ray Tracing doppler * f0"]=float(dr)
                                    ray["Middle Number"]=len(m)
                                    path_index+=1
                                    
        ssp.utils.increaseCurrentFrame()
        break

def processing_HandGesture1642():
    ssp.utils.trimUserInputs()  
    ssp.config.restart()
    ssp.config.DopplerProcessingMethod_FFT_Winv = (True)

    ssp.config.directReceivefromTX =  False 
    ssp.config.RadarRX_only_fromscatters_itsTX = True
    ssp.config.RadarRX_only_fromitsTX = True
    ssp.config.Radar_TX_RX_isolation = True

    ssp.utils.useCUDA(True)

    # ssp.utils.set_RayTracing_light()
    ssp.utils.set_RayTracing_balanced()
    # ssp.utils.set_RayTracing_advanced_intense()
    fig, FigsAxes = plt.subplots(2,3)
    FigsAxes[1, 2] = fig.add_subplot(2, 3, 6, projection='3d')
    
    all_pcs = []
    # Real-time processing loop
    while ssp.config.run():
            
        # Initialize 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # Generate data for the current frame
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        pcs = []  # Collect point clouds for the current frame

        # Process each radar's signal
        for isuite, radarSpecifications in enumerate(ssp.RadarSpecifications):
            for iradar, specifications in enumerate(radarSpecifications):
                for XRadar, timeX in Signals[isuite]['radars'][iradar]:
                    # Perform range processing
                    X_fft_fast, d_fft = ssp.radar.utils.rangeprocessing(XRadar, specifications)

                    # Perform Doppler processing
                    rangeDopplerTXRX, f_Doppler = ssp.radar.utils.dopplerprocessing_mimodemodulation(X_fft_fast, specifications)

                    # Clear and update the first plot: ADC Samples vs. ADC Output Level
                    FigsAxes[0, 0].clear()
                    FigsAxes[0, 0].plot(np.real(XRadar[:, 0, 0]), label="Real Part")
                    FigsAxes[0, 0].plot(np.imag(XRadar[:, 0, 0]), label="Imaginary Part")
                    # FigsAxes[0, 0].set_title("ADC Samples vs. ADC Output Level")
                    FigsAxes[0, 0].set_xlabel("ADC Samples")
                    FigsAxes[0, 0].set_ylabel("ADC Output Level")
                    FigsAxes[0, 0].legend(loc='upper right')

                    # Clear and update the second plot: Range vs. ADC Output Level
                    FigsAxes[0, 1].clear()
                    FigsAxes[0, 1].plot(d_fft, np.abs(X_fft_fast[:, 0, 0]))
                    # FigsAxes[0, 1].set_title("Range vs. ADC Output Level")
                    FigsAxes[0, 1].set_xlabel("Range (m)")
                    FigsAxes[0, 1].set_ylabel("Range Profile")

                    # Clear and update the third plot: Range-Doppler Heatmap
                    FigsAxes[0, 2].clear()
                    im = FigsAxes[0, 2].imshow(np.abs(rangeDopplerTXRX[:, :, 0, 0]),
                                            extent=[f_Doppler[0], f_Doppler[-1], d_fft[0], d_fft[-1]],
                                            aspect='auto', origin='lower')
                    # FigsAxes[0, 2].set_title("Range-Doppler Map")
                    FigsAxes[0, 2].set_xlabel("Doppler Frequency (Hz)")
                    FigsAxes[0, 2].set_ylabel("Range (m)")
                    if not hasattr(FigsAxes[0, 2], "colorbar_added"):  # Add colorbar only once
                        colorbar = fig.colorbar(im, ax=FigsAxes[0, 2], label="Amplitude")
                        FigsAxes[0, 2].colorbar_added = True
                    # CFAR detection
                    detections, cfar_threshold, rangeDoppler4CFAR = ssp.radar.utils.rangedoppler_detection_alpha(rangeDopplerTXRX, specifications)

                    # Plot CFAR detection results
                    FigsAxes[1, 0].clear()
                    im = FigsAxes[1, 0].imshow(rangeDoppler4CFAR, 
                                            extent=[f_Doppler[0], f_Doppler[-1], d_fft[0], d_fft[-1]],
                                            aspect='auto', origin='lower', cmap='viridis')
                    FigsAxes[1, 0].set_xlabel("Doppler Frequency (Hz)")
                    FigsAxes[1, 0].set_ylabel("Range (m)")
                    FigsAxes[1, 0].set_title("CFAR Detection Map")
                    if not hasattr(FigsAxes[1, 0], "colorbar_added"):
                        fig.colorbar(im, ax=FigsAxes[1, 0], label="Detection Amplitude")
                        FigsAxes[1, 0].colorbar_added = True

                    
                    # distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
                    # elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
                    # X, Y = np.meshgrid(elevation, distance)
                    # ax.plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
                    # ax.plot_surface(X, Y, 10*np.log10(cfar_threshold)+0, color='yellow', alpha=1)
                    # detected_points = np.where(detections == 1)
                    # ax.scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                    #        10*np.log10(rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')
                    # plt.show()
                    # Extract detected points
                    detected_points = np.where(detections == 1)

                    # Angle processing and point cloud generation
                    Azimuth_Angles,debug_spectrums = ssp.radar.utils.angleprocessing_capon1D(rangeDopplerTXRX, detections, specifications)
                    pc = ssp.radar.utils.pointscloud(d_fft[detected_points[0]], f_Doppler[detected_points[1]], Azimuth_Angles)
                    pcs.append(pc)
                    
                    FigsAxes[1, 1].clear()
                    for debug_spectrum in debug_spectrums:
                        FigsAxes[1, 1].plot(debug_spectrum)
                        
                    FigsAxes[1, 2].clear()
                    update_pointclouds(all_pcs,FigsAxes[1, 2])
                     
                    plt.draw()
                    plt.pause(0.1)
                    plt.gcf().canvas.flush_events()
        all_pcs.append(pcs)

        print(f'Processed frame = {ssp.config.CurrentFrame}')
        ssp.utils.increaseCurrentFrame()

    plt.legend()

    plt.show()


def update_pointclouds(all_pcs,ax):
    
    x,y,z,v=np.array([]),np.array([]),np.array([]),np.array([])
    for i, pcs in enumerate(all_pcs):
        for j,pc in enumerate(pcs):
            if len(pc)==0:
                continue
            y=np.append(y,pc[:, 1])
            z=np.append(z,pc[:, 0])
            v=np.append(v,pc[:, 2])
            x=np.append(x,np.ones_like(pc[:, 0])*i)
    ind_filter = np.where(np.abs(v)>-70)
    x=x[ind_filter]
    y=y[ind_filter]
    z=z[ind_filter]
    v=v[ind_filter]
    if len(v)==0:
        return
    # ax = fig.add_subplot(111, projection='3d')

    # Normalize the v values for better color mapping
    norm = plt.Normalize(v.min(), v.max())


    # Scatter plot with colors based on v values
    scatter = ax.scatter(x, y, z, c=v, cmap='viridis', norm=norm)

    # Add a colorbar to interpret the color scale
    # cbar = fig.colorbar(scatter, ax=ax, label='v value')

    # ssp.utils.set_axes_equal(ax)    
    ax.set_ylabel('X')
    ax.set_xlabel('Frame')
    ax.set_zlabel('Y')
    ax.set_xlim([0,30])
    
def predefine_Hand_Gesture_3Xethru_Nature_paper():
    addfile = os.path.join(ssp.config.temp_folder,'assets/HandGest/up_down_swipe.blend')
    addfile = ssp.utils.hub.fetch_file("animation","Hand")
    if "Extra Settings" in bpy.data.objects:
        
        if "Hand Gesture 3D Asset" in bpy.data.objects["Extra Settings"]:
            addfile = bpy.data.objects["Extra Settings"]["Hand Gesture 3D Asset"]
    
    ssp.utils.delete_all_objects()
    if "Extra Settings" not in bpy.data.objects:
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(.01, .01, .01))
        sim_axes = bpy.context.object
        sim_axes.name = f'Extra Settings'
        sim_axes["Hand Gesture output"] = os.path.join(ssp.config.temp_folder,'HandG.mat')
        sim_axes["Hand Gesture 3D Asset"] = addfile
    ssp.utils.define_settings()
    ssp.integratedSensorSuite.define_suite(0, location=Vector((-.14, -.5, -.1)), rotation=Vector((0, 0, np.pi/2)))
    radar1 = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9)
    radar2 = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=1, location=Vector((1, 0,0)), rotation=Vector((np.pi/2,0, np.pi/2)), f0=76e9)
    radar3 = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=2, location=Vector((.5, 0,.5)), rotation=Vector((0,0, 0)), f0=76e9)
    # During data acquisition, examiners recorded a subject repeating a particular hand gesture for 450 seconds, corresponding to 9000 (slow-time) rows. There is 1 complete gesture motion in 90 slow-time frames. As such, each radar signal matrix contains 100 complete hand gesture motion samples. The range of each UWB radar is 1.2 meters, corresponding to 189 fast-time bins.
    gesture_motion_slowTimeFrames = 90
    recordedTimePerSample = 4.5
    radarRange = 1.2
    fast_time_bins = 189
    rangebins = radarRange / fast_time_bins
    for radar in [radar1,radar2,radar3]:
        radar['RadarMode']='Pulse'
        radar['PulseWaveform']='UWB'
        radar['N_ADC']  = fast_time_bins
        radar['Fs_MHz'] =  ssp.LightSpeed / (2*rangebins) / 1e6
        radar['RF_AnalogNoiseFilter_Bandwidth_MHz'] =  1500
        radar['PRI_us']= 1e6*recordedTimePerSample/gesture_motion_slowTimeFrames
        radar['NPulse'] = gesture_motion_slowTimeFrames
        radar['Range_End']=100
    FramesNumber=int(1.1*recordedTimePerSample*bpy.context.scene.render.fps)
    
    
    # add_blenderfileobjects(os.path.join(ssp.config.temp_folder,'RealisticHand.blend'),RCS0=10)    
    add_blenderfileobjects(addfile,RCS0=10e3)    
    
    ssp.utils.set_RayTracing_balanced()
    ssp.utils.set_frame_start_end(start=1,end=FramesNumber)
    ssp.utils.save_Blender()
# def add_blenderfileobjects(blend_file_path,RCS0=1,decimatefactor=1):
#     with bpy.data.libraries.load(blend_file_path) as (data_from, data_to):
#         data_to.objects = data_from.objects  # Load all objects
#     for obj in data_to.objects:
#         if obj is not None:  # Avoid null references
#             bpy.context.scene.collection.objects.link(obj)
#             if obj.type == 'MESH':
#                 obj["RCS0"]=RCS0
    # if decimatefactor<1:
    #     ssp.utils.decimate_scene_all(decimation_ratio=decimatefactor)
def setRCSToMaterial(startname="",RCS0=1):
    for obj in bpy.data.objects:
        if obj is not None:
            if obj.type == 'MESH':
                if obj.name.startswith(startname):
                    obj["RCS0"]=RCS0
def add_blenderfileobjects(blend_file_path, RCS0=1, decimatefactor=1, rotation=(0, 0, 0), translation=(0, 0, 0)):
    """
    Load objects from a Blender file, link them to the current scene, and apply transformations while preserving armature relationships.
    
    Parameters:
        blend_file_path (str): Path to the .blend file containing objects.
        RCS0 (float): Custom property to assign to mesh objects.
        decimatefactor (float): Factor to simplify geometry for mesh objects (optional).
        rotation (tuple): Rotation to apply to all objects (in radians, as (x, y, z)).
        translation (tuple): Translation to apply to all objects (as (x, y, z)).
    
    Returns:
        list: A list of the added objects.
    """
    import mathutils

    added_objects = []  # List to store references to added objects
    
    # Load objects from the specified blend file
    with bpy.data.libraries.load(blend_file_path) as (data_from, data_to):
        data_to.objects = data_from.objects  # Load all objects

    # Transform matrix for rotation and translation
    rotation_matrix = mathutils.Euler(rotation).to_matrix().to_4x4()
    translation_matrix = mathutils.Matrix.Translation(translation)
    transform_matrix = translation_matrix @ rotation_matrix  # Combine translation and rotation
    
    # Process and add each object to the current scene
    for obj in data_to.objects:
        if obj is not None:  # Avoid null references
            bpy.context.scene.collection.objects.link(obj)
            added_objects.append(obj)  # Keep track of added objects

            # Apply the transformation while respecting armature relationships
            if obj.parent is None:  # Only transform root objects
                obj.matrix_world = transform_matrix @ obj.matrix_world
            
            # Process mesh objects
            if obj.type == 'MESH':
                obj["RCS0"] = RCS0
                
                # Apply decimation if needed
                if decimatefactor < 1:
                    modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
                    modifier.ratio = decimatefactor
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.modifier_apply(modifier="Decimate")
    
    return added_objects  # Return the list of added objects

def predefine_Hand_Gesture_1642():
    ssp.utils.delete_all_objects()
    # try:
    #     ssp.utils.open_Blend(os.path.join(ssp.config.temp_folder, 'arm_to_left 4.blend'))
    # except Exception as X:
    #     1
    blend_file_path = os.path.join(ssp.config.temp_folder, 'arm_to_left 4.blend')
    # blend_file_path = os.path.join(ssp.config.temp_folder, 'up_down_swipe.blend')
    

    # The directory inside the .blend file to access objects
    object_directory = blend_file_path + "/Object/"

    # Append all objects from the file
    with bpy.data.libraries.load(blend_file_path) as (data_from, data_to):
        # List all objects available in the .blend file
        print("Available objects:", data_from.objects)
        # Select objects you want to append (replace with specific names if needed)
        data_to.objects = data_from.objects  # Load all objects

    # Link the loaded objects to the current scene
    for obj in data_to.objects:
        if obj is not None:  # Avoid null references
            bpy.context.scene.collection.objects.link(obj)
    ssp.utils.define_settings()
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    radar = ssp.radar.utils.predefined_array_configs_TI_AWR1642(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9)
    rangeResolution,radialVelocityResolution,N_ADC,ChirpTimeMax,CentralFreq=0.039,0.13,256,60e-6,76e9

    ssp.radar.utils.FMCW_Chirp_Parameters(rangeResolution,N_ADC,ChirpTimeMax,radialVelocityResolution,CentralFreq)

    ssp.radar.utils.set_FMCW_Chirp_Parameters(radar,slope=64.06,fsps=4.2,N_ADC=256,NPulse=256,PRI_us=60)
    radar['RangeFFT_OverNextP2'] = 0
    radar['Range_End']=100*.6/9.98 
    radar['DopplerFFT_OverNextP2']=0
    radar['CFAR_RD_alpha']=15
    radar['CFAR_Angle_alpha']=2
    radar['Transmit_Antenna_Element_Pattern']='Directional-Sinc' # add rect pattern
    radar["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = 60
    radar["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = 60

    ssp.utils.set_frame_start_end(start=1,end=30)
    ssp.utils.save_Blender()
def predefine_Pattern_SISO():
    ssp.utils.delete_all_objects()
    ssp.utils.define_settings()
    R=10
    NF = 360*2
    cube = ssp.environment.add_cube(location=Vector((R, 0, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .5)), subdivision=0)
    cube["RCS0"]=1
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)), scale=(.01, .01, .01))
    empty = bpy.context.object
    cube.parent = empty
    empty.rotation_euler  = (0,0,2*np.pi)
    empty.keyframe_insert(data_path="rotation_euler", frame=1)
    empty.rotation_euler  = (0,0,0)
    empty.keyframe_insert(data_path="rotation_euler", frame=NF/2)
    empty.rotation_euler  = (0,2*np.pi,0)
    empty.keyframe_insert(data_path="rotation_euler", frame=NF)
    for fcurve in empty.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    ssp.utils.set_frame_start_end(start=1,end=NF)
    
def process_predefine_Hand_Gesture_3Xethru_Nature_paper():
    ssp.utils.trimUserInputs()  
    ssp.config.restart()
    # ssp.config.DopplerProcessingMethod_FFT_Winv = (True)

    # ssp.config.directReceivefromTX =  False 
    # ssp.config.RadarRX_only_fromscatters_itsTX = True
    # ssp.config.RadarRX_only_fromitsTX = True
    # ssp.config.Radar_TX_RX_isolation = True

    ssp.utils.useCUDA(False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    data_save = []
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        for isuite, radarSpecifications in enumerate(ssp.RadarSpecifications):
            for iradar, specifications in enumerate(radarSpecifications):
                for XRadar, timeX in Signals[isuite]['radars'][iradar]:
                    # Range and Doppler processing
                    # plt.figure()
                    s = 1500
                    datasample = np.real(XRadar[:,:,0].T)/s
                    data_save.append(datasample)
                    clutter_removal = True
                    if clutter_removal:
                        b = [1, -1]
                        a = [1, -0.9]
                        datasample = lfilter(b, a, datasample, axis=0)
                        
                    
                    axs[iradar].imshow(datasample, aspect='auto', cmap='viridis')
                    plt.title(f'{iradar}')
                    axs[iradar].axis("off")
        print(f'Processed frame = {ssp.config.CurrentFrame}')
        ssp.utils.increaseCurrentFrame()
    if len(data_save)==3:

        data_to_save = {
            "Left": data_save[0],  # Example 2D array for "Left"
            "Top": data_save[2],  # Example 2D array for "Top"
            "Right": data_save[1]  # Example 2D array for "Right"
        }
        savefilename = os.path.join(ssp.config.temp_folder,'HandG.mat')
        if "Extra Settings" in bpy.data.objects:    
            if "Hand Gesture output" in bpy.data.objects["Extra Settings"]:
                savefilename = bpy.data.objects["Extra Settings"]["Hand Gesture output"]
        savemat(savefilename, data_to_save)
        simulated_samples = ssp.ai.radarML.HandGestureMisoCNN.make_sample(data_save[0],data_save[1],data_save[2])

        # np.save(os.path.join(ssp.config.temp_folder, 'Left.npy'),data_save[0])
        # np.save(os.path.join(ssp.config.temp_folder, 'Right.npy'),data_save[1])
        # np.save(os.path.join(ssp.config.temp_folder, 'Top.npy'),data_save[2])
        import torch
        downloaded_model_path = ssp.utils.hub.fetch_pretrained_model("models", "HandGesture MisoCNN")
        if downloaded_model_path:
            model = ssp.ai.radarML.HandGestureMisoCNN.MultiInputModel(num_classes=12)
            model.load_state_dict(torch.load(downloaded_model_path))
            dataset = ssp.ai.radarML.HandGestureMisoCNN.RadarGestureDataset(data_folder='')  
            for radar_tensor in simulated_samples:
                with torch.no_grad():
                    outputs = model( radar_tensor )
                    _ , preds = torch.max( outputs , 1 )
                    Gi = f"G{preds.item()+1}"
                    print(f"Sample from Simulator -> classified as {Gi}: {dataset.gestureVocabulary[preds.item()]}")
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    classes,softmax_probs=[],[]
                    for idx, prob in enumerate(probabilities[0]):
                        class_name = dataset.gestureVocabulary[idx]
                        print(f"Class {class_name}: {prob:.4f} : {outputs[0,idx]:.4f}")
                        classes.append(class_name)
                        softmax_probs.append(prob.cpu())
                    for idx, prob in enumerate(softmax_probs):
                        bar_length = int(prob * 50)  
                        print(f"Class {classes[idx]:<20}: {prob:.2f}: {'#' * bar_length}")
                    plt.figure(figsize=(8, 5))
                    plt.bar(classes, softmax_probs, color='skyblue')
                    plt.xlabel('Classes', fontsize=12)
                    plt.ylabel('Probability', fontsize=12)
                    plt.title('Class Probabilities', fontsize=14)
                    plt.ylim(0, 1)
                    for i, prob in enumerate(softmax_probs):
                        plt.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=10)

                    plt.title(f"Sample from Simulator : classified as {Gi}: {dataset.gestureVocabulary[preds.item()]}")
                    
        

    plt.tight_layout()
    plt.show()


def deform_scenario_1(angLim=30,Lengths = [.2,.1,.1],elps=[.25,.1,.5],sbd=4,cycles=8,cycleHLen=15):
    ssp.utils.delete_all_objects()

    # === Create Armature (Bones) ===
    # Add a new armature object
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0), radius=Lengths[0])
    armature = bpy.context.object
    armature.name = "sspDeformArmature"

    # Extrude a new bone from the initial bone
    bpy.ops.armature.extrude_move(
        TRANSFORM_OT_translate={"value": (0, 0, Lengths[1])}
    )
    bpy.ops.armature.extrude_move(TRANSFORM_OT_translate={"value": (0, 0, Lengths[2])})
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # === Create an Icosphere ===
    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), scale=(elps[0],elps[1], elps[2]), subdivisions=sbd)
    icosphere = bpy.context.object

    # === Parent the Icosphere to the Armature ===
    # Select both objects (armature must be active)
    bpy.ops.object.select_all(action='DESELECT')
    icosphere.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Parent the icosphere to the armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # Deselect all objects for clarity
    bpy.ops.object.select_all(action='DESELECT')

    # === Switch to Pose Mode and Animate Bone ===
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Access the pose bone (second bone in the armature)
    pose_bone = bpy.context.object.pose.bones[1]
    pose_bone.rotation_mode = 'XYZ'
    pose_bone2 = bpy.context.object.pose.bones[2]
    pose_bone2.rotation_mode = 'XYZ'
    for i in range(cycles):
        bpy.context.scene.frame_set(1+i*cycleHLen*2)
        pose_bone.rotation_euler = (np.radians(angLim), 0.0, 0.0)  # Initial rotation (no rotation)
        pose_bone2.rotation_euler = (np.radians(-angLim), 0.0, 0.0)  # Initial rotation (no rotation)
        pose_bone.keyframe_insert(data_path="rotation_euler")
        pose_bone2.keyframe_insert(data_path="rotation_euler")

        bpy.context.scene.frame_set(cycleHLen+i*cycleHLen*2)
        pose_bone.rotation_euler = (np.radians(-angLim), 0, 0.0)  # Simulate Y-direction movement with rotation
        pose_bone2.rotation_euler = (np.radians(angLim), 0, 0.0)  # Simulate Y-direction movement with rotation
        pose_bone.keyframe_insert(data_path="rotation_euler")
        pose_bone2.keyframe_insert(data_path="rotation_euler")

    # Exit pose mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # # === Save Blender File ===
    # ssp.utils.save_Blender()
    return armature
def handGesture_simple(G=1):
    ssp.utils.delete_all_objects()

    # === Create an Icosphere ===
    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), scale=(.2,.15, .02), subdivisions=4)
    icosphere = bpy.context.object
    if G==1:
        fr=1
        for i in range(3):
            bpy.context.scene.frame_set(fr);fr+=10
            icosphere.location=(0, 0, -.4)
            icosphere.keyframe_insert(data_path="location")
            bpy.context.scene.frame_set(fr);fr+=110
            icosphere.location=(0, 0, 0)
            icosphere.keyframe_insert(data_path="location")
    elif G==2:
        fr=1
        for i in range(3):
            bpy.context.scene.frame_set(fr);fr+=10
            icosphere.location=(0, 0, 0)
            icosphere.keyframe_insert(data_path="location")
            bpy.context.scene.frame_set(fr);fr+=110
            icosphere.location=(0, 0, -.4)
            icosphere.keyframe_insert(data_path="location")
        
    
    # === Save Blender File ===
    ssp.utils.save_Blender()



def raytracing_rays():
    o =[]
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    while ssp.config.run():    
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
            for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
                for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
                    for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
                        for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
                            for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
                                for ip,p in enumerate(TX_d_drate_amp):
                                    d,dr,a,m=p
                                    tx = ssp.lastSuite_Position[istx]['Radar'][irtx]['TX-Position'][itx] 
                                    rx = ssp.lastSuite_Position[isrx]['Radar'][irrx]['RX-Position'][irx]
                                    v = [tx]
                                    for mi in m:
                                        if isinstance(mi, list) and len(mi) == 2 and isinstance(mi[1], Vector):
                                            v.append(mi[1])
                                        if isinstance(mi, str):
                                            _, is0, ir0, iris0 = mi.split('_')
                                            is0 = int(is0)
                                            ir0 = int(ir0)
                                            iris0 = int(iris0)
                                            v.append(ssp.lastSuite_Position[is0]['RIS'][ir0]['Position'][iris0])
                                    v.append(rx)
                                    ray={}
                                    ray['v']=v
                                    ray["Ray Tracing 20log Amp (dB)"]=float(20*np.log10(a))
                                    ray["Ray Tracing distance"]=float(d)
                                    ray["Ray Tracing doppler * f0"]=float(dr)
                                    ray["Middle Number"]=len(m)
                                    ray["suiteradar"]=[isrx,irrx,irx,istx,irtx,itx]
                                    o.append(ray)
        ssp.utils.increaseCurrentFrame()
        break
    return o

