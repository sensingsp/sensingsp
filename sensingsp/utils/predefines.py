import bpy 
import sensingsp as ssp
import numpy as np
from mathutils import Vector

def add_radar_string(st):
    if st=='App&File':
        ssp.radar.utils.apps.runradarArrayapp()
        return
    if st == 'RadarVis':
        ssp.radar.utils.apps.runRadarVis()
        return
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        radar_planes = ssp.environment.BlenderSuiteFinder().find_radar_planes(obj)
        radarIndex = max([int(plane.name.split('_')[2]) for plane in radar_planes if plane.parent == obj] or [-1]) + 1
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        radarIndex = 0  
    if st=='TI AWR 1642 (2TX-4RX)':
        freq = 76e9
        ssp.radar.utils.predefined_array_configs_TI_AWR1642(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=freq)
    if st=='TI IWR 6843 (3TX-4RX)':
        freq = 76e9
        ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=freq)
    if st=='TI Cascade AWR 2243 (12TX-16RX)':
        freq = 76e9
        ssp.radar.utils.predefined_array_configs_TI_Cascade_AWR2243(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=freq)
    if st=='SISO':
        freq = 76e9
        ssp.radar.utils.predefined_array_configs_SISO(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=freq)
    
def add_sensors_string(st):
    1

def add_radar(context,option):
    freq = float(context.scene.radar_properties.freq)*1e9  # Convert string to float
    if "Simulation Settings" in bpy.data.objects:
        if "RF Frequency (GHz)" not in bpy.data.objects["Simulation Settings"]:
          bpy.data.objects["Simulation Settings"]["RF Frequency (GHz)"]=1
        freq = bpy.data.objects["Simulation Settings"]["RF Frequency (GHz)"]*1e9
          
    
    if freq<=0:
        freq=1
    # finder = BlenderSuiteFinder()
    # suite_manager = BlenderRadarSuiteManager()
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        radar_planes = ssp.environment.BlenderSuiteFinder().find_radar_planes(obj)
        radarIndex = max([int(plane.name.split('_')[2]) for plane in radar_planes if plane.parent == obj] or [-1]) + 1
        if option=="Cascade":
            ssp.radar.utils.predefined_array_configs_TI_Cascade_AWR2243(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
        if option=="6843":
            ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
        if option=="SISO":
            ssp.radar.utils.predefined_array_configs_SISO(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
        if option=="awr1642":
            ssp.radar.utils.predefined_array_configs_TI_AWR1642(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=freq)
        if option=="JSON":
            ssp.radar.utils.predefined_array_configs_JSON(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        if option=="Cascade":
            ssp.radar.utils.predefined_array_configs_TI_Cascade_AWR2243(isuite=suiteIndex, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
        if option=="6843":
            ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=suiteIndex, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
        if option=="SISO":
            ssp.radar.utils.predefined_array_configs_SISO(isuite=suiteIndex, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)
        if option=="awr1642":
            ssp.radar.utils.predefined_array_configs_TI_AWR1642(isuite=suiteIndex, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=freq)
        if option=="JSON":
            ssp.radar.utils.predefined_array_configs_JSON(isuite=suiteIndex, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=freq)

def add_radar_Json(file_path):
    if not file_path.lower().endswith('.json'):
        print("Error: The selected file is not a .json file.")
        return None
    
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        radar_planes = ssp.environment.BlenderSuiteFinder().find_radar_planes(obj)
        radarIndex = max([int(plane.name.split('_')[2]) for plane in radar_planes if plane.parent == obj] or [-1]) + 1
        ssp.radar.utils.predefined_array_configs_JSON(isuite=suiteIndex, iradar=radarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), file_path=file_path)
        
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        ssp.radar.utils.predefined_array_configs_JSON(isuite=suiteIndex, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), file_path=file_path)
        
    
def predefine_cube_cascade():
    ssp.utils.delete_all_objects()
    ssp.environment.add_cube(location=Vector((5, 0, 0)), direction=Vector((1, 0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    ssp.radar.utils.predefined_array_configs_TI_Cascade_AWR2243(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)), f0=70e9)
    ssp.utils.set_frame_start_end(start=1,end=2)

def predefine_movingcube_6843():
    ssp.utils.delete_all_objects()
    ssp.utils.define_settings()
    cube = ssp.environment.add_cube(location=Vector((0, 0, 0)), direction=Vector((1, 0, 0)), scale=Vector((.1, .1, .1)), subdivision=0)
    cube["RCS0"]=.0001
    cube.location = (3,0,0)
    cube.keyframe_insert(data_path="location", frame=1)
    cube.location = (3, 3,0)
    cube.keyframe_insert(data_path="location", frame=30)
    cube.location = (3, -3,0)
    cube.keyframe_insert(data_path="location", frame=100)
    for fcurve in cube.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'
    ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
    ssp.utils.set_frame_start_end(start=1,end=100)


def add_camera(context,option):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        camera_planes = ssp.environment.BlenderSuiteFinder().find_camera_planes(obj)
        cameraIndex = max([int(plane.name.split('_')[2]) for plane in camera_planes if plane.parent == obj] or [-1]) + 1
        ssp.camera.utils.add_camera(isuite=suiteIndex, icamera=cameraIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        ssp.camera.utils.add_camera(isuite=suiteIndex, icamera=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))

def add_lidar(context,option):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        lidar_planes = ssp.environment.BlenderSuiteFinder().find_lidar_planes(obj)
        lidarIndex = max([int(plane.name.split('_')[2]) for plane in lidar_planes if plane.parent == obj] or [-1]) + 1
        ssp.lidar.utils.add_lidar(isuite=suiteIndex, ilidar=lidarIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        ssp.lidar.utils.add_lidar(isuite=suiteIndex, ilidar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
def add_ris(context,option):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        ris_planes = ssp.environment.BlenderSuiteFinder().find_ris_planes(obj)
        risIndex = max([int(plane.name.split('_')[2]) for plane in ris_planes if plane.parent == obj] or [-1]) + 1
        ssp.ris.utils.add_ris(isuite=suiteIndex, iris=risIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        ssp.ris.utils.add_ris(isuite=suiteIndex, iris=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
def add_JRC(context,option):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        # ris_planes = ssp.environment.BlenderSuiteFinder().find_ris_planes(obj)
        # risIndex = max([int(plane.name.split('_')[2]) for plane in ris_planes if plane.parent == obj] or [-1]) + 1
        # ssp.ris.utils.add_ris(isuite=suiteIndex, iris=risIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        # ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        # ssp.ris.utils.add_ris(isuite=suiteIndex, iris=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
def add_Probe(context,option):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = bpy.context.view_layer.objects.active
    if obj in suite_planes:
        suiteIndex = int(obj.name.split('_')[-1])
        probe_planes = ssp.environment.BlenderSuiteFinder().find_probe_planes(obj)
        probeIndex = max([int(plane.name.split('_')[2]) for plane in probe_planes if plane.parent == obj] or [-1]) + 1
        ssp.probe.utils.add_probe(isuite=suiteIndex, iprobe=probeIndex, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        
    else:
        suiteIndex = max([int(plane.name.split('_')[-1]) for plane in suite_planes] or [-1]) + 1
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        ssp.probe.utils.add_probe(isuite=suiteIndex, iprobe=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,-np.pi/2, -np.pi/2)))
        