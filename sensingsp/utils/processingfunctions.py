import bpy 
import sensingsp as ssp
# import sensingsp as ssp
import numpy as np
from mathutils import Vector
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import bmesh

def processing_functions_run1():
  ssp.utils.set_frame_start_end(start=bpy.context.scene.frame_start,end=2+0*bpy.context.scene.frame_end)
  ssp.utils.trimUserInputs() # input from bpy and blender UI -> ssp.RadarSpecifications
  ssp.raytracing.Path_RayTracing() # input from bpy, output ssp.Paths
  
  ssp.config.setDopplerProcessingMethod_FFT_Winv(1)
  ssp.config.restart()
  while ssp.config.run():
      Signals = ssp.integratedSensorSuite.SensorsSignalGeneration() # input is ssp.Paths in ssp.CurrentFrame
      ProcessingOutputs = ssp.integratedSensorSuite.SensorsSignalProccessing(Signals)
      ssp.visualization.visualizeProcessingOutputs(ProcessingOutputs)
      ssp.utils.saveMatFile(ProcessingOutputs)
      ssp.utils.increaseCurrentFrame()



def processing_functions_run2_old():
    ssp.utils.trimUserInputs() 
    ssp.raytracing.Path_RayTracing() 
    ssp.config.restart()
    ssp.config.define_axes(1)
    ssp.config.setDopplerProcessingMethod_FFT_Winv(1)
    while ssp.config.run():
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration() # input is ssp.Paths in ssp.CurrentFrame
        ssp.integratedSensorSuite.SensorsSignalProccessing_Angle(Signals)
        ssp.utils.increaseCurrentFrame()

def processing_functions_run2():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    # ssp.config.define_axes(2)
    ssp.config.setDopplerProcessingMethod_FFT_Winv(1)
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        plt.plot(np.sin(np.arange(100)*.1))
        plt.pause(.1)
        # ssp.integratedSensorSuite.SensorsSignalProccessing_Angle_frame(Signals)
        # print(ssp.config.CurrentFrame)
        ssp.utils.increaseCurrentFrame()
    plt.show()
    
def processing_functions_run3():
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    ssp.config.define_axes()
    while ssp.config.run():
        ssp.raytracing.Path_RayTracing() 
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration() # input is ssp.Paths in ssp.CurrentFrame
        ssp.integratedSensorSuite.SensorsSignalProccessing_Angle(Signals)
        ssp.utils.increaseCurrentFrame()
    
def processing_functions_RISProbe1():
    db = True
    range_effect = True
    add_color = True
    add_open3d = True
    colormap = 'viridis' 
    size_scale = 0.05
    if "Simulation Settings" in bpy.data.objects:
        option = bpy.data.objects["Simulation Settings"]["Add Probe"]
        parts = option.split(',')
        db = str(parts[4])=='db1'
        range_effect = str(parts[5])=='range_effect1'
        add_color = str(parts[6])=='add_color1'
        add_open3d = str(parts[7])=='add_open3d1'
        if str(parts[8])=='colormap1':
            colormap = 'viridis'
        elif str(parts[8])=='colormap2':
            colormap = 'plasma'
        elif str(parts[8])=='colormap3':
            colormap = 'inferno'
        elif str(parts[8])=='colormap4':
            colormap = 'magma'
        elif str(parts[8])=='colormap5':
            colormap = 'jet'
        size_scale = float(parts[9])
    add_open3d = False
    
    ssp.utils.set_frame_start_end(start=1,end=2)
    ssp.utils.trimUserInputs() 
    ssp.config.restart()
    # Simple analysis for Probe (RIS)
    bpy.context.scene.frame_set(ssp.config.CurrentFrame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geocalculator = ssp.raytracing.BlenderGeometry()
    rayTracingFunctions = ssp.raytracing.RayTracingFunctions()
    suite_information = ssp.raytracing.BlenderSuiteFinder().find_suite_information()
    Suite_Position,ScattersGeo,HashFaceIndex_ScattersGeo,ScattersGeoV = geocalculator.get_Position_Velocity(bpy.context.scene, suite_information, ssp.config.CurrentFrame, 1)
    Lambda = 3e8 / 70e9
    sourcePos=Suite_Position[0]['Radar'][0]['TX-Position'][0]
    p1=ssp.raytracing.Vector3D(sourcePos.x,sourcePos.y,sourcePos.z)
    probe_values = []
    probe_xys = []
    for i in range(len(Suite_Position[0]['Probe'])):
        for probe_Pos in Suite_Position[0]['Probe'][i]:
            p3=ssp.raytracing.Vector3D(probe_Pos.x,probe_Pos.y,probe_Pos.z)
            o = 0
            for iriselement , ris_element_position in enumerate(Suite_Position[0]['RIS'][0]['Position']):
                p2=ssp.raytracing.Vector3D(ris_element_position.x,ris_element_position.y,ris_element_position.z)
                d1 = p2.distance_to(p1)
                d2 = p2.distance_to(p3)
                d=d1+d2
                phase = -2*np.pi*d / Lambda
                if range_effect:
                    amp = 1  / d1 / d2
                else:
                    amp = 1
                amp *= Suite_Position[0]['RIS'][0]['PhaseAmplitude'][iriselement][0]
                phase -= Suite_Position[0]['RIS'][0]['PhaseAmplitude'][iriselement][1]
                o += amp*np.exp(1j*phase)
            if db:
                probe_values.append(np.log10(np.abs(o)))
            else:
                probe_values.append(np.abs(o))
            probe_xys.append([probe_Pos.x,probe_Pos.y,probe_Pos.z])
    probe_xys = np.array(probe_xys)
    probe_values = np.array(probe_values)
    amplitudes_normalized = (probe_values - probe_values.min()) / (probe_values.max() - probe_values.min())
    amplitudes_normalized = .01+.99*amplitudes_normalized
    spheres = []
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'Probe Measurements'
    # empty = bpy.data.collections.new(f'Probe Measurements')
    # Create spheres based on the normalized amplitudes and probe positions
    
    for i in range(len(probe_xys)):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=amplitudes_normalized[i] * size_scale, location=(probe_xys[i][0],probe_xys[i][1],probe_xys[i][2]))
        sphere = bpy.context.object
        
        sphere.parent=empty
        if add_color:
            parula_colormap = cm.get_cmap(colormap)
            mat = bpy.data.materials.new(name="SphereMaterial")
            # mat.diffuse_color = (amplitudes_normalized[i], 0, 0, 1)  
            color = parula_colormap(amplitudes_normalized[i])[:3]  # Get RGB color from colormap
            mat.diffuse_color = (*color, 1)  # Set color based on amplitude
            sphere.data.materials.append(mat)

    if add_open3d:
        import open3d as o3d
        for i in range(len(probe_xys)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=amplitudes_normalized[i] * size_scale)
            sphere.translate(probe_xys[i])
            color = [amplitudes_normalized[i], 0, 0] 
            sphere.paint_uniform_color(color)
            spheres.append(sphere)
        combined_geometry = o3d.geometry.TriangleMesh()
        for sphere in spheres:
            combined_geometry += sphere
        o3d.visualization.draw_geometries([combined_geometry])

    
def extrafunctions(st):
    if st == "Environment Meshes":
        ssp.utils.trimUserInputs() 
        ssp.config.restart()
        framesdata=[]
        while ssp.config.run():
            bpy.context.scene.frame_set(ssp.config.CurrentFrame)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            xi=[]
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    if obj.name.startswith('Probe_')==False:
                        bm = bmesh.new()
                        bm.from_object(obj, depsgraph)
                        bm.transform(obj.matrix_world)
                        for face in bm.faces:
                            vertices = np.array([[vert.co.x,vert.co.y,vert.co.z] for vert in face.verts])
                            xi.append(vertices)
                        bm.free()
            # print("xi:              ",xi)
            framesdata.append(xi)
            ssp.utils.increaseCurrentFrame()
            # break
        # print("sent x :",framesdata)
        ssp.radar.utils.pyqtgraph3DApp(framesdata)
    if st == "Ray Tracing Simulation":
        ssp.environment.scenarios.raytracing_test()
    if st == "CSI Simulation":
        ssp.utils.trimUserInputs()
        ssp.utils.set_RayTracing_balanced()
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        ssp.utils.channels_info(path_d_drate_amp)
    if st == "RIS Simulation":
        ssp.utils.processingfunctions.processing_functions_RISProbe1()
    if st == "Antenna Pattern":
        ssp.visualization.plot_pattern_button()
    if st == "Open Temp Folder":
        ssp.utils.open_temp_folder()
        
    if st == "Light RayTracing":
        ssp.utils.set_RayTracing_light()
        
    if st == "Balanced RayTracing":
        ssp.utils.set_RayTracing_balanced()
    if st == "Advanced Intense RayTracing":
        ssp.utils.set_RayTracing_advanced_intense()
    if st == "Surface Materials":
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
    if st == "Wifi Sensing Settings":
        if not("Wifi Sensing Settings" in bpy.data.objects):
            import os
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(.01, .01, .01))
            ws_axes = bpy.context.object
            ws_axes.name = f'Wifi Sensing Settings'
            # --- Simulation Settings ---
            ws_axes["Sampling Frequency (MHz)"]    = 80.0    # Fs = 80 MHz
            ws_axes["Channel Bandwidth (MHz)"]     = 80.0    # VHT80
            ws_axes["FFT Size"]                    = 256     # NFFT
            ws_axes["Guard Interval (µs)"]         = 0.4     # short GI
            ws_axes["Modulation Order"]            = 16      # 16-QAM
            ws_axes["MCS Index"]                   = 7       # highest VHT MCS for 1 SS
            ws_axes["Carrier Frequency (GHz)"]     = 5.8     # 5.8 GHz center
            ws_axes["Noise Std (linear)"]          = 1e-3    # AWGN sigma
            ws_axes["Output mat file"]          =   "wifi_sensing_output.mat"
            ws_axes["Load environment"]          =   os.path.join(ssp.config.temp_folder, "WifiSensing-2.blend")
            ws_axes["Random TX bit length"]      = 100
            ws_axes["Deterministic TX message"] = "SensingSP™ is an open-source library for simulating sensing systems with signal processing and ML tools. Install with: pip install sensingsp"
    if st == "SensingSP Version":
        # Check if property is already defined to avoid re-register errors
        
        if not hasattr(bpy.types.Scene, "my_custom_text"):
            bpy.types.Scene.my_custom_text = bpy.props.StringProperty(
                name="Settings",
                default=f'SensingSP {ssp.__version__}; {ssp.config.updateTime}; To Update: BlenderPyhtonPath/python -m pip install --upgrade sensingsp',
            )
        else:
            bpy.context.scene.my_custom_text = f'SensingSP {ssp.__version__}; {ssp.config.updateTime}; To Update: BlenderPyhtonPath/python -m pip install --upgrade sensingsp'
            
        def draw(self, context):
            self.layout.label(text="Version:")  # Simple label
            self.layout.prop(context.scene, "my_custom_text", text="")  # Text box to copy from

        bpy.context.window_manager.popup_menu(draw, title="Sensing Signal Processing", icon='INFO')
        # bpy.ops.message.dialog('INVOKE_DEFAULT')
        # bpy.context.window_manager.popup_menu(title="Settings Info", icon='INFO')
        # bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(.01, .01, .01))
        # ws_axes = bpy.context.object
        # ws_axes.name = f'ssp Version: {ssp.__version__}'
    
    if st == "Array Visualization":
        ssp.visualization.visualize_array()
        
    if st == "Environment information":
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        geocalculator = ssp.raytracing.BlenderGeometry()
        suite_information = ssp.raytracing.BlenderSuiteFinder().find_suite_information()
        Suite_Position, ScattersGeo, HashFaceIndex_ScattersGeo, ScattersGeoV = geocalculator.get_Position_Velocity(
            bpy.context.scene, suite_information, ssp.config.CurrentFrame, 1
        )

        # Build information text
        text = f'Triangles Number: {len(ScattersGeo)}\n  Suites Number: {len(Suite_Position)} , '
        for i, suite in enumerate(Suite_Position):
            text += f'\nSuite {i+1}: '
            for j, radar in enumerate(suite['Radar']):
                tx_count = len(radar['TX-Position'])
                rx_count = len(radar['RX-Position'])
                text += f'\n  Radar {j+1}: {tx_count} TX, {rx_count} RX, '
        if not hasattr(bpy.types.Scene, "ssp_env_info"):
            bpy.types.Scene.ssp_env_info = bpy.props.StringProperty(
                name="Settings",
                default=text,
            )
        else:
            bpy.context.scene.ssp_env_info = text

        # Popup draw function
        def draw(self, context):
            self.layout.label(text="Environment information:")
            self.layout.prop(context.scene, "ssp_env_info", text="")

        bpy.context.window_manager.popup_menu(draw, title="Sensing Signal Processing", icon='INFO')

    if st == "Load Hub Environment":
        ssp.radar.utils.apps.runHubLoad()
        return
        if not("Load Hub Environment" in bpy.data.objects):
            import os
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(.01, .01, .01))
            ws_axes = bpy.context.object
            ws_axes.name = f'Load Hub Environment'
            metadata = ssp.utils.hub.load_metadata()
            if not metadata:
                return
            ws_axes["1: Auto Next"]  = False
            for category, files in metadata.items():
                category_folder = os.path.join(ssp.config.temp_folder, "hub")
                os.makedirs(category_folder, exist_ok=True)
                category_folder = os.path.join(category_folder, category)
                os.makedirs(category_folder, exist_ok=True)
                for item in files:
                    name = item["name"]
                    local_file_path = os.path.join(category_folder, f"{name}.blend")
                    ws_axes[f"{category}_{name}"]  = False
        else:
            ws_axes = bpy.data.objects["Load Hub Environment"]
            metadata = ssp.utils.hub.load_metadata()
            if not metadata:
                return
            import os
            find = False
            Auto_Next = ws_axes["1: Auto Next"] 
            for category, files in metadata.items():
                if find:
                    break
                category_folder = os.path.join(ssp.config.temp_folder, "hub")
                os.makedirs(category_folder, exist_ok=True)
                category_folder = os.path.join(category_folder, category)
                os.makedirs(category_folder, exist_ok=True)
                for item in files:
                    name = item["name"]
                    if ws_axes[f"{category}_{name}"]  != 0:
                        blend_path = ssp.utils.hub.fetch_file(category, name)
                        find = True
                        break
            find2 = False
            if Auto_Next:
                b = False
                for category, files in metadata.items():
                    for item in files:
                        name = item["name"]
                        if find2:
                            ws_axes[f"{category}_{name}"] = True
                            b = True
                            break
                        if ws_axes[f"{category}_{name}"]  != 0:
                            ws_axes[f"{category}_{name}"] = False
                            find2 = True
                    if b:
                        break
                if find2 == False:
                    b = False
                    for category, files in metadata.items():
                        for item in files:
                            name = item["name"]
                            ws_axes[f"{category}_{name}"] = True
                            b = True
                            break
                        if b:
                            break
                view_layer = bpy.context.view_layer
                bpy.ops.object.select_all(action='DESELECT')
                for obj in bpy.data.objects:
                    if obj.name in view_layer.objects.keys():
                        view_layer.objects.active = obj
                        obj.select_set(True)
                if "Simulation Settings" in bpy.data.objects:
                    bpy.data.objects["Simulation Settings"].select_set(False)
                if "Wifi Sensing Settings" in bpy.data.objects:
                    bpy.data.objects["Wifi Sensing Settings"].select_set(False)
                if "Load Hub Environment" in bpy.data.objects:
                    bpy.data.objects["Load Hub Environment"].select_set(False)
                
                bpy.ops.object.delete()        
                
            if find:
                if os.path.exists(blend_path):
                    ssp.environment.add_blenderfileobjects(blend_path)
                    ssp.utils.define_settings()