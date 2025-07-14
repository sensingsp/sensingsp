import sensingsp as ssp
from ..environment import BlenderSuiteFinder
from ..radar.utils import MIMO_Functions
import os
import sys
import bpy
import numpy as np
from mathutils import Vector
import bmesh
import subprocess
import platform
import time
import matplotlib.pyplot as plt
import cv2
import scipy.io
# def subdivide_object(obj, level=2):
#     if obj is None:
#         return
#     modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
#     modifier.levels = level
#     bpy.context.view_layer.objects.active = obj
#     bpy.ops.object.modifier_apply(modifier="Subdivision")
def export_radar_positions():
    bpy.context.scene.frame_set(ssp.config.CurrentFrame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geocalculator = ssp.raytracing.BlenderGeometry()
    rayTracingFunctions = ssp.raytracing.RayTracingFunctions()
    suite_information = BlenderSuiteFinder().find_suite_information()
    blender_frame_Jump_for_Velocity = 1
    Suite_Position,ScattersGeo,HashFaceIndex_ScattersGeo,ScattersGeoV = geocalculator.get_Position_Velocity(bpy.context.scene, suite_information, ssp.config.CurrentFrame, blender_frame_Jump_for_Velocity)
    return Suite_Position,ScattersGeo
    


def subdivide_object(obj, level=2):
    bpy.ops.object.mode_set(mode='OBJECT')

    # Make the object active and enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Subdivide the object without smoothing
    for _ in range(level):
        bpy.ops.mesh.subdivide(smoothness=0)

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
def initialize_environment():
   ssp.utils.delete_all_objects()
   ssp.utils.define_settings()
def initialize_simulation(startframe=1,endframe=2,RunonGPU=False,rayTracing=2):
  ssp.utils.define_settings()
  ssp.utils.set_frame_start_end(start=startframe,end=endframe)
  ssp.utils.save_Blender()
  ssp.utils.trimUserInputs() 
  ssp.config.restart()  
  ssp.utils.useCUDA(RunonGPU)
  if rayTracing==1:
    ssp.utils.set_RayTracing_light()
  elif rayTracing==2:
    ssp.utils.set_RayTracing_balanced()
  elif rayTracing==3:
    ssp.utils.set_RayTracing_advanced_intense()

def set_RayTracing_advanced_intense():  
  bpy.data.objects["Simulation Settings"]["do RayTracing LOS"] = True
  bpy.data.objects["Simulation Settings"]["do RayTracing Simple"] = False

def setRadar_oneCPI_in_oneFrame(radar):
  radar['continuousCPIsTrue_oneCPIpeerFrameFalse']=False
def setRadar_multipleCPI_in_oneFrame(radar):
  radar['continuousCPIsTrue_oneCPIpeerFrameFalse']=True


def set_configurations(configurations):
  for config in configurations:
    if config == ssp.radar.utils.RadarSignalGenerationConfigurations.RayTracing_Balanced:
      set_RayTracing_balanced()
    elif config == ssp.radar.utils.RadarSignalGenerationConfigurations.RayTracing_Light:
      set_RayTracing_light()
    elif config == ssp.radar.utils.RadarSignalGenerationConfigurations.RayTracing_Advanced:
      set_RayTracing_advanced_intense()
    elif config == ssp.radar.utils.RadarSignalGenerationConfigurations.Spillover_Disabled:
      ssp.config.directReceivefromTX =  False 
      ssp.config.RadarRX_only_fromscatters_itsTX = True
      ssp.config.RadarRX_only_fromitsTX = True
      ssp.config.Radar_TX_RX_isolation = True
    elif config == ssp.radar.utils.RadarSignalGenerationConfigurations.Spillover_Enabled:
      ssp.config.directReceivefromTX =  True
    elif config == ssp.radar.utils.RadarSignalGenerationConfigurations.CUDA_Enabled:
      ssp.utils.useCUDA(True)
    elif config == ssp.radar.utils.RadarSignalGenerationConfigurations.CUDA_Disabled:
      ssp.utils.useCUDA(False)
      
    

def set_RayTracing_balanced():  
  bpy.data.objects["Simulation Settings"]["do RayTracing LOS"] = True
  bpy.data.objects["Simulation Settings"]["do RayTracing Simple"] = True
  
def set_RayTracing_light():  
  bpy.data.objects["Simulation Settings"]["do RayTracing LOS"] = False
  bpy.data.objects["Simulation Settings"]["do RayTracing Simple"] = False

def set_raytracing_bounce(N):
  bpy.data.objects["Simulation Settings"]["Bounce Number"] = N

def set_debugSettings(x):
  bpy.data.objects["Simulation Settings"]["debug Settings"] = x
def get_debugSettings():
  return bpy.data.objects["Simulation Settings"]["debug Settings"] 
        
def tic():
  start_time = time.perf_counter()
  return start_time
def toc(start_time,s=''):
  elapsed_time = time.perf_counter() - start_time
  print(f"Elapsed time: {elapsed_time} seconds {s}")

def pointCloud_RangeAzimuthElevation_THR(RangeAzimuthElevation=None,rangeResolution_maxUnambigiousRange=1,THR=0.05):
  if RangeAzimuthElevation is None:
      return
  X=np.abs(RangeAzimuthElevation)
  max_value = np.max(X)
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111, projection='3d')
  x,y,z,values_above_threshold=[],[],[],[]
  AzimuthElevationRangeindexofMaxValue = np.argmax(X, axis=0)
  for iaz in range(AzimuthElevationRangeindexofMaxValue.shape[0]):
    az = np.arcsin(-1+2*iaz/AzimuthElevationRangeindexofMaxValue.shape[0])
    for iel in range(AzimuthElevationRangeindexofMaxValue.shape[1]):
      el = np.arcsin(-1+2*iel/AzimuthElevationRangeindexofMaxValue.shape[1])
      ir = AzimuthElevationRangeindexofMaxValue[iaz,iel]
      # if X[ir,iaz,iel] > 500*np.mean(X[:,iaz,iel]):
      if X[ir,iaz,iel] > max_value*THR:
        x0,y0,z0=sph2cart(ir*rangeResolution_maxUnambigiousRange, az, el)
        x.append(x0)
        y.append(y0)
        z.append(z0)
        values_above_threshold.append(X[ir,iaz,iel])
  sc = ax.scatter(x, y, z, c=values_above_threshold, cmap='viridis', marker='o')
  plt.colorbar(sc)
  # Make the axes limits equal
  max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
  
  mean_x = np.mean([max(x), min(x)])
  mean_y = np.mean([max(y), min(y)])
  mean_z = np.mean([max(z), min(z)])
  
  ax.set_xlim(mean_x - max_range, mean_x + max_range)
  ax.set_ylim(mean_y - max_range, mean_y + max_range)
  ax.set_zlim(mean_z - max_range, mean_z + max_range)
  plt.show()
  
def azel2uv(azimuth, elevation):
  if np.abs(azimuth) > np.pi/2:
    return None
  u = np.cos(elevation) * np.sin(azimuth)
  v = np.sin(elevation)
  return u, v

def sph2cart(r, azimuth, elevation):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
  
  
def cart2sph(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)  # Compute the radial distance
    azimuth = np.arctan2(y, x)            # Compute the azimuth angle
    elevation = np.arcsin(z / radius)     # Compute the elevation angle
    
    return radius, azimuth, elevation
def imshow(X=None, xlabel='', ylabel='', title=''):
    if X is None:
        return
    plt.figure(figsize=(10, 6))
    plt.imshow(X.T, extent=None, aspect='auto')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.show()
                

def plot(x=None, y=None, option=1, xlabel='', ylabel='', title=''):
    # Initialize x and y to empty lists if None is provided
    if x is None:
        x = []
    if y is None:
        y = []
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Check if y is empty or not
    if not y:  # If y is empty, plot only x
        plt.plot(x)
    else:
        if len(x) == len(y):  # If x and y have the same length, plot x vs y
            plt.plot(x, y)
    
    # Optionally add labels and title
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    # Add grid and show plot
    plt.grid(True)
    plt.show()
def plot_text(s):
  if len(s)==0:
    return
  plt.figure(figsize=(10, 6))
  for i, a in enumerate(s):
      plt.text(0.01, 1-i * 0.1, a, fontsize=12)  # Adjust y-coordinate to start from the top.
  plt.axis('off')                               # Hides the axis for a cleaner look.
  plt.tight_layout()
  plt.show()
def initEnvironment():
  delete_all_objects()
def delete_all_objects():
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
      
    bpy.ops.object.delete()
    ssp.RadarSpecifications = [] 
    ssp.config.restart()
def save_Blender(folder="",file="save_frompython.blend"):
  if folder == "":
    folder = ssp.config.temp_folder

  bpy.ops.wm.save_as_mainfile(filepath = os.path.join(folder, file))
  print(os.path.join(folder, file))
  
def getRadarSpecs():
  return ssp.RadarSpecifications

def useCUDA(value=True):
    bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"] = value
    
def set_frame_start_end(start=1,end=2):
    
    bpy.context.scene.frame_start   =   start
    bpy.context.scene.frame_end     =   end
def increaseCurrentFrame(step=1):
    ssp.config.CurrentFrame += step
    c= sys.executable.lower()
    if "blender" in c:
      bpy.context.view_layer.update()
      bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

def total_paths(path_d_drate_amp):
  N=0
  for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
    for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
      for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
        for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
          for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
            for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
              N+=len(TX_d_drate_amp)
  return N
def force_zeroDoppler_4Simulation(path_d_drate_amp):
  d=[]
  for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
    for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
      for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
        for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
          for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
            for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
              for d_drate_amp in TX_d_drate_amp:
                d_drate_amp[1]=0
                d.append(d_drate_amp[0])
  return np.array(d)
def zeroDopplerCancellation_4Simulation(path_d_drate_amp,attenuation_dB=80):
  attenuation = 10**(-attenuation_dB/20)
  for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
    for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
      for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
        for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
          for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
            for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
              for d_drate_amp in TX_d_drate_amp:
                if d_drate_amp[1]==0:
                  d_drate_amp[2]*=attenuation

def trimUserInputs():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
      if obj.type == 'MESH':
            if "RCS0" in obj:
                obj["RCS0"] = float(obj["RCS0"])
    current_working_directory = os.getcwd()
    if "Simulation Settings" in bpy.data.objects:
        sim_axes = bpy.data.objects["Simulation Settings"]
        RenderBlenderFrames = bpy.data.objects["Simulation Settings"]["Render Blender Frames"]
        video_directory = bpy.data.objects["Simulation Settings"]["Video Directory"]
        open_output_folder = bpy.data.objects["Simulation Settings"]["Open Output Folder"]
    else:
        RenderBlenderFrames = True
        video_directory = current_working_directory
        open_output_folder = True
    RadarSpecifications = []
    suite_information = BlenderSuiteFinder().find_suite_information()
    
    ssp.suite_information = suite_information
    # suite_information= finder.find_suite_information()
    mimo_Functions = MIMO_Functions()
    for isuite,suiteobject in enumerate(suite_information):
      radarSpecifications=[]
      for iradar,radarobject in enumerate(suiteobject['Radar']):
        specifications={}

        # empty["Transmit_Power_dBm"] = 12
        # empty["Center_Frequency_GHz"] = f0/1e9
        # empty['Fs_MHz']=5
        # empty['FMCW_ChirpTime_us'] = 60 automatically set to = N_ADC * Ts

        specifications['PRI']=radarobject['GeneralRadarSpec_Object']['PRI_us']*1e-6
        specifications['Ts']=1e-6/radarobject['GeneralRadarSpec_Object']['Fs_MHz']
        specifications['NPulse'] = radarobject['GeneralRadarSpec_Object']['NPulse']
        specifications['N_ADC']  = radarobject['GeneralRadarSpec_Object']['N_ADC']
        
        specifications['Lambda']=ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9
        specifications['RadarMode']=radarobject['GeneralRadarSpec_Object']['RadarMode']
        specifications['PulseWaveform']=radarobject['GeneralRadarSpec_Object']['PulseWaveform']
        # specifications['FMCW_Bandwidth']=radarobject['GeneralRadarSpec_Object']['FMCW_Bandwidth_GHz']*1e9
        specifications['Tempreture_K']=radarobject['GeneralRadarSpec_Object']['Tempreture_K']
        specifications['FMCW_ChirpSlobe'] = radarobject['GeneralRadarSpec_Object']['FMCW_ChirpSlobe_MHz_usec']*1e12
        # specifications['PrecodingMatrix'] = np.eye(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position']),dtype=np.complex128)
        specifications['M_TX'] = len(radarobject['TX'])
        specifications['N_RX'] = len(radarobject['RX'])
        specifications['MIMO_Tech'] = radarobject['GeneralRadarSpec_Object']['MIMO_Tech']
        specifications['RangeFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['RangeFFT_OverNextP2']
        specifications['Range_Start'] = radarobject['GeneralRadarSpec_Object']['Range_Start']
        specifications['Range_End'] = radarobject['GeneralRadarSpec_Object']['Range_End']
        specifications['CFAR_RD_guard_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_guard_cells']
        specifications['CFAR_RD_training_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_training_cells']
        specifications['CFAR_RD_false_alarm_rate'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_false_alarm_rate']
        specifications['STC_Enabled'] = radarobject['GeneralRadarSpec_Object']['STC_Enabled']
        specifications['MTI_Enabled'] = radarobject['GeneralRadarSpec_Object']['MTI_Enabled']
        specifications['DopplerFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['DopplerFFT_OverNextP2']
        specifications['AzFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['AzFFT_OverNextP2']
        specifications['ElFFT_OverNextP2'] = radarobject['GeneralRadarSpec_Object']['ElFFT_OverNextP2']
        specifications['CFAR_Angle_guard_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_guard_cells']
        specifications['CFAR_Angle_training_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_training_cells']
        specifications['CFAR_Angle_false_alarm_rate'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_false_alarm_rate']
        specifications['CFAR_RD_alpha'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_alpha']
        specifications['CFAR_Angle_alpha'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_alpha']
        
        specifications['PrecodingMatrix'] = mimo_Functions.AD_matrix(NPulse=specifications['NPulse'],
                                                                    M=len(radarobject['TX']),
                                                                    tech=specifications['MIMO_Tech'])
        specifications['DopplerProcessingMIMODemod'] = radarobject['GeneralRadarSpec_Object']['DopplerProcessingMIMODemod']
        
        specifications['ADC_peak2peak'] = radarobject['GeneralRadarSpec_Object']['ADC_peak2peak']
        specifications['ADC_levels'] = radarobject['GeneralRadarSpec_Object']['ADC_levels']
        specifications['ADC_ImpedanceFactor'] = radarobject['GeneralRadarSpec_Object']['ADC_ImpedanceFactor']
        specifications['ADC_LNA_Gain'] = 10**(radarobject['GeneralRadarSpec_Object']['ADC_LNA_Gain_dB']/10)
        specifications['ADC_SaturationEnabled'] = radarobject['GeneralRadarSpec_Object']['ADC_SaturationEnabled']
        specifications['RF_NoiseFiguredB'] = radarobject['GeneralRadarSpec_Object']['RF_NoiseFiguredB']
        specifications['RF_AnalogNoiseFilter_Bandwidth'] = radarobject['GeneralRadarSpec_Object']['RF_AnalogNoiseFilter_Bandwidth_MHz']*1e6
        specifications['MIMO_Antenna_Azimuth_Elevation_Order']=radarobject['GeneralRadarSpec_Object']['antenna2azelIndex']
        specifications['matrix_world']=radarobject['GeneralRadarSpec_Object'].matrix_world.decompose()
        if "ArrayInfofile" in radarobject["GeneralRadarSpec_Object"].keys():
          specifications['ArrayInfofile']=radarobject['GeneralRadarSpec_Object']['ArrayInfofile']
        else:
          specifications['ArrayInfofile']=None
          
        modifyArrayinfowithFile(specifications)
        k=0
        global_location_Center = Vector((0,0,0))
        global_location_TX = []
        for itx,txobj in enumerate(radarobject['TX']):
          global_location, global_rotation, global_scale = txobj.matrix_world.decompose()
          global_location_TX.append(global_location)
          global_location_Center += global_location
          k+=1
        global_location_RX = []
        for irx,rxobj in enumerate(radarobject['RX']):
          global_location, global_rotation, global_scale = rxobj.matrix_world.decompose()
          global_location_RX.append(global_location)
          global_location_Center += global_location
          k+=1
        global_location_Center /= k
        specifications['global_location_TX_RX_Center'] = [global_location_TX,global_location_RX,global_location_Center]
        azindex = []
        elindex = []
        for itx,txobj in enumerate(radarobject['TX']):
          # global_location, global_rotation, global_scale = txobj.matrix_world.decompose()
          local_location, local_rotation, local_scale = txobj.matrix_local.decompose()
          local_location_HW = local_location / (ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9)* 2
          azTx=round(local_location_HW.x)
          elTx=round(local_location_HW.y)
          # print("itx,local_location:",itx,local_location,txobj.name)
          for irx,rxobj in enumerate(radarobject['RX']):
              local_location, local_rotation, local_scale = rxobj.matrix_local.decompose()
              local_location_HW = local_location / (ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9) * 2
              azRx=round(local_location_HW.x)
              elRx=round(local_location_HW.y)
              # print(iradar,azTx+azRx,elRx+elTx)
              azindex.append(azTx+azRx)
              elindex.append(elTx+elRx)
        #       print("irx,local_location:",irx,local_location,rxobj.name)
        # print("azindex:",azindex)
        # print("elindex:",elindex)


        azindex = azindex - np.min(azindex)+1
        elindex = elindex - np.min(elindex)+1
        antennaIndex2VAx = np.zeros((len(radarobject['TX']),len(radarobject['RX'])))
        antennaIndex2VAy = np.zeros((len(radarobject['TX']),len(radarobject['RX'])))
        k=0
        for itx in range(antennaIndex2VAx.shape[0]):
          for irx in range(antennaIndex2VAx.shape[1]):
            antennaIndex2VAx[itx,irx] = azindex[k]-1
            antennaIndex2VAy[itx,irx] = elindex[k]-1
            k+=1

        specifications['MIMO_AntennaIndex2VA']=[antennaIndex2VAx,antennaIndex2VAy,np.max(elindex),np.max(azindex)]
        antenna_Pos0_Wavelength_TX=[]
        for itx,txobj in enumerate(radarobject['TX']):
          local_location, local_rotation, local_scale = txobj.matrix_local.decompose()
          local_location_HW = local_location / (ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9)
          antenna_Pos0_Wavelength_TX.append(local_location_HW)
        antenna_Pos0_Wavelength_RX=[]
        for irx,rxobj in enumerate(radarobject['RX']):
              local_location, local_rotation, local_scale = rxobj.matrix_local.decompose()
              local_location_HW = local_location / (ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9)
              antenna_Pos0_Wavelength_RX.append(local_location_HW)
        specifications['antenna_Pos0_Wavelength']=[antenna_Pos0_Wavelength_TX,antenna_Pos0_Wavelength_RX]


        PosIndex = []
        for itx,txobj in enumerate(radarobject['TX']):
          # global_location, global_rotation, global_scale = txobj.matrix_world.decompose()
          local_location, local_rotation, local_scale = txobj.matrix_local.decompose()
          local_location_HW = local_location / (ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9)* 2
          azTx=local_location_HW.x
          elTx=local_location_HW.y
          # print("itx,local_location:",itx,local_location,txobj.name)
          for irx,rxobj in enumerate(radarobject['RX']):
              local_location, local_rotation, local_scale = rxobj.matrix_local.decompose()
              local_location_HW = local_location / (ssp.constants.LightSpeed/radarobject['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9) * 2
              azRx=local_location_HW.x
              elRx=local_location_HW.y
              PosIndex.append([azTx+azRx,elTx+elRx,itx,irx])
        specifications['Local_location_TXplusRX_Center'] = PosIndex

        # x = np.zeros((np.max(elindex),np.max(azindex)))
        # for itx,txobj in enumerate(radarobject['TX']):
        #   for irx,rxobj in enumerate(radarobject['RX']):
        #     x[int(antennaIndex2VAy[itx,irx]),int(antennaIndex2VAx[itx,irx])]=1

        # print(iradar,azindex,elindex,np.max(azindex),np.max(elindex),x)
        # specifications['RangePulseRX']= np.zeros((specifications['N_ADC'],specifications['NPulse'],len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])),dtype=np.complex128)
        
        specifications['RadarTiming'] = RadarTiming(t_start_radar=radarobject['GeneralRadarSpec_Object']['t_start_radar'], 
                                                    t_start_manual_restart_tx=0,
                                                    t_last_pulse=0.0,
                                                    t_current_pulse=0, 
                                                    pri_sequence=[specifications['PRI']], 
                                                    n_pulse=0,
                                                    n_last_cpi=0)
        specifications['CPI_Buffer']=[]
        specifications['RadarBuffer']=RadarBuffer(specifications['NPulse'])
        specifications['SaveSignalGenerationTime']=radarobject['GeneralRadarSpec_Object']['SaveSignalGenerationTime']
        specifications['MaxRangeScatter']=radarobject['GeneralRadarSpec_Object']['MaxRangeScatter']
        specifications['continuousCPIsTrue_oneCPIpeerFrameFalse']=radarobject['GeneralRadarSpec_Object']['continuousCPIsTrue_oneCPIpeerFrameFalse']

        tx_positions,rx_positions=radarobject['GeneralRadarSpec_Object']["TXRXPos"]
        specifications['TXRXPos']=radarobject['GeneralRadarSpec_Object']["TXRXPos"]
        vainfo = ssp.radar.utils.virtualArray_info(tx_positions,rx_positions)
        specifications['ULA_TXRX_Lx_Ly_NonZ']=vainfo
        
        radarSpecifications.append(specifications)
      RadarSpecifications.append(radarSpecifications)
      

    
    # if os.path.exists('frames'):
    #     shutil.rmtree('frames')
    # os.makedirs('frames')
    ssp.RadarSpecifications = RadarSpecifications

def modifyArrayinfowithFile(specifications):
  if specifications['ArrayInfofile'] == '':
    return
  if os.path.exists(specifications['ArrayInfofile'])==False:
    return
  data = scipy.io.loadmat(specifications['ArrayInfofile'])
  tx_positions   = data.get('tx_positions', None)
  rx_positions   = data.get('rx_positions', None)
  rx_bias =  data.get('rx_bias', None)[0]
  for i in range(len(rx_positions)):
      rx_positions[i] = (rx_positions[i][0]+rx_bias[0], rx_positions[i][1]+rx_bias[1])

  vaorder        = data.get('vaorder', None)
  vaorder2        = data.get('vaorder2', None)
  PrecodingMatrix = data.get('PrecodingMatrix', None)
  scale          = str(data.get('scale', None)[0])
  vaprocessing   = str(data.get('vaprocessing', None)[0])
  AzELscale       = data.get('AzELscale', None)[0]

  id = str(data.get('id', None)[0])
  if len(rx_positions) != specifications['N_RX']:
    return
  if len(tx_positions) != specifications['M_TX']:
    return
  
  specifications['PrecodingMatrix'] = np.tile(PrecodingMatrix, (int(np.ceil(specifications['NPulse'] / PrecodingMatrix.shape[0])), 1))[:specifications['NPulse'], :]
  specifications['vaprocessing']=vaprocessing
  specifications['vaorder']=vaorder
  specifications['vaorder2']=vaorder2
  specifications['AzELscale']=AzELscale
  
    
    
def open_Blend(file):
  bpy.ops.wm.open_mainfile(filepath=file)
def applyDecimate(obj, decimateFactor):
    if obj.data.shape_keys is not None:
        bpy.context.view_layer.objects.active = obj 
        blocks = obj.data.shape_keys.key_blocks
        for ind in reversed(range(len(blocks))):
            obj.active_shape_key_index = ind
            bpy.ops.object.shape_key_remove()
    if len(obj.data.vertices) * decimateFactor > 10: 
        decimate_mod = obj.modifiers.new(type='DECIMATE', name='decimate')
        decimate_mod.ratio = decimateFactor
        decimate_mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier='decimate')

def cleanAllDecimateModifiers(obj):
    for m in obj.modifiers:
        if(m.type=="DECIMATE"):
            obj.modifiers.remove(modifier=m)

def mesh2triangles(mesh):
    out = []
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles: 
        xyz0 = mesh.vertices[tri.vertices[0]].co.to_tuple()
        xyz1 = mesh.vertices[tri.vertices[1]].co.to_tuple()
        xyz2 = mesh.vertices[tri.vertices[2]].co.to_tuple()
        out.append([xyz0,xyz1,xyz2])
    return out
def createMeshesCollection_fromTrianglesRCS0(triangleList,vertex_colors,sigma0=1.0,addMatrial=False):
  bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
  p = bpy.context.object
  k=0
  for mesh in triangleList:
      for triangle in mesh:
          if vertex_colors is None:
            color=1
          else:  
            color = vertex_colors[k]
          # if color<128:
          #     color=0
          # else:
          #     color=255
          mesh = bpy.data.meshes.new(f"Rectangle_Mesh_{k}")
          obj = bpy.data.objects.new(f"Rectangle_{k}", mesh)
          bpy.context.collection.objects.link(obj)
          faces = [(0, 1, 2)]
          mesh.from_pydata(triangle, [], faces)
          mesh.update()
          if addMatrial:
              mat = bpy.data.materials.new(name=f"Material_{k}")
              mat.use_nodes = True
              bsdf = mat.node_tree.nodes["Principled BSDF"]
              mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (color, color, color, 1)  
              if obj.data.materials:
                  obj.data.materials[0] = mat
              else:
                  obj.data.materials.append(mat)
          obj["RCS0"] = color*sigma0
          obj.parent = p
          k+=1

def scale_unit_sphere(x,y,z):        
  x=x-np.mean(x)
  x=x/np.max(x)
  y=y-np.mean(y)
  y=y/np.max(y)
  z=z-np.mean(z)
  z=z/np.max(z)
  return x,y,z
def vectorize_triangleList(triangleList):        
  x,y,z=[],[],[]
  for mesh in triangleList:
      for triangle in mesh:
          v0, v1, v2 = triangle
          x.append((v0[0]+v1[0]+v2[0])/3.0)
          y.append((v0[1]+v1[1]+v2[1])/3.0)
          z.append((v0[2]+v1[2]+v2[2])/3.0)
  x=np.array(x)
  y=np.array(y)
  z=np.array(z)
  return x,y,z
def showTileImages(images):
    if len(images) == 1:
        # Special handling for a single image
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(images[0][0])
        ax.set_title(f'{images[0][1]}, {images[0][2]}')
        ax.axis('off')
    else:
        # General case for multiple images
        N = int(np.ceil(np.sqrt(len(images))))
        M = int(np.ceil(len(images) / N))
        fig, axs = plt.subplots(M, N, figsize=(20, 10))
        for i, ax in enumerate(axs.flat):
            if i < len(images):
                ax.imshow(images[i][0])
                title = f'{images[i][1]}, {images[i][2]}'
                
                ax.set_title(title)
                ax.axis('off')
            else:
                ax.axis('off')
    output_path = os.path.join(ssp.config.temp_folder,'triangles_rendered.png')
    fig.savefig(output_path, dpi=300)
    plt.show()

def renderBlenderTriangles(Triangles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # k=0
    # for obj in Triangles:
    #     for triangle in obj:
    #         k+=1
    #         # if k>100:
    #         #     break
    #         ax.plot([triangle[i][0] for i in [0,1,2,0]], [triangle[i][1] for i in [0,1,2,0]],
    #                 [triangle[i][2] for i in [0,1,2,0]],'k-')

    x,y,z=[],[],[]   
    for obj in Triangles:
        for triangle in obj:
            x.append(triangle[0][0])
            x.append(triangle[1][0])
            x.append(triangle[2][0])
            x.append(triangle[0][0])
            y.append(triangle[0][1])
            y.append(triangle[1][1])
            y.append(triangle[2][1])
            y.append(triangle[0][1])
            z.append(triangle[0][2])
            z.append(triangle[1][2])
            z.append(triangle[2][2])
            z.append(triangle[0][2])
            # if k>100:
            #     break
        
    ax.plot(x,y,z,'-')

    # Set axis limits
    ax.set_aspect('equal', 'box')
    # plt.show()
    # Save the rendered image
    output_path = os.path.join(ssp.config.temp_folder,'triangles_rendered.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    img = plt.imread(output_path)
    return img
  
  
def exportBlenderTriangles():
  frame = ssp.config.CurrentFrame
  out = []
  bpy.context.scene.frame_set(frame)
  bpy.context.view_layer.update()
  for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
      depgraph = bpy.context.evaluated_depsgraph_get()
      bm = bmesh.new()
      bm.verts.ensure_lookup_table()
      bm.from_object(obj, depgraph)
      bm.transform(obj.matrix_world)
      mesh = bpy.data.meshes.new('new_mesh')
      bm.to_mesh(mesh)
      trianglesList=mesh2triangles(mesh)
      out.append(trianglesList)
  return out
def exportBlenderFaceCenters(frame=1):
  out = []
  bpy.context.scene.frame_set(frame)
  bpy.context.view_layer.update()
  depsgraph = bpy.context.evaluated_depsgraph_get()
  for obj in bpy.context.scene.objects:
      if obj.type == 'MESH':
          if obj.name.startswith('Probe_')==False:
              bm = bmesh.new()
              bm.from_object(obj, depsgraph)
              bm.transform(obj.matrix_world)
              for face in bm.faces:
                  face_center = face.calc_center_median()
                  out.append(face_center)
              bm.free()
  return out

def decimate_scene(decimation_ratio = 0.5):
  for obj in bpy.context.selected_objects:
      if obj.type == 'MESH':
          # Add a Decimate modifier
          decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
          decimate_modifier.ratio = decimation_ratio
          
          # Apply the Decimate modifier
          bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
def decimate_scene_all(decimation_ratio = 0.5):
  for obj in bpy.data.objects:
      if obj.type == 'MESH':
          # Set the object as active
          bpy.context.view_layer.objects.active = obj
          obj.select_set(True)

          # Add a decimate modifier
          modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
          modifier.ratio = decimation_ratio

          # Apply the decimate modifier
          bpy.ops.object.modifier_apply(modifier="Decimate")

          # Deselect the object
          obj.select_set(False)

def plot1D(ax,x,xlabel="",ylabel="",title="",ylim=[]):
  ax.cla()
  ax.plot(x)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  if len(ylim)==2:
    ax.set_ylim(ylim)
  ax.grid(True)
def plot2D(ax,x,xlabel="",ylabel="",title=""):
  ax.cla()
  ax.imshow(x)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  

def open_folder(folder_path):
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    # Detect the OS and use the appropriate method to open the folder
    current_os = platform.system()

    if current_os == "Windows":
        # For Windows
        os.startfile(folder_path)
    elif current_os == "Linux":
        # For Linux
        subprocess.run(['xdg-open', folder_path])
    elif current_os == "Darwin":  # For macOS
        subprocess.run(['open', folder_path])
    else:
        print("Unsupported operating system.")


def open_temp_folder():
  open_folder(ssp.config.temp_folder)

def readImage_tempfolder(imagefilename,resize=(400, 120),display=False):
  moonTexture_file = os.path.join(ssp.config.temp_folder, imagefilename)
  image = cv2.imread(moonTexture_file)
  if resize is not None:
    image = cv2.resize(image, resize)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if display:
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  return gray_image
class RadarTiming:
    def __init__(self, t_start_radar: float, t_start_manual_restart_tx: float, t_last_pulse: float,
                 t_current_pulse: float, pri_sequence: list, n_pulse: int, n_last_cpi: int):
        self.t_start_radar = np.float64(t_start_radar)
        self.t_start_manual_restart_tx = np.float64(t_start_manual_restart_tx)
        self.t_last_pulse = np.float64(t_last_pulse)
        self.t_current_pulse = np.float64(t_current_pulse)
        self.pri_sequence = np.array(pri_sequence, dtype=np.float64)
        self.n_pulse = n_pulse
        self.n_last_cpi = np.int64(n_last_cpi)
        self.Frame_PulseTimes = np.array([],dtype=np.float64)
        self.Frame_PulseWidths = np.array([],dtype=np.float64)
        self.Started = False

    def update(self,currentTime,FPS): # [ )
      self.Frame_PulseTimes = np.array([],dtype=np.float64)
      self.Frame_PulseWidths = np.array([],dtype=np.float64)
      T = 1/float(FPS) # 1/24 = 0.04167
      if self.t_start_radar >= currentTime+T:
        return
      if self.Started == False:
        if self.t_start_radar >= currentTime:
          self.Started = True
          self.t_current_pulse = self.t_start_radar
          self.n_pulse = 1
          self.Frame_PulseTimes = np.append(self.Frame_PulseTimes, self.t_current_pulse)
          self.Frame_PulseWidths = np.append(self.Frame_PulseWidths, self.pri_sequence[0])
      if self.Started == False:
        return
        
      while 1:
        k=self.n_pulse-1
        k=0
        CurrentPRI = self.pri_sequence[k]
        if self.t_current_pulse + CurrentPRI >= currentTime+T:
          break
        self.t_current_pulse += CurrentPRI
        self.n_pulse+=1
        self.Frame_PulseTimes = np.append(self.Frame_PulseTimes, self.t_current_pulse)
        self.Frame_PulseWidths = np.append(self.Frame_PulseWidths, self.pri_sequence[0])
        
# class RadarTiming:
#     def __init__(self, t_start_radar: float, pri_sequence: list, cpi_length: int):
#         self.t_start_radar = np.float64(t_start_radar)
#         self.pri_sequence = np.array(pri_sequence, dtype=np.float64)
#         self.cpi_length = cpi_length
#         self.t_current_pulse = t_start_radar
#         self.n_pulse = 0
#         self.Frame_PulseTimes = np.array([], dtype=np.float64)
#         self.Started = False
#         self.pri_index = 0

#     def update(self, currentTime, FPS):
#         self.Frame_PulseTimes = []  # Use list for efficiency during appending
#         T = 1 / float(FPS)  # Frame duration

#         # Check if radar should start
#         if self.t_start_radar >= currentTime + T:
#             return

#         # Initialize radar start if needed
#         if not self.Started:
#             if self.t_start_radar >= currentTime:
#                 self.Started = True
#                 self.t_current_pulse = self.t_start_radar
#                 self.n_pulse = 1
#                 self.Frame_PulseTimes.append(self.t_current_pulse)

#         if not self.Started:
#             return

#         # Process pulses within the current frame
#         while True:
#             current_pri = self.pri_sequence[self.pri_index]
#             if self.t_current_pulse + current_pri >= currentTime + T:
#                 break

#             self.t_current_pulse += current_pri
#             self.n_pulse += 1
#             self.Frame_PulseTimes.append(self.t_current_pulse)

#             # Move to the next PRI, wrapping around if needed
#             self.pri_index = (self.pri_index + 1) % len(self.pri_sequence)

#         # Convert the list to numpy array after appending
#         self.Frame_PulseTimes = np.array(self.Frame_PulseTimes, dtype=np.float64)


class RadarBuffer:
    def __init__(self, cpi_length: int):
        self.cpi_length = cpi_length
        self.buffer = []
        self.remaining_pulses = []

    def buffer_pulses(self, pulses, option=1):
        if option == 1:
            # Buffer all generated pulses
            self.buffer.extend(pulses)
        elif option == 2:
            # Buffer only complete CPIs and store remaining pulses
            total_pulses = self.remaining_pulses + list(pulses)
            while len(total_pulses) >= self.cpi_length:
                complete_cpi = total_pulses[:self.cpi_length]
                self.buffer.append(complete_cpi)
                total_pulses = total_pulses[self.cpi_length:]
            self.remaining_pulses = total_pulses

    def get_buffered_cpis(self):
        return self.buffer

    def get_remaining_pulses(self):
        return self.remaining_pulses


def channels_info(path_d_drate_amp,isuiteRX=0,iradarRX=0,isuiteTX=0,iradarTX=0):
  fig, ax = plt.subplots(2,2)
  for irx  in range(len(path_d_drate_amp[isuiteRX][iradarRX])):
    PathNumber = [len(path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX][itx]) for itx in range(len(path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX]))]
    ax[0,0].plot(PathNumber)
    ax[0,0].set_xlabel('TX index')
    ax[0,0].set_ylabel('Path Number from RXi (legend) ') 
  itx,irx=0,0
  Channel_d_fd_amp=np.array([[float(d),float(dr),float(a)] for d,dr,a,m in path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX][itx]])
  if len(Channel_d_fd_amp)==0:
    plt.show()
    return
  ax[0,1].plot(Channel_d_fd_amp[:,0]/(3e8/1e9),20*np.log10(Channel_d_fd_amp[:,2]),'.')
  ax[0,1].set_xlabel('delay (nsec)')
  ax[0,1].set_ylabel('amp (dB) Tx0RX0') 
  ax[1,1].plot(Channel_d_fd_amp[:,0]/(3e8/1e9),Channel_d_fd_amp[:,1]/0.005,'.')
  ax[1,1].set_xlabel('delay (nsec)')
  ax[1,1].set_ylabel('Doppler (Hz) for 5mm wavelength Tx0RX0') 
  d,dr,a,m = path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX][itx][0]
  all_phase = []
  for irx  in range(len(path_d_drate_amp[isuiteRX][iradarRX])):
    for itx in range(len(path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX])):
      for di,dri,ai,mi in path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX][itx]:
        if mi==m:
          all_phase.append((di-d)/0.005*360)
          break
  ax[1,0].plot(all_phase,'.')
  ax[1,0].plot(sorted(all_phase),'.')
  ax[1,0].set_xlabel('TXRX pair index')
  ax[1,0].set_ylabel('Phase (deg) for 5mm wavelength for scatter 0') 
  plt.show()
  
#   Channel_d_fd_amp=np.array([[float(d),float(dr),float(a)] for d,dr,a,m in path_d_drate_amp[isuiteRX][iradarRX][irx][isuiteTX][iradarTX][itx]])
#   fig, ax = plt.subplots(1,1)
#   ax = fig.add_subplot(1, 1, 1 , projection='3d')
#   ax.scatter(Channel_d_fd_amp[:,0],Channel_d_fd_amp[:,1],Channel_d_fd_amp[:,2])
#   ax.set_xlabel('Distance (m)')
#   ax.set_ylabel('Distance Rate (m/s)') 
#   ax.set_zlabel('Amplitude (v)')
#   # ax.set_box_aspect([1,1,1])  
#   plt.show()
#   # plt.draw()
#   # plt.pause(.1)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale so the units are equal in all directions."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def exportBlenderPolygons(frame=1):
    """
    Export all mesh polygons in the Blender scene at a specific frame,
    returning their world-space vertex coordinates.

    Args:
        frame (int): The frame number to evaluate the scene at.

    Returns:
        List[List[List[float]]]: A list of mesh objects, each containing
                                 a list of polygons, each containing
                                 a list of 3D coordinates [x, y, z].
    """
    # Set the scene to the desired frame
    bpy.context.scene.frame_set(frame)
    
    all_meshes = []
    
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        world_matrix = obj.matrix_world
        polygons = []

        for poly in mesh.polygons:
            world_coords = [
                list(world_matrix @ mesh.vertices[vidx].co)
                for vidx in poly.vertices
            ]
            polygons.append(world_coords)

        all_meshes.append(polygons)

        # Important: free memory
        obj.to_mesh_clear()
    
    return all_meshes




# def exportBlenderTriangles():
#   frame = ssp.config.CurrentFrame
#   out = []
#   bpy.context.scene.frame_set(frame)
#   bpy.context.view_layer.update()
#   for obj in bpy.context.scene.objects:
#     if obj.type == 'MESH':
#       depgraph = bpy.context.evaluated_depsgraph_get()
#       bm = bmesh.new()
#       bm.verts.ensure_lookup_table()
#       bm.from_object(obj, depgraph)
#       bm.transform(obj.matrix_world)
#       mesh = bpy.data.meshes.new('new_mesh')
#       bm.to_mesh(mesh)
#       trianglesList=mesh2triangles(mesh)
#       out.append(trianglesList)
#   return out