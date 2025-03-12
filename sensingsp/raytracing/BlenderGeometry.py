import bpy
import numpy as np
from mathutils import Vector, Euler
import bmesh
import sensingsp as ssp
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r
def azel_fromRotMatrix_dir(Mat,Dir):
    inv = Mat.inverted()
    A=Euler(np.radians((90,0,-90)), 'XYZ').to_matrix()
    dir = inv @ Dir
    dir = A @ dir
    az, el, r = cart2sph(dir.x, dir.y, dir.z)
    return az , el

class BlenderGeometry:
    def get_Target_Position(self, scene, depsgraph):
        face_center_all = []
        HashFaceIndex_ScattersGeo={}
        for obj in scene.objects:
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
                    Backscatter_dev = obj["Backscatter Dev (deg)"]
                    if "SpecularDiffusionFactor" not in obj:
                        obj["SpecularDiffusionFactor"] = 1.0
                    SpecularDiffusionFactor = obj["SpecularDiffusionFactor"]
                    bm = bmesh.new()
                    bm.from_object(obj, depsgraph)
                    bm.transform(obj.matrix_world)
                    obj_hash = hash(obj)
                    HashFaceIndex_ScattersGeo[obj_hash]={}
                    for face in bm.faces:
                        face_center = face.calc_center_median()
                        fn = obj.matrix_world.to_3x3() @ face.normal
                        fn.normalize()
                        face_center_all.append([face_center,obj_hash,face.index,face.calc_area(),fn,RCS0,Backscatter_N,Backscatter_dev,SpecularDiffusionFactor])
                        # print(face_center,face.normal,fn)
                        HashFaceIndex_ScattersGeo[obj_hash][face.index]=len(face_center_all)-1
                    bm.free()
        return face_center_all,HashFaceIndex_ScattersGeo
    def get_Suite_Position(self, suite_information,depsgraph):
        Suite_Position = []
        for suite in suite_information:  # suite_planes[i]['radars'][j]["TXs"]

            suite_info={}

            suite_info['Radar'] = []

            for radar in suite['Radar']:

                txPos = []
                txDir = []
                rxPos = []
                rxDir = []
                txpower = []
                TX_PatternType=[]
                TX_PatternMaxGain=[]
                TX_PatternAzimuthBeamWidth = []
                TX_PatternElevationBeamWidth=[]
                RX_PatternType=[]
                RX_PatternMaxGain=[]
                RX_PatternAzimuthBeamWidth = []
                RX_PatternElevationBeamWidth=[]
                LightSpeed=299792458.0
                WaveLength=LightSpeed/radar['GeneralRadarSpec_Object']["Center_Frequency_GHz"]/1e9
                Pt = 10**((radar['GeneralRadarSpec_Object']['Transmit_Power_dBm']-30)/10)
                Pt /= len(radar['TX'])
                Gt = 10**((radar['GeneralRadarSpec_Object']['Transmit_Antenna_Element_Gain_db'])/10)
                Gr = 10**((radar['GeneralRadarSpec_Object']['Receive_Antenna_Element_Gain_db'])/10)

                for c in radar['TX']:
                    global_location, global_rotation, global_scale = c.matrix_world.decompose()
                    txPos.append(global_location)
                    txDir.append(global_rotation)
                    txpower.append(Pt)
                    TX_PatternType.append(radar['GeneralRadarSpec_Object']["Transmit_Antenna_Element_Pattern"])
                    # TX_PatternType.append("Omni")
                    TX_PatternMaxGain.append(Gt)
                    TX_PatternAzimuthBeamWidth.append(radar['GeneralRadarSpec_Object']["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"])
                    TX_PatternElevationBeamWidth.append(radar['GeneralRadarSpec_Object']["Transmit_Antenna_Element_Elevation_BeamWidth_deg"])
                for c in radar['RX']:
                    global_location, global_rotation, global_scale = c.matrix_world.decompose()
                    rxPos.append(global_location)
                    rxDir.append(global_rotation)
                    # RX_PatternType.append("Omni")
                    RX_PatternType.append(radar['GeneralRadarSpec_Object']["Receive_Antenna_Element_Pattern"])
                    RX_PatternMaxGain.append(Gr)
                    RX_PatternAzimuthBeamWidth.append(radar['GeneralRadarSpec_Object']["Receive_Antenna_Element_Azimuth_BeamWidth_deg"])
                    RX_PatternElevationBeamWidth.append(radar['GeneralRadarSpec_Object']["Receive_Antenna_Element_Elevation_BeamWidth_deg"])


                suite_info['Radar'].append({'TX-Position':txPos,'RX-Position':rxPos,
                                            'TX-Velocity':[],'RX-Velocity':[],
                                            'TX-Direction':txDir,'RX-Direction':rxDir,
                                            'TX-Direction_Next':[],'RX-Direction_Next':[],
                                            'TX-Power':txpower,'TX-PatternType':TX_PatternType,'TX-PatternMaxGain':TX_PatternMaxGain,
                                            'TX-PatternAzimuthBeamWidth':TX_PatternAzimuthBeamWidth,'TX-PatternElevationBeamWidth':TX_PatternElevationBeamWidth,
                                            'RX-PatternType':RX_PatternType,'RX-PatternMaxGain':RX_PatternMaxGain,
                                            'RX-PatternAzimuthBeamWidth':RX_PatternAzimuthBeamWidth,'RX-PatternElevationBeamWidth':RX_PatternElevationBeamWidth,
                                            'WaveLength':WaveLength
                                            })

                    # az,el = azel_fromRotMatrix_dir(TX-Direction,Exact_start-EndRay0)
            suite_info['RIS'] = []
            LightSpeed = 299792458.0
            for ris in suite['RIS']:
                Pos = []
                Dir = []
                PhaseAmp = []
                PatternType=[]
                PatternMaxGain=[]
                PatternAzimuthBeamWidth=[]
                PatternElevationBeamWidth=[]
                WaveLength=LightSpeed/70e9
                for c in ris:
                    global_location, global_rotation, global_scale = c.matrix_world.decompose()
                    Pos.append(global_location)
                    Dir.append(global_rotation)
                    PhaseAmp.append(np.array([c['amplitude'],c['phase']]))
                    PatternType.append("Omni")
                    PatternMaxGain.append(1)
                    PatternAzimuthBeamWidth.append(60)
                    PatternElevationBeamWidth.append(80)

                suite_info['RIS'].append({'Position':Pos,'Velocity':[],'Direction':Dir,'PhaseAmplitude':PhaseAmp,
                                          'PatternType':PatternType,'PatternMaxGain':PatternMaxGain,
                                          'PatternAzimuthBeamWidth':PatternAzimuthBeamWidth,'PatternElevationBeamWidth':PatternElevationBeamWidth,
                                          'WaveLength':WaveLength})
            if ssp.config.myglobal_outsidedefined_RIS is not None:
                suite_info['RIS'] = ssp.config.myglobal_outsidedefined_RIS
                 
            suite_info['Probe'] = []

            # print(suite['Probe'])
            for probe in suite['Probe']:
                Pos = []
                for obj in probe:
                    # print(obj)
                    # obj.hide_viewport = False
                    # probe_points = []
                    # if obj.type == 'MESH':
                    #     object_id = hash(obj.name)
                    #     bm = bmesh.new()
                    #     bm.verts.ensure_lookup_table()
                    #     bm.from_object(obj, depsgraph)
                    #     bm.transform(obj.matrix_world)
                    #     mesh = bpy.data.meshes.new('new_mesh')
                    #     bm.to_mesh(mesh)
                    #     for face in bm.faces:
                    #         # probe_points.append(face.calc_center_median())
                    #         probe_points.append([face.calc_center_median(),[vert.co.copy() for vert in face.verts],0])
                    #     bm.free()
                    # obj.hide_viewport = True
                    global_location, global_rotation, global_scale = obj.matrix_world.decompose()
                    
                    Pos.append(global_location)

                suite_info['Probe'].append(Pos)

            Suite_Position.append(suite_info)
        return Suite_Position

    def get_Position_Velocity(self, scene, suite_information, frame, frame_jump_for_velocity):

        # for isuite,suiteobject in enumerate(suite_information):
        #   for probes in suiteobject['Probe']:
        #     for probe in probes:
        #       probe.hide_viewport = False

        JumpTime = frame_jump_for_velocity / float(scene.render.fps)
        frame_jump_for_velocity = int(frame_jump_for_velocity)
        scene.frame_set(frame+frame_jump_for_velocity)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        face_center_all_next,_ = self.get_Target_Position( scene, depsgraph)
        Suite_Position_next = self.get_Suite_Position(suite_information,depsgraph)

        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        face_center_all,HashFaceIndex_ScattersGeo = self.get_Target_Position( scene, depsgraph)
        Suite_Position = self.get_Suite_Position(suite_information,depsgraph)

        for i,s in enumerate(Suite_Position):
            for j,r in enumerate(s['Radar']):
                Suite_Position[i]['Radar'][j]['TX-Velocity']=[]
                for k,t in enumerate(r['TX-Position']):
                    v = Suite_Position_next[i]['Radar'][j]['TX-Position'][k] - t
                    Suite_Position[i]['Radar'][j]['TX-Velocity'].append(v/JumpTime)
                Suite_Position[i]['Radar'][j]['TX-Direction_Next']=Suite_Position_next[i]['Radar'][j]['TX-Direction']
                Suite_Position[i]['Radar'][j]['RX-Velocity']=[]
                for k,t in enumerate(r['RX-Position']):
                    v = Suite_Position_next[i]['Radar'][j]['RX-Position'][k] - t
                    Suite_Position[i]['Radar'][j]['RX-Velocity'].append(v/JumpTime)
                Suite_Position[i]['Radar'][j]['RX-Direction_Next']=Suite_Position_next[i]['Radar'][j]['RX-Direction']

            for iris , ris in enumerate(Suite_Position[i]['RIS']):
              # Suite_Position[isuite]['RIS'][iris]['Position'][iriselement]
                Suite_Position[i]['RIS'][iris]['Velocity']=[]
                for iriselement ,ris_element_position in enumerate(ris['Position']):
                    v = Suite_Position_next[i]['RIS'][iris]['Position'][iriselement] - ris_element_position
                    Suite_Position[i]['RIS'][iris]['Velocity'].append(v/JumpTime)
        v = []
        for i,c in enumerate(face_center_all):
            v.append((face_center_all_next[i][0] - c[0]) / JumpTime)

        # for isuite,suiteobject in enumerate(suite_information):
        #   for probes in suiteobject['Probe']:
        #     for probe in probes:
        #       probe.hide_viewport = True

        return Suite_Position,face_center_all,HashFaceIndex_ScattersGeo,v


