import bpy
# from .. import constants
from ..constants import *
from ..environment import BlenderSuiteFinder
from .BlenderGeometry import *
from .RayTracingFunctions import *
import sensingsp as ssp
import math
class Vector3D:
    def __init__(self, x, y, z):
        self.x = np.longdouble(x)
        self.y = np.longdouble(y)
        self.z = np.longdouble(z)

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def diff(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __repr__(self):
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
# x = Vector3D(1,2,3)
# print(x)
def Path_RayTracing(progressbar=[]):
    all_info = []
    all_renderImages = []
    all_info_TXRXPos = []
    ssp.lastScatterInfo = []
    
    geocalculator = BlenderGeometry()
    rayTracingFunctions = RayTracingFunctions()


    if "Simulation Settings" in bpy.data.objects:
        sim_axes = bpy.data.objects["Simulation Settings"]

        if "Velocity Calc. Jump" not in sim_axes:
          sim_axes["Velocity Calc. Jump"]=1
        blender_frame_Jump_for_Velocity = sim_axes["Velocity Calc. Jump"]

        BounceNumber = bpy.data.objects["Simulation Settings"]["Bounce Number"]
        RenderBlenderFrames = bpy.data.objects["Simulation Settings"]["Render Blender Frames"]
        render_imade_dir = bpy.data.objects["Simulation Settings"]["Video Directory"]
        open_output_folder = bpy.data.objects["Simulation Settings"]["Open Output Folder"]

    else:
        blender_frame_Jump_for_Velocity = 1
        BounceNumber = 2
        RenderBlenderFrames = False
        render_imade_dir = current_working_directory
        open_output_folder = True

    scene = bpy.context.scene
    suite_information = BlenderSuiteFinder().find_suite_information()

    anything2do = True
    blender_frame_index = scene.frame_start
    while anything2do:
        print(f'frame {blender_frame_index}')
        if blender_frame_index+blender_frame_Jump_for_Velocity>scene.frame_end:
            break
        if not(progressbar==[]):
            progressbar.progress_update(0+50*blender_frame_index/scene.frame_end)
        anything2do = False

        scene.frame_set(blender_frame_index)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        Suite_Position,ScattersGeo,HashFaceIndex_ScattersGeo,ScattersGeoV = geocalculator.get_Position_Velocity(scene, suite_information, blender_frame_index, blender_frame_Jump_for_Velocity)

        ssp.lastScatterInfo.append(ScattersGeo)
        print(f"ScattersGeo {len(ScattersGeo)}")
        # plotscene(Suite_Position,ScattersGeo)

        ## RIS init
        for isuite,suiteobject in enumerate(suite_information):
          if 1: # Zero Phase
            for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
                for iriselement ,ris_element_position in enumerate(ris['Position']):
                    ris['PhaseAmplitude'][iriselement][0] = 1
                    ris['PhaseAmplitude'][iriselement][1] = 0
          else: # Random Phase
            for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
                for iriselement ,ris_element_position in enumerate(ris['Position']):
                    ris['PhaseAmplitude'][iriselement][0] = 1
                    ris['PhaseAmplitude'][iriselement][1] = 2*np.pi*np.random.rand()


        # Probes Reset
        for isuite,suiteobject in enumerate(suite_information):
          for iprobe , probe in enumerate(Suite_Position[isuite]['Probe']):
            for iprobe2 , probe2 in enumerate(probe):
              for iprobev , probev in enumerate(probe2):
                probev[2]=np.longdouble(0)+1j*np.longdouble(0)

        all_d_drate_amp = {}
        for isuite in range(len(suite_information)):
            all_d_drate_amp[isuite] = {}
            for iradar in range(len(suiteobject['Radar'])):
                all_d_drate_amp[isuite][iradar] = {}
                for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                    all_d_drate_amp[isuite][iradar][irx] = {}
                    for isuite2 in range(len(suite_information)):
                        all_d_drate_amp[isuite][iradar][irx][isuite2] = {}
                        for iradar2 in range(len(suiteobject['Radar'])):
                            all_d_drate_amp[isuite][iradar][irx][isuite2][iradar2] = {}
                            for itx in range(len(Suite_Position[isuite2]['Radar'][iradar2]['TX-Position'])):
                                all_d_drate_amp[isuite][iradar][irx][isuite2][iradar2][itx] = []


        for isuite,suiteobject in enumerate(suite_information):
          ## Radar Analysis
          for iradar,radarobject in enumerate(suiteobject['Radar']):
            # Timing should be done
            print(isuite,iradar)
            raysets = []
            for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
              rayset = rayTracingFunctions.rayset_gen_TX(itx,Suite_Position,isuite,iradar,ScattersGeo,depsgraph)
              raysets.append(rayset)

            Paths = []
            # BounceNumber_Radar = 2
            # BounceNumber_RIS = 2
            # bounce = 0
            # while 1:
            #   if

            for bounce in range(BounceNumber):

              raysetsNext = []
              for rayset in raysets:
                rxPath = rayTracingFunctions.RXPath(rayset)
                for path in rxPath:
                  Paths.append(path)
                for _ in rayset.metaInformation['RIS']:
                  rayset_new = rayTracingFunctions.rayset_gen_RIS(_[2],Suite_Position,_[0],_[1],ScattersGeo,depsgraph)
                  rayset_new.source_ray = rayset
                  raysetsNext.append(rayset_new)
                for targetLOS_hitpoint_startreflectedpoints_randomvectors in rayset.metaInformation['Target']:
                  rayset_new = rayTracingFunctions.rayset_gen_Scatter(targetLOS_hitpoint_startreflectedpoints_randomvectors,Suite_Position,isuite,iradar,depsgraph)
                  rayset_new.source_ray = rayset
                  raysetsNext.append(rayset_new)
              raysets = raysetsNext

            for path in Paths:
              amp = 1
              d = np.longdouble(0)
              drate = 0
              # DESIRED FOR EACH PATH: AMP D DRATE

              # if len(path.middle_elements)==0: # Strong Interference + Friis Equation
              #   continue

              isuiteTX = path.transmitter[0]
              iradarTX = path.transmitter[1]
              itx = path.transmitter[2]
              tx = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Position'][itx]
              tx_v = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Velocity'][itx]
              TX = Vector3D(tx.x,tx.y,tx.z)
              Scatters_in_the_Middle = []
              if len(path.middle_elements)==0:

                isuiteRX = path.receiver[0]
                iradarRX = path.receiver[1]
                irx = path.receiver[2]
                next = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Position'][irx]
                Next = Vector3D(next.x,next.y,next.z)
                nextv = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Velocity'][irx]

                dir = next-tx
                distanceLP =  dir.magnitude
                dir/= distanceLP
                # Distance
                d+=Next.distance_to(TX)
                # DistanceRate
                drate += (-tx_v.dot(dir)+nextv.dot(dir))
                # Amplitude 1
                if Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx]=='Omni':
                  gain = 1
                else:
                  az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Direction'][itx],dir)
                  gain = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternMaxGain'][itx]
                  gain*= np.sinc(az/Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternAzimuthBeamWidth'][itx])*np.sinc(el/Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternElevationBeamWidth'][itx])

                Total_Transmit_Power = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Power'][itx]
                PTX = gain * Total_Transmit_Power
                amp*=np.sqrt(PTX)/distanceLP/np.sqrt(4*np.pi)

                # Amplitude 2
                if Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternType'][irx]=='Omni':
                  gain = 1
                else:
                  az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-'][irx],dir)
                  gain = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternMaxGain'][irx]
                  gain*= np.sinc(az/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternAzimuthBeamWidth'][irx])*np.sinc(el/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternElevationBeamWidth'][irx])
                Lambda = Suite_Position[isuiteRX]['Radar'][iradarRX]['WaveLength']
                Aeff = gain * Lambda * Lambda / (4 * np.pi)
                amp*=np.sqrt(Aeff)
                all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append([d,drate,amp,Scatters_in_the_Middle])
                continue
              # find Next for TX (Step 0)
              next_element = path.middle_elements[len(path.middle_elements)-1]
              if next_element.source_type=='scatter':
                next = next_element.ids[1]
                nextv = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
                next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
                Next = Vector3D(next.x,next.y,next.z)
                Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]],next])
              if next_element.source_type=='RIS':
                isuiteRIS = next_element.ids[0]
                iris = next_element.ids[1]
                iriselement = next_element.ids[2]
                next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
                Next = Vector3D(next.x,next.y,next.z)
                nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
              dir = next-tx
              distanceLP =  dir.magnitude
              dir/= distanceLP
              # Distance
              d+=Next.distance_to(TX)
              # DistanceRate
              drate += (-tx_v.dot(dir)+nextv.dot(dir))
              # Amplitude 1
              if Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx]=='Omni':
                gain = 1
              else:
                az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Direction'][itx],dir)
                gain = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternMaxGain'][itx]
                gain*= np.sinc(az/Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternAzimuthBeamWidth'][itx])*np.sinc(el/Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternElevationBeamWidth'][itx])

              Total_Transmit_Power = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Power'][itx]
              PTX = gain * Total_Transmit_Power
              amp*=np.sqrt(PTX)
              amp/=distanceLP*np.sqrt(4*np.pi)
              # print(amp,"1,1",PTX,distanceLP,"np.sqrt(PTX)/distanceLP/np.sqrt(4*np.pi)")
              # Amplitude 2
              if next_element.source_type=='scatter':
                area = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][3]
                RCS0 = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][5]
                area*=np.abs(next_normal.dot(-dir))
                rcs = np.sqrt(area * RCS0)
                amp*=np.sqrt(rcs)
              # print(amp,"1,2",rcs,"RCS")
              if next_element.source_type=='RIS':
                isuiteRIS = next_element.ids[0]
                iris = next_element.ids[1]
                iriselement = next_element.ids[2]
                if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
                  gain = 1
                else:
                  az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
                  gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
                  gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

                Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
                Aeff = gain * Lambda * Lambda / (4 * np.pi)
                amp*=np.sqrt(Aeff)
              # Middle Elements
              for imiddle_elements in range(len(path.middle_elements)-1):
                middle_element = path.middle_elements[len(path.middle_elements)-1-imiddle_elements]
                if middle_element.source_type=='scatter':
                  p = middle_element.ids[1]
                  currentv = ScattersGeoV[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]]
                  currentPoint = Vector3D(p.x,p.y,p.z)
                  normal = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][4]
                  area     = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][3]
                  RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][5]
                  Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]],currentPoint])
                  next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
                  if next_element.source_type=='scatter':
                    next = next_element.ids[1]
                    nextv            = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
                    next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
                    Next = Vector3D(next.x,next.y,next.z)
                    dir = next-p
                    distanceLP =  dir.magnitude
                    dir/= distanceLP
                    # Distance
                    d+=Next.distance_to(currentPoint)
                    # DistanceRate
                    drate += (-currentv.dot(dir)+nextv.dot(dir))
                    # Amplitude 1
                    area*=np.abs(normal.dot(dir))
                    rcs = np.sqrt(area * RCS0)
                    amp*=np.sqrt(rcs)
                    amp/=distanceLP*np.sqrt(4*np.pi)
                    # Amplitude 2
                    area     = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][3]
                    area*=np.abs(next_normal.dot(-dir))
                    RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][5]
                    rcs = np.sqrt(area * RCS0)
                    amp*=np.sqrt(rcs)

                  if next_element.source_type=='RIS':
                    isuiteRIS = next_element.ids[0]
                    iris = next_element.ids[1]
                    iriselement = next_element.ids[2]
                    next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
                    Next = Vector3D(next.x,next.y,next.z)
                    nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
                    dir = next-p
                    distanceLP =  dir.magnitude
                    dir/= distanceLP
                    # Distance
                    d+=Next.distance_to(currentPoint)
                    # DistanceRate
                    drate += (-currentv.dot(dir)+nextv.dot(dir))
                    # Amplitude 1
                    area*=np.abs(normal.dot(dir))
                    rcs = np.sqrt(area * RCS0)
                    amp*=np.sqrt(rcs)
                    amp/=distanceLP*np.sqrt(4*np.pi)
                    # Amplitude 2
                    if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
                      gain = 1
                    else:
                      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
                      gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
                      gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

                    Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
                    Aeff = gain * Lambda * Lambda / (4 * np.pi)
                    amp*=np.sqrt(Aeff)
                if middle_element.source_type=='RIS':
                  isuiteRIS0 = middle_element.ids[0]
                  iris0 = middle_element.ids[1]
                  iriselement0 = middle_element.ids[2]
                  p = Suite_Position[isuiteRIS0]['RIS'][iris0]['Position'][iriselement0]
                  currentPoint = Vector3D(p.x,p.y,p.z)
                  currentv = Suite_Position[isuiteRIS0]['RIS'][iris0]['Velocity'][iriselement0]
                  next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
                  if next_element.source_type=='scatter':
                    next = next_element.ids[1]
                    nextv            = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
                    next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
                    Next = Vector3D(next.x,next.y,next.z)
                    dir = next-p
                    distanceLP =  dir.magnitude
                    dir/= distanceLP
                    # Distance
                    d+=Next.distance_to(currentPoint)
                    # DistanceRate
                    drate += (-currentv.dot(dir)+nextv.dot(dir))
                    # Amplitude 1
                    if Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternType'][iriselement0]=='Omni':
                      gain = 1
                    else:
                      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS0]['RIS'][iris0]['Direction'][iriselement0],dir)
                      gain = Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternMaxGain'][iriselement0]
                      gain*= np.sinc(az/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternAzimuthBeamWidth'][iriselement0])*np.sinc(el/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternElevationBeamWidth'][iriselement0])
                    amp*=np.sqrt(gain)
                    amp/=distanceLP*np.sqrt(4*np.pi)
                    # Amplitude 2
                    area     = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][3]
                    area*=np.abs(next_normal.dot(-dir))
                    RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][5]
                    rcs = np.sqrt(area * RCS0)
                    amp*=np.sqrt(rcs)

                  if next_element.source_type=='RIS':
                    isuiteRIS = next_element.ids[0]
                    iris = next_element.ids[1]
                    iriselement = next_element.ids[2]
                    next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
                    Next = Vector3D(next.x,next.y,next.z)
                    nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
                    dir = next-p
                    distanceLP =  dir.magnitude
                    dir/= distanceLP
                    # Distance
                    d+=Next.distance_to(currentPoint)
                    # DistanceRate
                    drate += (-currentv.dot(dir)+nextv.dot(dir))
                    # Amplitude 1
                    if Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternType'][iriselement0]=='Omni':
                      gain = 1
                    else:
                      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS0]['RIS'][iris0]['Direction'][iriselement0],dir)
                      gain = Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternMaxGain'][iriselement0]
                      gain*= np.sinc(az/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternAzimuthBeamWidth'][iriselement0])*np.sinc(el/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternElevationBeamWidth'][iriselement0])
                    amp*=np.sqrt(gain)
                    amp/=distanceLP*np.sqrt(4*np.pi)
                    # Amplitude 2
                    if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
                      gain = 1
                    else:
                      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
                      gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
                      gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

                    Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
                    Aeff = gain * Lambda * Lambda / (4 * np.pi)
                    amp*=np.sqrt(Aeff)

              # Last
              middle_element = path.middle_elements[0]
              isuiteRX = path.receiver[0]
              iradarRX = path.receiver[1]
              irx = path.receiver[2]
              next = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Position'][irx]
              Next = Vector3D(next.x,next.y,next.z)
              nextv = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Velocity'][irx]
              if middle_element.source_type=='scatter':
                p = middle_element.ids[1]
                currentv = ScattersGeoV[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]]
                currentPoint = Vector3D(p.x,p.y,p.z)
                normal = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][4]
                area     = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][3]
                RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][5]
              if middle_element.source_type=='RIS':
                isuiteRIS0 = middle_element.ids[0]
                iris0 = middle_element.ids[1]
                iriselement0 = middle_element.ids[2]
                p = Suite_Position[isuiteRIS0]['RIS'][iris0]['Position'][iriselement0]
                currentPoint = Vector3D(p.x,p.y,p.z)
                currentv = Suite_Position[isuiteRIS0]['RIS'][iris0]['Velocity'][iriselement0]
                # next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
              dir = next-p
              distanceLP =  dir.magnitude
              dir/= distanceLP
              # Distance
              d+=Next.distance_to(currentPoint)
              # DistanceRate
              drate += (-currentv.dot(dir)+nextv.dot(dir))
              # Amplitude 1
              if middle_element.source_type=='scatter':
                area*=np.abs(normal.dot(dir))
                rcs = np.sqrt(area * RCS0)
                amp*=np.sqrt(rcs)
              # print(amp,"1,3",rcs,"RCS")
              if middle_element.source_type=='RIS':
                if Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternType'][iriselement0]=='Omni':
                  gain = 1
                else:
                  az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS0]['RIS'][iris0]['Direction'][iriselement0],dir)
                  gain = Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternMaxGain'][iriselement0]
                  gain*= np.sinc(az/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternAzimuthBeamWidth'][iriselement0])*np.sinc(el/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternElevationBeamWidth'][iriselement0])
                amp*=np.sqrt(gain)

              amp/=distanceLP*np.sqrt(4*np.pi)
              # Amplitude 2
              if Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternType'][irx]=='Omni':
                gain = 1
              else:
                az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-'][irx],dir)
                gain = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternMaxGain'][irx]
                gain*= np.sinc(az/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternAzimuthBeamWidth'][irx])*np.sinc(el/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternElevationBeamWidth'][irx])
              Lambda = Suite_Position[isuiteRX]['Radar'][iradarRX]['WaveLength']
              Aeff = gain * Lambda * Lambda / (4 * np.pi)
              amp*=np.sqrt(Aeff)
              # print(amp,"1,4",Aeff,gain,Lambda,"gain * Lambda * Lambda / (4 * np.pi)")
              all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append([d,drate,amp,Scatters_in_the_Middle])


        # print(all_d_drate_amp)
        all_info.append(all_d_drate_amp)
        TXRXPos=[]
        for isuite in range(len(suite_information)):
            RXTX=[]
            for iradar in range(len(suiteobject['Radar'])):
                RXTX.append([Suite_Position[isuite]['Radar'][iradar]['TX-Position'],Suite_Position[isuite]['Radar'][iradar]['RX-Position']])
            TXRXPos.append(RXTX)
        all_info_TXRXPos.append(TXRXPos)
        # if RenderBlenderFrames:
        #   suiteims = []
        #   for isuite,suiteobject in enumerate(suite_information):
        #     radims=[]
        #     for iradar,radarobject in enumerate(suiteobject['Radar']):
        #       if len(radarobject['RX'])>0:
        #         bpy.context.scene.camera = radarobject['RX'][0]
        #         bpy.context.scene.render.resolution_x = 640
        #         bpy.context.scene.render.resolution_y = 480
        #         bpy.context.scene.render.resolution_percentage = 100
        #         bpy.context.scene.render.filepath = f"{render_imade_dir}/rendered_image.png"
        #         bpy.ops.render.render(write_still=True)
        #         image = Image.open(bpy.context.scene.render.filepath)
        #         image_np = np.array(image)
        #         radims.append(image_np)
        #     suiteims.append(radims)
        #   all_renderImages.append(suiteims)

        blender_frame_index+=1
        anything2do = True
        # break
    ssp.Paths = all_info
def run():
    empty = bpy.data.objects.get("Detection Cloud")
    if empty:
        # Select all children of the empty object
        for child in empty.children:
            child.select_set(True)
        # Select the empty itself
        empty.select_set(True)
        # Delete selected objects
        bpy.ops.object.delete()
    
    bpy.context.scene.frame_set(ssp.config.CurrentFrame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geocalculator = BlenderGeometry()
    rayTracingFunctions = RayTracingFunctions()
    suite_information = BlenderSuiteFinder().find_suite_information()
    
    if "Simulation Settings" in bpy.data.objects:
        sim_axes = bpy.data.objects["Simulation Settings"]
        if "Velocity Calc. Jump" not in sim_axes:
          sim_axes["Velocity Calc. Jump"]=1
        blender_frame_Jump_for_Velocity = sim_axes["Velocity Calc. Jump"]
        BounceNumber = bpy.data.objects["Simulation Settings"]["Bounce Number"]
        # RenderBlenderFrames = bpy.data.objects["Simulation Settings"]["Render Blender Frames"]
        # render_imade_dir = bpy.data.objects["Simulation Settings"]["Video Directory"]
        do_RayTracing_LOS = bpy.data.objects["Simulation Settings"]["do RayTracing LOS"]
        do_RayTracing_Simple = bpy.data.objects["Simulation Settings"]["do RayTracing Simple"]
    else:
        blender_frame_Jump_for_Velocity = 1
        BounceNumber = 2
        do_RayTracing_LOS = True 
        do_RayTracing_Simple = False
        # RenderBlenderFrames = False
        # render_imade_dir = current_working_directory
    Suite_Position,ScattersGeo,HashFaceIndex_ScattersGeo,ScattersGeoV = geocalculator.get_Position_Velocity(bpy.context.scene, suite_information, ssp.config.CurrentFrame, blender_frame_Jump_for_Velocity)
    # float(bpy.context.scene.render.fps)
    ssp.lastScatterInfo = ScattersGeo
    ssp.lastSuite_Position=Suite_Position
    ## RIS init
    # for isuite,suiteobject in enumerate(suite_information):
    #   if 1: # Zero Phase
    #     for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
    #         for iriselement ,ris_element_position in enumerate(ris['Position']):
    #             ris['PhaseAmplitude'][iriselement][0] = 1
    #             ris['PhaseAmplitude'][iriselement][1] = 0
    #   else: # Random Phase
    #     for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
    #         for iriselement ,ris_element_position in enumerate(ris['Position']):
    #             ris['PhaseAmplitude'][iriselement][0] = 1
    #             ris['PhaseAmplitude'][iriselement][1] = 2*np.pi*np.random.rand()
    # Probes Reset
    for isuite,suiteobject in enumerate(suite_information):
      for iprobe , probe in enumerate(Suite_Position[isuite]['Probe']):
        for iprobe2 , probe2 in enumerate(probe):
          for iprobev , probev in enumerate(probe2):
            probev[2]=np.longdouble(0)+1j*np.longdouble(0)
    all_d_drate_amp = {}
    for isuite in range(len(suite_information)):
        all_d_drate_amp[isuite] = {}
        for iradar in range(len(suiteobject['Radar'])):
            all_d_drate_amp[isuite][iradar] = {}
            for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                all_d_drate_amp[isuite][iradar][irx] = {}
                for isuite2 in range(len(suite_information)):
                    all_d_drate_amp[isuite][iradar][irx][isuite2] = {}
                    for iradar2 in range(len(suiteobject['Radar'])):
                        all_d_drate_amp[isuite][iradar][irx][isuite2][iradar2] = {}
                        for itx in range(len(Suite_Position[isuite2]['Radar'][iradar2]['TX-Position'])):
                            all_d_drate_amp[isuite][iradar][irx][isuite2][iradar2][itx] = []
    for isuite,suiteobject in enumerate(suite_information):
      ## Radar Analysis
      for iradar,radarobject in enumerate(suiteobject['Radar']):
        # Timing should be done
        # print(isuite,iradar)
        Paths = []
        if do_RayTracing_LOS:
          if do_RayTracing_Simple:
            rad = Suite_Position[isuite]['Radar'][iradar]['TX-Position']
            cent = Vector((0,0,0))
            for pos in rad:
              cent+=pos
            cent/=len(rad)
            LOS_target = []
            for target in ScattersGeo:
              if rayTracingFunctions.check_line_of_sight_checkID(cent, target, depsgraph):
                LOS_target.append(target)
              # if rayTracingFunctions.check_line_of_sight(cent, target[0], depsgraph):
              #   LOS_target.append(target)
            for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
              for target in LOS_target: #face_center_all.append([face_center,obj_hash,face.index,face.calc_area(),fn,RCS0])
                for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                    path = Path4WavePropagation(None,None,[])
                    path.transmitter = [isuite,iradar,itx]
                    mid=SourceType("scatter", [[-1,target[1],target[2]],target[0]])
                    path.middle_elements.append(mid)
                    path.receiver = [isuite,iradar,irx]
                    Paths.append(path)
          else:
            raysets = []
            for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
              rayset = rayTracingFunctions.rayset_gen_TX(itx,Suite_Position,isuite,iradar,ScattersGeo,depsgraph)
              raysets.append(rayset)
            for bounce in range(BounceNumber):
              raysetsNext = []
              for rayset in raysets:
                rxPath = rayTracingFunctions.RXPath(rayset)
                for path in rxPath:
                  Paths.append(path)
                for _ in rayset.metaInformation['RIS']:
                  rayset_new = rayTracingFunctions.rayset_gen_RIS(_[2],Suite_Position,_[0],_[1],ScattersGeo,depsgraph)
                  rayset_new.source_ray = rayset
                  raysetsNext.append(rayset_new)
                # Test = len(rayset.metaInformation['Target'])
                # print("Test ",Test)
                for targetLOS_hitpoint_startreflectedpoints_randomvectors in rayset.metaInformation['Target']:
                  rayset_new = rayTracingFunctions.rayset_gen_Scatter(targetLOS_hitpoint_startreflectedpoints_randomvectors,Suite_Position,isuite,iradar,depsgraph)
                  if len(rayset_new.metaInformation['RX'])+len(rayset_new.metaInformation['Target'])+len(rayset_new.metaInformation['RIS'])>0:
                    rayset_new.source_ray = rayset
                    raysetsNext.append(rayset_new)
              raysets = raysetsNext
        else:
          # import sys
          size=0
          for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
            for itarget,target in enumerate(ScattersGeo): #face_center_all.append([face_center,obj_hash,face.index,face.calc_area(),fn,RCS0])
              for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                  path = Path4WavePropagation(None,None,[])
                  path.transmitter = [isuite,iradar,itx]
                  mid=SourceType("scatter", [[-1,target[1],target[2]],target[0]])
                  path.middle_elements.append(mid)
                  path.receiver = [isuite,iradar,irx]
                  Paths.append(path)
          #         size += sys.getsizeof(path)
          # size2 = sys.getsizeof(Paths)

        if 1:
          paths_processing(Paths,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,all_d_drate_amp)
        else:
          for path in Paths:
            amp = np.longdouble(1)
            d = np.longdouble(0)
            drate = 0
            # DESIRED FOR EACH PATH: AMP D DRATE

            # if len(path.middle_elements)==0: # Strong Interference + Friis Equation
            #   continue
            d_drate_amp_Scatters_in_the_Middle,is_end=calc_Path_TX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
            if is_end:
              RXfromTX = ssp.config.directReceivefromTX
              if RXfromTX:
                all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append(d_drate_amp_Scatters_in_the_Middle)
              continue
            d,drate,amp,Scatters_in_the_Middle = d_drate_amp_Scatters_in_the_Middle
            
            # Middle Elements
            d,drate,amp,Scatters_in_the_Middle=calc_Path_Middle(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle)
            
            d,drate,amp=calc_Path_RX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
            
            all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append([d,drate,amp,Scatters_in_the_Middle])
    return all_d_drate_amp
def Path_RayTracing_frame(progressbar=[]):
    empty = bpy.data.objects.get("Detection Cloud")
    if empty:
        # Select all children of the empty object
        for child in empty.children:
            child.select_set(True)
        # Select the empty itself
        empty.select_set(True)
        # Delete selected objects
        bpy.ops.object.delete()
    
    bpy.context.scene.frame_set(ssp.config.CurrentFrame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geocalculator = BlenderGeometry()
    rayTracingFunctions = RayTracingFunctions()
    suite_information = BlenderSuiteFinder().find_suite_information()
    
    if "Simulation Settings" in bpy.data.objects:
        sim_axes = bpy.data.objects["Simulation Settings"]
        if "Velocity Calc. Jump" not in sim_axes:
          sim_axes["Velocity Calc. Jump"]=1
        blender_frame_Jump_for_Velocity = sim_axes["Velocity Calc. Jump"]
        BounceNumber = bpy.data.objects["Simulation Settings"]["Bounce Number"]
        # RenderBlenderFrames = bpy.data.objects["Simulation Settings"]["Render Blender Frames"]
        # render_imade_dir = bpy.data.objects["Simulation Settings"]["Video Directory"]
        do_RayTracing_LOS = bpy.data.objects["Simulation Settings"]["do RayTracing LOS"]
        do_RayTracing_Simple = bpy.data.objects["Simulation Settings"]["do RayTracing Simple"]
    else:
        blender_frame_Jump_for_Velocity = 1
        BounceNumber = 2
        do_RayTracing_LOS = True 
        do_RayTracing_Simple = False
        # RenderBlenderFrames = False
        # render_imade_dir = current_working_directory
    Suite_Position,ScattersGeo,HashFaceIndex_ScattersGeo,ScattersGeoV = geocalculator.get_Position_Velocity(bpy.context.scene, suite_information, ssp.config.CurrentFrame, blender_frame_Jump_for_Velocity)
    # float(bpy.context.scene.render.fps)
    ssp.lastScatterInfo = ScattersGeo
    ssp.lastSuite_Position=Suite_Position
    ## RIS init
    # for isuite,suiteobject in enumerate(suite_information):
    #   if 1: # Zero Phase
    #     for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
    #         for iriselement ,ris_element_position in enumerate(ris['Position']):
    #             ris['PhaseAmplitude'][iriselement][0] = 1
    #             ris['PhaseAmplitude'][iriselement][1] = 0
    #   else: # Random Phase
    #     for iris , ris in enumerate(Suite_Position[isuite]['RIS']):
    #         for iriselement ,ris_element_position in enumerate(ris['Position']):
    #             ris['PhaseAmplitude'][iriselement][0] = 1
    #             ris['PhaseAmplitude'][iriselement][1] = 2*np.pi*np.random.rand()
    # Probes Reset
    for isuite,suiteobject in enumerate(suite_information):
      for iprobe , probe in enumerate(Suite_Position[isuite]['Probe']):
        for iprobe2 , probe2 in enumerate(probe):
          for iprobev , probev in enumerate(probe2):
            probev[2]=np.longdouble(0)+1j*np.longdouble(0)
    all_d_drate_amp = {}
    for isuite in range(len(suite_information)):
        all_d_drate_amp[isuite] = {}
        for iradar in range(len(suiteobject['Radar'])):
            all_d_drate_amp[isuite][iradar] = {}
            for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                all_d_drate_amp[isuite][iradar][irx] = {}
                for isuite2 in range(len(suite_information)):
                    all_d_drate_amp[isuite][iradar][irx][isuite2] = {}
                    for iradar2 in range(len(suiteobject['Radar'])):
                        all_d_drate_amp[isuite][iradar][irx][isuite2][iradar2] = {}
                        for itx in range(len(Suite_Position[isuite2]['Radar'][iradar2]['TX-Position'])):
                            all_d_drate_amp[isuite][iradar][irx][isuite2][iradar2][itx] = []
    for isuite,suiteobject in enumerate(suite_information):
      ## Radar Analysis
      for iradar,radarobject in enumerate(suiteobject['Radar']):
        # Timing should be done
        # print(isuite,iradar)
        Paths = []
        if do_RayTracing_LOS:
          if do_RayTracing_Simple:
            rad = Suite_Position[isuite]['Radar'][iradar]['TX-Position']
            cent = Vector((0,0,0))
            for pos in rad:
              cent+=pos
            cent/=len(rad)
            LOS_target = []
            for target in ScattersGeo:
              if rayTracingFunctions.check_line_of_sight_checkID(cent, target, depsgraph):
                LOS_target.append(target)
              # if rayTracingFunctions.check_line_of_sight(cent, target[0], depsgraph):
              #   LOS_target.append(target)
            for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
              for target in LOS_target: #face_center_all.append([face_center,obj_hash,face.index,face.calc_area(),fn,RCS0])
                for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                    path = Path4WavePropagation(None,None,[])
                    path.transmitter = [isuite,iradar,itx]
                    mid=SourceType("scatter", [[-1,target[1],target[2]],target[0]])
                    path.middle_elements.append(mid)
                    path.receiver = [isuite,iradar,irx]
                    Paths.append(path)
          else:
            raysets = []
            for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
              rayset = rayTracingFunctions.rayset_gen_TX(itx,Suite_Position,isuite,iradar,ScattersGeo,depsgraph)
              raysets.append(rayset)
            for bounce in range(BounceNumber):
              raysetsNext = []
              for rayset in raysets:
                rxPath = rayTracingFunctions.RXPath(rayset)
                for path in rxPath:
                  Paths.append(path)
                for _ in rayset.metaInformation['RIS']:
                  rayset_new = rayTracingFunctions.rayset_gen_RIS(_[2],Suite_Position,_[0],_[1],ScattersGeo,depsgraph)
                  rayset_new.source_ray = rayset
                  raysetsNext.append(rayset_new)
                # Test = len(rayset.metaInformation['Target'])
                # print("Test ",Test)
                for targetLOS_hitpoint_startreflectedpoints_randomvectors in rayset.metaInformation['Target']:
                  rayset_new = rayTracingFunctions.rayset_gen_Scatter(targetLOS_hitpoint_startreflectedpoints_randomvectors,Suite_Position,isuite,iradar,depsgraph)
                  if len(rayset_new.metaInformation['RX'])+len(rayset_new.metaInformation['Target'])+len(rayset_new.metaInformation['RIS'])>0:
                    rayset_new.source_ray = rayset
                    raysetsNext.append(rayset_new)
              raysets = raysetsNext
        else:
          # import sys
          size=0
          for itx in range(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position'])):
            for itarget,target in enumerate(ScattersGeo): #face_center_all.append([face_center,obj_hash,face.index,face.calc_area(),fn,RCS0])
              for irx in range(len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])):
                  path = Path4WavePropagation(None,None,[])
                  path.transmitter = [isuite,iradar,itx]
                  mid=SourceType("scatter", [[-1,target[1],target[2]],target[0]])
                  path.middle_elements.append(mid)
                  path.receiver = [isuite,iradar,irx]
                  Paths.append(path)
          #         size += sys.getsizeof(path)
          # size2 = sys.getsizeof(Paths)

        if 1:
          paths_processing(Paths,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,all_d_drate_amp)
        else:
          for path in Paths:
            amp = np.longdouble(1)
            d = np.longdouble(0)
            drate = 0
            # DESIRED FOR EACH PATH: AMP D DRATE

            # if len(path.middle_elements)==0: # Strong Interference + Friis Equation
            #   continue
            d_drate_amp_Scatters_in_the_Middle,is_end=calc_Path_TX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
            if is_end:
              RXfromTX = ssp.config.directReceivefromTX
              if RXfromTX:
                all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append(d_drate_amp_Scatters_in_the_Middle)
              continue
            d,drate,amp,Scatters_in_the_Middle = d_drate_amp_Scatters_in_the_Middle
            
            # Middle Elements
            d,drate,amp,Scatters_in_the_Middle=calc_Path_Middle(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle)
            
            d,drate,amp=calc_Path_RX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
            
            all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append([d,drate,amp,Scatters_in_the_Middle])
    return all_d_drate_amp
def calc_Path_TX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp):
  isuiteTX = path.transmitter[0]
  iradarTX = path.transmitter[1]
  itx = path.transmitter[2]
  tx = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Position'][itx]
  tx_v = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Velocity'][itx]
  TX = Vector3D(tx.x,tx.y,tx.z)
  Scatters_in_the_Middle = []
  if len(path.middle_elements)==0:

    isuiteRX = path.receiver[0]
    iradarRX = path.receiver[1]
    irx = path.receiver[2]
    next = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Position'][irx]
    Next = Vector3D(next.x,next.y,next.z)
    nextv = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Velocity'][irx]
    dir = next-tx
    distanceLP =  dir.magnitude
    dir/= distanceLP
    # Distance
    d+=Next.distance_to(TX)
    # DistanceRate
    drate += (-tx_v.dot(dir)+nextv.dot(dir))
    # Amplitude 1
    if Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx]=='Omni':
      gain = 1
    else:
      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Direction'][itx],dir)
      gain = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternMaxGain'][itx]
      # gain = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternMaxGain'][itx]
    
      gain*=ssp.radar.utils.antenna_gain_from_beamwidth(np.rad2deg(az), np.rad2deg(el), 
                                                  1, Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternAzimuthBeamWidth'][itx],
                                                    Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternElevationBeamWidth'][itx],
                                                    Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx])
      
          
      # gain*= np.sinc(az/Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternAzimuthBeamWidth'][itx])*np.sinc(el/Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternElevationBeamWidth'][itx])

    Total_Transmit_Power = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Power'][itx]
    PTX = gain * Total_Transmit_Power
    amp*=np.sqrt(PTX)/distanceLP/np.sqrt(4*np.pi)

    # Amplitude 2
    if Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternType'][irx]=='Omni':
      gain = 1
    else:
      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Direction'][irx],-dir)
      gain = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternMaxGain'][irx]
      gain*=ssp.radar.utils.antenna_gain_from_beamwidth(np.rad2deg(az), np.rad2deg(el), 
                                                1, Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternAzimuthBeamWidth'][irx],
                                                  Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternElevationBeamWidth'][irx],
                                                  Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternType'][irx])
      # gain*= np.sinc(az/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternAzimuthBeamWidth'][irx])*np.sinc(el/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternElevationBeamWidth'][irx])
    Lambda = Suite_Position[isuiteRX]['Radar'][iradarRX]['WaveLength']
    Aeff = gain * Lambda * Lambda / (4 * np.pi)
    amp*=np.sqrt(Aeff)
    # all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append([d,drate,amp,Scatters_in_the_Middle])
    return [d,drate,amp,Scatters_in_the_Middle],True
  # find Next for TX (Step 0)
  next_element = path.middle_elements[len(path.middle_elements)-1]
  if next_element.source_type=='scatter':
    next = next_element.ids[1]
    nextv = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
    next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
    Next = Vector3D(next.x,next.y,next.z)
    Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]],next])
  if next_element.source_type=='RIS':
    isuiteRIS = next_element.ids[0]
    iris = next_element.ids[1]
    iriselement = next_element.ids[2]
    next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
    Next = Vector3D(next.x,next.y,next.z)
    nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
    
    risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
    risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
    risd *= (math.fmod(risphase,360)/360)/2
    amp*=np.sqrt(np.sqrt(risamp))
    d+=risd
    # print(next_element.ids,next,nextv)
    
  dir = next-tx
  distanceLP =  dir.magnitude
  dir/= distanceLP
  # Distance
  d+=Next.distance_to(TX)
  # DistanceRate
  drate += (-tx_v.dot(dir)+nextv.dot(dir))
  # Amplitude 1
  if Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx]=='Omni':
    gain = 1
  else:
    az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Direction'][itx],dir)
    gain = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternMaxGain'][itx]
    
    gain*=ssp.radar.utils.antenna_gain_from_beamwidth(np.rad2deg(az), np.rad2deg(el), 
                                                1, Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternAzimuthBeamWidth'][itx],
                                                  Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternElevationBeamWidth'][itx],
                                                  Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx])
    
    if 1:
      azNext,elNext = azel_fromRotMatrix_dir(Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Direction_Next'][itx],dir)
      gainNext = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternMaxGain'][itx]
      gainNext*=ssp.radar.utils.antenna_gain_from_beamwidth(np.rad2deg(azNext), np.rad2deg(elNext), 
                                                  1, Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternAzimuthBeamWidth'][itx],
                                                    Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternElevationBeamWidth'][itx],
                                                    Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-PatternType'][itx])

  Total_Transmit_Power = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Power'][itx]
  PTX = gain * Total_Transmit_Power
  amp*=np.sqrt(PTX)
  amp/=distanceLP*np.sqrt(4*np.pi)
  # Amplitude 2
  if next_element.source_type=='scatter':
    area = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][3]
    RCS0 = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][5]
    area*=np.abs(next_normal.dot(-dir))
    rcs = np.sqrt(area * RCS0)
    amp*=np.sqrt(rcs)
  # print(amp,"1,2",rcs,"RCS")
  if next_element.source_type=='RIS':
    isuiteRIS = next_element.ids[0]
    iris = next_element.ids[1]
    iriselement = next_element.ids[2]
    if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
      gain = 1
    else:
      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
      gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
      gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

    Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
    Aeff = gain * Lambda * Lambda / (4 * np.pi)
    amp*=np.sqrt(Aeff)
  
  return [d,drate,amp,Scatters_in_the_Middle],False

def calc_Path_Middle(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle):
  for imiddle_elements in range(len(path.middle_elements)-1):
    middle_element = path.middle_elements[len(path.middle_elements)-1-imiddle_elements]
    if middle_element.source_type=='scatter':
      p = middle_element.ids[1]
      currentv = ScattersGeoV[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]]
      currentPoint = Vector3D(p.x,p.y,p.z)
      normal = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][4]
      area     = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][3]
      RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][5]
      next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
      if next_element.source_type=='scatter':
        next = next_element.ids[1]
        nextv            = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
        next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
        Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]],next])

        Next = Vector3D(next.x,next.y,next.z)
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        area*=np.abs(normal.dot(dir))
        rcs = np.sqrt(area * RCS0)
        amp*=np.sqrt(rcs)
        amp/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        area     = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][3]
        area*=np.abs(next_normal.dot(-dir))
        RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][5]
        rcs = np.sqrt(area * RCS0)
        amp*=np.sqrt(rcs)

      if next_element.source_type=='RIS':
        isuiteRIS = next_element.ids[0]
        iris = next_element.ids[1]
        iriselement = next_element.ids[2]
        next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
        Next = Vector3D(next.x,next.y,next.z)
        nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        area*=np.abs(normal.dot(dir))
        rcs = np.sqrt(area * RCS0)
        amp*=np.sqrt(rcs)
        amp/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
          gain = 1
        else:
          az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
          gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
          gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

        Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
        Aeff = gain * Lambda * Lambda / (4 * np.pi)
        amp*=np.sqrt(Aeff)
        
            
        risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
        risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
        risd *= (math.fmod(risphase,360)/360)
        amp*=np.sqrt(risamp)
        d+=risd
    if middle_element.source_type=='RIS':
      isuiteRIS0 = middle_element.ids[0]
      iris0 = middle_element.ids[1]
      iriselement0 = middle_element.ids[2]
      p = Suite_Position[isuiteRIS0]['RIS'][iris0]['Position'][iriselement0]
      currentPoint = Vector3D(p.x,p.y,p.z)
      currentv = Suite_Position[isuiteRIS0]['RIS'][iris0]['Velocity'][iriselement0]
      next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
      if next_element.source_type=='scatter':
        next = next_element.ids[1]
        nextv            = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
        next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
        Next = Vector3D(next.x,next.y,next.z)
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        if Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternType'][iriselement0]=='Omni':
          gain = 1
        else:
          az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS0]['RIS'][iris0]['Direction'][iriselement0],dir)
          gain = Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternMaxGain'][iriselement0]
          gain*= np.sinc(az/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternAzimuthBeamWidth'][iriselement0])*np.sinc(el/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternElevationBeamWidth'][iriselement0])
        amp*=np.sqrt(gain)
        amp/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        area     = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][3]
        area*=np.abs(next_normal.dot(-dir))
        RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][5]
        rcs = np.sqrt(area * RCS0)
        amp*=np.sqrt(rcs)

      if next_element.source_type=='RIS':
        isuiteRIS = next_element.ids[0]
        iris = next_element.ids[1]
        iriselement = next_element.ids[2]
        next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
        Next = Vector3D(next.x,next.y,next.z)
        nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        if Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternType'][iriselement0]=='Omni':
          gain = 1
        else:
          az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS0]['RIS'][iris0]['Direction'][iriselement0],dir)
          gain = Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternMaxGain'][iriselement0]
          gain*= np.sinc(az/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternAzimuthBeamWidth'][iriselement0])*np.sinc(el/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternElevationBeamWidth'][iriselement0])
        amp*=np.sqrt(gain)
        amp/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
          gain = 1
        else:
          az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
          gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
          gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

        Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
        Aeff = gain * Lambda * Lambda / (4 * np.pi)
        amp*=np.sqrt(Aeff)
            
        risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
        risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
        risd *= (math.fmod(risphase,360)/360)
        amp*=np.sqrt(risamp)
        d+=risd
  return d,drate,amp,Scatters_in_the_Middle



def calc_Path_RX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp):
    middle_element = path.middle_elements[0]
    isuiteRX = path.receiver[0]
    iradarRX = path.receiver[1]
    irx = path.receiver[2]
    next = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Position'][irx]
    Next = Vector3D(next.x,next.y,next.z)
    nextv = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Velocity'][irx]
    if middle_element.source_type=='scatter':
      p = middle_element.ids[1]
      currentv = ScattersGeoV[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]]
      currentPoint = Vector3D(p.x,p.y,p.z)
      normal = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][4]
      area     = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][3]
      RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][5]
    if middle_element.source_type=='RIS':
      isuiteRIS0 = middle_element.ids[0]
      iris0 = middle_element.ids[1]
      iriselement0 = middle_element.ids[2]
      p = Suite_Position[isuiteRIS0]['RIS'][iris0]['Position'][iriselement0]
      currentPoint = Vector3D(p.x,p.y,p.z)
      currentv = Suite_Position[isuiteRIS0]['RIS'][iris0]['Velocity'][iriselement0]
        
      risamp,risphase=Suite_Position[isuiteRIS0]['RIS'][iris0]['PhaseAmplitude'][iriselement0]
      risd=Suite_Position[isuiteRIS0]['RIS'][iris0]['WaveLength']
      risd *= (math.fmod(risphase,360)/360)
      amp*=np.sqrt(risamp)
      d+=risd
      # next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
    dir = next-p
    distanceLP =  dir.magnitude
    dir/= distanceLP
    # Distance
    d+=Next.distance_to(currentPoint)
    # DistanceRate
    drate += (-currentv.dot(dir)+nextv.dot(dir))
    # Amplitude 1
    if middle_element.source_type=='scatter':
      area*=np.abs(normal.dot(dir))
      rcs = np.sqrt(area * RCS0)
      amp*=np.sqrt(rcs)
    # print(amp,"1,3",rcs,"RCS")
    if middle_element.source_type=='RIS':
      if Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternType'][iriselement0]=='Omni':
        gain = 1
      else:
        az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS0]['RIS'][iris0]['Direction'][iriselement0],dir)
        gain = Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternMaxGain'][iriselement0]
        gain*= np.sinc(az/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternAzimuthBeamWidth'][iriselement0])*np.sinc(el/Suite_Position[isuiteRIS0]['RIS'][iris0]['PatternElevationBeamWidth'][iriselement0])
      amp*=np.sqrt(gain)

    amp/=distanceLP*np.sqrt(4*np.pi)
    # Amplitude 2
    if Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternType'][irx]=='Omni':
      gain = 1
    else:
      az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Direction'][irx],-dir)
      gain = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternMaxGain'][irx]
      gain = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternMaxGain'][irx]
      gain*=ssp.radar.utils.antenna_gain_from_beamwidth(np.rad2deg(az), np.rad2deg(el), 
                                                1, Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternAzimuthBeamWidth'][irx],
                                                  Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternElevationBeamWidth'][irx],
                                                  Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternType'][irx])
      
      # gain*= np.sinc(az/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternAzimuthBeamWidth'][irx])*np.sinc(el/Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-PatternElevationBeamWidth'][irx])
    Lambda = Suite_Position[isuiteRX]['Radar'][iradarRX]['WaveLength']
    Aeff = gain * Lambda * Lambda / (4 * np.pi)
    amp*=np.sqrt(Aeff)
    return d,drate,amp


def paths_processing(Paths,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,all_d_drate_amp):
  for path in Paths:
    # if len(path.middle_elements)==3:
    #   1
    amp = np.longdouble(1)
    d = np.longdouble(0)
    drate = 0
    # DESIRED FOR EACH PATH: AMP D DRATE

    # if len(path.middle_elements)==0: # Strong Interference + Friis Equation
    #   continue
    d_drate_amp_Scatters_in_the_Middle,is_end=calc_Path_TX_functional(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
    if is_end:
      RXfromTX = ssp.config.directReceivefromTX
      if RXfromTX:
        if amp==0:
          continue
        all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append(d_drate_amp_Scatters_in_the_Middle)
      continue
    d,drate,amp,Scatters_in_the_Middle = d_drate_amp_Scatters_in_the_Middle
    if amp==0:
      continue
    
    # Middle Elements
    # d,drate,amp,Scatters_in_the_Middle=calc_Path_Middle(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle)
    d,drate,amp,Scatters_in_the_Middle=calc_Path_Middle_functional(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle)
    if amp==0:
      continue
    
    # d,drate,amp=calc_Path_RX(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
    d,drate,amp=calc_Path_RX_functional(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp)
    if amp==0:
      continue
    all_d_drate_amp[path.receiver[0]][path.receiver[1]][path.receiver[2]][path.transmitter[0]][path.transmitter[1]][path.transmitter[2]].append([d,drate,amp,Scatters_in_the_Middle])

def tx2rx(path,Suite_Position,d,drate,amp):
  
  isuiteTX = path.transmitter[0]
  iradarTX = path.transmitter[1]
  itx = path.transmitter[2]
  tx = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Position'][itx]
  tx_v = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Velocity'][itx]
  TX = Vector3D(tx.x,tx.y,tx.z)
  
  isuiteRX = path.receiver[0]
  iradarRX = path.receiver[1]
  irx = path.receiver[2]
  next = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Position'][irx]
  Next = Vector3D(next.x,next.y,next.z)
  nextv = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Velocity'][irx]

  dir = next-tx
  distanceLP =  dir.magnitude
  dir/= distanceLP

  # Distance
  d+=Next.distance_to(TX)
  # DistanceRate
  drate += (-tx_v.dot(dir)+nextv.dot(dir))
  
  amp=tx_amp(distanceLP,Suite_Position[isuiteTX]['Radar'][iradarTX],itx,amp,dir)
  amp=rx_amp(distanceLP,Suite_Position[isuiteRX]['Radar'][iradarRX],irx,amp,-dir)
  return d,drate,amp

def tx2scatter_or_ris(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle):
  isuiteTX = path.transmitter[0]
  iradarTX = path.transmitter[1]
  itx = path.transmitter[2]
  tx = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Position'][itx]
  tx_v = Suite_Position[isuiteTX]['Radar'][iradarTX]['TX-Velocity'][itx]
  TX = Vector3D(tx.x,tx.y,tx.z)
  
  # find Next for TX (Step 0)
  next_element = path.middle_elements[len(path.middle_elements)-1]
  if next_element.source_type=='scatter':
    next = next_element.ids[1]
    nextv = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
    # next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
    Next = Vector3D(next.x,next.y,next.z)
    Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]],next])
  if next_element.source_type=='RIS':
    isuiteRIS = next_element.ids[0]
    iris = next_element.ids[1]
    iriselement = next_element.ids[2]
    next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
    Next = Vector3D(next.x,next.y,next.z)
    nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
    
    risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
    risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
    risd *= (math.fmod(risphase,360)/360)/2
    amp*=np.sqrt(np.sqrt(risamp))
    d+=risd
    # print(next_element.ids,next,nextv)
    Scatters_in_the_Middle.append(f'RIS_{isuiteRIS}_{iris}_{iriselement}')
  
  dir = next-tx
  distanceLP =  dir.magnitude
  dir/= distanceLP
  # Distance
  d+=Next.distance_to(TX)
  # DistanceRate
  drate += (-tx_v.dot(dir)+nextv.dot(dir))
  # Amplitude 1
  amp=tx_amp(distanceLP,Suite_Position[isuiteTX]['Radar'][iradarTX],itx,amp,dir)
  
  # Amplitude 2
  if next_element.source_type=='scatter':
    amp=scatter_amp(distanceLP,ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]],-dir,amp)
  if next_element.source_type=='RIS':
    isuiteRIS = next_element.ids[0]
    iris = next_element.ids[1]
    iriselement = next_element.ids[2]
    amp=rx_amp(distanceLP,Suite_Position[isuiteRIS]['RIS'][iris],iriselement,amp,-dir)
  return d,drate,amp
    # if Suite_Position[isuiteRIS]['RIS'][iris]['PatternType'][iriselement]=='Omni':
    #   gain = 1
    # else:
    #   az,el = azel_fromRotMatrix_dir(Suite_Position[isuiteRIS]['RIS'][iris]['Direction'][iriselement],dir)
    #   gain = Suite_Position[isuiteRIS]['RIS'][iris]['PatternMaxGain'][iriselement]
    #   gain*= np.sinc(az/Suite_Position[isuiteRIS]['RIS'][iris]['PatternAzimuthBeamWidth'][iriselement])*np.sinc(el/Suite_Position[isuiteRIS]['RIS'][iris]['PatternElevationBeamWidth'][iriselement])

    # Lambda = Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
    # Aeff = gain * Lambda * Lambda / (4 * np.pi)
    # amp*=np.sqrt(Aeff)
  #   def rx_amp(radar,irx,amp):
  # gain = antenna_gain(radar['RX-PatternType'][irx],
  #                     radar['RX-Direction'][irx],
  #                     radar['RX-PatternMaxGain'][irx], 
  #                     radar['RX-PatternAzimuthBeamWidth'][irx],
  #                     radar['RX-PatternElevationBeamWidth'][irx],
  #                     radar['RX-PatternType'][irx])
  # Lambda = radar['WaveLength']
  # Aeff = gain * Lambda * Lambda / (4 * np.pi)
  # amp*=np.sqrt(Aeff)


# def scatter_amp(distanceLP,SG,dir,amp):
#   area = SG[3]
#   normal = SG[4]
#   RCS0 = SG[5]
#   area*=np.abs(normal.dot(-dir))
#   rcs = np.sqrt(area * RCS0)
#   co = 1.0/ ( np.sqrt(4*np.pi) * distanceLP )
#   amp*=np.sqrt(co*rcs)
#   return amp
def scatter_amp(distanceLP,SG,dir,amp):
  area = SG[3]
  normal = SG[4]
  RCS0 = SG[5]
  SpecularDiffusionFactor = SG[8]
  area*=np.abs(normal.dot(-dir))
  area_SpecularDiffusionFactor = area ** SpecularDiffusionFactor
  rcs = np.sqrt( area_SpecularDiffusionFactor * RCS0)
  co = 1.0/ ( np.sqrt(4*np.pi) * distanceLP )
  amp*=np.sqrt(co*rcs)
  return amp
def tx_amp(distanceLP,radar,itx,amp,dir):
  gain = antenna_gain(radar['TX-PatternType'][itx],
                      radar['TX-Direction'][itx],
                      radar['TX-PatternMaxGain'][itx], 
                      radar['TX-PatternAzimuthBeamWidth'][itx],
                      radar['TX-PatternElevationBeamWidth'][itx],
                      radar['TX-PatternType'][itx],dir)
  Total_Transmit_Power = radar['TX-Power'][itx]
  PTX = gain * Total_Transmit_Power
  co = 1.0/ ( np.sqrt(4*np.pi) * distanceLP )
  amp*=np.sqrt( co * PTX)
  return amp
def antenna_gain(PatternType,antenna_direction,PatternMaxGain, azimuth_beamwidth, elevation_beamwidth,antennaType,dir):
  if PatternType=='Omni':
    gain = 1
  else:
    az,el = azel_fromRotMatrix_dir(antenna_direction,dir)
    gain=ssp.radar.utils.antenna_gain_from_beamwidth(np.rad2deg(az), np.rad2deg(el),PatternMaxGain, azimuth_beamwidth, elevation_beamwidth,antennaType)
    # gain = np.rad2deg(az)
    # gain = np.abs(az+el)
  return gain
def rx_amp(distanceLP,radar,irx,amp,dir):
  if 'RX-PatternType' in radar:
    gain = antenna_gain(radar['RX-PatternType'][irx],
                      radar['RX-Direction'][irx],
                      radar['RX-PatternMaxGain'][irx], 
                      radar['RX-PatternAzimuthBeamWidth'][irx],
                      radar['RX-PatternElevationBeamWidth'][irx],
                      radar['RX-PatternType'][irx],dir)
  else:
    gain = antenna_gain(radar['PatternType'][irx],
                      radar['Direction'][irx],
                      radar['PatternMaxGain'][irx], 
                      radar['PatternAzimuthBeamWidth'][irx],
                      radar['PatternElevationBeamWidth'][irx],
                      radar['PatternType'][irx],dir)
  Lambda = radar['WaveLength']
  Aeff = gain * Lambda * Lambda / (4 * np.pi)
  co = 1.0/ ( np.sqrt(4*np.pi) * distanceLP )
  amp*=np.sqrt(co *Aeff)
  return amp
def tx_amp_ris(distanceLP,radar,irx,amp,dir):
  gain = antenna_gain(radar['PatternType'][irx],
                      radar['Direction'][irx],
                      radar['PatternMaxGain'][irx], 
                      radar['PatternAzimuthBeamWidth'][irx],
                      radar['PatternElevationBeamWidth'][irx],
                      radar['PatternType'][irx],dir)  
  # co = 1.0/ ( np.sqrt(4*np.pi) * distanceLP )
  amp*=np.sqrt(gain/ np.sqrt(( (4 * np.pi) * distanceLP**2 )))
  # G_ris / np.sqrt(( (4 * np.pi) * R_ris_scatter**2 ))
  return amp

def calc_Path_TX_functional(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp):
  Scatters_in_the_Middle = []
  if len(path.middle_elements)==0:
    d,drate,amp=tx2rx(path,Suite_Position,d,drate,amp)
    return [d,drate,amp,Scatters_in_the_Middle],True
  d,drate,amp=tx2scatter_or_ris(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle)
  return [d,drate,amp,Scatters_in_the_Middle],False

def calc_Path_Middle_functional(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp,Scatters_in_the_Middle):
  for imiddle_elements in range(len(path.middle_elements)-1):
    middle_element = path.middle_elements[len(path.middle_elements)-1-imiddle_elements]
    if middle_element.source_type=='scatter':
      p = middle_element.ids[1]
      currentv = ScattersGeoV[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]]
      currentPoint = Vector3D(p.x,p.y,p.z)
      # normal = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][4]
      # area     = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][3]
      # RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][5]
      next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
      if next_element.source_type=='scatter':
        next = next_element.ids[1]
        nextv            = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
        # next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
        Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]],next])
        Next = Vector3D(next.x,next.y,next.z)
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        amp0=scatter_amp(distanceLP,ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]],dir,1.0)    
        # amp0/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        amp0=scatter_amp(distanceLP,ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]],-dir,amp0)    
        amp *= amp0
      if next_element.source_type=='RIS':
        isuiteRIS = next_element.ids[0]
        iris = next_element.ids[1]
        iriselement = next_element.ids[2]
        Scatters_in_the_Middle.append(f'RIS_{isuiteRIS}_{iris}_{iriselement}')
        next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
        Next = Vector3D(next.x,next.y,next.z)
        nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        amp0=scatter_amp(distanceLP,ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]],dir,1.0)    
        # amp0/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        amp0=rx_amp(distanceLP,Suite_Position[isuiteRIS]['RIS'][iris],iriselement,amp0,-dir)
        amp *= amp0
                #RIS
        risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
        risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
        risd *= (math.fmod(risphase,360)/360)/2
        amp*=np.sqrt(np.sqrt(risamp))
        d+=risd
    if middle_element.source_type=='RIS':
      isuiteRIS0 = middle_element.ids[0]
      iris0 = middle_element.ids[1]
      iriselement0 = middle_element.ids[2]
      p = Suite_Position[isuiteRIS0]['RIS'][iris0]['Position'][iriselement0]
      currentPoint = Vector3D(p.x,p.y,p.z)
      currentv = Suite_Position[isuiteRIS0]['RIS'][iris0]['Velocity'][iriselement0]
      next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
      if next_element.source_type=='scatter':
        next = next_element.ids[1]
        nextv            = ScattersGeoV[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]]
        # next_normal = ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]][4]
        Next = Vector3D(next.x,next.y,next.z)
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        amp0=tx_amp_ris(distanceLP,Suite_Position[isuiteRIS0]['RIS'][iris0],iriselement0,1.0,dir)
        # Amplitude 2
        
        amp0=scatter_amp(distanceLP,ScattersGeo[HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]]],-dir,amp0)
        amp *= amp0
        Scatters_in_the_Middle.append([HashFaceIndex_ScattersGeo[next_element.ids[0][1]][next_element.ids[0][2]],next])
        

      if next_element.source_type=='RIS':
        isuiteRIS = next_element.ids[0]
        iris = next_element.ids[1]
        iriselement = next_element.ids[2]
        Scatters_in_the_Middle.append(f'RIS_{isuiteRIS}_{iris}_{iriselement}')
        
        next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
        Next = Vector3D(next.x,next.y,next.z)
        nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
        dir = next-p
        distanceLP =  dir.magnitude
        dir/= distanceLP
        # Distance
        d+=Next.distance_to(currentPoint)
        # DistanceRate
        drate += (-currentv.dot(dir)+nextv.dot(dir))
        # Amplitude 1
        amp0=tx_amp_ris(distanceLP,Suite_Position[isuiteRIS0]['RIS'][iris0],iriselement0,1.0,dir)
        # amp0/=distanceLP*np.sqrt(4*np.pi)
        # Amplitude 2
        amp0=rx_amp(distanceLP,Suite_Position[isuiteRIS]['RIS'][iris],iriselement,amp0,-dir)
        amp *= amp0
        #RIS    
        risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
        risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
        risd *= (math.fmod(risphase,360)/360)/2
        amp*=np.sqrt(np.sqrt(risamp))
        d+=risd

      #RIS
      risamp,risphase=Suite_Position[isuiteRIS0]['RIS'][iris0]['PhaseAmplitude'][iriselement0]
      risd=Suite_Position[isuiteRIS0]['RIS'][iris0]['WaveLength']
      risd *= (math.fmod(risphase,360)/360)/2
      amp*=np.sqrt(np.sqrt(risamp))
      d+=risd
  
  return d,drate,amp,Scatters_in_the_Middle
  
def calc_Path_RX_functional(path,Suite_Position,ScattersGeoV,HashFaceIndex_ScattersGeo,ScattersGeo,d,drate,amp):
    middle_element = path.middle_elements[0]
    isuiteRX = path.receiver[0]
    iradarRX = path.receiver[1]
    irx = path.receiver[2]
    next = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Position'][irx]
    Next = Vector3D(next.x,next.y,next.z)
    nextv = Suite_Position[isuiteRX]['Radar'][iradarRX]['RX-Velocity'][irx]
    if middle_element.source_type=='scatter':
      p = middle_element.ids[1]
      currentv = ScattersGeoV[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]]
      currentPoint = Vector3D(p.x,p.y,p.z)
      normal = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][4]
      area     = ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][3]
      RCS0 =     ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]][5]
    if middle_element.source_type=='RIS':
      isuiteRIS0 = middle_element.ids[0]
      iris0 = middle_element.ids[1]
      iriselement0 = middle_element.ids[2]
      p = Suite_Position[isuiteRIS0]['RIS'][iris0]['Position'][iriselement0]
      currentPoint = Vector3D(p.x,p.y,p.z)
      currentv = Suite_Position[isuiteRIS0]['RIS'][iris0]['Velocity'][iriselement0]
      risamp,risphase=Suite_Position[isuiteRIS0]['RIS'][iris0]['PhaseAmplitude'][iriselement0]
      risd=Suite_Position[isuiteRIS0]['RIS'][iris0]['WaveLength']
      risd *= (math.fmod(risphase,360)/360)/2
      amp*=np.sqrt(np.sqrt(risamp))
      d+=risd
      # next_element   = path.middle_elements[len(path.middle_elements)-1-imiddle_elements-1]
    dir = next-p
    distanceLP =  dir.magnitude
    dir/= distanceLP
    # Distance
    d+=Next.distance_to(currentPoint)
    # DistanceRate
    drate += (-currentv.dot(dir)+nextv.dot(dir))
    # Amplitude 1
    if middle_element.source_type=='scatter':
      amp=scatter_amp(distanceLP,ScattersGeo[HashFaceIndex_ScattersGeo[middle_element.ids[0][1]][middle_element.ids[0][2]]],dir,amp)
    if middle_element.source_type=='RIS':
      amp=tx_amp_ris(distanceLP,Suite_Position[isuiteRIS0]['RIS'][iris0],iriselement0,amp,dir)
        
      # amp=tx(distanceLP,Suite_Position[isuiteRIS0]['RIS'][iris0],iriselement0,amp)
    # amp/=distanceLP*np.sqrt(4*np.pi)
    
    # Amplitude 2
    amp=rx_amp(distanceLP,Suite_Position[isuiteRX]['Radar'][iradarRX],irx,amp,-dir)
  
    return d,drate,amp

# def path_element_processing(next_element,SG,H,Suite_Position):
#   if next_element.source_type=='scatter':
#     next = next_element.ids[1]
#     nextv = SG[H[next_element.ids[0][1]][next_element.ids[0][2]]]
#     next_normal = SG[H[next_element.ids[0][1]][next_element.ids[0][2]]][4]
#     Next = Vector3D(next.x,next.y,next.z)
#     Scatters_in_the_Middle.append(H[next_element.ids[0][1]][next_element.ids[0][2]])
#   if next_element.source_type=='RIS':
#     isuiteRIS = next_element.ids[0]
#     iris = next_element.ids[1]
#     iriselement = next_element.ids[2]
#     next = Suite_Position[isuiteRIS]['RIS'][iris]['Position'][iriselement]
#     Next = Vector3D(next.x,next.y,next.z)
#     nextv = Suite_Position[isuiteRIS]['RIS'][iris]['Velocity'][iriselement]
    
#     risamp,risphase=Suite_Position[isuiteRIS]['RIS'][iris]['PhaseAmplitude'][iriselement]
#     risd=Suite_Position[isuiteRIS]['RIS'][iris]['WaveLength']
#     risd *= (math.fmod(risphase,360)/360)/2
#     amp*=np.sqrt(np.sqrt(risamp))
#     d+=risd
#     # print(next_element.ids,next,nextv)
  