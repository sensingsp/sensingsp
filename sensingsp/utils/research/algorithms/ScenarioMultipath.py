import sensingsp as ssp
import numpy as np
from matplotlib import pyplot as plt

def runSimpleScenario():
    downloaded_file = ssp.utils.hub.fetch_file("environments", "CarGuardRail")
    ssp.utils.initialize_environment()
    ssp.environment.add_blenderfileobjects(downloaded_file,RCS0=100)
    ssp.environment.setRCSToMaterial("car",1)
    SensorType = ssp.radar.utils.RadarSensorsCategory.ULA_SameTXRX
    radar = ssp.radar.utils.addRadar( radarSensor = SensorType, location_xyz=[0, 0, .5])

    ssp.radar.utils.printAntennaInfp(radar)

    azBW = 80
    elBW = 20
    gaindB = 6
    radar['Transmit_Power_dBm']=18
    radar["Transmit_Antenna_Element_Pattern"] = "NotOmni"
    radar["Transmit_Antenna_Element_Gain_db"] = gaindB
    radar["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = azBW
    radar["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = elBW
    radar["Receive_Antenna_Element_Gain_db"] = gaindB
    radar["Receive_Antenna_Element_Pattern"] = "NotOmni"
    radar["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = azBW
    radar["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = elBW


    ssp.utils.set_RayTracing_advanced_intense()
    ssp.utils.set_raytracing_bounce(3)
    ssp.utils.set_frame_start_end(start=1,end=2)


    ssp.utils.save_Blender()


    ssp.utils.trimUserInputs()


    s = ssp.RadarSpecifications[0][0]
    ssp.config.restart()
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        ssp.visualization.visualize_radar_path_d_drate_amp(path_d_drate_amp)
        print(f'Processed frame = {ssp.config.CurrentFrame}')
        ssp.utils.increaseCurrentFrame()
