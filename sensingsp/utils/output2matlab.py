import numpy as np
import scipy.io as sio
import sensingsp as ssp
import os
def savemat_in_tmpfolder(filename,data_to_save):
    savefilename = os.path.join(ssp.config.temp_folder,filename)
    sio.savemat(savefilename, data_to_save)
def file_in_tmpfolder(filename):
    return os.path.join(ssp.config.temp_folder,filename)
def saveMatFile(ProcessingOutputs):
    grid_points , grid_velocities , all_outputs = ProcessingOutputs
    data_to_save = {
        'grid_points': grid_points,
        'grid_velocities': grid_velocities,
        'all_outputs': all_outputs,
        # 'RadarSpecifications': ssp.RadarSpecifications,
        # 'ScatterInfo': ssp.lastScatterInfo
    }
    
    # sio.savemat('simulation_data.mat', data_to_save)

def saveScenario():
    P = []
    for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
        S = []        
        for iradar,specifications in enumerate(radarSpecifications):
            global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
            S.append([np.array(global_location_TX),np.array(global_location_RX)])
        P.append(S)
    P_array = np.array(P, dtype=object)

    data_to_save = {
        'Triangles': ssp.utils.exportBlenderTriangles(),
        'ArrayLocation': P_array
    }
    sio.savemat(f'scenario_data_{ssp.config.CurrentFrame}.mat', data_to_save)
    