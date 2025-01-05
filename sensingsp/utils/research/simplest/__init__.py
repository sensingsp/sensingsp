import sensingsp as ssp
import numpy as np
def RayTracing():
    distance = np.array([])
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.run()
        alld = []
        NRX = len(path_d_drate_amp[0][0])
        NTX = len(path_d_drate_amp[0][0][0][0][0])
        distance = np.zeros((NTX,NRX))
        d0 = path_d_drate_amp[0][0][0][0][0][0][0][0]
        for irx in range(NRX):
            for itx in range(NTX):
                for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                    d = d_drate_amp[0]
                    distance[itx,irx]=d-d0
                    break
        break
    return distance 
def RadarRawData():
    radarRawData = np.array([])
    radarParameters = []
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.run()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        for isuite, radarSpecifications in enumerate(ssp.RadarSpecifications):
            for iradar, specifications in enumerate(radarSpecifications):
                for XRadar, timeX in Signals[isuite]['radars'][iradar]:
                    radarRawData = XRadar
                    radarParameters = specifications
                    break
                break
            break
        break
    return radarRawData, radarParameters

def txNumber(radarParameters):
    return len(radarParameters['global_location_TX_RX_Center'][0])
def rxNumber(radarParameters):
    return len(radarParameters['global_location_TX_RX_Center'][1])
def PRF(radarParameters):
    return 1.0/radarParameters['PRI']
def ADCrate(radarParameters):
    return 1.0/radarParameters['Ts']
def FMCW_slew_rate(radarParameters):
    return radarParameters['FMCW_ChirpSlobe']

def wavelength(radarParameters):
    return radarParameters['Lambda']

def txrxplot(radarParameters):
    tx = np.array([[v.x,v.y,v.z] for v in radarParameters['global_location_TX_RX_Center'][0]])
    rx = np.array([[v.x,v.y,v.z] for v in radarParameters['global_location_TX_RX_Center'][1]])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tx[:,0], tx[:,1], tx[:,2], c='r', marker='o')
    for txt in range(len(tx)):
        ax.text(tx[txt,0], tx[txt,1], tx[txt,2], f'TX{txt+1}', color='black')
    for txt in range(len(rx)):
        ax.text(rx[txt,0], rx[txt,1], rx[txt,2], f'RX{txt+1}', color='black')
    ax.scatter(rx[:,0], rx[:,1], rx[:,2], c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()