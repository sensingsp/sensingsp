import sensingsp as ssp
import numpy as np

from matplotlib import pyplot as plt
        
def RayTracing():
    distance = np.array([])
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.run()
        alld = []
        NRX = len(path_d_drate_amp[0][0])
        NTX = len(path_d_drate_amp[0][0][0][0][0])
        distance = np.zeros((NTX,NRX))
        d0 = 0*path_d_drate_amp[0][0][0][0][0][0][0][0]
        for irx in range(NRX):
            for itx in range(NTX):
                for d_drate_amp in path_d_drate_amp[0][0][irx][0][0][itx]:
                    d = d_drate_amp[0]
                    distance[itx,irx]=d-d0
                    break
        break
    return distance 
def tx_rx_Location():
    ssp.utils.trimUserInputs()
    if len(ssp.RadarSpecifications)==0:
        return [],[],[]
    if len(ssp.RadarSpecifications[0])==0:
        return [],[],[]
    radarParameters = ssp.RadarSpecifications[0][0]
    tx = np.array([[v.x,v.y,v.z] for v in radarParameters['global_location_TX_RX_Center'][0]])
    rx = np.array([[v.x,v.y,v.z] for v in radarParameters['global_location_TX_RX_Center'][1]])
    return tx,rx,wavelength(radarParameters)
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
                    continue
                continue
            continue
        ssp.utils.increaseCurrentFrame()
        if len(radarRawData) > 0:
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
    ax.axis('equal')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for itx in range(len(tx)):
        for irx in range(len(rx)):
            ax.scatter(tx[itx,0]+rx[irx,0], tx[itx,1]+rx[irx,1], tx[itx,2]+rx[irx,2], c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis('equal')
    plt.show()
    
    
def RangeProcessing(s_IF_nfast_mslow_prx,radarParameters,plot=True):
    FastTime_axe, SlowTime_axe, RX_axe = 0,1,2
    win = np.hamming(s_IF_nfast_mslow_prx.shape[FastTime_axe])
    s_IF_nfast_mslow_prx *= win[:,np.newaxis, np.newaxis]
    S_IF_krange_mslow_prx = np.fft.fft( s_IF_nfast_mslow_prx , axis = FastTime_axe , n = 5*s_IF_nfast_mslow_prx.shape[FastTime_axe] )    
    ranges = np.arange(0, S_IF_krange_mslow_prx.shape[0]) / S_IF_krange_mslow_prx.shape[0] * ssp.LightSpeed * ssp.utils.research.simplest.ADCrate(radarParameters)/2/ssp.utils.research.simplest.FMCW_slew_rate(radarParameters)
    if plot:
        plt.plot(ranges,np.abs(S_IF_krange_mslow_prx[:,0,0]))
        plt.show()
    return S_IF_krange_mslow_prx,ranges


def DopplerProcessing(S_IF_krange_mslow_prx,radarParameters,plot=True):
    MTX = ssp.utils.research.simplest.txNumber(radarParameters)
    NRX = ssp.utils.research.simplest.rxNumber(radarParameters) 
    Leff = int(S_IF_krange_mslow_prx.shape[1]/MTX)
    S_IF_krange_mslow_pva = S_IF_krange_mslow_prx.reshape(-1, Leff , MTX * NRX)

    
    tech,mat=ssp.utils.research.simplest.MIMORadarTechnique(radarParameters)
    if tech == 'TDM':
        for l in range(Leff):
            for m in range(MTX):
                n = np.arange(NRX)
                S_IF_krange_mslow_pva[:, l, m * NRX + n] = S_IF_krange_mslow_prx[:, l * MTX + m, :]
    elif tech == 'BPM':
        precodemat = mat[:mat.shape[1],:].T 
        for l in range(Leff):
            x = S_IF_krange_mslow_prx[:, l * MTX + np.arange(MTX), :]
            for k in range(x.shape[0]):
                for n in range(x.shape[2]):
                    y = precodemat @ x[k,:,n]
                    m = np.arange(y.shape[0])
                    S_IF_krange_mslow_pva[k, l, m * NRX + n] = y
    else:
        xxxxxxxxxxxxxxx
    # if tech == 'TDM':
    #     S_IF_krange_mslow_pva[:, :, :MTX * NRX] = S_IF_krange_mslow_prx.reshape(
    #         S_IF_krange_mslow_prx.shape[0], Leff, MTX * NRX
    #     )
    # elif tech == 'BPM':
    #     x = S_IF_krange_mslow_prx[:, np.arange(Leff)[:, None] * MTX + np.arange(MTX), :]
    #     y = np.einsum('ij,klmj->klmi', mat, x)
    #     S_IF_krange_mslow_pva[:, :, :MTX * NRX] = y.reshape(
    #         y.shape[0], Leff, MTX * NRX
    #     )
    # else:
    #     xxxxxxxxxxxxxxx
                    
    range_axe, SlowTime_axe, VA_axe = 0,1,2

    nDopplerFFT = 1*S_IF_krange_mslow_pva.shape[SlowTime_axe]
    S_IF_krange_ldoppler_pva = np.fft.fft( S_IF_krange_mslow_pva , n = nDopplerFFT , axis = SlowTime_axe )

    PRF_TDM = ssp.utils.research.simplest.PRF(radarParameters) / MTX
    fd = np.arange(0, S_IF_krange_ldoppler_pva.shape[1]) / S_IF_krange_ldoppler_pva.shape[1] * PRF_TDM 
    v = -fd * ssp.utils.research.simplest.wavelength(radarParameters) / 2
    if plot:
        for i in range(S_IF_krange_ldoppler_pva.shape[0]):
            plt.plot(fd,np.abs(S_IF_krange_ldoppler_pva[i,:,0]))
        plt.figure()
        plt.imshow(np.sum(np.abs(S_IF_krange_ldoppler_pva),axis=VA_axe), aspect='auto', cmap='viridis')
        plt.title('Range Doppler Map')
        plt.show()
    return S_IF_krange_ldoppler_pva,fd,v

def TDM_PhaseCompensation(S_IF_krange_ldoppler_pva,radarParameters,fd):
    tech,mat=ssp.utils.research.simplest.MIMORadarTechnique(radarParameters)
    if tech != 'TDM':
        return S_IF_krange_ldoppler_pva
    MTX = ssp.utils.research.simplest.txNumber(radarParameters)
    NRX = ssp.utils.research.simplest.rxNumber(radarParameters)
    PRF_TDM = ssp.utils.research.simplest.PRF(radarParameters) / MTX

    TDM_MIMO_phase_compensation = np.ones((S_IF_krange_ldoppler_pva.shape[1], S_IF_krange_ldoppler_pva.shape[2]),dtype=S_IF_krange_ldoppler_pva.dtype)
    T = 1 / PRF_TDM / MTX
    for ifd in range(len(fd)):
        for m in range(MTX):
            n = np.arange(NRX)
            fdi = -fd[ifd]
            # fdi = fdix
            TDM_MIMO_phase_compensation[ ifd , m * NRX + n] = np.exp(1j * 2 * np.pi * fdi * m * T)
    S_IF_krange_ldoppler_pva *= TDM_MIMO_phase_compensation[np.newaxis, : , : ]
    return S_IF_krange_ldoppler_pva

# def CFAR2D(S_IF_krange_ldoppler_pva,alpha=5,MaxOnly=True):
#     range_axe, SlowTime_axe, VA_axe = 0,1,2
#     THR = alpha*np.mean(np.abs(S_IF_krange_ldoppler_pva))
#     if MaxOnly:
#         detections = 
#         detectionSignal = S_IF_krange_ldoppler_pva[detections[0],detections[1],:]
#         return detections, detectionSignal    
#     detections = np.where(np.mean(np.abs(S_IF_krange_ldoppler_pva),axis=VA_axe)>THR)
#     detectionSignal = S_IF_krange_ldoppler_pva[detections[0],detections[1],:]
#     return detections, detectionSignal     

def CFAR2D(S_IF_krange_ldoppler_pva, alpha=5, MaxOnly=True):
    range_axe, SlowTime_axe, VA_axe = 0, 1, 2
    
    # Calculate the threshold value
    THR = alpha * np.mean(np.abs(S_IF_krange_ldoppler_pva))

    if MaxOnly:
        # Identify the maximum value location (indices) in the array
        max_idx = np.unravel_index(np.argmax(np.abs(S_IF_krange_ldoppler_pva)), 
                                   S_IF_krange_ldoppler_pva.shape)
        detections = (np.array([max_idx[range_axe]]), np.array([max_idx[SlowTime_axe]]))
        detectionSignal = S_IF_krange_ldoppler_pva[detections[0], detections[1], :]
        return detections, detectionSignal    

    # General case: Apply CFAR thresholding
    detections = np.where(np.mean(np.abs(S_IF_krange_ldoppler_pva), axis=VA_axe) > THR)
    detectionSignal = S_IF_krange_ldoppler_pva[detections[0], detections[1], :]
    
    return detections, detectionSignal

def MIMOArrayOrder(radarParameters):
    p2y,p2z=radarParameters['MIMO_Antenna_Azimuth_Elevation_Order']
    p2y,p2z=list(p2y),list(p2z)
    Ny = max(p2y)+1
    Nz = max(p2z)+1
    return list(p2y),list(p2z),Ny,Nz
def DOAProcessing(S_IF_khatrange_lhatdoppler_pva,radarParameters,plot=True,N_UpSampling=0):
    rangedopplerindex_axe, VA_axe = 0,1
    p2y,p2z,Ny,Nz=MIMOArrayOrder(radarParameters)
    S_IF_khatrange_lhatdoppler_yaz_zel = np.zeros_like(S_IF_khatrange_lhatdoppler_pva).reshape(-1,Ny,Nz)
    S_IF_khatrange_lhatdoppler_yaz_zel[:,p2y,p2z] = S_IF_khatrange_lhatdoppler_pva
    rangedopplerindex_axe , y_axe, z_axe = 0,1,2
    winY = np.hanning(S_IF_khatrange_lhatdoppler_yaz_zel.shape[y_axe])
    winZ = np.hanning(S_IF_khatrange_lhatdoppler_yaz_zel.shape[z_axe])
    
    Py_nAzimuthAngleFFT = 2**N_UpSampling*S_IF_khatrange_lhatdoppler_yaz_zel.shape[y_axe]
    Pz_nElevationAngleFFT = 2**N_UpSampling*S_IF_khatrange_lhatdoppler_yaz_zel.shape[z_axe]
    if Nz==1:
        Pz_nElevationAngleFFT = 1
    if Ny==1:
        Py_nAzimuthAngleFFT = 1
    S_IF_khatrange_lhatdoppler_qazimuth_uelevation = np.fft.fft2( S_IF_khatrange_lhatdoppler_yaz_zel, s=(Py_nAzimuthAngleFFT, Pz_nElevationAngleFFT), axes=(y_axe, z_axe) )
    S_IF_khatrange_lhatdoppler_qazimuth_uelevation = np.fft.fftshift(S_IF_khatrange_lhatdoppler_qazimuth_uelevation,axes=(y_axe, z_axe) )
    ky_index = np.arange(-Py_nAzimuthAngleFFT//2, Py_nAzimuthAngleFFT//2 + (Py_nAzimuthAngleFFT % 2))
    kz_index = np.arange(-Pz_nElevationAngleFFT//2, Pz_nElevationAngleFFT//2 + (Pz_nElevationAngleFFT % 2))
    if Nz==1:
        kz_index = np.arange(1)
    if Ny==1:
        ky_index = np.arange(1)
    KY, KZ = np.meshgrid(ky_index, kz_index, indexing='xy')
    Lambda = wavelength(radarParameters)
    dy = .5 * Lambda
    dz = .5 * Lambda
    cosphi_sintheta_est = (KY / Py_nAzimuthAngleFFT) * (Lambda / dy)
    sinphi_est          = (KZ / Pz_nElevationAngleFFT) * (Lambda / dz)
    sinphi_est_clipped = np.clip(sinphi_est, -1.0, 1.0)
    phi_est = np.arcsin(sinphi_est_clipped)
    cosphi_est = np.cos(phi_est)
    cosphi_est[ np.abs(cosphi_est) < 1e-12 ] = 1e-12
    sin_theta_est = cosphi_sintheta_est / cosphi_est
    sin_theta_est_clipped = np.clip(sin_theta_est, -1.0, 1.0)
    theta_est = np.arcsin(sin_theta_est_clipped)
    azimuth = np.rad2deg(theta_est)
    elevation = -np.rad2deg(phi_est) 
    
    # fy = np.arange(Py_nAzimuthAngleFFT)/Py_nAzimuthAngleFFT / dy   # [0 , 1 / dy] or   [0, K / Lambda]
    # fz = np.arange(Pz_nElevationAngleFFT)/Pz_nElevationAngleFFT / dz   # [0 , 1 / dz] or   [0, Q / Lambda]
    # # a = 2*pi/Lambda
    # # kz = sqrt(a**2-kx**2-ky**2)
    # # 2*pi*fz = sqrt(a**2-(2*pi*kx)**2-(2*pi*ky)**2)
    # fy = fy[:, np.newaxis]  
    # fz = fz[np.newaxis, :]  
    # cos_el = Lambda * fz
    # sin_az = fy / np.sqrt( (1/Lambda)**2 - fz**2 )
    # azimuth = np.arcsin( sin_az )
    # elevation_zangle = np.arccos( cos_el )
    # elevation = np.pi/2 - elevation_zangle
    if plot:
        N = S_IF_khatrange_lhatdoppler_qazimuth_uelevation.shape[0]
        M = int(np.ceil(np.sqrt(N)))
        M2=int(np.ceil(N/M))
        fig, axes = plt.subplots(M, M2, figsize=(8, 10))
        if M*M2>1:
            axes=axes.flatten()
        for i in range(N):
            x=np.abs(S_IF_khatrange_lhatdoppler_qazimuth_uelevation[i,:,:]).T
            if M*M2>1:
                ax=axes[i]
            else:
                ax=axes
            if Nz==1:
                ax.plot(x[0,:])
            else:
                ax.imshow(x)
            # c = ax.pcolormesh(azimuth,elevation, x , cmap='jet', shading='auto')
            
            # ax.set_xlabel(r'$\theta$ (deg)')
            # ax.set_ylabel(r'$\phi$ (deg)')
            # ax.set_title('2D FFT Magnitude (dB)')
            # fig.colorbar(c, ax=ax, label='Magnitude (dB)')

            # plt.tight_layout()
            # axes[i].imshow(np.abs(S_IF_khatrange_lhatdoppler_qazimuth_uelevation[i,:,:]).T)
        plt.show()
    return S_IF_khatrange_lhatdoppler_qazimuth_uelevation,azimuth,elevation
def CFAR2DAngle(S_IF_khatrange_lhatdoppler_qazimuth_uelevation, alpha=5, MaxOnly=True):
    IndexrangeDoppler_axe, azimuth_axe, elevation_axe = 0, 1, 2
    detections = []
    for i in range(S_IF_khatrange_lhatdoppler_qazimuth_uelevation.shape[IndexrangeDoppler_axe]):
        x = S_IF_khatrange_lhatdoppler_qazimuth_uelevation[i, :, :]
        THR = alpha * np.mean(np.abs(x))  # Calculate threshold
        
        if MaxOnly:
            max_idx = np.argmax(np.abs(x))  # Get the flat index of the max value
            max_value = np.abs(x).flatten()[max_idx]  # Find its value
            if max_value > THR:  
                ind = np.unravel_index(max_idx, x.shape)
                detections.append(([ind[0]],[ind[1]])) 
            else:
                detections.append(([],[]))
        else:
            # Find all indices where the condition is met
            detections.append(np.where(np.abs(x) > THR))
    
    return detections

def printDetections(detectionsBF,detections,ranges,dopplers,azimuths,elevations):
    s=''
    for i,d in enumerate(detectionsBF):
        r = ranges[detections[0][i]]
        fd = dopplers[detections[1][i]]
        for j in range(len(d[0])):
            az = azimuths[d[1][j],d[0][j]]
            el = elevations[d[1][j],d[0][j]]
            si=f'detection {i}: range={r:.2f},doppler={fd:.1f},azimuth={az:.2f} deg, elevation={el:.2f} deg'
            print(si)  
            # s+=si  
            # s+='\n'
    # ssp.utils.plot_text(s)

def MIMORadarTechnique(radarParameters):
    return radarParameters['MIMO_Tech'],radarParameters['PrecodingMatrix']

def MIMORadarTechniqueObj(radar):
    return radar['MIMO_Tech']

def setMIMO(radar , inputMIMORadarTechnique):
    if inputMIMORadarTechnique == ssp.radar.utils.MIMORadarTechnique.TDM:
        radar['MIMO_Tech']='TDM'
    if inputMIMORadarTechnique == ssp.radar.utils.MIMORadarTechnique.BPM:
        radar['MIMO_Tech']='BPM'
    # elif 