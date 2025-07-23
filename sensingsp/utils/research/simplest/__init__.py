import sensingsp as ssp
import numpy as np

from matplotlib import pyplot as plt
import scipy
import os
from mathutils import Vector
import bpy
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

def ui_frame_processing(QtApp):
    for a in QtApp.axes:
        a.cla()
    if ssp.config.CurrentFrame + 1 > ssp.config.get_frame_end():
        QtApp.statusBar().showMessage("No more frames to process.")
        ssp.radar.utils.apps.process_events()
        return
    if QtApp.combo2.count()==0:
        return
    if QtApp.combo.currentText()==["RD-Det-Ang-Det", "VitalSign"][1]:
        QtApp.vs_processing()
        return
    isuite = int(QtApp.combo2.currentText().split(',')[0])
    iradar = int(QtApp.combo2.currentText().split(',')[1])
    specifications = ssp.RadarSpecifications[isuite][iradar]
    QtApp.statusBar().showMessage("raytracing ...")
    ssp.radar.utils.apps.process_events()
    path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
    mem = ssp.radar.utils.apps.guess_memory_usage(path_d_drate_amp)
    QtApp.statusBar().showMessage(f"{mem} bytes rays, SensorsSignalGeneration ...")
    ssp.radar.utils.apps.process_events()
    Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
    
    pointcloud_axe_parent = QtApp.pointcloud_axe()
    
    if len(Signals[isuite]['radars'][iradar])==0:
        return
    XRadar,t = Signals[isuite]['radars'][iradar][0]
    
    FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
    Ts = specifications['Ts']
    PRI = specifications['PRI']
    PrecodingMatrix = specifications['PrecodingMatrix']
    RadarMode = specifications['RadarMode']
    
    QtApp.axes[1].plot(np.real(XRadar[:, 0, 0]), label="Real Part")
    QtApp.axes[1].plot(np.imag(XRadar[:, 0, 0]), label="Imaginary Part")
    QtApp.axes[1].set_xlabel("ADC Samples")
    QtApp.axes[1].set_ylabel("ADC Output Level")
    QtApp.axes[1].legend(loc='upper right')
    QtApp.axes[0].imshow(np.abs(XRadar[:, :, 0]),extent=[1*PRI, XRadar.shape[1]*PRI, 1, XRadar.shape[0]],
                        aspect='auto', origin='lower')
    QtApp.axes[0].set_xlabel("Slow time")
    QtApp.axes[0].set_ylabel("ADC Samples")
    
    QtApp.statusBar().showMessage("Range Doppler Processing ...")
    ssp.radar.utils.apps.process_events()
    
    if specifications['RangeWindow'] == "Hamming":
        fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
    elif specifications['RangeWindow'] == "Hann":
        fast_time_window = scipy.signal.windows.hann(XRadar.shape[0])
    elif specifications['RangeWindow'] == "Rectangular":
        fast_time_window = np.ones(XRadar.shape[0])
    else:
        QtApp.statusBar().showMessage(f"Unknown RangeWindow: {specifications['RangeWindow']}")
        return
    
    X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

    NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
    if NFFT_Range_OverNextPow2 < 0:
        NFFT_Range = XRadar.shape[0]
    else:
        NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
    X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
    d_fft = np.arange(NFFT_Range) * ssp.LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
    Range_Start = specifications['Range_Start']
    Range_End = specifications['Range_End']
    d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
    d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
    d_fft = d_fft[d1i:d2i]
    X_fft_fast = X_fft_fast[d1i:d2i,:,:] 
    
    QtApp.axes[3].plot(d_fft, np.abs(X_fft_fast[:, 0, 0]))
    QtApp.axes[3].set_xlabel("Range (m)")
    QtApp.axes[3].set_ylabel("Range Profile")
    QtApp.axes[2].imshow(np.abs(X_fft_fast[:, :, 0]),
                        extent=[1*PRI, XRadar.shape[1]*PRI, d_fft[0], d_fft[-1]],
                        aspect='auto', origin='lower')
    QtApp.axes[2].set_xlabel("Slow time")
    QtApp.axes[2].set_ylabel("Range")
    
    
    # specifications['Pulse_Buffering'] = BlenderAddon['Pulse_Buffering']
    
    rangeDopplerTXRX, f_Doppler = ssp.radar.utils.dopplerprocessing_mimodemodulation(X_fft_fast, specifications)

    

    if specifications['RangeDoppler CFAR Mean']:
        rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
    else:
        rangeDoppler4CFAR = np.abs(rangeDopplerTXRX[:, :, 0, 0])

    if specifications['RangeDopplerCFARLogScale']==True:
        rangeDoppler4CFAR = np.log10(rangeDoppler4CFAR + 1e-10)
        rangeDoppler4CFAR -= np.min(rangeDoppler4CFAR)

    im = QtApp.axes[4].imshow(np.abs(rangeDopplerTXRX[:, :, 0, 0]),
                            extent=[f_Doppler[0], f_Doppler[-1], d_fft[0], d_fft[-1]],
                            aspect='auto', origin='lower')
    QtApp.axes[4].set_xlabel("Doppler Frequency (Hz)")
    QtApp.axes[4].set_ylabel("Range (m)")
    
    distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
    elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
    X, Y = np.meshgrid(elevation, distance)
    # FigsAxes[1,2].plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
    QtApp.axes[5].plot_surface(X, Y, (rangeDoppler4CFAR), cmap='viridis', alpha=1)
    QtApp.axes[5].set_xlabel('Doppler (Hz)')
    QtApp.axes[5].set_ylabel('Distance (m)')
    QtApp.axes[5].set_zlabel('Magnitude (normalized, dB)')

    if QtApp.doCFARCB.isChecked() == False:
        QtApp.canvas.draw()
        QtApp.canvas.flush_events()
        ssp.utils.increaseCurrentFrame()
        return
    QtApp.statusBar().showMessage("Range Doppler CFAR ...")
    ssp.radar.utils.apps.process_events()
    num_train = [int(specifications['CFAR_RD_training_cells'].split(',')[0]), int(specifications['CFAR_RD_training_cells'].split(',')[1])]
    num_guard = [int(specifications['CFAR_RD_guard_cells'].split(',')[0]), int(specifications['CFAR_RD_guard_cells'].split(',')[1])]
    alpha = float(specifications['CFAR_RD_alpha'])
    if specifications['CFAR_RD_type'] == "CA CFAR":
        if ssp.config.GPU_run_available():
            detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, num_train, num_guard, alpha)
        else:
            detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR,num_train, num_guard, alpha)
        cfar_threshold*=alpha
    elif specifications['CFAR_RD_type'] == "OS CFAR":
        QtApp.cfar_rd_type.setCurrentText("OS CFAR")
        return
    elif specifications['CFAR_RD_type'] == "Fixed Threshold":
        T = alpha 
        cfar_threshold = T * np.ones_like(rangeDoppler4CFAR)
        detections = np.zeros_like(rangeDoppler4CFAR)
        detections[rangeDoppler4CFAR > T] = 1
    elif specifications['CFAR_RD_type'] == "Fixed Threshold a*mean":
        T = alpha * np.mean(rangeDoppler4CFAR)
        cfar_threshold = T * np.ones_like(rangeDoppler4CFAR)
        detections = np.zeros_like(rangeDoppler4CFAR)
        detections[rangeDoppler4CFAR > T] = 1
    elif specifications['CFAR_RD_type'] == "Fixed Threshold a*KSort":
        
        # N = int(.9*rangeDoppler4CFAR.shape[0] * rangeDoppler4CFAR.shape[1])
        N = rangeDoppler4CFAR.shape[0] * rangeDoppler4CFAR.shape[1]-1-num_guard[0]
        if N<0:
            N=0
        T = alpha * np.partition(rangeDoppler4CFAR.ravel(), N)[N]
        cfar_threshold = T * np.ones_like(rangeDoppler4CFAR)
        detections = np.zeros_like(rangeDoppler4CFAR)
        detections[rangeDoppler4CFAR > T] = 1
        
    elif specifications['CFAR_RD_type'] == "No CFAR (max)":
        max_idx = np.unravel_index(np.argmax(rangeDoppler4CFAR), rangeDoppler4CFAR.shape)
        detections = np.zeros_like(rangeDoppler4CFAR)
        detections[max_idx] = 1
        cfar_threshold = np.full_like(rangeDoppler4CFAR, rangeDoppler4CFAR[max_idx])
    all_xyz=[]

    # detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # FigsAxes[1,2].plot_surface(X, Y, (cfar_threshold)+0, color='yellow', alpha=1)
    detected_points = np.where(detections == 1)
    QtApp.axes[5].scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                (rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')
    
    # Labels and legend
    if  QtApp.thrCH.isChecked():
        
        QtApp.axes[5].plot_surface(X,Y,(cfar_threshold), color='orange', alpha=1)
    NDetection = detected_points[0].shape[0]

    if specifications["AngleSpectrum"] == "FFT":
        rangeVA = np.zeros((int(np.max(specifications['vaorder'][:,2])),int(np.max(specifications['vaorder'][:,3]))),dtype=rangeDopplerTXRX.dtype)
        Lambda = 1.0
        AzELscale = specifications['distance scaling'].split(',')
        AzELscale = [float(AzELscale[0]), float(AzELscale[1])]
        dy = .5*Lambda*AzELscale[0]
        dz = .5*Lambda*AzELscale[1]
        ampmax = 0
        for id in range(NDetection):
            QtApp.statusBar().showMessage(f"Angle Processing {id} from {NDetection} ...")
            ssp.radar.utils.apps.process_events()
            rangeTarget = d_fft[detected_points[0][id]]
            dopplerTarget = f_Doppler[detected_points[1][id]]
            
            antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
            for indx in specifications['vaorder']:
                rangeVA[int(indx[2]-1), int(indx[3]-1)] = antennaSignal[int(indx[0]-1), int(indx[1]-1)]
            NFFT_Angle_OverNextPow2 = specifications['AzFFT_OverNextP2']
            if NFFT_Angle_OverNextPow2 < 0:
                NFFT_Angle = rangeVA.shape[0]
            else:
                NFFT_Angle = int(2 ** (np.ceil(np.log2(rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
            NFFT_Angle_OverNextPow_elevation = specifications['ElFFT_OverNextP2']
            if NFFT_Angle_OverNextPow_elevation < 0:
                NFFT_Angle_elevation = rangeVA.shape[1]
            else:
                NFFT_Angle_elevation = int(2 ** (np.ceil(np.log2(rangeVA.shape[1]))+NFFT_Angle_OverNextPow_elevation))
            AngleMap0 = np.fft.fft(rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            AngleMap0 = np.fft.fftshift(AngleMap0, axes=0)
            AngleMap = np.fft.fft(AngleMap0, axis=1, n=NFFT_Angle_elevation)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            AngleMap = np.fft.fftshift(AngleMap, axes=1)
            AngleMap = np.abs(AngleMap)
            a = np.fft.fftshift(np.fft.fftfreq(AngleMap.shape[0]))
            b = np.fft.fftshift(np.fft.fftfreq(AngleMap.shape[1]))
                
            X, Y = np.meshgrid(b, a)
            num_train = [int(specifications['CFAR_Angle_training_cells'].split(',')[0]), int(specifications['CFAR_Angle_training_cells'].split(',')[1])]
            num_guard = [int(specifications['CFAR_Angle_guard_cells'].split(',')[0]), int(specifications['CFAR_RD_guard_cells'].split(',')[1])]
            alpha = float(specifications['CFAR_Angle_alpha'])
            if specifications['CFAR_Angle_type'] == "CA CFAR":
                if ssp.config.GPU_run_available():
                    detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*AngleMap, num_train, num_guard, alpha)
                else:
                    detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*AngleMap,num_train, num_guard, alpha)
                cfar_threshold*=alpha
            elif specifications['CFAR_Angle_type'] == "OS CFAR":
                QtApp.cfar_rd_type.setCurrentText("OS CFAR")
                return
            elif specifications['CFAR_Angle_type'] == "Fixed Threshold":
                T = alpha 
                cfar_threshold = T * np.ones_like(AngleMap)
                detections = np.zeros_like(AngleMap)
                detections[AngleMap > T] = 1
            elif specifications['CFAR_Angle_type'] == "Fixed Threshold a*mean":
                T = alpha * np.mean(AngleMap)
                cfar_threshold = T * np.ones_like(AngleMap)
                detections = np.zeros_like(AngleMap)
                detections[AngleMap > T] = 1
            elif specifications['CFAR_Angle_type'] == "Fixed Threshold a*KSort":
                N = int(.9*AngleMap.shape[0] * AngleMap.shape[1])
                T=alpha * np.partition(AngleMap.ravel(), N)[N]
                cfar_threshold = T * np.ones_like(AngleMap)
                detections = np.zeros_like(AngleMap)
                detections[AngleMap > T] = 1
                
            elif specifications['CFAR_Angle_type'] == "No CFAR (max)":
                max_idx = np.unravel_index(np.argmax(AngleMap), AngleMap.shape)
                detections = np.zeros_like(AngleMap)
                detections[max_idx] = 1
                cfar_threshold = np.full_like(AngleMap, AngleMap[max_idx])
            detected_points_angle = np.where(detections == 1)
            NDetection_angle = detected_points_angle[0].shape[0]
            for id_angle in range(NDetection_angle):
                amp = AngleMap[detected_points_angle[0][id_angle],detected_points_angle[1][id_angle]]
                
                if ampmax < amp:
                    ampmax = amp
                    AngleMapmax = AngleMap
                    detected_points_anglemax = detected_points_angle
                    cfar_thresholdmax = cfar_threshold
                fy = a[detected_points_angle[0][id_angle]] / dy
                fz = b[detected_points_angle[1][id_angle]] / dz
                azhat = np.arcsin(fy / np.sqrt((1/Lambda)**2 - fz**2))
                elhat = np.arccos(fz * Lambda)
                elhat = np.pi/2 - elhat
                x, y, z = ssp.utils.sph2cart(rangeTarget, azhat, elhat)
                x, y, z = y,z,-x
                global_location, global_rotation, global_scale = specifications['matrix_world']  
                local_point = Vector((x, y, z))
                global_point = global_location + global_rotation @ (local_point * global_scale)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
                sphere = bpy.context.object
                sphere.parent=pointcloud_axe_parent
                x = global_point.x
                y = global_point.y
                z = global_point.z
                all_xyz.append([x,y,z,amp,dopplerTarget])
        QtApp.axes[6].cla()
        if NDetection>0:
            if AngleMap.shape[1]>1:
                QtApp.axes[6].plot_surface(X,Y,(AngleMapmax), cmap='viridis', alpha=1)
                QtApp.axes[6].scatter(b[detected_points_anglemax[1]],a[detected_points_anglemax[0]], 
                            (AngleMapmax[detected_points_anglemax]), color='red', s=20, label='Post-CFAR Point Cloud')
                if  QtApp.thrCH.isChecked():
                    QtApp.axes[6].plot_surface(X,Y,cfar_thresholdmax, color='orange', alpha=1)
            else:
                QtApp.axes[6].plot(Y[:,0],AngleMapmax[:,0])
                QtApp.axes[6].plot(a[detected_points_anglemax[0]],AngleMapmax[detected_points_anglemax],'or')
                if  QtApp.thrCH.isChecked():
                    QtApp.axes[6].plot(Y[:,0],cfar_thresholdmax[:,0])
                    
    elif specifications["AngleSpectrum"] == "Capon":
        min_deg, res_deg, max_deg, fine_res_deg = specifications['Capon Azimuth min:res:max:fine_res']
        min_deg2, res_deg2, max_deg2, fine_res_deg2 = specifications['Capon Elevation min:res:max:fine_res']
        ampmax = 0
        for id in range(NDetection):
            QtApp.statusBar().showMessage(f"Angle Processing {id} from {NDetection} ...")
            ssp.radar.utils.apps.process_events()
            rangeTarget = d_fft[detected_points[0][id]]
            dopplerTarget = f_Doppler[detected_points[1][id]]
            
            antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
            sig0 = antennaSignal.reshape(-1, 1)
            Ndim = specifications['array_geometry'].shape[0]
            
            if specifications['Capon DL']>10:
                R_inv = np.eye(Ndim)
            else:
                R = np.zeros((Ndim, Ndim), dtype=np.complex64)
                k=0
                for i in range(rangeDopplerTXRX.shape[1]):
                    if np.abs(i-detected_points[1][id]) <= .2*rangeDopplerTXRX.shape[1]:
                        continue
                    sig = rangeDopplerTXRX[detected_points[0][id],i,:,:]
                    sig = sig.reshape(-1, 1)  
                    R += sig @ sig.conj().T 
                    k += 1
                if k == 0:
                    R = np.eye(Ndim)
                R = R / k  # average covariance matrix

                R = R / np.linalg.norm(R)  # normalize the covariance matrix
                R_inv = np.linalg.inv(R + specifications['Capon DL']*np.eye(Ndim))
            
            azimuth_angles = np.arange(min_deg, max_deg + res_deg, res_deg)
            elevation_angles = np.arange(min_deg2, max_deg2 + res_deg2, res_deg2)
            
            AngleMap = np.zeros((len(azimuth_angles), len(elevation_angles)), dtype=np.float32)


            for j, el in enumerate(elevation_angles):
                for i, az in enumerate(azimuth_angles):
                    theta = np.deg2rad(az)       # azimuth
                    phi = np.deg2rad(el)         # elevation
                    wavelength = 2
                    k_vec = (2 * np.pi / wavelength) * np.array([
                        np.cos(phi) * np.cos(theta),
                        np.cos(phi) * np.sin(theta),
                        np.sin(phi)
                    ])
                    steering = np.exp(-1j * (specifications['array_geometry'] @ k_vec)).reshape(-1, 1)
                    w = R_inv @ steering / (np.conj(steering.T) @ R_inv @ steering)
                    p = np.conj(w.T) @ sig0
                    AngleMap[i, j] = np.abs(p.squeeze())
            
            
            num_train = [int(specifications['CFAR_Angle_training_cells'].split(',')[0]), int(specifications['CFAR_Angle_training_cells'].split(',')[1])]
            num_guard = [int(specifications['CFAR_Angle_guard_cells'].split(',')[0]), int(specifications['CFAR_RD_guard_cells'].split(',')[1])]
            alpha = float(specifications['CFAR_Angle_alpha'])
            if specifications['CFAR_Angle_type'] == "CA CFAR":
                if ssp.config.GPU_run_available():
                    detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*AngleMap, num_train, num_guard, alpha)
                else:
                    detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*AngleMap,num_train, num_guard, alpha)
                cfar_threshold*=alpha
            elif specifications['CFAR_Angle_type'] == "OS CFAR":
                QtApp.cfar_rd_type.setCurrentText("OS CFAR")
                return
            elif specifications['CFAR_Angle_type'] == "Fixed Threshold":
                T = alpha 
                cfar_threshold = T * np.ones_like(AngleMap)
                detections = np.zeros_like(AngleMap)
                detections[AngleMap > T] = 1
            elif specifications['CFAR_Angle_type'] == "Fixed Threshold a*mean":
                T = alpha * np.mean(AngleMap)
                cfar_threshold = T * np.ones_like(AngleMap)
                detections = np.zeros_like(AngleMap)
                detections[AngleMap > T] = 1
            elif specifications['CFAR_Angle_type'] == "Fixed Threshold a*KSort":
                N = int(.9*AngleMap.shape[0] * AngleMap.shape[1])
                T=alpha * np.partition(AngleMap.ravel(), N)[N]
                cfar_threshold = T * np.ones_like(AngleMap)
                detections = np.zeros_like(AngleMap)
                detections[AngleMap > T] = 1
                
            elif specifications['CFAR_Angle_type'] == "No CFAR (max)":
                max_idx = np.unravel_index(np.argmax(AngleMap), AngleMap.shape)
                detections = np.zeros_like(AngleMap)
                detections[max_idx] = 1
                cfar_threshold = np.full_like(AngleMap, AngleMap[max_idx])
            detected_points_angle = np.where(detections == 1)
            NDetection_angle = detected_points_angle[0].shape[0]
            for id_angle in range(NDetection_angle):
                amp = AngleMap[detected_points_angle[0][id_angle],detected_points_angle[1][id_angle]]
                if ampmax < amp:
                    ampmax = amp
                    AngleMapmax = AngleMap
                    detected_points_anglemax = detected_points_angle
                    cfar_thresholdmax = cfar_threshold

                azhat = np.deg2rad(azimuth_angles[detected_points_angle[0][id_angle]])
                elhat = np.deg2rad(elevation_angles[detected_points_angle[1][id_angle]])
                x, y, z = ssp.utils.sph2cart(rangeTarget, azhat, elhat)
                x, y, z = y,z,-x
                global_location, global_rotation, global_scale = specifications['matrix_world']  
                local_point = Vector((x, y, z))
                global_point = global_location + global_rotation @ (local_point * global_scale)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
                sphere = bpy.context.object
                sphere.parent=pointcloud_axe_parent
                x = global_point.x
                y = global_point.y
                z = global_point.z
                all_xyz.append([x,y,z,amp,dopplerTarget])
            X, Y = np.meshgrid(elevation_angles, azimuth_angles)
        QtApp.axes[6].cla()
        if NDetection>0:
            if AngleMap.shape[1]>1:
                QtApp.axes[6].plot_surface(X,Y,(AngleMapmax), cmap='viridis', alpha=1)
                QtApp.axes[6].scatter(elevation_angles[detected_points_anglemax[1]],azimuth_angles[detected_points_anglemax[0]], 
                            (AngleMapmax[detected_points_anglemax]), color='red', s=20, label='Post-CFAR Point Cloud')
                if  QtApp.thrCH.isChecked():
                    QtApp.axes[6].plot_surface(X,Y,cfar_thresholdmax, color='orange', alpha=1)
            else:
                QtApp.axes[6].plot(Y[:,0],AngleMapmax[:,0])
                QtApp.axes[6].plot(azimuth_angles[detected_points_anglemax[0]],AngleMapmax[detected_points_anglemax],'or')
                if  QtApp.thrCH.isChecked():
                    QtApp.axes[6].plot(Y[:,0],alpha*cfar_thresholdmax[:,0])
                    
        
    points = np.array(all_xyz)
    if points.shape[0]>0:
        ampcolor = QtApp.combocolor.currentText()
        if ampcolor == 'Amp':
            QtApp.axes[7].scatter(points[:, 0], points[:, 1],points[:, 2], c=points[:, 3], marker='o') 
        if ampcolor == 'Doppler':
            QtApp.axes[7].scatter(points[:, 0], points[:, 1],points[:, 2], c=points[:, 4], marker='o') 
            
    QtApp.axes[7].plot(0, 0, 0, 'o', markersize=5, color='red')
    QtApp.axes[7].plot([0,5], [0,0], [0,0], color='red')
    QtApp.axes[7].set_xlabel('X (m)')
    QtApp.axes[7].set_ylabel('Y (m)')
    QtApp.axes[7].set_zlabel('Z (m)')
    QtApp.axes[7].set_title("Detected Points")
    ssp.utils.set_axes_equal(QtApp.axes[7])
    
    QtApp.canvas.draw()
    QtApp.canvas.flush_events()
    if QtApp.combosave.currentText()=="Point Cloud":
        fn = f'PointCloud_frame_{ssp.config.CurrentFrame}.mat'
        if "Simulation Settings" in bpy.data.objects:
            if "Radar Outputs Folder" in bpy.data.objects["Simulation Settings"]:
                fp= bpy.data.objects["Simulation Settings"]["Radar Outputs Folder"]
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fn = os.path.join(fp,fn)
                scipy.io.savemat(fn, {
                        'points'    : points,
                        'frame': ssp.config.CurrentFrame
                    })
    if QtApp.combosave.currentText()=="Raw DataCube":
        fn = f'Signals_frame_{ssp.config.CurrentFrame}.mat'
        if "Simulation Settings" in bpy.data.objects:
            if "Radar Outputs Folder" in bpy.data.objects["Simulation Settings"]:
                fp= bpy.data.objects["Simulation Settings"]["Radar Outputs Folder"]
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fn = os.path.join(fp,fn)
                rangePulseTXRX = []
                XRadar,X_windowed_fast,rangePulseTXRX,d_fft,rangeDopplerTXRX,f_Doppler,rangeDoppler4CFAR
                scipy.io.savemat(fn, {
                        'frame'             : ssp.config.CurrentFrame,
                        'XRadar'            : XRadar,
                        'X_windowed_fast'   : X_windowed_fast,
                        'rangePulseTXRX'    : rangePulseTXRX,
                        'd_fft'             : d_fft,
                        'rangeDopplerTXRX'  : rangeDopplerTXRX,
                        'f_Doppler'         : f_Doppler,
                        'rangeDoppler4CFAR' : rangeDoppler4CFAR,
                    })
    
    
    
    
    
    QtApp.statusBar().showMessage(f'processed frame = {ssp.config.CurrentFrame}')
    ssp.utils.increaseCurrentFrame()