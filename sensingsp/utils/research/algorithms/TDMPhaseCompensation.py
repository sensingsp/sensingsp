import sensingsp as ssp
import numpy as np
from matplotlib import pyplot as plt

def runSimpleScenario(MIMO_phase_compensation_Enable = True, Use_General_MIMO_MOdel = False,
                      radial_velocity=5,azimuth=-20,Optionfd0=0):
    ssp.utils.initialize_environment()
    ssp.config.AddRadar_ULA_N = 6
    radar = ssp.radar.utils.addRadar(
        radarSensor=ssp.radar.utils.RadarSensorsCategory.ULA_SameTXRX,
        location_xyz=[-3, 0, 1])
    ssp.radar.utils.set_FMCW_Chirp_Parameters(radar,slope=12.49,fsps=.67,N_ADC=40,
                                            NPulse=ssp.config.AddRadar_ULA_N*(1*32),PRI_us=60)

    vmax = 1/60e-6 / ssp.config.AddRadar_ULA_N * 3e8 / 76e9 / 2 #= 5.48

    rangeTarget = 6.5
    ssp.radar.utils.addTarget(
        refRadar=radar,
        range=rangeTarget, azimuth=azimuth,
        RCS0=30, radial_velocity=radial_velocity)

    ssp.utils.initialize_simulation()

    ssp.utils.set_configurations([ssp.radar.utils.RadarSignalGenerationConfigurations.Spillover_Disabled,
                                ssp.radar.utils.RadarSignalGenerationConfigurations.RayTracing_Balanced])

    ssp.utils.useCUDA(False)



    s_IF_nfast_mslow_prx, radarParameters = ssp.utils.research.simplest.RadarRawData()
    fdix = -2 * radial_velocity / ssp.utils.research.simplest.wavelength(radarParameters)

    # ssp.utils.imshow(np.angle(s_IF_nfast_mslow_prx[0,:,:]), "slow","rx")
    if 0:
        distanceMatrix = ssp.utils.research.simplest.RayTracing()
        print(distanceMatrix)
        plt.imshow(distanceMatrix)
        plt.figure()
        distancVA = distanceMatrix.reshape(distanceMatrix.shape[0]*distanceMatrix.shape[1])
        plt.plot(distancVA,'.--')
        plt.show()

    print(s_IF_nfast_mslow_prx.shape)

    if Use_General_MIMO_MOdel:
        S_IF_krange_mslow_prx, ranges = ssp.radar.utils.rangeprocessing(s_IF_nfast_mslow_prx, radarParameters)
        radarParameters['DopplerProcessingMIMODemod'] = 'General'
        rangeDopplerTXRX, f_Doppler = ssp.radar.utils.dopplerprocessing_mimodemodulation(S_IF_krange_mslow_prx, radarParameters)
        S_IF_krange_ldoppler_pva = rangeDopplerTXRX.reshape(*rangeDopplerTXRX.shape[:2], -1)
        VA_axe = 2
        nAngleFFT = 100*S_IF_krange_ldoppler_pva.shape[VA_axe]
        S_IF_krange_ldoppler_qangle = np.fft.fft( S_IF_krange_ldoppler_pva , n = nAngleFFT, axis = VA_axe )
        S_IF_krange_ldoppler_qangle = np.fft.fftshift(S_IF_krange_ldoppler_qangle, axes = VA_axe)
        d_Wavelength = .5
        normalized_freq = np.arange(-S_IF_krange_ldoppler_qangle.shape[2]/2, S_IF_krange_ldoppler_qangle.shape[2]/2) / S_IF_krange_ldoppler_qangle.shape[2] 
        sintheta = normalized_freq / d_Wavelength
        azimuths = -np.arcsin(sintheta)
        r, theta = np.meshgrid(ranges, azimuths)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        # Plot both sharp and blurred range-azimuth maps
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))

        if Optionfd0 == 0:
            RangeAngleMap = 20*np.log10(np.sum(np.abs(S_IF_krange_ldoppler_qangle),axis=1))
        else:
            DominantDopplerProfile = np.sum(np.sum(np.abs(S_IF_krange_ldoppler_pva),axis=0),axis=-1)
            lfd0 = np.argmax(DominantDopplerProfile)
            RangeAngleMap = 20*np.log10(np.abs(S_IF_krange_ldoppler_qangle[:,lfd0,:]))
        # RangeAngleMap = (np.sum(np.abs(S_IF_krange_ldoppler_qangle),axis=1))
        RangeAngleMap = RangeAngleMap.T

        x = np.concatenate((x, x[::-1,:]), axis=0)
        y = np.concatenate((y, -y[::-1,:]), axis=0)
        RangeAngleMap = np.concatenate((RangeAngleMap, RangeAngleMap[::-1,:]), axis=0)

        RangeAngleMap -= np.max(RangeAngleMap)
        # With phase compensation (sharp target)

        mesh = axes.pcolormesh(x, y, RangeAngleMap,vmin=-20,vmax=0, shading='auto', cmap='hot')
        axes.set_title("Range-azimuth map with phase compensation")
        # axes.axis('equal')
        axes.set_xlabel("y (m)")
        axes.set_ylabel("x (m)")
        fig.colorbar(mesh, ax=axes, label="Amplitude")
        axes.plot(rangeTarget*np.sin(np.deg2rad(azimuth)),rangeTarget*np.cos(np.deg2rad(azimuth)),'r*')

        # plt.tight_layout()
        axes.set_xlim(-8,8)
        axes.set_ylim(0,2*8)
        plt.show()


        return

    FastTime_axe, SlowTime_axe, RX_axe = 0,1,2

    win = np.hamming(s_IF_nfast_mslow_prx.shape[FastTime_axe])
    s_IF_nfast_mslow_prx *= win[:,np.newaxis, np.newaxis]
    S_IF_krange_mslow_prx = np.fft.fft( s_IF_nfast_mslow_prx , axis = FastTime_axe , n = 5*s_IF_nfast_mslow_prx.shape[FastTime_axe] )    

    ranges = np.arange(0, S_IF_krange_mslow_prx.shape[0]) / S_IF_krange_mslow_prx.shape[0] * ssp.LightSpeed * ssp.utils.research.simplest.ADCrate(radarParameters)/2/ssp.utils.research.simplest.FMCW_slew_rate(radarParameters)


    # ssp.utils.research.simplest.txrxplot(radarParameters)

    MTX = ssp.utils.research.simplest.txNumber(radarParameters)
    NRX = ssp.utils.research.simplest.rxNumber(radarParameters) 
    Leff = int(S_IF_krange_mslow_prx.shape[1]/MTX)
    S_IF_krange_mslow_pva = S_IF_krange_mslow_prx.reshape(-1, Leff , MTX * NRX)

    for l in range(Leff):
        for m in range(MTX):
            n = np.arange(NRX)
            S_IF_krange_mslow_pva[:, l, m * NRX + n] = S_IF_krange_mslow_prx[:, l * MTX + m, :]
        
    range_axe, SlowTime_axe, VA_axe = 0,1,2

    nDopplerFFT = 1*S_IF_krange_mslow_pva.shape[SlowTime_axe]
    S_IF_krange_ldoppler_pva = np.fft.fft( S_IF_krange_mslow_pva , n = nDopplerFFT , axis = SlowTime_axe )

    PRF_TDM = ssp.utils.research.simplest.PRF(radarParameters) / MTX
    fd = np.arange(0, S_IF_krange_ldoppler_pva.shape[1]) / S_IF_krange_ldoppler_pva.shape[1] * PRF_TDM 
    v = -fd * ssp.utils.research.simplest.wavelength(radarParameters) / 2



    DominantDopplerProfile = np.sum(np.sum(np.abs(S_IF_krange_ldoppler_pva),axis=0),axis=-1)
    lfd0 = np.argmax(DominantDopplerProfile)
    fd0 = fd[lfd0]
    
    
    if MIMO_phase_compensation_Enable:
        TDM_MIMO_phase_compensation = np.ones((S_IF_krange_ldoppler_pva.shape[1], S_IF_krange_ldoppler_pva.shape[2]),dtype=S_IF_krange_ldoppler_pva.dtype)
        T = 1/PRF_TDM / MTX
        for ifd in range(len(fd)):
            for m in range(MTX):
                n = np.arange(NRX)
                fdi = -fd[ifd]
                # fdi = fdix
                TDM_MIMO_phase_compensation[ ifd , m * NRX + n] = np.exp(1j * 2 * np.pi * fdi * m * T)
        S_IF_krange_ldoppler_pva *= TDM_MIMO_phase_compensation[np.newaxis, : , : ]

    range_axe, doppler_axe, VA_axe = 0,1,2

    nAngleFFT = 100*S_IF_krange_ldoppler_pva.shape[VA_axe]

    win = np.hanning(S_IF_krange_ldoppler_pva.shape[VA_axe])
    # S_IF_krange_ldoppler_pva = S_IF_krange_ldoppler_pva * win[np.newaxis, np.newaxis, :]

    S_IF_krange_ldoppler_qangle = np.fft.fft( S_IF_krange_ldoppler_pva , n = nAngleFFT, axis = VA_axe )
    S_IF_krange_ldoppler_qangle = np.fft.fftshift(S_IF_krange_ldoppler_qangle, axes = VA_axe)
    # -d/lambda sin theta
    d_Wavelength = .5
    normalized_freq = np.arange(-S_IF_krange_ldoppler_qangle.shape[2]/2, S_IF_krange_ldoppler_qangle.shape[2]/2) / S_IF_krange_ldoppler_qangle.shape[2] 
    sintheta = normalized_freq / d_Wavelength
    # sintheta[sintheta>1] = sintheta[sintheta>1] -2
    # plot(Fs/L*(-L/2:L/2-1),abs(fftshift(Y)),"LineWidth",3)

    azimuths = -np.arcsin(sintheta)




    # Compute range-Doppler map
    range_doppler_map = 20 * np.log10(np.sum(np.abs(S_IF_krange_ldoppler_pva), axis=2))

    # Create the plot
    plt.figure(figsize=(10, 6))
    extent = [v[0], v[-1], ranges[-1], ranges[0]]  # Define extent for the axes (velocity and range)

    plt.imshow(range_doppler_map, aspect='auto', cmap='jet', extent=extent)

    # Add labels and titles
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Range-Doppler Map")
    plt.colorbar(label="Amplitude (dB)")


    # Create polar-to-Cartesian conversion
    r, theta = np.meshgrid(ranges, azimuths)
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    # Plot both sharp and blurred range-azimuth maps
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    if Optionfd0 == 0:
        RangeAngleMap = 20*np.log10(np.sum(np.abs(S_IF_krange_ldoppler_qangle),axis=1))
    else:
        RangeAngleMap = 20*np.log10(np.abs(S_IF_krange_ldoppler_qangle[:,lfd0,:]))
    # RangeAngleMap = (np.sum(np.abs(S_IF_krange_ldoppler_qangle),axis=1))
    RangeAngleMap = RangeAngleMap.T

    x = np.concatenate((x, x[::-1,:]), axis=0)
    y = np.concatenate((y, -y[::-1,:]), axis=0)
    RangeAngleMap = np.concatenate((RangeAngleMap, RangeAngleMap[::-1,:]), axis=0)

    RangeAngleMap -= np.max(RangeAngleMap)
    # With phase compensation (sharp target)

    mesh = axes.pcolormesh(x, y, RangeAngleMap,vmin=-20,vmax=0, shading='auto', cmap='hot')
    axes.set_title("Range-azimuth map with phase compensation")
    # axes.axis('equal')
    axes.set_xlabel("y (m)")
    axes.set_ylabel("x (m)")
    fig.colorbar(mesh, ax=axes, label="Amplitude")

    axes.plot(rangeTarget*np.sin(np.deg2rad(azimuth)),rangeTarget*np.cos(np.deg2rad(azimuth)),'r*')

    # plt.tight_layout()
    axes.set_xlim(-8,8)
    axes.set_ylim(0,2*8)
    plt.show()
