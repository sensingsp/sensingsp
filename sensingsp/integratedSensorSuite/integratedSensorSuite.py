import bpy
import sensingsp as ssp
import numpy as np
from mathutils import Vector
from ..constants import *
from ..utils.stochastics import Complex_Noise_Buffer
import scipy
from matplotlib import pyplot as plt
from ..radar.utils.rss import *
from numba import cuda
import cv2

# class integratedSensorSuite:
def define_suite(isuite=0, location=Vector((0, 0, 0)), rotation=Vector((np.pi/2, 0, -np.pi/2))):
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    suite = bpy.context.object
    suite.name = f'SuitePlane_{isuite}'
    return suite


def SensorsSignalGeneration(): # input is ssp.Paths in ssp.config.CurrentFrame
    # def Generate_RadarSignal(self,frame_d_drate_amp,RadarSpecifications):
    SuiteRadarRangePulseRXSignals = []
    for radarSpecifications in ssp.RadarSpecifications:
        RadarRangePulseRX = {'radars':[],'lidars':[],'cameras':[]}
        for specifications in radarSpecifications:
          # B should be analog Filter Bandwidth
          RadarRangePulseRX['radars'].append(Complex_Noise_Buffer(specifications['N_ADC'],specifications['NPulse'],specifications['N_RX'],T=specifications['Tempreture_K'],B=specifications['FMCW_Bandwidth']))
        SuiteRadarRangePulseRXSignals.append(RadarRangePulseRX)
    for i,suite_info in enumerate(ssp.suite_information):
      for cam in suite_info['Camera']:
        SuiteRadarRangePulseRXSignals[i]['cameras'].append(ssp.camera.utils.render(cam))
      for lidar in suite_info['Lidar']:
        SuiteRadarRangePulseRXSignals[i]['lidars'].append(ssp.lidar.utils.pointcloud(lidar))
      
        
    Frame2ArrayIndex = ssp.config.CurrentFrame-bpy.context.scene.frame_start
    # Frame2ArrayIndex = 0
    for isrx,suiteRX_d_drate_amp in ssp.Paths[Frame2ArrayIndex].items():
        for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
          for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
            for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
              if istx == isrx:
                for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
                  if irtx == irrx:
                    PRI = ssp.RadarSpecifications[isrx][irrx]['PRI']
                    Ts = ssp.RadarSpecifications[isrx][irrx]['Ts']
                    NPulse = ssp.RadarSpecifications[isrx][irrx]['NPulse']
                    N_ADC = ssp.RadarSpecifications[isrx][irrx]['N_ADC']
                    iADC = np.arange(N_ADC)

                    Lambda = ssp.RadarSpecifications[isrx][irrx]['Lambda']
                    FMCW_ChirpSlobe = ssp.RadarSpecifications[isrx][irrx]['FMCW_ChirpSlobe']
                    PrecodingMatrix = ssp.RadarSpecifications[isrx][irrx]['PrecodingMatrix']
                    
                    RadarMode = ssp.RadarSpecifications[isrx][irrx]['RadarMode']
                    FMCWRadar = 1
                    if RadarMode=='FMCW':
                      FMCWRadar = 1
                    if RadarMode=='Pulse':
                      FMCWRadar = 0
                      PulseWaveform = ssp.RadarSpecifications[isrx][irrx]['PulseWaveform']
                      Waveform = ssp.radar.radarwaveforms.barker_code(11)
                      ssp.RadarSpecifications[isrx][irrx]['PulseWaveform_Loaded']=Waveform
                      
                      
                            
                    for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():

                      for d_drate_amp in TX_d_drate_amp:
                        if len(d_drate_amp[3])==0:
                          continue
                        HardwareDCBlockerAttenuation = 1 # d_drate_amp[1]
                        # if d_drate_amp[1]==0:
                        #   continue
                        # if d_drate_amp[0]==0: TX RX same position
                        #   continue
                        # if d_drate_amp[0] > PRI*LightSpeed:
                        #   Save it in a buffer for next pulses or fix PRF : effective_d = mod(d_drate_amp[0] , PRI*LightSpeed)
                        #   continue
                        
                        SimDopplerEffect = 1 # Should be 1; for test and analysis, can set to 0
                        for ip in range(NPulse):
                          d_of_t =  d_drate_amp[0] + (ip * PRI + iADC * Ts) * d_drate_amp[1]*SimDopplerEffect
                          
                          if FMCWRadar == 1:
                            phase2 = 2*np.pi*(
                              d_of_t/Lambda
                              +FMCW_ChirpSlobe/LightSpeed*iADC * Ts*d_of_t
                              -.5*FMCW_ChirpSlobe*(d_of_t/LightSpeed)*(d_of_t/LightSpeed)
                              )
                            ipPM = ip % PrecodingMatrix.shape[0]
                            SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += PrecodingMatrix[ipPM][itx]*d_drate_amp[2]*np.exp(1j*phase2) * HardwareDCBlockerAttenuation
                        #   ssp.RadarSpecifications[isrx][irrx]['RangePulseRX'][iADC,ip,irx]+=PrecodingMatrix[ipPM][itx]*d_drate_amp[2]*np.exp(1j*phase2) * HardwareDCBlockerAttenuation
                          elif FMCWRadar == 0:
                            Window_Waveform = np.zeros_like(d_of_t,dtype=complex)
                            ind1 = int( (d_drate_amp[0] + (ip * PRI + 0 * Ts) * d_drate_amp[1]) / LightSpeed / Ts)
                            if ind1 < Window_Waveform.shape[0]:
                              ind1 = np.arange(ind1,min([ind1+Waveform.shape[0],Window_Waveform.shape[0]]))
                              ind2 = np.arange(0,ind1.shape[0])
                              Window_Waveform[ind1]=Waveform[ind2]
                              phase2 = 2*np.pi*(d_of_t/Lambda)
                              ipPM = ip % PrecodingMatrix.shape[0]
                              SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += PrecodingMatrix[ipPM][itx]*d_drate_amp[2]*np.exp(1j*phase2)*Window_Waveform
    ## Ampedance and ADC 
    for isuite in range(len(SuiteRadarRangePulseRXSignals)):
      for iradar in range(len(SuiteRadarRangePulseRXSignals[isuite]['radars'])):
        specifications = ssp.RadarSpecifications[isuite][iradar]
        ImpedanceFactor = np.sqrt(specifications['ADC_ImpedanceFactor'])
        LNA_Gain = specifications['ADC_LNA_Gain']
        ADC_Peak2Peak = specifications['ADC_peak2peak']
        ADC_Levels = specifications['ADC_levels']
        SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx] = apply_adc(SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx],ADC_Peak2Peak,ADC_Levels,ImpedanceFactor,LNA_Gain,specifications['ADC_SaturationEnabled'])
        # real = np.real(SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx])*ImpedanceFactor*LNA_Gain
        # real = (real + (ADC_Peak2Peak / 2)) / ADC_Peak2Peak
        # real = np.clip(real, 0, 1)
        # real = np.round(real * (ADC_Levels - 1))-(ADC_Levels - 1)/2
        # imag = np.imag(SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx])*ImpedanceFactor*LNA_Gain
        # imag = (imag + (ADC_Peak2Peak / 2)) / ADC_Peak2Peak
        # imag = np.clip(imag, 0, 1)
        # imag = np.round(imag * (ADC_Levels - 1)) - (ADC_Levels - 1)/2
        # SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx] = real+1j*imag
    return SuiteRadarRangePulseRXSignals

def SensorsSignalGeneration_frame(path_d_drate_amp): # input is ssp.Paths in ssp.config.CurrentFrame
    # def Generate_RadarSignal(self,frame_d_drate_amp,RadarSpecifications):
    SuiteRadarRangePulseRXSignals = []
    for radarSpecifications in ssp.RadarSpecifications:
        RadarRangePulseRX = {'radars':[],'lidars':[],'cameras':[]}
        for specifications in radarSpecifications:
          specifications['RadarTiming'].update(ssp.config.CurrentTime,bpy.context.scene.render.fps)
          # B should be analog Filter Bandwidth
          TotalBandwidth = specifications['RF_AnalogNoiseFilter_Bandwidth'] 
          TotalBandwidth *= 10**(specifications['RF_NoiseFiguredB']/10)
          NPulseFrame = specifications['RadarTiming'].Frame_PulseTimes.shape[0]  # specifications['NPulse'] 
          RadarRangePulseRX['radars'].append(
            Complex_Noise_Buffer(specifications['N_ADC'],NPulseFrame,
                                 specifications['N_RX'],T=specifications['Tempreture_K'],
                                 B=TotalBandwidth))#specifications['FMCW_Bandwidth']))
          # if len(specifications['CPI_Buffer'])==0:
          #   iCalc=NPulseFrame
          # else:
          #   iCalc=NPulseFrame+specifications['CPI_Buffer'].shape[1]
          # theLastCompleteCPI = int(iCalc/specifications['NPulse'])-1    
          # if theLastCompleteCPI<0:
          #   theLastCompleteCPI=0
          # usefullIIndices_buffer_start = theLastCompleteCPI * specifications['NPulse']
          # # usefullIIndices_Frame_PulseTimes_start = f(usefullIIndices_buffer_start) 
          # if len(specifications['CPI_Buffer'])==0:
          #   usefullIIndices_Frame_PulseTimes_start=usefullIIndices_buffer_start
          # else:
          #   usefullIIndices_Frame_PulseTimes_start=usefullIIndices_buffer_start-specifications['CPI_Buffer'].shape[1]
          # specifications['usefullIIndices_Frame_PulseTimes_start']=usefullIIndices_Frame_PulseTimes_start
          
        SuiteRadarRangePulseRXSignals.append(RadarRangePulseRX)
        
        
        
    for i,suite_info in enumerate(ssp.suite_information):
      for cam in suite_info['Camera']:
        SuiteRadarRangePulseRXSignals[i]['cameras'].append(ssp.camera.utils.render(cam))
      for lidar in suite_info['Lidar']:
        SuiteRadarRangePulseRXSignals[i]['lidars'].append(ssp.lidar.utils.pointcloud(lidar))
    CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
    if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled:
      for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
          for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
            N_data_rx = 0
            for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
              for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
                if istx == isrx:
                  for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
                    if irtx == irrx:
                      for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
                        N_data_rx += (0+4*len(TX_d_drate_amp))
            array_data = np.zeros(N_data_rx,dtype=np.longdouble)
            array_data_i = 0
            indices =[]
            for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
              indices.append(array_data_i)
              for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
                if istx == isrx:
                  for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
                    if irtx == irrx:
                      for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():
                        for d_drate_amp in TX_d_drate_amp:
                          array_data[array_data_i]=itx
                          array_data_i+=1
                          array_data[array_data_i]=d_drate_amp[0]
                          array_data_i+=1
                          array_data[array_data_i]=d_drate_amp[1]
                          array_data_i+=1
                          array_data[array_data_i]=d_drate_amp[2]
                          array_data_i+=1
              indices.append(array_data_i)
            indices = np.array(indices, dtype=np.int64)
            NRX=int(indices.shape[0]/2)
            # NRX=
            PRI = ssp.RadarSpecifications[isrx][irrx]['PRI']
            Ts = ssp.RadarSpecifications[isrx][irrx]['Ts']
            Frame_PulseTimes = 1.0*ssp.RadarSpecifications[isrx][irrx]['RadarTiming'].Frame_PulseTimes
            Frame_PulseTimes -= ssp.config.CurrentTime
            NPulseFrame = Frame_PulseTimes.shape[0]
            if NPulseFrame==0:
              continue
            # StartPulseIndex = ssp.RadarSpecifications[isrx][irrx]['RadarTiming'].n_pulse - NPulseFrame 
            RadarMode = ssp.RadarSpecifications[isrx][irrx]['RadarMode']
            if RadarMode != 'FMCW':
              raise NotImplementedError(f"Radar mode '{RadarMode}' is not implemented in GPU yet. Use CPU")

            N_ADC = ssp.RadarSpecifications[isrx][irrx]['N_ADC']
            Lambda = ssp.RadarSpecifications[isrx][irrx]['Lambda']
            FMCW_ChirpSlobe = ssp.RadarSpecifications[isrx][irrx]['FMCW_ChirpSlobe']
            PrecodingMatrix = ssp.RadarSpecifications[isrx][irrx]['PrecodingMatrix']
                        
            PrecodingMatrix_real = np.real(PrecodingMatrix).flatten()
            PrecodingMatrix_imag = np.imag(PrecodingMatrix).flatten()
            d_PrecodingMatrix_real = cuda.to_device(PrecodingMatrix_real)
            d_PrecodingMatrix_imag = cuda.to_device(PrecodingMatrix_imag)

            blocksize = (16, 16, 4)
            gridsize = (int(np.ceil(N_ADC / blocksize[0])),int( np.ceil(NPulseFrame / blocksize[1])),int( np.ceil(NRX / blocksize[2])))
            d_array_data = cuda.device_array(array_data.shape[0], dtype=array_data.dtype)
            d_indices = cuda.device_array(indices.shape[0], dtype=np.int64)
            cuda.to_device(array_data, to=d_array_data)
            cuda.to_device(indices, to=d_indices)
            NRangePulse = int(N_ADC*NPulseFrame*NRX*2)
            d_RangePulse = cuda.device_array(NRangePulse, dtype=np.float64)
            pulsebias = ssp.RadarSpecifications[isrx][irrx]['RadarTiming'].n_pulse - NPulseFrame 
            d_Frame_PulseTimes = cuda.device_array(Frame_PulseTimes.shape[0], dtype=Frame_PulseTimes.dtype)
            cuda.to_device(Frame_PulseTimes, to=d_Frame_PulseTimes)
            CUDA_signalGeneration[gridsize, blocksize](d_RangePulse,d_array_data, d_indices, indices.shape[0], N_ADC, NPulseFrame, Lambda, Ts,FMCW_ChirpSlobe,
                d_PrecodingMatrix_real,d_PrecodingMatrix_imag,PrecodingMatrix.shape[0],PrecodingMatrix.shape[1],
                                                      d_Frame_PulseTimes,pulsebias
            )
            h_RangePulse = d_RangePulse.copy_to_host()
            del d_array_data
            del d_indices
            del d_RangePulse
            del d_Frame_PulseTimes
            h_RangePulse_complex = h_RangePulse.reshape((int(NRX), int(NPulseFrame), int(N_ADC), 2))
            RangePulse = h_RangePulse_complex[..., 0] + 1j * h_RangePulse_complex[..., 1]
            # RangePulse = np.transpose(RangePulse, (2, 1, 0))
            if 0:
              RangePulseCPU = np.zeros(NRangePulse,dtype=complex)
              RangePulse_Matrix,RangePulse_Matrix2 = Sim_CUDA_signalGeneration(RangePulseCPU,array_data, indices, indices.shape[0], N_ADC, NPulseFrame, Lambda, Ts,FMCW_ChirpSlobe,
                  PrecodingMatrix_real,PrecodingMatrix_imag,PrecodingMatrix.shape[0],PrecodingMatrix.shape[1],
                                                        Frame_PulseTimes,pulsebias
              )
              er = np.linalg.norm(h_RangePulse-RangePulseCPU)
              er1 = np.linalg.norm(RangePulse-RangePulse_Matrix)
              er2 = np.linalg.norm(RangePulse-RangePulse_Matrix)/np.linalg.norm(RangePulse)
              er3 = np.linalg.norm(RangePulse-RangePulse_Matrix)/np.linalg.norm(RangePulse_Matrix)
              RangePulseCPU_complex = RangePulseCPU.reshape((int(NRX), int(NPulseFrame), int(N_ADC), 2))
              RangePulse2 = RangePulseCPU_complex[..., 0] + 1j * RangePulseCPU_complex[..., 1]
              er4 = np.linalg.norm(RangePulse2-RangePulse_Matrix)
              er5 = np.linalg.norm(RangePulse2-RangePulse)
              RangePulse_xx = np.zeros((NRX,NPulseFrame,N_ADC),dtype=complex)        
              er6Large = np.linalg.norm(RangePulse_xx-RangePulse_Matrix)  
              for iADC in range(N_ADC):
                for ip in range(NPulseFrame):
                  for irx in range(NRX):
                    index = 2 * (iADC + ip * N_ADC + irx * N_ADC * NPulseFrame)
                    RangePulse_xx[irx,ip,iADC]=RangePulseCPU[index]+1j*RangePulseCPU[index+1]
              er6 = np.linalg.norm(RangePulse_xx-RangePulse_Matrix)
              er7 = np.linalg.norm(RangePulse_xx-RangePulse2)
              er8 = np.linalg.norm(RangePulse_Matrix2-RangePulse_Matrix)
                    
            print("from cuda")
            # Add Noise
            for iADC in range(RangePulse.shape[2]):
              for ip in range(RangePulse.shape[1]):
                for irx in range(RangePulse.shape[0]):
                  SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += RangePulse[irx,ip,iADC]
    else: # on CPU           
      for isrx,suiteRX_d_drate_amp in path_d_drate_amp.items():
          for irrx,radarRX_d_drate_amp in suiteRX_d_drate_amp.items():
            for irx,RX_d_drate_amp in radarRX_d_drate_amp.items():
              for istx,suiteTX_d_drate_amp in RX_d_drate_amp.items():
                if istx == isrx:
                  for irtx,radarTX_d_drate_amp in suiteTX_d_drate_amp.items():
                    if irtx == irrx:
                      PRI = ssp.RadarSpecifications[isrx][irrx]['PRI']
                      Ts = ssp.RadarSpecifications[isrx][irrx]['Ts']
                      # NPulse = ssp.RadarSpecifications[isrx][irrx]['NPulse']
                      Frame_PulseTimes = 1.0*ssp.RadarSpecifications[isrx][irrx]['RadarTiming'].Frame_PulseTimes
                      Frame_PulseTimes -= ssp.config.CurrentTime
                      NPulseFrame = Frame_PulseTimes.shape[0]
                      pulsebias = ssp.RadarSpecifications[isrx][irrx]['RadarTiming'].n_pulse - NPulseFrame 
                      RadarMode = ssp.RadarSpecifications[isrx][irrx]['RadarMode']
                      N_ADC = ssp.RadarSpecifications[isrx][irrx]['N_ADC']
                      # if RadarMode=='CW':
                      #   N_ADC = 1
                      iADC = np.arange(N_ADC)
                      Lambda = ssp.RadarSpecifications[isrx][irrx]['Lambda']
                      FMCW_ChirpSlobe = ssp.RadarSpecifications[isrx][irrx]['FMCW_ChirpSlobe']
                      PrecodingMatrix = ssp.RadarSpecifications[isrx][irrx]['PrecodingMatrix']
                      # SaveSignalGenerationTime = ssp.RadarSpecifications[isrx][irrx]['SaveSignalGenerationTime']
                      # usefullIIndices_Frame_PulseTimes_start = ssp.RadarSpecifications[isrx][irrx]['usefullIIndices_Frame_PulseTimes_start']
                      # start_Pulse_index = 0
                      # if SaveSignalGenerationTime:
                      #   start_Pulse_index = usefullIIndices_Frame_PulseTimes_start
                      # if ssp.RadarSpecifications[isrx][irrx]['continuousCPIsTrue_oneCPIpeerFrameFalse']:
                      #   start_Pulse_index = 0
                      FMCWRadar = 1
                      if RadarMode=='FMCW':
                        FMCWRadar = 1
                      if RadarMode=='Pulse':
                        FMCWRadar = 0
                        PulseWaveform = ssp.RadarSpecifications[isrx][irrx]['PulseWaveform']
                        Waveform = ssp.radar.radarwaveforms.barker_code(11)
                        if PulseWaveform.startswith('waveform_'):
                          waveform_file = PulseWaveform[len('waveform_'):]
                          Waveform = np.load(waveform_file)
                        if PulseWaveform=='UWB':
                          FMCWRadar = 100
                          Lwaveform = int(1/Ts/ssp.RadarSpecifications[isrx][irrx]['RF_AnalogNoiseFilter_Bandwidth'])
                          # Waveform = ssp.radar.radarwaveforms.gaussian_waveform(Lwaveform,std_dev=.3)
                          Waveform = ssp.radar.radarwaveforms.gaussian_waveform(Lwaveform,std_dev=0.6)
                          f0T=Ts*(ssp.LightSpeed/ssp.RadarSpecifications[isrx][irrx]['Lambda']-.5*ssp.RadarSpecifications[isrx][irrx]['RF_AnalogNoiseFilter_Bandwidth'])
                          Waveform *= np.sin(np.pi*2*np.arange(Waveform.shape[0])*f0T)
                          ssp.RadarSpecifications[isrx][irrx]['PulseWaveform_Loaded']=Waveform
                      if RadarMode=='CW':
                        FMCWRadar = 2
                        
                        
                      start_Pulse_index = 0
                              
                      for itx,TX_d_drate_amp in radarTX_d_drate_amp.items():

                        for d_drate_amp in TX_d_drate_amp:
                          
                          # if len(d_drate_amp[3])==0:
                          #   continue
                          HardwareDCBlockerAttenuation = 1 # d_drate_amp[1]
                          # if d_drate_amp[1]==0:
                          #   continue
                          # if d_drate_amp[0]==0: TX RX same position
                          #   continue
                          # if d_drate_amp[0] > PRI*LightSpeed:
                          #   Save it in a buffer for next pulses or fix PRF : effective_d = mod(d_drate_amp[0] , PRI*LightSpeed)
                          #   continue
                          # byPass = bpy.data.objects["Simulation Settings"]["Debug_BypassCPITiming"]
                          # if byPass:
                          #   start_Pulse_index = 0
                          #   StartPulseIndex = 0
                          #   if NPulseFrame > ssp.RadarSpecifications[isrx][irrx]['NPulse']:
                          #     NPulseFrame = ssp.RadarSpecifications[isrx][irrx]['NPulse']
                          # SimDopplerEffect = 1 # Should be 1; for test and analysis, can set to 0
                          for ip in range(start_Pulse_index,NPulseFrame):
                            ip2 = pulsebias + ip
                            ipPM = ip2 % PrecodingMatrix.shape[0]
                            MIMO_Precoding = PrecodingMatrix[ipPM][itx]
                            amp=MIMO_Precoding*d_drate_amp[2]
                            # if itx == 0: ######### Temp
                            #   amp=0
                            if amp == 0:
                              continue
                            # d_of_t =  d_drate_amp[0] + (ip * PRI + iADC * Ts) * d_drate_amp[1]*SimDopplerEffect
                            d_of_t =  d_drate_amp[0] + (Frame_PulseTimes[ip] + iADC * Ts) * d_drate_amp[1]#*SimDopplerEffect
                            
                            if FMCWRadar == 1:
                              phase2 = 2*np.pi*(
                                d_of_t/Lambda
                                +FMCW_ChirpSlobe/LightSpeed*iADC * Ts*d_of_t
                                -.5*FMCW_ChirpSlobe*(d_of_t/LightSpeed)*(d_of_t/LightSpeed)
                                )
                              PhaseNoiseTerm = 0
                              phase2+=PhaseNoiseTerm 
                              # ipPM = ip % PrecodingMatrix.shape[0]
                              SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += amp*np.exp(1j*phase2) * HardwareDCBlockerAttenuation
                          #   ssp.RadarSpecifications[isrx][irrx]['RangePulseRX'][iADC,ip,irx]+=PrecodingMatrix[ipPM][itx]*d_drate_amp[2]*np.exp(1j*phase2) * HardwareDCBlockerAttenuation
                            elif FMCWRadar == 2: # CW
                              phase2 = 2*np.pi*d_of_t/Lambda
                              SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += amp*np.exp(1j*phase2) * HardwareDCBlockerAttenuation
                            elif FMCWRadar == 0:
                              Window_Waveform = np.zeros_like(d_of_t,dtype=complex)
                              ind1 = int( (d_drate_amp[0] + (Frame_PulseTimes[ip] + 0 * Ts) * d_drate_amp[1]) / LightSpeed / Ts)
                              if ind1 < Window_Waveform.shape[0]:
                                ind1 = np.arange(ind1,min([ind1+Waveform.shape[0],Window_Waveform.shape[0]]))
                                ind2 = np.arange(0,ind1.shape[0])
                                Window_Waveform[ind1]=Waveform[ind2]
                                phase2 = 2*np.pi*(d_of_t/Lambda)
                                SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += amp*np.exp(1j*phase2)*Window_Waveform
                            elif FMCWRadar == 100:
                              Lwaveform = int(1.5*1/Ts/ssp.RadarSpecifications[isrx][irrx]['RF_AnalogNoiseFilter_Bandwidth'])
                              Waveform = ssp.radar.radarwaveforms.gaussian_waveform(Lwaveform,std_dev=0.6).astype(np.complex128)
                              f00=(ssp.LightSpeed/ssp.RadarSpecifications[isrx][irrx]['Lambda']-.5*ssp.RadarSpecifications[isrx][irrx]['RF_AnalogNoiseFilter_Bandwidth'])
                              t = Ts*np.arange((Waveform.shape[0]))
                              # t-= (d_drate_amp[0] + (Frame_PulseTimes[ip] + 0 * Ts) * d_drate_amp[1]) / LightSpeed
                              Waveform *= np.exp(1j*np.pi*2*f00*t)
                              
                              Window_Waveform = np.zeros_like(d_of_t,dtype=complex)
                              ind1 = int( (d_drate_amp[0] + (Frame_PulseTimes[ip] + 0 * Ts) * d_drate_amp[1]) / LightSpeed / Ts)
                              if ind1 < Window_Waveform.shape[0]:
                                ind1 = np.arange(ind1,min([ind1+Waveform.shape[0],Window_Waveform.shape[0]]))
                                ind2 = np.arange(0,ind1.shape[0])
                                Window_Waveform[ind1]=Waveform[ind2]
                                phase2 = 2*np.pi*(d_of_t/Lambda)
                                SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += amp*Window_Waveform
                                # SuiteRadarRangePulseRXSignals[isrx]['radars'][irrx][iADC,ip,irx] += amp*np.exp(1j*phase2)*Window_Waveform
      
          
    ## Ampedance and ADC 
    
    for isuite in range(len(SuiteRadarRangePulseRXSignals)):
      for iradar in range(len(SuiteRadarRangePulseRXSignals[isuite]['radars'])):
        specifications = ssp.RadarSpecifications[isuite][iradar]
        ImpedanceFactor = np.sqrt(specifications['ADC_ImpedanceFactor'])
        LNA_Gain = specifications['ADC_LNA_Gain']
        ADC_Peak2Peak = specifications['ADC_peak2peak']
        ADC_Levels = specifications['ADC_levels']
        SuiteRadarRangePulseRXSignals[isuite]['radars'][iradar] = apply_adc(SuiteRadarRangePulseRXSignals[isuite]['radars'][iradar],ADC_Peak2Peak,ADC_Levels,ImpedanceFactor,LNA_Gain,specifications['ADC_SaturationEnabled'])
        signal_time = [SuiteRadarRangePulseRXSignals[isuite]['radars'][iradar],ssp.RadarSpecifications[isuite][iradar]['RadarTiming'].Frame_PulseTimes]
        specifications['CPI_Buffer']=push_radar_buffer(specifications['CPI_Buffer'],signal_time)
        specifications['CPI_Buffer'],SuiteRadarRangePulseRXSignals[isuite]['radars'][iradar]=pop_radar_buffer(specifications['CPI_Buffer'],specifications['NPulse'],ssp.RadarSpecifications[isuite][iradar]['continuousCPIsTrue_oneCPIpeerFrameFalse'])
        
          
    return SuiteRadarRangePulseRXSignals

@cuda.jit
def CUDA_signalGeneration(RangePulse, array_data, indices, indices_shape, N_ADC, NPulseFrame, Lambda, Ts, FMCW_ChirpSlobe,
                          PrecodingMatrix_real, PrecodingMatrix_imag, PrecodingMatrix_shape0, PrecodingMatrix_shape1,
                          Frame_PulseTimes, pulsebias):
    NRX = indices_shape // 2
    
    iADC = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ip = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    irx = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z
    LightSpeed = 299792458.0

    if iADC < N_ADC and ip < NPulseFrame and irx < NRX:
        sig_real = 0.0
        sig_imag = 0.0
        for iscatter in range(indices[2 * irx]//4, indices[2 * irx + 1] // 4):
            itx = array_data[iscatter * 4 + 0]
            ip2 = (pulsebias + ip) % PrecodingMatrix_shape0
            index = int(ip2 * PrecodingMatrix_shape1 + itx)
            d = array_data[iscatter * 4 + 1]
            dr = array_data[iscatter * 4 + 2]
            a = array_data[iscatter * 4 + 3]
            d_of_t = d + (Frame_PulseTimes[ip] + iADC * Ts) * dr
            
            phase2 = 2 * np.pi * (d_of_t / Lambda + FMCW_ChirpSlobe / LightSpeed * iADC * Ts * d_of_t
                                    - 0.5 * FMCW_ChirpSlobe * (d_of_t / LightSpeed) * (d_of_t / LightSpeed))
            
            cos_phase = np.cos(phase2)
            sin_phase = np.sin(phase2)
            
            w_real = PrecodingMatrix_real[index]
            w_imag = PrecodingMatrix_imag[index]
            
            sig_real += (w_real * cos_phase - w_imag * sin_phase) * a
            sig_imag += (w_real * sin_phase + w_imag * cos_phase) * a

        index = 2 * (iADC + ip * N_ADC + irx * N_ADC * NPulseFrame)
        RangePulse[index] = sig_real
        RangePulse[index + 1] = sig_imag
def Sim_CUDA_signalGeneration(RangePulse, array_data, indices, indices_shape, N_ADC, NPulseFrame, Lambda, Ts, FMCW_ChirpSlobe,
                          PrecodingMatrix_real, PrecodingMatrix_imag, PrecodingMatrix_shape0, PrecodingMatrix_shape1,
                          Frame_PulseTimes, pulsebias):
    NRX = indices_shape // 2
    LightSpeed = 299792458.0
    RangePulse_Matrix = np.zeros((NRX,NPulseFrame,N_ADC),dtype=complex)
    RangePulse_Matrix2 = np.zeros((NRX,NPulseFrame,N_ADC),dtype=complex)
    
    for iADC in range(N_ADC):
      for ip in range(NPulseFrame):
        for irx in range(NRX):
          sig_real = 0.0
          sig_imag = 0.0
          for iscatter in range(indices[2 * irx]//4, indices[2 * irx + 1] // 4):
              itx = array_data[iscatter * 4 + 0]
              ip2 = (pulsebias + ip) % PrecodingMatrix_shape0
              index = int(ip2 * PrecodingMatrix_shape1 + itx)
              d = array_data[iscatter * 4 + 1]
              dr = array_data[iscatter * 4 + 2]
              a = array_data[iscatter * 4 + 3]
              d_of_t = d + (Frame_PulseTimes[ip] + iADC * Ts) * dr
              
              phase2 = 2 * np.pi * (d_of_t / Lambda + FMCW_ChirpSlobe / LightSpeed * iADC * Ts * d_of_t
                                      - 0.5 * FMCW_ChirpSlobe * (d_of_t / LightSpeed) * (d_of_t / LightSpeed))
              
              cos_phase = np.cos(phase2)
              sin_phase = np.sin(phase2)
              
              w_real = PrecodingMatrix_real[index]
              w_imag = PrecodingMatrix_imag[index]
              
              sig_real += (w_real * cos_phase - w_imag * sin_phase) * a
              sig_imag += (w_real * sin_phase + w_imag * cos_phase) * a
              RangePulse_Matrix[irx,ip,iADC]=(sig_real+1j*sig_imag)
              RangePulse_Matrix2[irx,ip,iADC]+=(w_real+1j*w_imag)*a*np.exp(1j*phase2)
          index = 2 * (iADC + ip * N_ADC + irx * N_ADC * NPulseFrame)
          RangePulse[index] = sig_real
          RangePulse[index + 1] = sig_imag
    return RangePulse_Matrix,RangePulse_Matrix2    
def push_radar_buffer(buf,data):
  if len(buf)==0:
    return data
  else:
    return [np.concatenate((buf[0], data[0]), axis=1),np.concatenate((buf[1], data[1]), axis=0)]
  
def pop_radar_buffer(buf_time,NPulse,continuousCPIsTrue_oneCPIpeerFrameFalse):
  M = int(buf_time[0].shape[1]/NPulse)
  o=[]
  if buf_time[0].shape[1]>=NPulse:
    if continuousCPIsTrue_oneCPIpeerFrameFalse:
      for i in range(M):
        o.append([buf_time[0][:,i * NPulse : (i+1) * NPulse,:],buf_time[1][i * NPulse : (i+1) * NPulse]])
      buf_time[0]=buf_time[0][:, M * NPulse:, :]
      buf_time[1]=buf_time[1][ M * NPulse:]
    else:
      o.append([buf_time[0][:,(M-1) * NPulse : M * NPulse,:],buf_time[1][(M-1) * NPulse : M * NPulse]])
      buf_time[0]=buf_time[0][:, M * NPulse:, :]
      buf_time[1]=buf_time[1][ M * NPulse:]
  return buf_time,o
def SensorsSignalProccessing(Signals):
      
      RangeDopplerDistributed = []
      for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
          RangeDopplerDistributed0 = []
          All_RangeResolutions = []
          for iradar,specifications in enumerate(radarSpecifications):

            XRadar = Signals[isuite]['radars'][iradar]
            FMCW_ChirpSlobe = ssp.RadarSpecifications[isuite][iradar]['FMCW_ChirpSlobe']
            Ts = ssp.RadarSpecifications[isuite][iradar]['Ts']
            PRI = ssp.RadarSpecifications[isuite][iradar]['PRI']
            Lambda = ssp.RadarSpecifications[isuite][iradar]['Lambda']
            PrecodingMatrix = ssp.RadarSpecifications[isuite][iradar]['PrecodingMatrix']

            All_RangeResolutions.append( LightSpeed/2/ssp.RadarSpecifications[isuite][iradar]['FMCW_Bandwidth'] )

            fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
            X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

            NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
            NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
            X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts

            # plt.plot(d_fft,np.abs(X_fft_fast[:,0,0]))
            # yyyyyyy


            Range_Start = specifications['Range_Start']
            Range_End = specifications['Range_End']
            d1i = int(NFFT_Range*Range_Start/100.0)
            d2i = int(NFFT_Range*Range_End/100.0)
            d_fft = d_fft[d1i:d2i]
            X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

            # STC_Enabled = 0
            # if STC_Enabled:
            #   STC_Signal = (d_fft + .001) **2
            #   X_fft_fast = X_fft_fast * STC_Signal[:, np.newaxis, np.newaxis]

            # print("XRadar shape:", XRadar.shape)
            # XRadar shape: (128, 192, 16)

            M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
            L = X_fft_fast.shape[1]
            Leff = int(L/M_TX)
            # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
            # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
            N_Doppler = Leff
            f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
            f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

            PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

            rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
            for idop , f_Doppler_i in enumerate(f_Doppler):
              dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
              X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
              if 1:
                rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
              else:
                rangeTXRX = np.zeros((X_doppler_comp.shape[0], PrecodingMatrixInv.shape[0], X_doppler_comp.shape[2]),dtype=X_doppler_comp.dtype)
                for i_r in range(X_doppler_comp.shape[0]):
                  for i_rx in range(X_doppler_comp.shape[2]):
                    temp = PrecodingMatrixInv @ X_doppler_comp[i_r,:,i_rx]
                    rangeTXRX[i_r,:,i_rx] = temp
                # print("with conj",np.linalg.norm(np.conj(np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv))-rangeTXRX)) % Not correct
                # print(np.linalg.norm((np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv))-rangeTXRX)) % Correct
                # print(np.linalg.norm(rangeTXRX))
                # ccccccccccccc

              # print(rangeTXRX.shape) (64, 12, 16)
              rangeDopplerTXRX[:, idop, :, :] = rangeTXRX

            # for irange in range(rangeDopplerTXRX.shape[0]):
            #   plt.plot(f_Doppler,np.abs(rangeDopplerTXRX[irange,:,0,0]))
            # plt.figure()
            # plt.imshow(np.abs(rangeDopplerTXRX[:,:,0,0]), aspect='auto')
            # xxxxxxxx
            RangeDopplerDistributed0.append([rangeDopplerTXRX,d_fft,f_Doppler])
          RangeDopplerDistributed.append(RangeDopplerDistributed0)
      # RangeAngleMapCalc = 1
      # if RangeAngleMapCalc:
      #   print("Range Angle Map Calculation ...")
      #   for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      #     for iradar,specifications in enumerate(radarSpecifications):
      #       global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
      #       PosIndex = np.array(specifications['Local_location_TXplusRX_Center'])
      #       azimuths = PosIndex[:, 0]
      #       elevations = PosIndex[:, 1]

      #       d_az = np.max(azimuths)-np.min(azimuths)
      #       d_el = np.max(elevations)-np.min(elevations)
      #       if d_az>d_el:
      #         sorted_indices = np.argsort(azimuths)
      #         sorted_PosIndex = PosIndex[sorted_indices,:]
      #         sorted_PosIndex[:,0]-=sorted_PosIndex[0,0]
      #         sorted_PosIndex[:,1]-=sorted_PosIndex[0,1]
      #         sorted_PosIndex[:,0]=np.round(sorted_PosIndex[:,0])
      #         sorted_PosIndex[:,1]=np.round(sorted_PosIndex[:,1])

      #         unique_azimuths, unique_indices = np.unique(sorted_PosIndex[:, 0], return_index=True)
      #         unique_PosIndex = sorted_PosIndex[unique_indices,:]

      #         # print(unique_PosIndex)




      #       rangeDopplerTXRX,d_fft,f_Doppler = RangeDopplerDistributed[isuite][iradar]
      #       rangeTXRX = np.mean(rangeDopplerTXRX,axis=1)

      #       # Selected_ind = [(0,0), (1,7), (2,6), (11,15)]
      #       # rows, cols = zip(*Selected_ind)
      #       rows = unique_PosIndex[:,2].astype(int)
      #       cols = unique_PosIndex[:,3].astype(int)
      #       # print(rows)
      #       # print(cols)
      #       rangeVA = rangeTXRX[:, rows, cols]
      #       angle_window = scipy.signal.windows.hamming(rangeVA.shape[1])
      #       X_windowed_rangeVA = rangeVA * angle_window[np.newaxis,:]

      #       NFFT_Angle_OverNextPow2 =  1
      #       NFFT_Angle = int(2 ** (np.ceil(np.log2(X_windowed_rangeVA.shape[1]))+NFFT_Angle_OverNextPow2))
      #       RangeAngleMap = np.fft.fft(X_windowed_rangeVA, axis=1, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
      #       RangeAngleMap = np.fft.fftshift(RangeAngleMap, axes=1)
      #       # sina_fft = np.rad2deg(np.arcsin(np.linspace(-1,1,NFFT_AngleNFFT_Angle)))
      #       extent = [-.5 ,.5,d_fft[-1],d_fft[0]]
      #       plt.figure()
      #       plt.imshow(np.abs(RangeAngleMap), extent=extent, aspect='auto')

      #       angles = np.linspace(-np.pi/2, np.pi/2, NFFT_Angle)

      #       # Define range bins from d_fft
      #       ranges = d_fft
      #       R, Theta = np.meshgrid(ranges, angles)
      #       plt.figure()
      #       ax = plt.subplot(111, polar=True)
      #       c = ax.pcolormesh(Theta, R, np.abs(RangeAngleMap).T, shading='auto')
      #       ax.set_thetalim(-np.pi/2, np.pi/2)
      #       ax.set_ylim(0, ranges[-1])
      #       plt.colorbar(c, label='Magnitude')
      #       plt.show()

      # Uncomment2SeePolarRangeAngle
      # xxxxxxxxxxxxxxxx
      print("Grid Search")
      x_start, y_start, z_start = ssp.config.Detection_Parameters_xyz_start
      N_x, N_y, N_z = ssp.config.Detection_Parameters_xyz_N
      gridlen = ssp.config.Detection_Parameters_gridlen
      x_points = x_start + (np.arange(N_x)) * gridlen
      y_points = y_start + (np.arange(N_y)) * gridlen
      z_points = z_start + (np.arange(N_z)) * gridlen
      X, Y, Z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
      grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T # 1000 x 3

      vx_start, vy_start, vz_start = 0, 0, 0
      N_vx, N_vy, N_vz = 1, 1, 1
      gridvlen = 0
      vx_points = vx_start + (np.arange(N_vx)) * gridvlen
      vy_points = vy_start + (np.arange(N_vy)) * gridvlen
      vz_points = vz_start + (np.arange(N_vz)) * gridvlen
      X, Y, Z = np.meshgrid(vx_points, vy_points, vz_points, indexing='ij')
      grid_velocities = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T # 1000 x 3
      all_outputs = []

      for i_grid_points in range(grid_points.shape[0]):
        # clear_output()
        print("Grid Search: ",i_grid_points,grid_points.shape[0])
        p0=Vector(grid_points[i_grid_points,:])
        for i_grid_velocities in range(grid_velocities.shape[0]):
          v0=Vector(grid_velocities[i_grid_velocities,:])
          # DetectionCount = 0
          Detection_data = []
          for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
            for iradar,specifications in enumerate(radarSpecifications):
              global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']

              R = np.linalg.norm(p0-global_location_Center)

              n = (p0-global_location_Center)/R
              fd = 2*v0.dot(n)/specifications['Lambda']

              rangeDopplerTXRX,d_fft,f_Doppler = RangeDopplerDistributed[isuite][iradar]

              indices = find_indices_within_distance(d_fft,R,1*np.sqrt(3)*gridlen)
              if indices.shape[0]==0:
                continue
              DopplerTXRX=rangeDopplerTXRX[indices,:,:,:]
              if 1:
                DopplerTXRX=np.mean(DopplerTXRX,axis=0)
              else:
                # max over axe 0
                DopplerTXRX=np.mean(DopplerTXRX,axis=0)

              indices_fd = find_indices_within_distance(f_Doppler,fd,2*gridvlen/specifications['Lambda'])

              if indices_fd.shape[0]==0:
                continue


              TXRX=DopplerTXRX[indices_fd,:,:]
              TXRX=np.mean(TXRX,axis=0)
              sv = 0*TXRX
              for itx , txPos in enumerate(global_location_TX):
                dtx = np.linalg.norm(p0-txPos)
                for irx , rxPos in enumerate(global_location_RX):
                  drx = np.linalg.norm(p0-rxPos)
                  sv[itx,irx]=np.exp(1j*2*np.pi/specifications['Lambda']*(dtx+drx))

              TXRX_vectorized = TXRX.reshape(-1, 1)
              sv_vectorized = sv.reshape(-1, 1)

              Guard_Len = 4
              Wing_Len = 2*sv.shape[0]*sv.shape[1]
              Complementary_indices_fd = CFAR_Window_Selection_F(f_Doppler.shape[0],indices_fd,Guard_Len,Wing_Len)

              Guard_Len_Range = 4
              Wing_Len_Range = 2*sv.shape[0]*sv.shape[1]
              Complementary_indices_range = CFAR_Window_Selection_F(d_fft.shape[0],indices,Guard_Len_Range,Wing_Len_Range)

              Complementary_indices_range = indices # temporary
              SecondaryData_RD = rangeDopplerTXRX[Complementary_indices_range,:,:,:]
              SecondaryData_RD = SecondaryData_RD[:,Complementary_indices_fd,:,:]
              energies = []
              indices = []
              for isd in range(SecondaryData_RD.shape[0]):
                  for isd1 in range(SecondaryData_RD.shape[1]):
                      TXRX_k = SecondaryData_RD[isd, isd1, :, :]
                      TXRX_k_vectorized = TXRX_k.reshape(-1, 1)
                      energy = np.abs(np.conj(TXRX_k_vectorized).T @ TXRX_k_vectorized)
                      energies.append(energy[0][0])
                      indices.append((isd, isd1))
              energies_array = np.array(energies)
              sorted_indices = np.argsort(energies_array)
              sorted_energies = energies_array[sorted_indices]
              sorted_indices_pairs = [indices[idx] for idx in sorted_indices]
              num_OSCFAR = int(0.7 * len(sorted_indices_pairs))
              Ntx = SecondaryData_RD.shape[2]
              Nrx = SecondaryData_RD.shape[3]
              print(num_OSCFAR,Ntx * Nrx)
              if 0:
                sum_matrix = np.zeros((Ntx * Nrx, Ntx * Nrx), dtype=SecondaryData_RD.dtype)
                for i in range(num_OSCFAR):
                    isd, isd1 = sorted_indices_pairs[i]
                    TXRX_k = SecondaryData_RD[isd, isd1, :, :]
                    TXRX_k_vectorized = TXRX_k.reshape(-1, 1)
                    product_matrix = TXRX_k_vectorized @ np.conj(TXRX_k_vectorized).T
                    sum_matrix += product_matrix

                Gamma_inv = np.linalg.pinv(sum_matrix)
              else:
                sum_en = 0
                for i in range(num_OSCFAR):
                    sum_en += sorted_energies[i]
                Gamma_inv = Ntx * Nrx / sum_en * np.eye(Ntx * Nrx, dtype=SecondaryData_RD.dtype)

              Kelly_num = np.abs(np.conj(sv_vectorized).T @ Gamma_inv @ TXRX_vectorized)[0][0]**2/np.abs(np.conj(sv_vectorized).T @ Gamma_inv @ sv_vectorized)[0][0]
              Kelly_denum = np.abs(np.conj(TXRX_vectorized).T @ Gamma_inv @ TXRX_vectorized)[0][0]
              # print(Kelly_num,Kelly_denum)
              Detection_data.append([Kelly_num,Kelly_denum])
          KellyNum = 0
          KellyDeNum = 1
          for idet,detdata in enumerate(Detection_data):
            Kelly_num,Kelly_denum = detdata
            KellyNum += Kelly_num
            KellyDeNum += Kelly_denum
          Kelly = KellyNum / KellyDeNum
          all_outputs.append(Kelly)

      return grid_points , grid_velocities , all_outputs
def SensorsSignalProccessing_Angle(Signals):
  RangeDopplerDistributed = []
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      RangeDopplerDistributed0 = []
      All_RangeResolutions = []
      for iradar,specifications in enumerate(radarSpecifications):
        XRadar = Signals[isuite]['radars'][iradar]
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        Lambda = specifications['Lambda']
        PrecodingMatrix = specifications['PrecodingMatrix']
        
        RadarMode = specifications['RadarMode']

        All_RangeResolutions.append( LightSpeed/2/specifications['FMCW_Bandwidth'] )

        fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
        X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

        if RadarMode == 'Pulse':
          waveform_MF = specifications['PulseWaveform_Loaded']
          matched_filter = np.conj(waveform_MF[::-1])
          X_fft_fast = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
          X_fft_fast=X_fft_fast[matched_filter.shape[0]-1:,:,:]
          d_fft = np.arange(X_fft_fast.shape[0]) * LightSpeed * Ts /2
          
        else:
          NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
          NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
          X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
          d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
        Range_Start = specifications['Range_Start']
        Range_End = specifications['Range_End']
        d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
        d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
        d_fft = d_fft[d1i:d2i]
        X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

        # print("XRadar shape:", XRadar.shape)
        # XRadar shape: (128, 192, 16)

        M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
        L = X_fft_fast.shape[1]
        Leff = int(L/M_TX)
        
        if ssp.config.DopplerProcessingMethod_FFT_Winv:
          # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
          # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
          N_Doppler = Leff
          f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
          f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

          PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

          rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
          for idop , f_Doppler_i in enumerate(f_Doppler):
            dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
            X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
            rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
            rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
        else:
          rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
          for ipulse in range(Leff):
            ind = ipulse*M_TX
            rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
          NFFT_Doppler_OverNextPow2=0
          NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
          rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
          rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
          f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
            
            
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        if len(global_location_TX)+len(global_location_RX)==2:
          ssp.config.ax[0,1].cla()
          ssp.config.ax[0,2].cla()
          ssp.config.ax[0,1].plot(np.real(XRadar[:,0,0]))
          ssp.config.ax[0,2].plot(np.abs(X_fft_fast[:,0,0]))
          
          plt.draw() 
          plt.pause(0.1)
          continue
        
        # RangeAngleMapCalc
        PosIndex = np.array(specifications['Local_location_TXplusRX_Center'])
        azimuths = PosIndex[:, 0]
        elevations = PosIndex[:, 1]
        
        d_az = np.max(azimuths)-np.min(azimuths)
        d_el = np.max(elevations)-np.min(elevations)
        if d_az>d_el:
          sorted_indices = np.argsort(azimuths)
          sorted_PosIndex = PosIndex[sorted_indices,:]
          sorted_PosIndex[:,0]-=sorted_PosIndex[0,0]
          sorted_PosIndex[:,1]-=sorted_PosIndex[0,1]
          sorted_PosIndex[:,0]=np.round(sorted_PosIndex[:,0])
          sorted_PosIndex[:,1]=np.round(sorted_PosIndex[:,1])

          unique_azimuths, unique_indices = np.unique(sorted_PosIndex[:, 0], return_index=True)
          unique_PosIndex = sorted_PosIndex[unique_indices,:]

          # print(unique_PosIndex)



        if 0:
          rangeTXRX = np.mean(rangeDopplerTXRX,axis=1)
        else:
          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          # print(ssp.config.ax)
          # print(ssp.config.ax[1,0])
          # print(len(ssp.config.ax))
          ssp.config.ax[1,0].imshow(rangeDoppler4CFAR, aspect='auto')
          ssp.config.ax[1,0].set_title("Range Doppler abs(mean) (CFAR)")
          ssp.config.ax[1,0].set_xlabel('Doppler')
          ssp.config.ax[1,0].set_ylabel('Range')
          for irange in range(rangeDoppler4CFAR.shape[0]):
            doppler_ind = np.argmax(rangeDoppler4CFAR[irange])
            rangeTXRX[irange,:,:]=rangeDopplerTXRX[irange,doppler_ind,:,:]

        # Selected_ind = [(0,0), (1,7), (2,6), (11,15)]
        # rows, cols = zip(*Selected_ind)
        
        rows = unique_PosIndex[:,2].astype(int)
        cols = unique_PosIndex[:,3].astype(int)
        # print(rows)
        # print(cols)
        rangeVA = rangeTXRX[:, rows, cols]
        angle_window = scipy.signal.windows.hamming(rangeVA.shape[1])
        X_windowed_rangeVA = rangeVA * angle_window[np.newaxis,:]
        
        # Bartlet Angle 
        # Capon
        
        NFFT_Angle_OverNextPow2 =  1
        NFFT_Angle = int(2 ** (np.ceil(np.log2(X_windowed_rangeVA.shape[1]))+NFFT_Angle_OverNextPow2))
        RangeAngleMap = np.fft.fft(X_windowed_rangeVA, axis=1, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
        RangeAngleMap = np.fft.fftshift(RangeAngleMap, axes=1)
        # sina_fft = np.rad2deg(np.arcsin(np.linspace(-1,1,NFFT_AngleNFFT_Angle)))
        extent = [-.5 ,.5,d_fft[-1],d_fft[0]]
        ssp.config.ax[1,1].imshow(np.abs(RangeAngleMap), extent=extent, aspect='auto')
        ssp.config.ax[1,1].set_xlabel('sin(az)')
        ssp.config.ax[1,1].set_ylabel('Range (m)')
        ssp.config.ax[1,1].set_title("Bartlet Beamformer")
        ssp.config.ax[0,1].cla()
        ssp.config.ax[0,2].cla()
        ssp.config.ax[0,1].plot(np.real(Signals[isuite]['radars'][iradar][:,0,0]))
        ssp.config.ax[0,2].plot(np.abs(np.fft.fft(Signals[isuite]['radars'][iradar][:,0,0])))
        
        ssp.config.ax[0,0].cla()
        for __ in ssp.lastScatterInfo[ssp.config.CurrentFrame-bpy.context.scene.frame_start]:
          _=__[0]
          ssp.config.ax[0,0].scatter(_[0],_[1],c='k',marker='x',s=20)
          # ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        for _ in global_location_TX:
          ssp.config.ax[0,0].scatter(_[0],_[1],c='r',marker='x')
          # ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='r',marker='x')
        for _ in global_location_RX:
          ssp.config.ax[0,0].scatter(_[0],_[1],c='b',marker='x')
          # ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='b',marker='x')
          
        ssp.config.ax[2,2].cla()
        for __ in ssp.lastScatterInfo[ssp.config.CurrentFrame-bpy.context.scene.frame_start]:
          _=__[0]
          ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        for _ in global_location_TX:
          ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='r',marker='x')
        for _ in global_location_RX:
          ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='b',marker='x')
          
        
        ssp.config.ax[0,1].set_xlabel('ADC sample')
        ssp.config.ax[0,1].set_ylabel('Amp')
        ssp.config.ax[0,1].set_title("ADC")
        ssp.config.ax[0,2].set_xlabel('Range')
        ssp.config.ax[0,2].set_ylabel('Amp')
        ssp.config.ax[0,2].set_title("Range FFT")
                
        
        plt.draw()  # Redraw the figure
        plt.pause(0.1)
        
        RangeDopplerDistributed0.append([rangeDopplerTXRX,d_fft,f_Doppler])
      RangeDopplerDistributed.append(RangeDopplerDistributed0)

      for icam in range(len(Signals[isuite]['cameras'])):
        ssp.config.ax[2,0].imshow(Signals[isuite]['cameras'][icam])
        plt.draw() 
        plt.pause(0.1)
        break
      
      ssp.config.ax[2,1].cla()
      for ilid in range(len(Signals[isuite]['lidars'])):
        pc = Signals[isuite]['lidars'][ilid]
        if pc.shape[0]==0:
          continue
        ssp.config.ax[2,1].scatter(pc[:,0],pc[:,1],pc[:,2])
        plt.draw() 
        plt.pause(0.1)
        break
  
  Triangles = ssp.utils.exportBlenderTriangles()
  for _ in Triangles:
      for __ in _:
          i,j=0,1
          ssp.config.ax[2,1].plot([__[i][0],__[j][0]],[__[i][1],__[j][1]],[__[i][2],__[j][2]])
          i,j=0,2
          ssp.config.ax[2,1].plot([__[i][0],__[j][0]],[__[i][1],__[j][1]],[__[i][2],__[j][2]])
          i,j=2,1
          ssp.config.ax[2,1].plot([__[i][0],__[j][0]],[__[i][1],__[j][1]],[__[i][2],__[j][2]])      
  plt.draw()  # Redraw the figure
  plt.pause(0.1)
def SensorsSignalProccessing_Angle_frame(Signals):
  RangeDopplerDistributed = []
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      RangeDopplerDistributed0 = []
      All_RangeResolutions = []
      for iradar,specifications in enumerate(radarSpecifications):
        if len(Signals[isuite]['radars'][iradar])==0:
          continue
        XRadar = Signals[isuite]['radars'][iradar][0]
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        Lambda = specifications['Lambda']
        PrecodingMatrix = specifications['PrecodingMatrix']
        
        RadarMode = specifications['RadarMode']

        All_RangeResolutions.append( LightSpeed/2/specifications['FMCW_Bandwidth'] )

        fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
        X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

        if RadarMode == 'Pulse':
          waveform_MF = specifications['PulseWaveform_Loaded']
          matched_filter = np.conj(waveform_MF[::-1])
          X_fft_fast = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
          X_fft_fast=X_fft_fast[matched_filter.shape[0]-1:,:,:]
          d_fft = np.arange(X_fft_fast.shape[0]) * LightSpeed * Ts /2
          
        else:
          NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
          NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
          X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
          d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
        Range_Start = specifications['Range_Start']
        Range_End = specifications['Range_End']
        d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
        d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
        d_fft = d_fft[d1i:d2i]
        X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

        # print("XRadar shape:", XRadar.shape)
        # XRadar shape: (128, 192, 16)

        M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
        L = X_fft_fast.shape[1]
        Leff = int(L/M_TX)
        
        if ssp.config.DopplerProcessingMethod_FFT_Winv:
          # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
          # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
          N_Doppler = Leff
          f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
          f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

          PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

          rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
          for idop , f_Doppler_i in enumerate(f_Doppler):
            dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
            X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
            rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
            rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
        else:
          rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
          for ipulse in range(Leff):
            ind = ipulse*M_TX
            rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
          NFFT_Doppler_OverNextPow2=0
          NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
          rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
          rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
          f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
            
            
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        if len(global_location_TX)+len(global_location_RX)==2:
          ssp.config.ax[0,1].cla()
          ssp.config.ax[0,2].cla()
          ssp.config.ax[0,1].plot(np.real(XRadar[:,0,0]))
          ssp.config.ax[0,2].plot(np.abs(X_fft_fast[:,0,0]))
          
          plt.draw() 
          plt.pause(0.1)
          continue
        
        # RangeAngleMapCalc
        PosIndex = np.array(specifications['Local_location_TXplusRX_Center'])
        azimuths = PosIndex[:, 0]
        elevations = PosIndex[:, 1]
        
        d_az = np.max(azimuths)-np.min(azimuths)
        d_el = np.max(elevations)-np.min(elevations)
        if d_az>d_el:
          sorted_indices = np.argsort(azimuths)
          sorted_PosIndex = PosIndex[sorted_indices,:]
          sorted_PosIndex[:,0]-=sorted_PosIndex[0,0]
          sorted_PosIndex[:,1]-=sorted_PosIndex[0,1]
          sorted_PosIndex[:,0]=np.round(sorted_PosIndex[:,0])
          sorted_PosIndex[:,1]=np.round(sorted_PosIndex[:,1])

          unique_azimuths, unique_indices = np.unique(sorted_PosIndex[:, 0], return_index=True)
          unique_PosIndex = sorted_PosIndex[unique_indices,:]

          # print(unique_PosIndex)



        if 0:
          rangeTXRX = np.mean(rangeDopplerTXRX,axis=1)
        else:
          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          
          ssp.config.ax[1,0].imshow(rangeDoppler4CFAR, aspect='auto')
          ssp.config.ax[1,0].set_title("Range Doppler abs(mean) (CFAR)")
          ssp.config.ax[1,0].set_xlabel('Doppler')
          ssp.config.ax[1,0].set_ylabel('Range')
          for irange in range(rangeDoppler4CFAR.shape[0]):
            doppler_ind = np.argmax(rangeDoppler4CFAR[irange])
            rangeTXRX[irange,:,:]=rangeDopplerTXRX[irange,doppler_ind,:,:]

        # Selected_ind = [(0,0), (1,7), (2,6), (11,15)]
        # rows, cols = zip(*Selected_ind)
        
        rows = unique_PosIndex[:,2].astype(int)
        cols = unique_PosIndex[:,3].astype(int)
        # print(rows)
        # print(cols)
        rangeVA = rangeTXRX[:, rows, cols]
        angle_window = scipy.signal.windows.hamming(rangeVA.shape[1])
        X_windowed_rangeVA = rangeVA * angle_window[np.newaxis,:]
        
        # Bartlet Angle 
        # Capon
        
        NFFT_Angle_OverNextPow2 =  1
        NFFT_Angle = int(2 ** (np.ceil(np.log2(X_windowed_rangeVA.shape[1]))+NFFT_Angle_OverNextPow2))
        RangeAngleMap = np.fft.fft(X_windowed_rangeVA, axis=1, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
        RangeAngleMap = np.fft.fftshift(RangeAngleMap, axes=1)
        # sina_fft = np.rad2deg(np.arcsin(np.linspace(-1,1,NFFT_AngleNFFT_Angle)))
        extent = [-.5 ,.5,d_fft[-1],d_fft[0]]
        ssp.config.ax[1,1].imshow(np.abs(RangeAngleMap), extent=extent, aspect='auto')
        ssp.config.ax[1,1].set_xlabel('sin(az)')
        ssp.config.ax[1,1].set_ylabel('Range (m)')
        ssp.config.ax[1,1].set_title("Bartlet Beamformer")
        if iradar==0:
          polar_range_angle(RangeAngleMap,d_fft,np.linspace(-np.pi/2, np.pi/2, NFFT_Angle),ssp.config.ax[1,2])
        else:
          polar_range_angle(RangeAngleMap,d_fft,np.linspace(-np.pi/2, np.pi/2, NFFT_Angle),ssp.config.ax[2,0])
        # ssp.config.ax[1,1].set_xlabel('sin(az)')
        # ssp.config.ax[1,1].set_ylabel('Range (m)')
        # ssp.config.ax[1,1].set_title("Bartlet Beamformer")
        
        ssp.config.ax[0,1].cla()
        ssp.config.ax[0,2].cla()
        ssp.config.ax[0,1].plot(np.real(XRadar[:,0,0]))
        ssp.config.ax[0,2].plot(np.abs(np.fft.fft(XRadar[:,0,0])))
        
        
        
        ssp.config.ax[0,0].cla()
        for __ in ssp.lastScatterInfo:
          _=__[0]
          ssp.config.ax[0,0].scatter(_[0],_[1],c='k',marker='x',s=20)
          # ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        for _ in global_location_TX:
          ssp.config.ax[0,0].scatter(_[0],_[1],c='r',marker='x')
          # ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='r',marker='x')
        for _ in global_location_RX:
          ssp.config.ax[0,0].scatter(_[0],_[1],c='b',marker='x')
          # ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='b',marker='x')
          
        ssp.config.ax[2,2].cla()
        for __ in ssp.lastScatterInfo:
          _=__[0]
          ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='k',marker='x',s=20)
        global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
        for _ in global_location_TX:
          ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='r',marker='x')
        for _ in global_location_RX:
          ssp.config.ax[2,2].scatter(_[0],_[1],_[2],c='b',marker='x')
          
        
        ssp.config.ax[0,1].set_xlabel('ADC sample')
        ssp.config.ax[0,1].set_ylabel('Amp')
        ssp.config.ax[0,1].set_title("ADC")
        ssp.config.ax[0,2].set_xlabel('Range')
        ssp.config.ax[0,2].set_ylabel('Amp')
        ssp.config.ax[0,2].set_title("Range FFT")
                
        
        plt.draw()  # Redraw the figure
        plt.pause(0.1)
        
        RangeDopplerDistributed0.append([rangeDopplerTXRX,d_fft,f_Doppler])
      RangeDopplerDistributed.append(RangeDopplerDistributed0)

      for icam in range(len(Signals[isuite]['cameras'])):
        ssp.config.ax[2,0].imshow(Signals[isuite]['cameras'][icam])
        plt.draw() 
        plt.pause(0.1)
        break
      
      ssp.config.ax[2,1].cla()
      for ilid in range(len(Signals[isuite]['lidars'])):
        pc = Signals[isuite]['lidars'][ilid]
        if pc.shape[0]==0:
          continue
        ssp.config.ax[2,1].scatter(pc[:,0],pc[:,1],pc[:,2])
        plt.draw() 
        plt.pause(0.1)
        break
  
  Triangles = ssp.utils.exportBlenderTriangles()
  for _ in Triangles:
      for __ in _:
          i,j=0,1
          ssp.config.ax[2,1].plot([__[i][0],__[j][0]],[__[i][1],__[j][1]],[__[i][2],__[j][2]])
          i,j=0,2
          ssp.config.ax[2,1].plot([__[i][0],__[j][0]],[__[i][1],__[j][1]],[__[i][2],__[j][2]])
          i,j=2,1
          ssp.config.ax[2,1].plot([__[i][0],__[j][0]],[__[i][1],__[j][1]],[__[i][2],__[j][2]])      
  plt.draw()  # Redraw the figure
  plt.pause(0.1)
        
        

def polar_range_angle(RangeAngleMap,ranges,angles,ax_polar_True):
    
    R, Theta = np.meshgrid(ranges, angles)
    # plt.figure()
    # ax = plt.subplot(111, polar=True)
    c = ax_polar_True.pcolormesh(Theta, R, (np.abs(RangeAngleMap).T), shading='auto')
    # c = ax_polar_True.pcolormesh(Theta, R, np.log10(np.abs(RangeAngleMap).T), shading='auto')
    ax_polar_True.set_thetalim(-np.pi/2, np.pi/2)
    ax_polar_True.set_ylim(0, ranges[-1]*.4)
    # plt.colorbar(c, label='Magnitude')
    # ax_polar_True.axis('off')
    # plt.show()
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

def SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler(Signals,FigsAxes=None,Fig=None):
  # RangeDopplerDistributed = []

  bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
  empty = bpy.context.object
  empty.name = f'Detection Cloud'
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      # RangeDopplerDistributed0 = []
      # All_RangeResolutions = []
      for iradar,specifications in enumerate(radarSpecifications):
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        PrecodingMatrix = specifications['PrecodingMatrix']
        RadarMode = specifications['RadarMode']
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
          X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

          if RadarMode == 'Pulse':
            waveform_MF = specifications['PulseWaveform_Loaded']
            matched_filter = np.conj(waveform_MF[::-1])
            X_fft_fast = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
            X_fft_fast=X_fft_fast[matched_filter.shape[0]-1:,:,:]
            d_fft = np.arange(X_fft_fast.shape[0]) * LightSpeed * Ts /2
            
          else:
            NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
            NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
            X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
          Range_Start = specifications['Range_Start']
          Range_End = specifications['Range_End']
          d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
          d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
          d_fft = d_fft[d1i:d2i]
          X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

          # print("XRadar shape:", XRadar.shape)
          # XRadar shape: (128, 192, 16)

          M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
          L = X_fft_fast.shape[1]
          Leff = int(L/M_TX)
          
          if ssp.config.DopplerProcessingMethod_FFT_Winv:
            # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
            # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
            N_Doppler = Leff
            f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
            f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

            PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

            rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
            for idop , f_Doppler_i in enumerate(f_Doppler):
              dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
              X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
              rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
              rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
          else:
            rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
            for ipulse in range(Leff):
              ind = ipulse*M_TX
              rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
            NFFT_Doppler_OverNextPow2=0
            NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
            rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
            rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
            f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
              
              
          global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
          if len(global_location_TX)+len(global_location_RX)==2:
            if FigsAxes is not None:
              FigsAxes[0,0].cla()
              FigsAxes[0,1].cla()
              FigsAxes[0,0].plot(np.arange(XRadar.shape[0])*Ts*1e6,np.real(XRadar[:,0,0]))
              FigsAxes[0,0].set_xlabel('t (usec)')
              FigsAxes[0,0].set_ylabel('ADC Amp.')

              FigsAxes[0,1].plot(d_fft,np.abs(X_fft_fast[:,0,:]))
              FigsAxes[0,1].set_xlabel('Range (m)')
              FigsAxes[0,1].set_ylabel('Amp.')

              maxDoppler =.5 /PRI / M_TX
              extent = [-maxDoppler* specifications['Lambda']/2 ,maxDoppler * specifications['Lambda']/2,d_fft[-1],d_fft[0]]
              rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
              FigsAxes[1,0].imshow(10*np.log10(rangeDoppler4CFAR), extent=extent, aspect='auto')
              FigsAxes[1,0].set_title("Range Doppler abs(mean) (CFAR)")
              # FigsAxes[1,0].set_xlabel('Doppler (Hz)')
              FigsAxes[1,0].set_xlabel('Velocity (m/s)')
              FigsAxes[1,0].set_ylabel('Range (m)')
              extent = [-.5 ,.5,d_fft[-1],d_fft[0]]
            
            continue
          PosIndex = np.array(specifications['Local_location_TXplusRX_Center'])
          azimuths = PosIndex[:, 0]
          elevations = PosIndex[:, 1]
          
          
          d_az = np.max(azimuths)-np.min(azimuths)
          d_el = np.max(elevations)-np.min(elevations)
          if d_az>d_el:
            sorted_indices = np.argsort(azimuths)
            sorted_PosIndex = PosIndex[sorted_indices,:]
            sorted_PosIndex[:,0]-=sorted_PosIndex[0,0]
            sorted_PosIndex[:,1]-=sorted_PosIndex[0,1]
            sorted_PosIndex[:,0]=np.round(sorted_PosIndex[:,0])
            sorted_PosIndex[:,1]=np.round(sorted_PosIndex[:,1])

            unique_azimuths, unique_indices = np.unique(sorted_PosIndex[:, 0], return_index=True)
            unique_PosIndex = sorted_PosIndex[unique_indices,:]

            # print(unique_PosIndex)



          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          if 1:
            num_train = 50  # Number of training cells
            num_guard = 4   # Number of guard cells
            prob_fa = 1e-3  # Desired probability of false alarm
            # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D(1.0*rangeDoppler4CFAR, num_train, num_guard, prob_fa)
            CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
            if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled:
              detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
            else:
              # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
              detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            FigsAxes[1,2].cla()
        # FigsAxes[1,2].set_xlabel('scatter index')
        # FigsAxes[1,2].set_ylabel('dr (m/s)')
    
            distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
            elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
            X, Y = np.meshgrid(elevation, distance)
            FigsAxes[1,2].plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
            FigsAxes[1,2].plot_surface(X, Y, 10*np.log10(cfar_threshold)+0, color='yellow', alpha=1)
            detected_points = np.where(detections == 1)
            FigsAxes[1,2].scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                       10*np.log10(rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')

            # Labels and legend
            FigsAxes[1,2].set_xlabel('Doppler (Hz)')
            FigsAxes[1,2].set_ylabel('Distance (m)')
            FigsAxes[1,2].set_zlabel('Magnitude (normalized, dB)')
            FigsAxes[1,2].xaxis.set_visible(False)
            FigsAxes[1,2].yaxis.set_visible(False)
            FigsAxes[1,2].zaxis.set_visible(False)
            # FigsAxes[1,2].legend()
            # plt.show()
            NDetection = detected_points[0].shape[0]
            # rows = unique_PosIndex[:,2].astype(int)
            # cols = unique_PosIndex[:,3].astype(int)
            rows,cols=specifications['MIMO_Antenna_Azimuth_Elevation_Order']
            global_location, global_rotation, global_scale = specifications['matrix_world']  
            for id in range(NDetection):
              antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
              rangeVA = np.zeros((np.max(np.array(rows))+1,np.max(np.array(cols))+1),dtype=antennaSignal.dtype)
              rangeVA[np.array(rows), np.array(cols)] = antennaSignal.ravel()
              NFFT_Angle_OverNextPow2 =  1
              NFFT_Angle = int(2 ** (np.ceil(np.log2(rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap = np.fft.fft(rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap = np.abs(np.fft.fftshift(AngleMap, axes=0))
              THR = 3*np.mean(AngleMap)
              detected_points_Angles = np.where(AngleMap > THR)
              # normalized_sinaz = np.linspace(-1,1,NFFT_Angle)
              normalized_sinaz = np.fft.fftshift(np.fft.fftfreq(NFFT_Angle))
              dpL = .5
              angles1 = -np.arcsin(1/dpL*normalized_sinaz[detected_points_Angles[0]])
              rangeTarget = d_fft[detected_points[0][id]]
              angles2 = 0
              dopplerTarget = f_Doppler[detected_points[1][id]]
              denominator = np.sqrt(1 + np.tan(angles1)**2 + np.tan(angles2)**2)
              xTarget = (rangeTarget * np.tan(angles1)) / denominator
              yTarget = (rangeTarget * np.tan(angles2)) / denominator
              zTarget = rangeTarget / denominator
              for iTarget in range(xTarget.shape[0]):
                local_point = Vector((-xTarget[iTarget], yTarget[iTarget], -zTarget[iTarget]))
                global_point = global_location + global_rotation @ (local_point * global_scale)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
                sphere = bpy.context.object
                sphere.parent=empty
                # sphere.hide_viewport = False
                # sphere.hide_render = False
              # print(rangeTarget,dopplerTarget)
              FigsAxes[0,2].plot(normalized_sinaz,10*np.log10(AngleMap))
              FigsAxes[0,2].set_xlabel('sin(Angle)')
              FigsAxes[0,2].set_ylabel('Amp.')
              
            
          for irange in range(rangeDoppler4CFAR.shape[0]):
            doppler_ind = np.argmax(rangeDoppler4CFAR[irange])
            rangeTXRX[irange,:,:]=rangeDopplerTXRX[irange,doppler_ind,:,:]

          # Selected_ind = [(0,0), (1,7), (2,6), (11,15)]
          # rows, cols = zip(*Selected_ind)
          rows,cols=specifications['MIMO_Antenna_Azimuth_Elevation_Order']
            
          # print(rows)
          # print(cols)
          # rangeVA = rangeTXRX[:, rows, cols]
          rangeVA = np.zeros((rangeTXRX.shape[0],np.max(np.array(rows))+1,np.max(np.array(cols))+1),dtype=antennaSignal.dtype)
          for irva in range(rangeTXRX.shape[0]):
            rangeVA[irva,np.array(rows), np.array(cols)] = rangeTXRX[irva,:,:].ravel()
              
          angle_window = scipy.signal.windows.hamming(rangeVA.shape[1])
          X_windowed_rangeVA = rangeVA * angle_window[np.newaxis,:,np.newaxis]
          
          # Bartlet Angle 
          # Capon
          
          NFFT_Angle_OverNextPow2 =  1
          NFFT_Angle = int(2 ** (np.ceil(np.log2(X_windowed_rangeVA.shape[1]))+NFFT_Angle_OverNextPow2))
          RangeAngleMap = np.fft.fft(X_windowed_rangeVA, axis=1, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
          RangeAngleMap = np.fft.fftshift(RangeAngleMap, axes=1)
          
          # sina_fft = np.rad2deg(np.arcsin(np.linspace(-1,1,NFFT_AngleNFFT_Angle)))
          if FigsAxes is not None:
            FigsAxes[0,0].cla()
            FigsAxes[0,1].cla()
            FigsAxes[0,0].plot(np.arange(XRadar.shape[0])*Ts*1e6,np.real(XRadar[:,0,0]))
            FigsAxes[0,0].set_xlabel('t (usec)')
            FigsAxes[0,0].set_ylabel('ADC Amp.')

            FigsAxes[0,1].plot(d_fft,np.abs(X_fft_fast[:,0,:]))
            FigsAxes[0,1].set_xlabel('Range (m)')
            FigsAxes[0,1].set_ylabel('Amp.')

            maxDoppler =.5 /PRI /M_TX
            extent = [-maxDoppler* specifications['Lambda']/2 ,maxDoppler * specifications['Lambda']/2,d_fft[-1],d_fft[0]]
            FigsAxes[1,0].imshow(10*np.log10(rangeDoppler4CFAR), extent=extent, aspect='auto')
            FigsAxes[1,0].set_title("Range Doppler abs(mean) (CFAR)")
            # FigsAxes[1,0].set_xlabel('Doppler (Hz)')
            FigsAxes[1,0].set_xlabel('Velocity (m/s)')
            FigsAxes[1,0].set_ylabel('Range (m)')
            extent = [-.5 ,.5,d_fft[-1],d_fft[0]]
            FigsAxes[1,1].imshow(10*np.log10(np.abs(RangeAngleMap)), extent=extent, aspect='auto')
            FigsAxes[1,1].set_xlabel('sin(az)')
            FigsAxes[1,1].set_ylabel('Range (m)')
            FigsAxes[1,1].set_title("Bartlet Beamformer")
            
  if FigsAxes is not None:
    plt.draw()  # Redraw the figure
    plt.pause(0.1)
    plt.gcf().canvas.flush_events() 
    # image=ssp.visualization.captureFig(Fig)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('moein', image)
    # cv2.waitKey(50)

def SensorsSignalProccessing_Chain_DAR(Signals,FigsAxes=None,Fig=None):
  # RangeDopplerDistributed = []
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      # RangeDopplerDistributed0 = []
      # All_RangeResolutions = []
      for iradar,specifications in enumerate(radarSpecifications):
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        PrecodingMatrix = specifications['PrecodingMatrix']
        RadarMode = specifications['RadarMode']
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
          X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

          if RadarMode == 'Pulse':
            waveform_MF = specifications['PulseWaveform_Loaded']
            matched_filter = np.conj(waveform_MF[::-1])
            X_fft_fast = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
            X_fft_fast=X_fft_fast[matched_filter.shape[0]-1:,:,:]
            d_fft = np.arange(X_fft_fast.shape[0]) * LightSpeed * Ts /2
            
          else:
            NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
            NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
            X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
          Range_Start = specifications['Range_Start']
          Range_End = specifications['Range_End']
          d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
          d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
          d_fft = d_fft[d1i:d2i]
          X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

          plt.plot(d_fft,np.abs(X_fft_fast[:,0,:]),label='range profile')
          # plt.show()

          # print("XRadar shape:", XRadar.shape)
          # XRadar shape: (128, 192, 16)

          M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
          L = X_fft_fast.shape[1]
          Leff = int(L/M_TX)
          
          if ssp.config.DopplerProcessingMethod_FFT_Winv:
            # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
            # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
            N_Doppler = Leff
            f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
            f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

            PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

            rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
            for idop , f_Doppler_i in enumerate(f_Doppler):
              dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
              X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
              rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
              rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
          else:
            rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
            for ipulse in range(Leff):
              ind = ipulse*M_TX
              rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
            NFFT_Doppler_OverNextPow2=0
            NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
            rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
            rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
            f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
              
          
          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          if 1:
            num_train = 50  # Number of training cells
            num_guard = 4   # Number of guard cells
            prob_fa = 1e-3  # Desired probability of false alarm
            # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D(1.0*rangeDoppler4CFAR, num_train, num_guard, prob_fa)
            # detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
            detections,cfar_threshold = ssp.radar.utils.cfar_simple_max(1.0*rangeDoppler4CFAR)
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # FigsAxes[1,2].cla()
        # FigsAxes[1,2].set_xlabel('scatter index')
        # FigsAxes[1,2].set_ylabel('dr (m/s)')
    
            distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
            elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
            X, Y = np.meshgrid(elevation, distance)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
            ax.plot_surface(X, Y, 10*np.log10(cfar_threshold), color='yellow', alpha=.3)
            detected_points = np.where(detections == 1)
            ax.scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                       10*np.log10(rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')

            # Labels and legend
            ax.set_xlabel('Doppler (Hz)')
            ax.set_ylabel('Distance (m)')
            ax.set_zlabel('Magnitude (normalized, dB)')
            # plt.show()
            # FigsAxes[1,2].legend()
            # plt.show()
            NDetection = detected_points[0].shape[0]
            # rows = unique_PosIndex[:,2].astype(int)
            # cols = unique_PosIndex[:,3].astype(int)
            rows,cols=specifications['MIMO_Antenna_Azimuth_Elevation_Order']
            rows = np.array(rows)
            cols = np.array(cols)
            
            full_range = set(range(np.max(rows) + 1))
            missing_indices = sorted(full_range - set(rows))
            ULA_TXRX_Lx_Ly_NonZ_AllRX=specifications['ULA_TXRX_Lx_Ly_NonZ']

            global_location, global_rotation, global_scale = specifications['matrix_world']  
            for id in range(NDetection):
              antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
              new_rangeVA = np.zeros((ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
              for new_ix in range(new_rangeVA.shape[0]):
                for new_iy in range(new_rangeVA.shape[1]):
                  itxirxs = ULA_TXRX_Lx_Ly_NonZ_AllRX[0][new_ix][new_iy]
                  for itxirx in itxirxs:
                    new_rangeVA[new_ix,new_iy] += antennaSignal[itxirx[0],itxirx[1]]
                  if len(itxirxs)>0:
                    new_rangeVA[new_ix,new_iy] /= len(itxirxs)
              new_rangeVA_NZ = new_rangeVA[ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
              # print(new_rangeVA)
              # print(new_rangeVA_NZ)
              # for itxtest in range(antennaSignal.shape[0]):
              #   for irxtest in range(antennaSignal.shape[0]):
              #     antennaSignal[itxtest,irxtest]=100*itxtest+irxtest
              rangeVA = np.zeros((np.max(rows)+1,np.max(cols)+1),dtype=antennaSignal.dtype)
              rangeVAcounter = np.zeros((np.max(rows)+1,np.max(cols)+1),dtype=np.int32)
              antennaSignal_ravel = antennaSignal.ravel()
              unique_rows=[]
              for irow in range(rows.shape[0]):
                rangeVA[rows[irow], cols[irow]] += antennaSignal_ravel[irow]
                rangeVAcounter[rows[irow], cols[irow]] += 1 
                if rangeVAcounter[rows[irow], cols[irow]]==1:
                  unique_rows.append(rows[irow])
              unique_rows=np.array(unique_rows)
                
              ind = np.where(rangeVAcounter>1)
              rangeVA[ind]/=rangeVAcounter[ind]
              rangeVA2 = np.delete(rangeVA, missing_indices, axis=0)
              # coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(unique_rows, rangeVA2)
              coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0], new_rangeVA_NZ)
              NFFT_Angle_OverNextPow2 =  3
              NFFT_Angle = int(2 ** (np.ceil(np.log2(new_rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap = np.fft.fft(new_rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap = np.abs(np.fft.fftshift(AngleMap, axes=0))
              NFFT_Angle = int(2 ** (np.ceil(np.log2(new_rangeVA_NZ.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap0 = np.fft.fft(new_rangeVA_NZ, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap0 = np.abs(np.fft.fftshift(AngleMap0, axes=0))
              NFFT_Angle2 = int(2 ** (np.ceil(np.log2(coArraySignal.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap2 = np.fft.fft(coArraySignal, axis=0, n=NFFT_Angle2)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap2 = np.abs(np.fft.fftshift(AngleMap2, axes=0))
              # THR = 3*np.mean(AngleMap)
              # detected_points_Angles = np.where(AngleMap > THR)
              # normalized_sinaz = np.linspace(-1,1,NFFT_Angle)
              # angles1 = np.asin(normalized_sinaz[detected_points_Angles[0]])
              # rangeTarget = d_fft[detected_points[0][id]]
              # angles2 = 0
              # dopplerTarget = f_Doppler[detected_points[1][id]]
              # denominator = np.sqrt(1 + np.tan(angles1)**2 + np.tan(angles2)**2)
              # xTarget = (rangeTarget * np.tan(angles1)) / denominator
              # yTarget = (rangeTarget * np.tan(angles2)) / denominator
              # zTarget = rangeTarget / denominator
              # for iTarget in range(xTarget.shape[0]):
              #   local_point = Vector((xTarget[iTarget], yTarget[iTarget], -zTarget[iTarget]))
              #   global_point = global_location + global_rotation @ (local_point * global_scale)
              #   bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
              #   # sphere = bpy.context.object
              #   # sphere.hide_viewport = False
              #   # sphere.hide_render = False
              # # print(rangeTarget,dopplerTarget)
              plt.figure()
              spatial_freq = np.linspace(-1, 1, AngleMap.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap))
              spatial_freq = np.linspace(-1, 1, AngleMap0.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap0))
              spatial_freq = np.linspace(-1, 1, AngleMap2.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap2))
              plt.show()
            break


def SensorsSignalProccessing_WinvTDM(Signals,FigsAxes=None,Fig=None):
  # RangeDopplerDistributed = []
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      # RangeDopplerDistributed0 = []
      # All_RangeResolutions = []
      for iradar,specifications in enumerate(radarSpecifications):
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        PrecodingMatrix = specifications['PrecodingMatrix']
        RadarMode = specifications['RadarMode']
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
          X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

          if RadarMode == 'Pulse':
            waveform_MF = specifications['PulseWaveform_Loaded']
            matched_filter = np.conj(waveform_MF[::-1])
            X_fft_fast = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
            X_fft_fast=X_fft_fast[matched_filter.shape[0]-1:,:,:]
            d_fft = np.arange(X_fft_fast.shape[0]) * LightSpeed * Ts /2
            
          else:
            NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
            NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
            X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
          Range_Start = specifications['Range_Start']
          Range_End = specifications['Range_End']
          d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
          d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
          d_fft = d_fft[d1i:d2i]
          X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

          plt.plot(d_fft,np.abs(X_fft_fast[:,0,:]),label='range profile')
          # plt.show()

          # print("XRadar shape:", XRadar.shape)
          # XRadar shape: (128, 192, 16)

          M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
          L = X_fft_fast.shape[1]
          Leff = int(L/M_TX)
          
          if ssp.config.DopplerProcessingMethod_FFT_Winv:
            # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
            # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
            N_Doppler = Leff
            f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
            f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

            PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

            rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
            for idop , f_Doppler_i in enumerate(f_Doppler):
              dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
              X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
              rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
              rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
          else:
            rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
            for ipulse in range(Leff):
              ind = ipulse*M_TX
              rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
            NFFT_Doppler_OverNextPow2=0
            NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
            rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
            rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
            f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
              
          
          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          if 1:
            num_train = 50  # Number of training cells
            num_guard = 4   # Number of guard cells
            prob_fa = 1e-3  # Desired probability of false alarm
            # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D(1.0*rangeDoppler4CFAR, num_train, num_guard, prob_fa)
            # detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
            detections,cfar_threshold = ssp.radar.utils.cfar_simple_max(1.0*rangeDoppler4CFAR)
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # FigsAxes[1,2].cla()
        # FigsAxes[1,2].set_xlabel('scatter index')
        # FigsAxes[1,2].set_ylabel('dr (m/s)')
    
            distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
            elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
            X, Y = np.meshgrid(elevation, distance)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
            ax.plot_surface(X, Y, 10*np.log10(cfar_threshold), color='yellow', alpha=.3)
            detected_points = np.where(detections == 1)
            ax.scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                       10*np.log10(rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')

            # Labels and legend
            ax.set_xlabel('Doppler (Hz)')
            ax.set_ylabel('Distance (m)')
            ax.set_zlabel('Magnitude (normalized, dB)')
            # plt.show()
            # FigsAxes[1,2].legend()
            # plt.show()
            NDetection = detected_points[0].shape[0]
            # rows = unique_PosIndex[:,2].astype(int)
            # cols = unique_PosIndex[:,3].astype(int)
            rows,cols=specifications['MIMO_Antenna_Azimuth_Elevation_Order']
            rows = np.array(rows)
            cols = np.array(cols)
              
            full_range = set(range(np.max(rows) + 1))
            missing_indices = sorted(full_range - set(rows))
            
            global_location, global_rotation, global_scale = specifications['matrix_world']  
            for id in range(NDetection):
              antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
              # for itxtest in range(antennaSignal.shape[0]):
              #   for irxtest in range(antennaSignal.shape[0]):
              #     antennaSignal[itxtest,irxtest]=100*itxtest+irxtest
              rangeVA = np.zeros((np.max(rows)+1,np.max(cols)+1),dtype=antennaSignal.dtype)
              rangeVAcounter = np.zeros((np.max(rows)+1,np.max(cols)+1),dtype=np.int32)
              antennaSignal_ravel = antennaSignal.ravel()
              unique_rows=[]
              for irow in range(rows.shape[0]):
                rangeVA[rows[irow], cols[irow]] += antennaSignal_ravel[irow]
                rangeVAcounter[rows[irow], cols[irow]] += 1 
                if rangeVAcounter[rows[irow], cols[irow]]==1:
                  unique_rows.append(rows[irow])
              unique_rows=np.array(unique_rows)
                
              ind = np.where(rangeVAcounter>1)
              rangeVA[ind]/=rangeVAcounter[ind]
              rangeVA2 = np.delete(rangeVA, missing_indices, axis=0)
              coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(unique_rows, rangeVA2)
              NFFT_Angle_OverNextPow2 =  3
              NFFT_Angle = int(2 ** (np.ceil(np.log2(rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap = np.fft.fft(rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap = np.abs(np.fft.fftshift(AngleMap, axes=0))
              NFFT_Angle = int(2 ** (np.ceil(np.log2(rangeVA2.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap0 = np.fft.fft(rangeVA2, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap0 = np.abs(np.fft.fftshift(AngleMap0, axes=0))
              NFFT_Angle2 = int(2 ** (np.ceil(np.log2(coArraySignal.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap2 = np.fft.fft(coArraySignal, axis=0, n=NFFT_Angle2)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap2 = np.abs(np.fft.fftshift(AngleMap2, axes=0))
              # THR = 3*np.mean(AngleMap)
              # detected_points_Angles = np.where(AngleMap > THR)
              # normalized_sinaz = np.linspace(-1,1,NFFT_Angle)
              # angles1 = np.asin(normalized_sinaz[detected_points_Angles[0]])
              # rangeTarget = d_fft[detected_points[0][id]]
              # angles2 = 0
              # dopplerTarget = f_Doppler[detected_points[1][id]]
              # denominator = np.sqrt(1 + np.tan(angles1)**2 + np.tan(angles2)**2)
              # xTarget = (rangeTarget * np.tan(angles1)) / denominator
              # yTarget = (rangeTarget * np.tan(angles2)) / denominator
              # zTarget = rangeTarget / denominator
              # for iTarget in range(xTarget.shape[0]):
              #   local_point = Vector((xTarget[iTarget], yTarget[iTarget], -zTarget[iTarget]))
              #   global_point = global_location + global_rotation @ (local_point * global_scale)
              #   bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
              #   # sphere = bpy.context.object
              #   # sphere.hide_viewport = False
              #   # sphere.hide_render = False
              # # print(rangeTarget,dopplerTarget)
              plt.figure()
              spatial_freq = np.linspace(-1, 1, AngleMap.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap))
              spatial_freq = np.linspace(-1, 1, AngleMap0.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap0))
              spatial_freq = np.linspace(-1, 1, AngleMap2.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap2))
              plt.show()
            break

def SensorsSignalProccessing_Chain_DAR_CoArray(Signals,FigsAxes=None,Fig=None):
  # RangeDopplerDistributed = []
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      # RangeDopplerDistributed0 = []
      # All_RangeResolutions = []
      for iradar,specifications in enumerate(radarSpecifications):
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        PrecodingMatrix = specifications['PrecodingMatrix']
        RadarMode = specifications['RadarMode']
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
          X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

          if RadarMode == 'Pulse':
            waveform_MF = specifications['PulseWaveform_Loaded']
            matched_filter = np.conj(waveform_MF[::-1])
            X_fft_fast = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
            X_fft_fast=X_fft_fast[matched_filter.shape[0]-1:,:,:]
            d_fft = np.arange(X_fft_fast.shape[0]) * LightSpeed * Ts /2
            
          else:
            NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
            NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
            X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
          Range_Start = specifications['Range_Start']
          Range_End = specifications['Range_End']
          d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
          d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
          d_fft = d_fft[d1i:d2i]
          X_fft_fast = X_fft_fast[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)

          plt.plot(d_fft,np.abs(X_fft_fast[:,0,:]),label='range profile')
          # plt.show()

          # print("XRadar shape:", XRadar.shape)
          # XRadar shape: (128, 192, 16)

          M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
          L = X_fft_fast.shape[1]
          Leff = int(L/M_TX)
          
          if ssp.config.DopplerProcessingMethod_FFT_Winv:
            # slow_time_window = scipy.signal.windows.hamming(X_fft_fast.shape[1])
            # X_windowed_slow = X_fft_fast * slow_time_window[np.newaxis,:,np.newaxis,np.newaxis]
            N_Doppler = Leff
            f_Doppler = np.hstack((np.linspace(-0.5/PRI/M_TX, 0, N_Doppler)[:-1], np.linspace(0, 0.5/PRI/M_TX, N_Doppler)))
            f_Doppler = np.arange(0,N_Doppler)/N_Doppler/PRI/M_TX - 1/PRI/M_TX / 2

            PrecodingMatrixInv = np.linalg.pinv(PrecodingMatrix)

            rangeDopplerTXRX = np.zeros((X_fft_fast.shape[0], f_Doppler.shape[0], M_TX, X_fft_fast.shape[2]),dtype=complex)
            for idop , f_Doppler_i in enumerate(f_Doppler):
              dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*PRI)
              X_doppler_comp = X_fft_fast * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
              rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
              rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
          else:
            rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
            for ipulse in range(Leff):
              ind = ipulse*M_TX
              rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
            NFFT_Doppler_OverNextPow2=0
            NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
            rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
            rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
            f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
              
          
          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          if 1:
            num_train = 50  # Number of training cells
            num_guard = 4   # Number of guard cells
            prob_fa = 1e-3  # Desired probability of false alarm
            # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D(1.0*rangeDoppler4CFAR, num_train, num_guard, prob_fa)
            # detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
            CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
            if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled:
              detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
            else:
              # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
              detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
            # detections,cfar_threshold = ssp.radar.utils.cfar_simple_max(1.0*rangeDoppler4CFAR)
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # FigsAxes[1,2].cla()
        # FigsAxes[1,2].set_xlabel('scatter index')
        # FigsAxes[1,2].set_ylabel('dr (m/s)')
    
            distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
            elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
            X, Y = np.meshgrid(elevation, distance)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
            ax.plot_surface(X, Y, 10*np.log10(cfar_threshold), color='yellow', alpha=.3)
            detected_points = np.where(detections == 1)
            ax.scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                       10*np.log10(rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')

            # Labels and legend
            ax.set_xlabel('Doppler (Hz)')
            ax.set_ylabel('Distance (m)')
            ax.set_zlabel('Magnitude (normalized, dB)')
            plt.show()
            # FigsAxes[1,2].legend()
            # plt.show()
            NDetection = detected_points[0].shape[0]
            # rows = unique_PosIndex[:,2].astype(int)
            # cols = unique_PosIndex[:,3].astype(int)
            rows,cols=specifications['MIMO_Antenna_Azimuth_Elevation_Order']
            rows = np.array(rows)
            cols = np.array(cols)
            
            full_range = set(range(np.max(rows) + 1))
            missing_indices = sorted(full_range - set(rows))
            ULA_TXRX_Lx_Ly_NonZ_AllRX=specifications['ULA_TXRX_Lx_Ly_NonZ']

            global_location, global_rotation, global_scale = specifications['matrix_world']  
            for id in range(NDetection):
              antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
              new_rangeVA = np.zeros((ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
              for new_ix in range(new_rangeVA.shape[0]):
                for new_iy in range(new_rangeVA.shape[1]):
                  itxirxs = ULA_TXRX_Lx_Ly_NonZ_AllRX[0][new_ix][new_iy]
                  for itxirx in itxirxs:
                    new_rangeVA[new_ix,new_iy] += antennaSignal[itxirx[0],itxirx[1]]
                  if len(itxirxs)>0:
                    new_rangeVA[new_ix,new_iy] /= len(itxirxs)
              new_rangeVA_NZ = new_rangeVA[ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
              # print(new_rangeVA)
              # print(new_rangeVA_NZ)
              # for itxtest in range(antennaSignal.shape[0]):
              #   for irxtest in range(antennaSignal.shape[0]):
              #     antennaSignal[itxtest,irxtest]=100*itxtest+irxtest
              rangeVA = np.zeros((np.max(rows)+1,np.max(cols)+1),dtype=antennaSignal.dtype)
              rangeVAcounter = np.zeros((np.max(rows)+1,np.max(cols)+1),dtype=np.int32)
              antennaSignal_ravel = antennaSignal.ravel()
              unique_rows=[]
              for irow in range(rows.shape[0]):
                rangeVA[rows[irow], cols[irow]] += antennaSignal_ravel[irow]
                rangeVAcounter[rows[irow], cols[irow]] += 1 
                if rangeVAcounter[rows[irow], cols[irow]]==1:
                  unique_rows.append(rows[irow])
              unique_rows=np.array(unique_rows)
                
              ind = np.where(rangeVAcounter>1)
              rangeVA[ind]/=rangeVAcounter[ind]
              rangeVA2 = np.delete(rangeVA, missing_indices, axis=0)
              # coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(unique_rows, rangeVA2)
              coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0], new_rangeVA_NZ)
              coArraySignal2,coArrayStructure2 = ssp.radar.utils.VAsignal2coArraySignal_1D(coArrayStructure[1], coArraySignal)
              NFFT_Angle_OverNextPow2 =  3
              NFFT_Angle = int(2 ** (np.ceil(np.log2(new_rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap = np.fft.fft(new_rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap = np.abs(np.fft.fftshift(AngleMap, axes=0))
              NFFT_Angle = int(2 ** (np.ceil(np.log2(new_rangeVA_NZ.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap0 = np.fft.fft(new_rangeVA_NZ, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap0 = np.abs(np.fft.fftshift(AngleMap0, axes=0))
              NFFT_Angle2 = int(2 ** (np.ceil(np.log2(coArraySignal.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap2 = np.fft.fft(coArraySignal, axis=0, n=NFFT_Angle2)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap2 = np.abs(np.fft.fftshift(AngleMap2, axes=0))
              NFFT_Angle3 = int(2 ** (np.ceil(np.log2(coArraySignal2.shape[0]))+NFFT_Angle_OverNextPow2))
              AngleMap3 = np.fft.fft(coArraySignal2, axis=0, n=NFFT_Angle3)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
              AngleMap3 = np.abs(np.fft.fftshift(AngleMap3, axes=0))
              # THR = 3*np.mean(AngleMap)
              # detected_points_Angles = np.where(AngleMap > THR)
              # normalized_sinaz = np.linspace(-1,1,NFFT_Angle)
              # angles1 = np.asin(normalized_sinaz[detected_points_Angles[0]])
              # rangeTarget = d_fft[detected_points[0][id]]
              # angles2 = 0
              # dopplerTarget = f_Doppler[detected_points[1][id]]
              # denominator = np.sqrt(1 + np.tan(angles1)**2 + np.tan(angles2)**2)
              # xTarget = (rangeTarget * np.tan(angles1)) / denominator
              # yTarget = (rangeTarget * np.tan(angles2)) / denominator
              # zTarget = rangeTarget / denominator
              # for iTarget in range(xTarget.shape[0]):
              #   local_point = Vector((xTarget[iTarget], yTarget[iTarget], -zTarget[iTarget]))
              #   global_point = global_location + global_rotation @ (local_point * global_scale)
              #   bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
              #   # sphere = bpy.context.object
              #   # sphere.hide_viewport = False
              #   # sphere.hide_render = False
              # # print(rangeTarget,dopplerTarget)
              plt.figure()
              spatial_freq = np.linspace(-1, 1, AngleMap.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap/np.max(AngleMap)))
              spatial_freq = np.linspace(-1, 1, AngleMap0.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap0/np.max(AngleMap0)))
              spatial_freq = np.linspace(-1, 1, AngleMap2.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,10*np.log10(AngleMap2/np.max(AngleMap2)))
              plt.figure()
              spatial_freq = np.linspace(-1, 1, AngleMap.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,(AngleMap/np.max(AngleMap)))
              spatial_freq = np.linspace(-1, 1, AngleMap0.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,(AngleMap0/np.max(AngleMap0)))
              spatial_freq = np.linspace(-1, 1, AngleMap2.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,(AngleMap2/np.max(AngleMap2)))
              spatial_freq = np.linspace(-1, 1, AngleMap3.shape[0])  # Normalized spatial frequencies (-1 to 1)
              fft_angles = np.degrees(np.arcsin(spatial_freq))
              plt.plot(fft_angles,(AngleMap3/np.max(AngleMap3)))
              plt.xlim([-10,10])
              plt.show()


def SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler_TruckHuman(Signals,FigsAxes=None,Fig=None):
  bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
  empty = bpy.context.object
  empty.name = f'Detection Cloud'
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      for iradar,specifications in enumerate(radarSpecifications):
        print(len(Signals[isuite]['radars'][iradar]))
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          # fig = plt.figure()
          # ax = fig.add_subplot(111, projection='3d')
          # x = np.arange(XRadar.shape[1])
          # y = np.arange(XRadar.shape[0])
          # X, Y = np.meshgrid(x, y)
          # ax.plot_surface(X, Y, 10*np.log10(np.abs(XRadar[:,:,0])), cmap='viridis', alpha=1)
          # plt.show()
          
          X_fft_fast,d_fft = ssp.radar.utils.rangeprocessing(XRadar,specifications)
          if ssp.utils.get_debugSettings():
            fig, (ax1, ax2) = plt.subplots(1, 2)
            FigsAxes.plot(d_fft,np.abs(X_fft_fast[:,0,0]))
            ax2.plot(d_fft,10*np.log10(np.abs(X_fft_fast[:,0,0])))
            plt.show()
          # fig = plt.figure()
          # ax = fig.add_subplot(111, projection='3d')
          # x = np.arange(X_fft_fast.shape[1])
          # y = np.arange(X_fft_fast.shape[0])
          # X, Y = np.meshgrid(x, y)
          # ax.plot_surface(X, Y, 10*np.log10(np.abs(X_fft_fast[:,:,0])), cmap='viridis', alpha=1)
          # fig = plt.figure()
          # plt.imshow( 10*np.log10(np.abs(X_fft_fast[:,:,0])))
          
          rangeDopplerTXRX,f_Doppler = ssp.radar.utils.dopplerprocessing_mimodemodulation(X_fft_fast,specifications)
          
          detections,cfar_threshold,rangeDoppler4CFAR = ssp.radar.utils.rangedoppler_detection(rangeDopplerTXRX)
          
          FigsAxes[1].cla()
          # distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
          # elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
          # X, Y = np.meshgrid(elevation, distance)
          # FigsAxes[1].plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
          # FigsAxes[1].plot_surface(X, Y, 10*np.log10(cfar_threshold)+0, color='yellow', alpha=1)
          # detected_points = np.where(detections == 1)
          # FigsAxes[1].scatter(elevation[detected_points[1]], distance[detected_points[0]], 
          #              10*np.log10(rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')

          # plt.show()
          FigsAxes[1].imshow(10 * np.log10(rangeDoppler4CFAR), extent=[f_Doppler[0],f_Doppler[-1], d_fft[0], d_fft[-1]], 
                aspect='auto', cmap='viridis', origin='lower')
          detected_points = np.where(detections == 1)
          FigsAxes[1].scatter(f_Doppler[detected_points[1]], d_fft[detected_points[0]], 
            color='red', s=20, label='Post-CFAR Point Cloud')
          # plt.figure()
          # plt.imshow(rangeDoppler4CFAR, aspect='auto')
          # plt.show()
          ssp.radar.utils.angleprocessing(rangeDopplerTXRX,detections,specifications,FigsAxes)
                
          plt.draw()  # Redraw the figure
          plt.pause(0.1)
          plt.gcf().canvas.flush_events() 



def SensorsSignalProccessing_RIS_Included(Signals,FigsAxes=None,Fig=None):
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      for iradar,specifications in enumerate(radarSpecifications):
        print(len(Signals[isuite]['radars'][iradar]))
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          X_fft_fast,d_fft = ssp.radar.utils.rangeprocessing(XRadar,specifications)
          plt.figure()
          plt.plot(np.real(XRadar[:,0,0]))
          plt.figure()
          plt.plot(d_fft,np.abs(X_fft_fast[:,0,0]))
          plt.draw() 
          plt.pause(0.1)
          plt.gcf().canvas.flush_events() 

def SensorsSignalProccessing_RIS_Included_Probe(Signals,FigsAxes=None,Fig=None):
  o=0
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      for iradar,specifications in enumerate(radarSpecifications):
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          X_fft_fast,d_fft = ssp.radar.utils.rangeprocessing(XRadar,specifications)
          o=np.max(np.abs(X_fft_fast[:,0,0]))
          break
  return o
          
def SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler_2D(Signals,FigsAxes=None,Fig=None):
  # RangeDopplerDistributed = []

  bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
  empty = bpy.context.object
  empty.name = f'Detection Cloud'
  for isuite,radarSpecifications in enumerate(ssp.RadarSpecifications):
      for iradar,specifications in enumerate(radarSpecifications):
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        PrecodingMatrix = specifications['PrecodingMatrix']
        RadarMode = specifications['RadarMode']
        for XRadar,timeX in Signals[isuite]['radars'][iradar]:
          fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
          X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

          NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
          NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
          X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
          d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
          Range_Start = specifications['Range_Start']
          Range_End = specifications['Range_End']
          d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
          d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
          d_fft = d_fft[d1i:d2i]
          X_fft_fast = X_fft_fast[d1i:d2i,:,:] 
          test1,test2 = np.min(np.abs(X_fft_fast)),np.max(np.abs(X_fft_fast))
          M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
          L = X_fft_fast.shape[1]
          Leff = int(L/M_TX)
          
          rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
          for ipulse in range(Leff):
            ind = ipulse*M_TX
            rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
          NFFT_Doppler_OverNextPow2=0
          NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
          rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
          rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
          f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)
          
          
          test1_1,test2_1 = np.min(np.abs(rangeDopplerTXRX)),np.max(np.abs(rangeDopplerTXRX))
          
          rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
          rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
          rangeDoppler4CFAR -= np.min(rangeDoppler4CFAR)
          # rangeDoppler4CFAR = np.abs(np.mean(rangeDopplerTXRX,axis=(2,3)))
          # rangeDoppler4CFAR = np.abs(rangeDopplerTXRX[:,:,0,0])
          test1_2,test2_2 = np.min(np.abs(rangeDoppler4CFAR)),np.max(np.abs(rangeDoppler4CFAR))
          all_xyz=[]
          CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
          num_train, num_guard, alpha = [25,15], [1,1], 5
          if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled:
            detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, num_train, num_guard, alpha)
          else:
            detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR,num_train, num_guard, alpha)
            # detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
          # fig = plt.figure(figsize=(10, 8))
          # ax = fig.add_subplot(111, projection='3d')
          FigsAxes[1,2].cla()
          distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
          elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
          X, Y = np.meshgrid(elevation, distance)
          # FigsAxes[1,2].plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
          FigsAxes[1,2].plot_surface(X, Y, (rangeDoppler4CFAR), cmap='viridis', alpha=1)
          # FigsAxes[1,2].plot_surface(X, Y, (cfar_threshold)+0, color='yellow', alpha=1)
          detected_points = np.where(detections == 1)
          FigsAxes[1,2].scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                      (rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')

          # Labels and legend
          FigsAxes[1,2].set_xlabel('Doppler (Hz)')
          FigsAxes[1,2].set_ylabel('Distance (m)')
          FigsAxes[1,2].set_zlabel('Magnitude (normalized, dB)')
          # FigsAxes[1,2].xaxis.set_visible(False)
          # FigsAxes[1,2].yaxis.set_visible(False)
          # FigsAxes[1,2].zaxis.set_visible(False)
          # break
          # FigsAxes[1,2].legend()
          # plt.show()
          NDetection = detected_points[0].shape[0]
          # rows = unique_PosIndex[:,2].astype(int)
          # cols = unique_PosIndex[:,3].astype(int)
          
          i_list, j_list = ssp.radar.utils.mimo.mimo_antenna_order(specifications)
          Lambda = 1.0
          dy = .5*Lambda
          dz = .5*Lambda
          
          for id in range(NDetection):
            
            rangeTarget = d_fft[detected_points[0][id]]
            dopplerTarget = f_Doppler[detected_points[1][id]]
            
            antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
            rangeVA = np.zeros((np.max(i_list)+1,np.max(j_list)+1),dtype=antennaSignal.dtype)
            rangeVA[i_list, j_list] = antennaSignal.ravel()
            NFFT_Angle_OverNextPow2 = 1
            NFFT_Angle = int(2 ** (np.ceil(np.log2(rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
            NFFT_Angle_elevation = int(2 ** (np.ceil(np.log2(rangeVA.shape[1]))+NFFT_Angle_OverNextPow2))
            AngleMap0 = np.fft.fft(rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            # AngleMap0 = np.fft.fftshift(AngleMap0, axes=0)
            AngleMap = np.fft.fft(AngleMap0, axis=1, n=NFFT_Angle_elevation)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            # AngleMap = np.fft.fftshift(AngleMap, axes=1)
            AngleMap = np.abs(AngleMap)
            FigsAxes[1,1].cla()
            a = np.linspace(-1, 1, AngleMap.shape[0])
            b = np.linspace(-1, 1, AngleMap.shape[1]) 
            X, Y = np.meshgrid(a, b)
            FigsAxes[1,1].plot_surface(X,Y,(AngleMap), cmap='viridis', alpha=1)
            num_train, num_guard, alpha = [25,15], [1,1], 5
            detections_angle,cfar_threshold_angle = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*AngleMap, num_train, num_guard, alpha)
            detected_points_angle = np.where(detections_angle == 1)
            FigsAxes[1,1].scatter(a[detected_points_angle[1]], b[detected_points_angle[0]], 
                      (AngleMap[detected_points_angle]), color='red', s=20, label='Post-CFAR Point Cloud')
            NDetection_angle = detected_points_angle[0].shape[0]
            for id_angle in range(NDetection_angle):
              amp = AngleMap[detected_points_angle[0][id_angle],detected_points_angle[1][id_angle]]
              fy = (detected_points_angle[0][id_angle]+1) / AngleMap.shape[0] / dy
              fz = (detected_points_angle[1][id_angle]+1) / AngleMap.shape[1] / dz
              if fy > 1:
                fy = fy - 2
              if fz > 1:
                fz = fz - 2
              azhat = np.arcsin(fy / np.sqrt((1/Lambda)**2 - fz**2))
              elhat = np.arccos(fz * Lambda)
              elhat = np.pi/2 - elhat
              
              x, y, z = ssp.utils.sph2cart(rangeTarget, -azhat, -elhat)
              all_xyz.append([x,y,z])
              #[]
              print(f"Detected angle: azimuth={np.degrees(azhat):.2f} degrees, elevation={np.degrees(elhat):.2f} degrees")
              print(x,y,z)
              
          points = np.array(all_xyz)
          FigsAxes[1,0].cla()
          FigsAxes[1,0].scatter(points[:, 0], points[:, 1],points[:, 2], marker='o')  
          FigsAxes[1,0].plot(0, 0, 0, 'o', markersize=5, color='red')
          FigsAxes[1,0].plot([0,5], [0,0], [0,0], color='red')
          FigsAxes[1,0].set_xlabel('X (m)')
          FigsAxes[1,0].set_ylabel('Y (m)')
          FigsAxes[1,0].set_zlabel('Z (m)')
          FigsAxes[1,0].set_title("Detected Points")
          
            
  if FigsAxes is not None:
    plt.draw() 
    plt.pause(0.1)
    plt.gcf().canvas.flush_events() 
