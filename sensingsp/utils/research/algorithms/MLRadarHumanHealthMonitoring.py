import sensingsp as ssp
import torch
import bpy
import numpy as np  
# def runSimpleScenario():        
# url,zfile = "https://ssd.mathworks.com/supportfiles/SPT/data/","SynchronizedRadarECGData.zip"
# save_path = ssp.utils.file_in_tmpfolder("datasets/SynchronizedRadarECGData") 
# save_path = ssp.utils.hub.download_zipfile_extract_remove(url,zfile,save_path,False)
# downloaded_model_path = ssp.utils.hub.fetch_pretrained_model("models", "HumanHealthMonitoring ConvAE_BiLSTM")
# if downloaded_model_path:
#     model = ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.Layer2_HumanHM()
#     model.load_state_dict(torch.load(downloaded_model_path))
#     N = 40
#     State = 'Resting' #, Apnea , Valsalva
#     matfile = save_path+f"/test/radar/GDN0006_{State}_radar_{N}.mat"
#     matfileecg = save_path+f"/test/ecg/GDN0006_{State}_ecg_{N}.mat"
#     radar_signal, ecg_signal = ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.load_sample(matfile,matfileecg)
#     from matplotlib import pyplot as plt
#     fig, ax = plt.subplots(3, 1)
#     with torch.no_grad():
#         outputs = model(radar_signal.unsqueeze(0))
#         ax[0].plot(radar_signal[0,:].cpu().numpy())
#         ax[1].plot(ecg_signal[0,:].cpu().numpy())
#         ax[2].plot(outputs[0,0,:].cpu().numpy())
#         plt.show()

def createChestMovement(frame, fps):
    time = (frame+1*158) / fps
    location = 0 + 1*3e8/70e9 / 4 * np.sin(2 * np.pi *20/60. * time+np.random.rand()*6 )+ .1*3e8/70e9 / 4 * np.sin(2 * np.pi *80/60.0 * time )+ 0*2*3e8/70e9 / 4 * np.sin(2 * np.pi / 50 * time )
    return location
def runSimpleScenario(trained_Model_index = 1, health_state_index = 1, sample_index = 20,sim=True):
    from matplotlib import pyplot as plt
    
    # Define URLs and paths
    dataset_url = "https://ssd.mathworks.com/supportfiles/SPT/data/"
    dataset_zipfile = "SynchronizedRadarECGData.zip"
    save_path = ssp.utils.file_in_tmpfolder("datasets/SynchronizedRadarECGData")

    # Download and extract dataset
    save_path = ssp.utils.hub.download_zipfile_extract_remove(dataset_url, dataset_zipfile, save_path, False)

    # Load pretrained model
    if trained_Model_index == 1:
        model_path = ssp.utils.hub.fetch_pretrained_model("models", "HumanHealthMonitoring ConvAE_BiLSTM 800")
    else:
        model_path = ssp.utils.hub.fetch_pretrained_model("models", "HumanHealthMonitoring ConvAE_BiLSTM")
        
        
    if not model_path:
        raise RuntimeError("Pretrained model could not be fetched. Ensure the model hub is accessible.")

    model = ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.Layer2_HumanHM()
    model.load_state_dict(torch.load(model_path))

    # Specify sample parameters
    if health_state_index == 1:
       health_state = "Resting"  
    elif health_state_index == 2:
       health_state = "Apnea"  
    else:
        health_state = "Valsalva"
    radar_file = f"{save_path}/test/radar/GDN0006_{health_state}_radar_{sample_index}.mat"
    ecg_file = f"{save_path}/test/ecg/GDN0006_{health_state}_ecg_{sample_index}.mat"

    # Load radar and ECG signals
    radar_signal, ecg_signal = ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.load_sample(radar_file, ecg_file)

    # Perform inference and visualize results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Human Health Monitoring: Radar and ECG Signals")

    with torch.no_grad():
        # Pass radar signal through the model
        output_signal = model(radar_signal.unsqueeze(0))

        # Plot input radar signal
        axes[0].plot(radar_signal[0, :].cpu().numpy(), label="Radar Signal")
        axes[0].set_title("Input Radar Signal")
        axes[0].legend()
        
        # Plot corresponding ECG signal
        axes[1].plot(ecg_signal[0, :].cpu().numpy(), label="ECG Signal", color="orange")
        axes[1].set_title("Input ECG Signal")
        axes[1].legend()

        # Plot model output
        axes[2].plot(output_signal[0, 0, :].cpu().numpy(), label="Model Output", color="green")
        axes[2].set_title("Model Output Signal")
        axes[2].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    if sim==False:
        return
    ssp.utils.initialize_environment()
    # 1024 samples
    # 200 Hz
    duration = 1024*1.1/200
    total_frames = int(duration * bpy.context.scene.render.fps)
    
    obj = ssp.environment.deform_scenario_1(angLim=.6,Lengths = [.2,.1,.1],elps=[.25,.14,.5],sbd=4,cycles=8,cycleHLen=15)
    

    SensorType = ssp.radar.utils.RadarSensorsCategory.SISO_mmWave76GHz
    radar = ssp.radar.utils.addRadar( radarSensor = SensorType, location_xyz=[0, -1, .3])
    radar.rotation_euler.z = 0
    
    # cube,par = ssp.radar.utils.addTarget(refRadar=radar,size=.1,range=1)
    # x0=cube.location.x
    # for i in range(total_frames):
    #     cube.location.x = x0+createChestMovement(i+.0,bpy.context.scene.render.fps+.0)
    #     cube.keyframe_insert(data_path="location", frame=i + 1)  
        
    ssp.utils.setRadar_multipleCPI_in_oneFrame(radar)
    radar['PRI_us']=1000/200*1000
    radar['NPulse']=1024
    radar['N_ADC']=1
    radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=1
    
    ssp.utils.initialize_simulation(endframe=total_frames,rayTracing=2)
    unwrap_signal=[]
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        for isuite, radarSpecifications in enumerate(ssp.RadarSpecifications):
            for iradar, specifications in enumerate(radarSpecifications):
                for XRadar, timeX in Signals[isuite]['radars'][iradar]:
                    X_fft_fast, d_fft = ssp.radar.utils.rangeprocessing(XRadar, specifications)
                    rangeBin = np.argmax(np.abs(X_fft_fast[:,0,0]))
                    unwrap_signal.append(X_fft_fast[rangeBin,:,0])
                    # time_signal[iradar].append(TimeRadar)
                    # unwrappedPhase=np.unwrap(np.angle(np.array(unwrap_signal[iradar])))
                    # unwrappedPhase2=np.unwrap(np.angle(np.imag(np.array(unwrap_signal[iradar]))+1j*np.real(np.array(unwrap_signal[iradar]))))
                    
        # plt.draw()  # Redraw the figure
        # plt.gcf().canvas.flush_events() 
        # plt.pause(.001)
        print(f'processed frame = {ssp.config.CurrentFrame}')
        ssp.utils.increaseCurrentFrame()
    simsignal = np.unwrap(np.angle(np.array(unwrap_signal[0])))
    plt.plot(simsignal)    
    plt.show()
    with torch.no_grad():
        # Pass radar signal through the model
        
        radar_signal = torch.tensor(simsignal, dtype=torch.float32).unsqueeze(0)  # (1, 1024)
        
        output_signal = model(radar_signal.unsqueeze(0))
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
        # Plot input radar signal
        axes[0].plot(radar_signal[0, :].cpu().numpy(), label="Radar Signal")
        axes[0].set_title("Input Radar Signal")
        axes[0].legend()
        
        # Plot model output
        axes[1].plot(output_signal[0, 0, :].cpu().numpy(), label="Model Output", color="green")
        axes[1].set_title("Model Output Signal")
        axes[1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    
def model_inference(radar_signal,trained_Model_index = 1):
    from matplotlib import pyplot as plt
    # Load pretrained model
    if trained_Model_index == 1:
        model_path = ssp.utils.hub.fetch_pretrained_model("models", "HumanHealthMonitoring ConvAE_BiLSTM 800")
    else:
        model_path = ssp.utils.hub.fetch_pretrained_model("models", "HumanHealthMonitoring ConvAE_BiLSTM")
        
        
    if not model_path:
        raise RuntimeError("Pretrained model could not be fetched. Ensure the model hub is accessible.")

    model = ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.Layer2_HumanHM()
    model.load_state_dict(torch.load(model_path))

    simsignal = np.unwrap(np.angle(np.array(radar_signal)))

    with torch.no_grad():
        radar_signal = torch.tensor(simsignal, dtype=torch.float32).unsqueeze(0)  # (1, 1024)
        output_signal = model(radar_signal.unsqueeze(0))
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
        # Plot input radar signal
        axes[0].plot(radar_signal[0, :].cpu().numpy(), label="Radar Signal")
        axes[0].set_title("Input Radar Signal")
        axes[0].legend()
        
        # Plot model output
        axes[1].plot(output_signal[0, 0, :].cpu().numpy(), label="Model Output", color="green")
        axes[1].set_title("Model Output Signal")
        axes[1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    