import sensingsp as ssp
import torch

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
def runSimpleScenario(trained_Model_index = 1, health_state_index = 1, sample_index = 20):
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
