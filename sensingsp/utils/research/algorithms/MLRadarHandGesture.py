
import sensingsp as ssp
import numpy as np
from mathutils import Vector
import torch

def runSimpleScenario( hubcategory = "animation", hubname = "Hand" , dataset_test = False):        
    ssp.utils.initialize_environment()
    hand_file_path = ssp.utils.hub.fetch_file(category=hubcategory,name=hubname)
    ssp.environment.add_blenderfileobjects(hand_file_path,RCS0=1,translation=(.55,.3,-.1),rotation=(0,0,np.pi/2))

    SensorType = ssp.radar.utils.RadarSensorsCategory.Xhetru_X4
    radar_left = ssp.radar.utils.addRadar( radarSensor = SensorType, location_xyz=[0, 0, 0])
    radar_top = ssp.radar.utils.addRadar(radarSensor = SensorType, location_xyz=[.55, 0, .5])
    radar_right = ssp.radar.utils.addRadar(radarSensor = SensorType, location_xyz=[1.1, 0, 0])

    radar_right.rotation_euler = Vector((np.pi/2,0, np.pi/2))
    radar_top.rotation_euler = Vector((0,0,0))

    ssp.utils.initialize_simulation(endframe=120)


    MultiSensorData = []
    while ssp.config.run():
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        for isuite, radarSpecifications in enumerate(ssp.RadarSpecifications):
            for iradar, specifications in enumerate(radarSpecifications):
                for XRadar, timeX in Signals[isuite]['radars'][iradar]:
                    datasample = np.real(XRadar[:,:,0].T)
                    MultiSensorData.append(datasample)
        minPulseNumber = 0 if len(MultiSensorData)==0 else min([len(ds) for ds in MultiSensorData])
        print(f'Processed frame = {ssp.config.CurrentFrame}')
        ssp.utils.increaseCurrentFrame()
        if minPulseNumber>=90:
            break
    simulated_samples = ssp.ai.radarML.HandGestureMisoCNN.make_sample(MultiSensorData[0],MultiSensorData[1],MultiSensorData[2])
    if dataset_test:
        url,zfile = "https://ssd.mathworks.com/supportfiles/SPT/data/","uwb-gestures.zip"
        save_path = ssp.utils.file_in_tmpfolder("datasets") 
        save_path = ssp.utils.hub.download_zipfile_extract_remove(url,zfile,save_path)
    downloaded_model_path = ssp.utils.hub.fetch_pretrained_model("models", "HandGesture MisoCNN")
    if downloaded_model_path:
        model = ssp.ai.radarML.HandGestureMisoCNN.MultiInputModel(num_classes=12)
        model.load_state_dict(torch.load(downloaded_model_path))
        if dataset_test:
            subject = torch.randint(low=1, high=8 , size=(1,)).item()
            gesture = torch.randint(low=1, high=12, size=(1,)).item()
            matfile_path = save_path+f"/HV_0{subject}/HV_0{subject}_G{gesture}_RawData.mat"
            samples = ssp.ai.radarML.HandGestureMisoCNN.load_sample(matfile_path)
            for radar_tensor in samples:
                with torch.no_grad():
                    outputs = model( radar_tensor )
                    _ , preds = torch.max( outputs , 1 )
                    Gi = f"G{preds.item()+1}"
                    print(f"Sample from Gesture {gesture} of subject {subject} -> classified as {Gi}")
        dataset = ssp.ai.radarML.HandGestureMisoCNN.RadarGestureDataset(data_folder='')  
        for radar_tensor in simulated_samples:
            with torch.no_grad():
                outputs = model( radar_tensor )
                _ , preds = torch.max( outputs , 1 )
                Gi = f"G{preds.item()+1}"
                print(f"Sample from Simulator -> classified as {Gi}: {dataset.gestureVocabulary[preds.item()]}")
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                classes,softmax_probs=[],[]
                for idx, prob in enumerate(probabilities[0]):
                    class_name = dataset.gestureVocabulary[idx]
                    print(f"Class {class_name}: {prob:.4f} : {outputs[0,idx]:.4f}")
                    classes.append(class_name)
                    softmax_probs.append(prob.cpu())
                for idx, prob in enumerate(softmax_probs):
                    bar_length = int(prob * 50)  
                    print(f"Class {classes[idx]:<20}: {prob:.2f}: {'#' * bar_length}")
                