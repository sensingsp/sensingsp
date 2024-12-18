import sys
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.signal import lfilter
from scipy.io import loadmat
from matplotlib import pyplot as plt
from collections import defaultdict
import glob
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QCheckBox,
    QDialog,QScrollArea,QMessageBox,QComboBox,QProgressBar,
)
from PyQt5.QtCore import Qt
import sensingsp as ssp
import requests
import zipfile
import random
class RadarWaveformDataset(Dataset):
    def __init__(self, radar_dir, ecg_dir, transform_ecg=True):
        self.radar_files = sorted(glob.glob(os.path.join(radar_dir, '*.mat')))
        self.ecg_files = sorted(glob.glob(os.path.join(ecg_dir, '*.mat')))
        allfiles = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for file_path in self.radar_files + self.ecg_files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            info = extract_file_info(file_name)
            # print(info['Subject'],info['Scenario'],info['Signal Type'],info['Segment'])
            allfiles[info['Subject']][info['Scenario']][info['Segment']][info['Signal Type']]=file_path
        
        # # allfiles[3]['Resting'][2]['Radar']
        # k1=allfiles.keys()
        # print(k1) # dict_keys([1, 2, 3, 4, 5])
        # k2=allfiles[list(k1)[0]].keys()
        # print(k2) # dict_keys(['Resting', 'Valsalva'])
        # k3=allfiles[list(k1)[0]][list(k2)[0]].keys()
        # print(k3) # dict_keys([1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 7, 8, 9])
        # k4=allfiles[list(k1)[0]][list(k2)[0]][list(k3)[0]].keys()
        # print(k4) # dict_keys(['Radar', 'Ecg'])
        # k5=allfiles[list(k1)[0]][list(k2)[0]][list(k3)[0]][list(k4)[0]]
        # print(k5) # C:\Users\MOEIN~1.AHM\AppData\Local\Temp\SensingSP\datasets\SynchronizedRadarECGData\trainVal\radar\GDN0001_Resting_radar_1.mat
        # radar_mat = loadmat(k5)
        # radar_mat = radar_mat['radar_l']
        # print(radar_mat.shape) # (1, 1024)
        
        self.transform_ecg = transform_ecg

    def __len__(self):
        return len(self.radar_files)

    def __getitem__(self, idx):
        radar_mat = loadmat(self.radar_files[idx])
        ecg_mat = loadmat(self.ecg_files[idx])

        # Adjust the key if necessary. Assuming 'signal' is correct.
        radar_signal = radar_mat['radar_l'].squeeze() # (1024,)
        ecg_signal = ecg_mat['ecg_l'].squeeze() # (1024,)

        if self.transform_ecg:
            ecg_signal = normalize_ecg(ecg_signal)

        radar_signal = torch.tensor(radar_signal, dtype=torch.float32).unsqueeze(0)  # (1, 1024)
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)      # (1, 1024)

        return radar_signal, ecg_signal

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayersGenerator(nn.Module):
    def __init__(self):
        super(LayersGenerator, self).__init__()
        self.fc = nn.Linear(128, 4096)  # Fully Connected Layer
        
        # Define transposed convolutional layers
        self.transposed_conv_1 = nn.ConvTranspose2d(4096, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.transposed_conv_2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.transposed_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.transposed_conv_4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.transposed_conv_5 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.transposed_conv_6 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        # Define LeakyReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        # Fully Connected Layer and Reshape
        x = self.fc(x)
        x = x.view(-1, 128, 2, 8)  # Reshape to [Batch Size, Channels, Height, Width]
        
        # Transposed Convolutions and Activations
        x = self.leaky_relu(self.transposed_conv_1(x))
        x = self.leaky_relu(self.transposed_conv_2(x))
        x = self.leaky_relu(self.transposed_conv_3(x))
        x = self.leaky_relu(self.transposed_conv_4(x))
        x = self.leaky_relu(self.transposed_conv_5(x))
        x = self.transposed_conv_6(x)  # Last layer doesn't need activation
        
        # Normalization (if applicable)
        x = F.normalize(x, p=2, dim=1)
        return x


class LayersDiscriminator(nn.Module):
    def __init__(self):
        super(LayersDiscriminator, self).__init__()
        
        # Define convolutional layers
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=6, stride=2, padding=2)  # Input: [Batch Size, 1, 512, 511]
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=6, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(256, 512, kernel_size=6, stride=2, padding=2)
        
        # Define LeakyReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.3)
        
        # Define dropout layers
        self.dropout = nn.Dropout(0.3)
        
        # Fully Connected Layer
        self.fc = nn.Linear(512 * 32 * 31, 1)  # Flattened size from last convolutional layer

    def forward(self, x):
        # Convolutional layers with LeakyReLU and Dropout
        x = self.leaky_relu(self.conv_1(x))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.conv_2(x))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.conv_3(x))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.conv_4(x))
        x = self.dropout(x)
        
        # Flatten and Fully Connected Layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        
        return x


class GAN:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid and binary cross-entropy

        # Optimizers
        self.optim_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    def train_discriminator(self, real_data, fake_data):
        # Labels for real (1) and fake (0)
        real_labels = torch.ones(real_data.size(0), 1, device=self.device)
        fake_labels = torch.zeros(fake_data.size(0), 1, device=self.device)
        
        # Real data loss
        real_output = self.discriminator(real_data)
        loss_real = self.criterion(real_output, real_labels)
        
        # Fake data loss
        fake_output = self.discriminator(fake_data.detach())
        loss_fake = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        loss_d = loss_real + loss_fake
        
        # Backpropagation and optimization
        self.optim_d.zero_grad()
        loss_d.backward()
        self.optim_d.step()
        
        return loss_d.item()
    
    def train_generator(self, fake_data):
        # Labels for fake data treated as real (1)
        real_labels = torch.ones(fake_data.size(0), 1, device=self.device)
        
        # Generator loss
        output = self.discriminator(fake_data)
        loss_g = self.criterion(output, real_labels)
        
        # Backpropagation and optimization
        self.optim_g.zero_grad()
        loss_g.backward()
        self.optim_g.step()
        
        return loss_g.item()
    
    def generate_fake_data(self, batch_size, latent_dim):
        # Generate random noise as input to the generator
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        fake_data = self.generator(noise)
        return fake_data


class RadarWaveformApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generate Novel Radar Waveforms Using GAN")
        self.default_folder = os.path.join(ssp.config.temp_folder, "datasets", "GenerateNovelRadarWaveformsData")
        self.initUI()
        self.initNet()

    def initUI(self):
        # Main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Form layout for inputs
        form_layout = QFormLayout()
        self.data_folder_input = QLineEdit(self)
        self.data_folder_input.setText(self.default_folder)  # Set default folder
        self.data_folder_browse_button = QPushButton("Browse")
        self.data_folder_browse_button.clicked.connect(self.browse_folder)
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.data_folder_input)
        folder_layout.addWidget(self.data_folder_browse_button)
        form_layout.addRow("Dataset Folder:", folder_layout)
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.loaddataset)
        form_layout.addRow("Load:", self.load_button)

        # self.folderN_input = QSpinBox(self)
        # self.folderN_input.setMinimum(1)
        # self.folderN_input.setValue(8)
        # form_layout.addRow("Folder read num:", self.folderN_input)
        self.epochs_input = QSpinBox(self)
        self.epochs_input.setMinimum(1)
        self.epochs_input.setMaximum(10000)
        self.epochs_input.setValue(800)
        form_layout.addRow("Epochs:", self.epochs_input)

        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setMinimum(1)
        self.batch_size_input.setMaximum(10000)
        self.batch_size_input.setValue(900)
        form_layout.addRow("Batch Size:", self.batch_size_input)

        self.learning_rate_input = QDoubleSpinBox(self)
        self.learning_rate_input.setMinimum(0.0001)
        self.learning_rate_input.setValue(1.0)
        self.learning_rate_input.setDecimals(4)
        form_layout.addRow("Learning Rate (x*1e-3):", self.learning_rate_input)

        self.split_train_input = QDoubleSpinBox(self)
        self.split_train_input.setMinimum(0.1)
        self.split_train_input.setMaximum(0.9)
        self.split_train_input.setSingleStep(0.05)
        self.split_train_input.setValue(0.85)
        form_layout.addRow("Train Split Ratio:", self.split_train_input)

        # self.split_val_input = QDoubleSpinBox(self)
        # self.split_val_input.setMinimum(0.05)
        # self.split_val_input.setMaximum(0.9)
        # self.split_val_input.setSingleStep(0.05)
        # self.split_val_input.setValue(0.15)
        # form_layout.addRow("Validation Split Ratio:", self.split_val_input)

        
        
        # Add "Clutter Removal" checkbox
        
        # self.clutter_removal_checkbox = QCheckBox("Enable Clutter Removal", self)
        # self.clutter_removal_checkbox.setChecked(True)
        # form_layout.addRow(self.clutter_removal_checkbox)

        main_layout.addLayout(form_layout)
        self.combobox_model = QComboBox()
        self.combobox_model.addItems(["Model: Layers1","Model: Layers2"])
        main_layout.addWidget(self.combobox_model)

        # Buttons for actions in rows
        # Row 0
        # row0_layout = QHBoxLayout()
        # self.load_button = QPushButton("Load Dataset")
        # self.load_button.clicked.connect(self.loaddataset)
        # self.defmodel_button = QPushButton("Define Model")
        # self.defmodel_button.clicked.connect(self.def_model)
        # row0_layout.addWidget(self.load_button)
        # # row0_layout.addWidget(self.defmodel_button)
        # main_layout.addLayout(row0_layout)

        
        # Buttons for actions in rows
        # Row 1
        row1_layout = QHBoxLayout()
        self.train_button = QPushButton("Train: NotImpYet")
        self.train_button.clicked.connect(self.train_model)
        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self.test_model)
        row1_layout.addWidget(self.train_button)
        row1_layout.addWidget(self.test_button)
        main_layout.addLayout(row1_layout)

        # Row 2
        row2_layout = QHBoxLayout()
        self.visualize_button = QPushButton("Visualize Network")
        self.visualize_button.clicked.connect(self.visualize_model)
        self.visualize_samples_button = QPushButton("Visualize Samples")
        self.visualize_samples_button.clicked.connect(self.visualize_samples)
        # row2_layout.addWidget(self.visualize_button)
        
        self.combobox = QComboBox()
        # row2_layout.addWidget(self.combobox)
        
        
        self.sampleVisN_input = QSpinBox(self)
        self.sampleVisN_input.setMinimum(1)
        self.sampleVisN_input.setValue(5)
        row2_layout.addWidget(self.sampleVisN_input)
        
        row2_layout.addWidget(self.visualize_samples_button)
        main_layout.addLayout(row2_layout)

        # Row 3
        # row3_layout = QHBoxLayout()
        # self.combobox.addItems(["Option 1", "Option 2", "Option 3", "Option 4"])
        # self.combobox.currentIndexChanged.connect(self.on_combobox_change)
        # row3_layout.addWidget(self.combobox)
        # self.visualize_button = QPushButton("Visualize Network")
        # self.visualize_button.clicked.connect(self.visualize_model)
        # self.visualize_samples_button = QPushButton("Visualize Samples")
        # self.visualize_samples_button.clicked.connect(self.visualize_samples)
        # row3_layout.addWidget(self.visualize_button)
        # row3_layout.addWidget(self.visualize_samples_button)
        # main_layout.addLayout(row3_layout)

        # # Row 3
        # row3_layout = QHBoxLayout()
        # self.visualize_image_button = QPushButton("Visualize Model as Image")
        # self.visualize_image_button.clicked.connect(self.visualize_model_as_image)
        # self.visualize_params_button = QPushButton("Visualize Parameter Updates (bash:tensorboard --logdir=runs)")
        # self.visualize_params_button.clicked.connect(self.visualize_parameter_updates)
        # row3_layout.addWidget(self.visualize_image_button)
        # row3_layout.addWidget(self.visualize_params_button)
        # main_layout.addLayout(row3_layout)

        # self.testfile_input = QLineEdit(self)
        # self.testfile_input.setText(os.path.join(ssp.config.temp_folder,'HandG.mat'))  # Set default folder
        # self.testfile_browse_button = QPushButton("Browse")
        # self.testfile_browse_button.clicked.connect(self.browse_testfile)
        # testfile_layout = QHBoxLayout()
        
        # testfile_layout.addWidget(self.testfile_input)
        # testfile_layout.addWidget(self.testfile_browse_button)
        # # form_layout.addRow("Test File:", testfile_layout)

        # row4_layout = QHBoxLayout()
        # self.testfile_button = QPushButton("Test input file")
        # self.testfile_button.clicked.connect(self.testinput_model)
        # row4_layout.addLayout(testfile_layout)
        # row4_layout.addWidget(self.testfile_button)
        # main_layout.addLayout(row4_layout)


        # Status display
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

    def initNet(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.savebestmodelpath = os.path.join(ssp.config.temp_folder,"best_model.pth")

    def initdataset(self):
        
        if 0:
            # Parameters
            num_samples = 13700
            length = 256
            sample_rate = 1e6
            save_dir = "phase_coded_waveforms"

            # Generate the dataset
            dataset = create_dataset(num_samples, length, sample_rate, save_dir)

            
            
        zip_path = "GenerateNovelRadarWaveformsData.zip"
        url = "https://ssd.mathworks.com/supportfiles/phased/data/GenerateNovelRadarWaveformsData.zip"
        zip_folder = self.default_folder

        # Check if the folder exists
        if not os.path.exists(zip_folder):

            # Download the ZIP file
            print("Downloading the ZIP file...")
            response = requests.get(url, stream=True)
            datasets_path = os.path.join(ssp.config.temp_folder, "datasets")
            if not os.path.exists(datasets_path):
                os.makedirs(datasets_path)
            zip_path = os.path.join(ssp.config.temp_folder, "datasets", "GenerateNovelRadarWaveformsData.zip")

            # Save the ZIP file locally
            with open(zip_path, "wb") as f:
                f.write(response.content)
            print("Download complete!")
            
            
            # Extract the ZIP file
            print("Extracting the ZIP file...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(zip_folder)
            print(f"Data extracted to {zip_folder}")
            
            # Clean up: Remove the ZIP file
            os.remove(zip_path)
            print("Temporary ZIP file removed.")
            
            sdpath=os.path.join(ssp.config.temp_folder, "datasets", "GenerateNovelRadarWaveformsData"
                         , "GenerateNovelRadarWaveformsData")
            sdfile=os.path.join(sdpath,"SyntheticPhaseCodedRadarWaveforms.zip")
            with zipfile.ZipFile(sdfile, "r") as zip_ref:
                zip_ref.extractall(sdpath)
            os.remove(sdfile)
            
        numChips = 256
        sampleRate = 1e6
        PRF = sampleRate/numChips
        chipWidth = 1/sampleRate
        dataset_folder = os.path.join(zip_folder, "GenerateNovelRadarWaveformsData", "SyntheticPhaseCodedRadarWaveforms")
        self.dataset_files = sorted(glob.glob(os.path.join(dataset_folder, '*.mat')))
        # N = len(self.dataset_files) # 13700
        
        # self.dataset = RadarECGDataset(dataset_folder)
        # train_ratio = self.split_train_input.value()
        # train_size = int(train_ratio * len(self.trainVal_dataset))
        # val_size = len(self.trainVal_dataset) - train_size
        # self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        
        # batch_size = self.batch_size_input.value() *0+1
        # self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.default_folder)
        if folder:
            self.default_folder = folder
            self.data_folder_input.setText(self.default_folder)
            self.loaddataset()
    def testinput_model(self):
        test_file_path = self.testfile_input.text()
        if test_file_path:
            fig, axs = plt.subplots(1, 5, figsize=(15, 6))
            # Load the best model
            model = MultiInputModel(num_classes=self.num_classes)
            model.load_state_dict(torch.load(self.savebestmodelpath))
            model.to(self.device)
            
            model.eval()
            mat_data = loadmat(test_file_path)
            left = mat_data.get("Left")
            top = mat_data.get("Top")
            right = mat_data.get("Right")
            if self.dataset.clutter_removal:
                b = [1, -1]
                a = [1, -0.9]
                left = lfilter(b, a, left, axis=0)
                top = lfilter(b, a, top, axis=0)
                right = lfilter(b, a, right, axis=0)

            # Normalize each row to [0,1]
            # Similar to MATLAB's normalize(...,2,"range",[0,1]), 
            # which normalizes along the second dimension (fast-time)
            left = self.dataset.normalize_along_rows(left)
            top = self.dataset.normalize_along_rows(top)
            right = self.dataset.normalize_along_rows(right)
            B = int(left.shape[0]/self.dataset.slow_time_per_sample)
            for i in range(B):
                x1 = left[i*self.dataset.slow_time_per_sample:(i+1)*self.dataset.slow_time_per_sample]
                x2 = top[i*self.dataset.slow_time_per_sample:(i+1)*self.dataset.slow_time_per_sample]
                x3 = right[i*self.dataset.slow_time_per_sample:(i+1)*self.dataset.slow_time_per_sample]
                
                radar_tensor = np.stack([x1, x2, x3], axis=0)  # shape: (3,90,189)
                radar_tensor = torch.tensor(radar_tensor, dtype=torch.float32).unsqueeze(0)
                radar_tensor = radar_tensor.to(self.device)
                with torch.no_grad():
                    outputs = model(radar_tensor)
                    _, preds = torch.max(outputs, 1)
                    Gi = f"G{preds.item()}"
                    Gn = self.dataset.gestureVocabulary[preds.item()]
                    print(Gi,Gn)
                
                radar=["Left","Top","Right"]
                SourceLocations = np.array([[0, 0], [0.5, 0.5], [1, 0]])
                results= []
                data = radar_tensor.cpu().numpy().squeeze()
                max_indices = np.argmax(np.abs(data), axis=2)
                for i2 in range(max_indices.shape[1]):
                    ranges = 1.2/189 * max_indices[:,i2]
                    # Define the objective function
                    def objective(xy):
                        x, y = xy
                        distances = np.sqrt((SourceLocations[:, 0] - x)**2 + (SourceLocations[:, 1] - y)**2)
                        return np.sum(np.abs(ranges - distances))
                    
                    # Minimize the objective function
                    res = minimize(objective, x0=[0.5, 0], bounds=[(-1, 2), (-1, 2)], method='L-BFGS-B')
                    results.append(res.x)
                    # estimate x,y such that min sum(distance(SourceLocations[i]-[x,y])) i 0:3
                results = np.array(results)
                for i2 in range(3):
                    axs[i2].clear()
                    axs[i2].imshow(data[i2,:,:], aspect='auto', cmap='viridis')
                    axs[i2].set_title(f"{Gn} : {radar[i2]}")
                i2=3
                axs[i2].clear()
                axs[i2].plot(results[:,0],results[:,1],'.')
                axs[i2].set_aspect('equal', adjustable='box')

                axs[i2].set_title(f"{Gn}")
                i2=4
                axs[i2].clear()
                axs[i2].plot(data[0,40:45,:].T)
                plt.draw()
                plt.tight_layout()
                plt.gcf().canvas.flush_events()
                plt.pause(0.001)
                if i>1:
                    break
        
            plt.show()
            
    def browse_testfile(self):
        # Open a file dialog to select a .mat file
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Test File", 
            os.path.join(ssp.config.temp_folder,""), 
            "MAT Files (*.mat)"  # Filter to show only .mat files
        )
        if file:
            # Set the selected file path to an input field or variable
            # self.test_file_path = file
            self.testfile_input.setText(file)  # Assuming there's a QLineEdit for this
            # self.load_testfile()  # Call a method to handle loading the test file

    
    def loaddataset(self):
        self.status_label.setText("Status: Loading...")
        self.initdataset()
        self.status_label.setText(f"Status: Dataset files : {len(self.dataset_files)}")
        
    def def_model(self):
        # self.model = MultiInputModel(num_classes=self.num_classes)
        1
        
    def train_model(self):
        latent_dim = 128
        batch_size = 64
        epochs = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate models
        generator = LayersGenerator()
        discriminator = LayersDiscriminator()

        # Initialize GAN
        gan = GAN(generator, discriminator, device)

        # Training loop
        for epoch in range(epochs):
            for _ in range(100):  # Assuming 100 batches per epoch (modify based on your data loader)
                # Generate real data (for example purposes, using random tensors with appropriate size)
                real_data = torch.rand(batch_size, 1, 512, 511, device=device)
                
                # Generate fake data
                fake_data = gan.generate_fake_data(batch_size, latent_dim)
                
                # Train discriminator
                loss_d = gan.train_discriminator(real_data, fake_data)
                
                # Train generator
                loss_g = gan.train_generator(fake_data)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

        mmmmmmmmmmmmmmmm
        # Placeholder for training logic
        self.status_label.setText("Status: Training started...")
        # -----------------------------
        # Model, Loss, Optimizer
        # -----------------------------
        if self.combobox_model.currentIndex()==0:
            self.model = Layer1_HumanHM()
        else:
            self.model = Layer2_HumanHM()
        
        self.model.to(self.device)
            
        self.criterion = nn.CrossEntropyLoss()
        learning_rate = self.learning_rate_input.value()*1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        num_epochs = self.epochs_input.value() 
        


        fig, axs = plt.subplots(2, 1)
        ax = axs.flatten()
        # -----------------------------
        # Training and Validation Loop
        # -----------------------------
        Validation_frequency = 40
        Validation_count = 0
        matlab_acc_t = [] 
        matlab_loss_t = []
        matlab_acc_v = [] 
        matlab_loss_v = []
        matlab_t_index = [] 
        matlab_v_index = []
        
        
        for epoch in range(num_epochs):
            val_loader_iter = iter(self.val_loader)
        # ---- Training ----
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)  # shape [B,3,90,189]
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # shape [B, num_classes]
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)

                matlab_acc = 100.0 * torch.sum(preds == labels).item() / labels.size(0) 
                matlab_loss = loss.item()
                matlab_acc_t.append(matlab_acc)
                matlab_loss_t.append(matlab_loss)
                matlab_t_index.append(len(matlab_acc_t))
                Validation_count+=1
                if Validation_count==1 or Validation_count>Validation_frequency:
                    Validation_count=1
                    self.model.eval()
                    with torch.no_grad():
                        inputs, labels = next(val_loader_iter)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        
                        matlab_acc = 100.0 * torch.sum(preds == labels).item() / labels.size(0) 
                        matlab_loss = loss.item()
                        matlab_acc_v.append(matlab_acc)
                        matlab_loss_v.append(matlab_loss)
                        matlab_v_index.append(len(matlab_acc_t))
                        if len(matlab_loss_v)==1:
                            best_loss=matlab_loss+1


                        if matlab_loss < best_loss:
                            best_loss = matlab_loss
                            torch.save(self.model.state_dict(), self.savebestmodelpath)
                    self.model.train()
                
                
                ax[0].clear()
                ax[0].plot(matlab_t_index,matlab_acc_t)
                ax[0].plot(matlab_v_index,matlab_acc_v,'--o')
                ax[0].set_title(f"Epoch {epoch + 1}")
                ax[0].set_xlabel("Iteration")
                ax[0].set_ylabel("Accuracy")
                ax[0].grid(True) 
                ax[1].clear()
                ax[1].plot(matlab_t_index,matlab_loss_t)
                ax[1].plot(matlab_v_index,matlab_loss_v,'--o')
                ax[1].set_xlabel("Iteration")
                ax[1].set_ylabel("Loss")
                ax[1].grid(True) 
                plt.draw()
                plt.pause(0.001)
                plt.gcf().canvas.flush_events()
            

        # Add actual training logic here
        plt.show()
        self.status_label.setText("Status: Training completed!")

    def test_model(self):
        # Placeholder for testing logic
        self.status_label.setText("Status: Testing started...")
        # -----------------------------
        # Testing Loop
        # -----------------------------
        # Load the best model
        self.model.load_state_dict(torch.load(self.savebestmodelpath))
        self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")

        # Confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_preds)
        # Plot confusion matrix
        class_labels = [str(i) for i in range(len(cm))]  # If using numeric classes

        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        plt.colorbar(cax)

        # Annotate each cell with the numeric value
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='black')

        # Set axis ticks and labels
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)

        # Add title (optional)
        plt.title('Confusion Matrix', fontsize=14)

        plt.show()

        print("Classification Report:")
        print(classification_report(all_labels, all_preds))
                          
        # Add actual testing logic here
        self.status_label.setText("Status: Testing completed!")

    def visualize_model(self):
        # Placeholder for model visualization logic
        self.status_label.setText("Status: Visualizing model...")
        try:
            import io
            from contextlib import redirect_stdout
            from torchsummary import summary
            s,l=self.dataset[0]
            input_width,input_height = s.shape[1] , s.shape[2]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                summary(self.model, input_size=[(1, input_width, input_height), (1, input_width, input_height), (1, input_width, input_height)])
            model_summary = buffer.getvalue()

            # Create a scrollable dialog
            scrollable_dialog = QDialog(self)
            scrollable_dialog.setWindowTitle("Model Summary")
            scrollable_dialog.setMinimumSize(600, 400)

            layout = QVBoxLayout(scrollable_dialog)

            scroll_area = QScrollArea(scrollable_dialog)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)

            inner_widget = QWidget()
            scroll_area.setWidget(inner_widget)
            inner_layout = QVBoxLayout(inner_widget)

            summary_label = QLabel(model_summary)
            summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Allow text selection
            summary_label.setWordWrap(True)
            inner_layout.addWidget(summary_label)

            # Add close button
            close_button = QPushButton("Close", scrollable_dialog)
            close_button.clicked.connect(scrollable_dialog.accept)
            layout.addWidget(close_button)

            scrollable_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during model visualization: {str(e)}")
    
    def visualize_samples(self):
        # Placeholder for sample visualization logic
        self.status_label.setText("Status: Visualizing samples...")
        # g = self.combobox.currentText()
        fig, axs = plt.subplots(2, 1, figsize=(15, 6))
        kd = 0
        MaxVis = self.sampleVisN_input.value()
        xv = 0
        for inputs, labels in self.train_loader:
            xax = np.arange(xv,xv+inputs.shape[2])
            xv+=inputs.shape[2]
            i=0
            # axs[i].clear()
            axs[i].plot(xax,inputs[0,0,:])
            # axs[i].set_aspect('equal', adjustable='box')
            i=1
            # axs[i].clear()
            axs[i].plot(xax,labels[0,0,:])
            # axs[i].set_aspect('equal', adjustable='box')
            plt.draw()
            plt.tight_layout()
            plt.gcf().canvas.flush_events()
            plt.pause(0.001)
            kd +=1
            if kd >= MaxVis:
                break

        plt.show()
   

    def visualize_model_as_image(self):
        # Placeholder for exporting model as image
        self.status_label.setText("Status: Exporting model as image...")

    def visualize_parameter_updates(self):
        # Placeholder for visualizing parameter updates (e.g., TensorBoard)
        self.status_label.setText("Status: Visualizing parameter updates...")

# ----------------------------
# Run Application Function
# ----------------------------
def runradarWaveformapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    appSTYLESHEET = ssp.config.appSTYLESHEET
    app.setStyleSheet(appSTYLESHEET)  # Replace with your desired stylesheet if any
    
    window = RadarWaveformApp()
    window.show()
    app.exec_()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    runradarWaveformapp()
    

def create_dataset(num_samples, length, sample_rate, save_dir):
    """
    Create a synthetic dataset of phase-coded waveforms.

    Parameters:
    - num_samples: Number of waveforms to generate
    - length: Length of each waveform
    - sample_rate: Sample rate of the waveforms
    - save_dir: Directory to save the dataset
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    phase_code_types = ['Frank', 'P1', 'P2', 'P3', 'P4', 'Zadoff-Chu']
    zadoff_chu_indices = [1, 5, 7]

    dataset = []

    for i in range(num_samples):
        code_type = random.choice(phase_code_types)

        if code_type == 'Frank':
            waveform = ssp.radar.radarwaveforms.frank_code(length)
        elif code_type == 'P1':
            waveform = ssp.radar.radarwaveforms.p1_code(length)
        elif code_type == 'P2':
            waveform = ssp.radar.radarwaveforms.p2_code(length)
        elif code_type == 'P3':
            waveform = ssp.radar.radarwaveforms.p3_code(length)
        elif code_type == 'P4':
            waveform = ssp.radar.radarwaveforms.p4_code(length)
        elif code_type == 'Zadoff-Chu':
            seq_idx = random.choice(zadoff_chu_indices)
            waveform = ssp.radar.radarwaveforms.zadoff_chu_code(length, seq_idx)
        else:
            raise ValueError(f"Unknown code type: {code_type}")

        dataset.append((waveform, code_type))

        # Save waveform to file
        np.save(os.path.join(save_dir, f"waveform_{i}.npy"), waveform)

    print(f"Dataset created with {num_samples} samples in {save_dir}.")
    return dataset


