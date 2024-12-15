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
import zipfile
import requests

def normalize_ecg(ecg_signal):
    ecg_signal = ecg_signal - np.median(ecg_signal)
    max_val = np.max(np.abs(ecg_signal))
    if max_val < 1e-12:
        max_val = 1.0
    ecg_signal = ecg_signal / max_val
    return ecg_signal

def extract_file_info(file_name):
    # Define the regular expression pattern
    pattern = r"GDN(?P<subject>\d+)_" \
              r"(?P<scenario>\w+)_" \
              r"(?P<signal>\w+)_" \
              r"(?P<segment>\d+)"

    # Match the file name with the pattern
    match = re.match(pattern, file_name)

    if match:
        # Extract details directly from the match groups
        subject = int(match.group('subject'))
        scenario = match.group('scenario').capitalize()
        signal = match.group('signal').capitalize()
        segment = int(match.group('segment'))

        return {
            'Subject': subject,
            'Scenario': scenario,
            'Signal Type': signal,
            'Segment': segment
        }
    else:
        raise ValueError("File name does not match the expected format")
class RadarECGDataset(Dataset):
    def __init__(self, radar_dir, ecg_dir, transform_ecg=True):
        self.radar_files = sorted(glob.glob(os.path.join(radar_dir, '*.mat')))
        self.ecg_files = sorted(glob.glob(os.path.join(ecg_dir, '*.mat')))
        allfiles = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for file_path in self.radar_files + self.ecg_files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            info = extract_file_info(file_name)
            # print(info['Subject'],info['Scenario'],info['Signal Type'],info['Segment'])
            allfiles[info['Subject']][info['Scenario']][info['Segment']][info['Signal Type']]=file_path
        # for 
        
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
def printnninfo(Lay,s=""):
    params = list(Lay.parameters())
    for i, p in enumerate(params):
        print(f"{s} Parameter {i} shape:", p.shape)
def printnnshape(x,s=""):
    return
    print(s,x.shape)
class Layer1_HumanHM(nn.Module):
    def __init__(self):
        super(Layer1_HumanHM, self).__init__()
        
        # Sequence Input Layer: Assuming input dimension = (batch_size, 1, 1024)
        
        self.conv1 = nn.Conv1d(1, 3, kernel_size=4, padding='same')
        printnninfo(self.conv1,"conv1")
        self.layer_norm1 = nn.LayerNorm(3)
        printnninfo(self.layer_norm1,"layer_norm1")
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=64, stride=8, padding=28)
        printnninfo(self.conv2,"conv2")
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(8)
        printnninfo(self.batch_norm2,"batch_norm2")
        self.dropout2 = nn.Dropout(0.2)
        
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1,padding=1, dilation=2)

        self.conv3 = nn.Conv1d(8, 8, kernel_size=32, stride=4, padding=14)
        printnninfo(self.conv3,"conv3")
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(8)
        printnninfo(self.batch_norm3,"batch_norm3")
        
        # MaxPooling1d (kernel size 2, padding same)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1,padding=1, dilation=2)
        
        self.transposed_conv1 = nn.ConvTranspose1d(8, 8, kernel_size=32, stride=4,padding=14)
        printnninfo(self.transposed_conv1,"transposed_conv1")
        self.relu4 = nn.ReLU()
        
        self.transposed_conv2 = nn.ConvTranspose1d(8, 8, kernel_size=64, stride=8, padding=28)
        printnninfo(self.transposed_conv2,"transposed_conv2")
        
        # BiLSTM Layer (8 units)
        # self.bilstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm = nn.LSTM(input_size=8,        # Number of input features (C from the MATLAB diagram)
                hidden_size=8,        # Number of hidden units per direction
                num_layers=1,         # Single layer
                bidirectional=True,   # Enable bidirectionality
                batch_first=False     # Matches MATLAB's default time-major format
            )
        printnninfo(self.bilstm,"bilstm")
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16, 4)
        printnninfo(self.fc1,"fc1")  
        self.fc2 = nn.Linear(4, 2)
        printnninfo(self.fc1,"fc2")
        self.fc3 = nn.Linear(2, 1)
        printnninfo(self.fc1,"fc3")

    def forward(self, x):
        # x shape: (batch_size, 1, 1024)
        printnnshape(x,"input")
        x = self.conv1(x)
        printnnshape(x,"conv1")
        
        x = x.permute(0, 2, 1)
        x = self.layer_norm1(x)  # LayerNorm expects (batch_size, seq_len, num_features)
        x = x.permute(0, 2, 1)
        printnnshape(x,"layer_norm1")
        
        x = self.dropout1(x)
        printnnshape(x,"dropout1")
        
        
        x = self.conv2(x)
        printnnshape(x,"conv2")
        
        x = self.relu2(x)
        
        x = self.batch_norm2(x)
        
        x = self.dropout2(x)
        

        x = self.maxpool1(x)
        printnnshape(x,"maxpool1")
        
        
        x = self.conv3(x)
        printnnshape(x,"conv3")
        
        x = self.relu3(x)
        
        x = self.batch_norm3(x)
        

        x = self.maxpool2(x)
        printnnshape(x,"maxpool2")
        
        
        x = self.transposed_conv1(x)
        printnnshape(x,"transposed_conv1")
        
        x = self.relu4(x)
        
        
        x = self.transposed_conv2(x)
        printnnshape(x,"transposed_conv2")
        

        x = x.permute(0, 2, 1)  # LSTM expects (batch_size, seq_len, features)
        x, _ = self.bilstm(x)
        printnnshape(x,"bilstm")
        # x = x.permute(0, 2, 1)
        
        
        # Fully Connected Layers
        x = self.fc1(x)  # Take the output of the last time step
        printnnshape(x,"fc1")
        
        x = self.fc2(x)
        printnnshape(x,"fc2")
        
        x = self.fc3(x)
        printnnshape(x,"fc3")
        
        x = x.permute(0, 2, 1)
        printnnshape(x,"last")
        

        return x

class MODWTLayer(nn.Module):
    def __init__(self, wavelet="sym4", level=5, include_lowpass=True,selected_Levels=[3,4,5],aggregate_levels=False, boundary="periodic", name="MODWT"):
        super(MODWTLayer, self).__init__()

        self.wavelet = wavelet
        self.level = level
        self.include_lowpass = include_lowpass
        self.aggregate_levels = aggregate_levels
        self.boundary = boundary
        self.selected_Levels = selected_Levels
        self.name = name

        # Get wavelet filters
        self.lowpass_filter, self.highpass_filter = self.get_wavelet_filters(wavelet)
        self.min_length = 2 ** self.level

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T), where B is batch size, C is channels, T is the time dimension.
        Returns: Tensor of shape (B, C, S, T), where S is the spatial (level) dimension.
        """
        if x.size(-1) < self.min_length:
            raise ValueError(f"Input length must be at least {self.min_length} for level {self.level}.")

        # Apply MODWT
        coeffs = self._modwt(x)

        if self.include_lowpass:
            coeffs.append(coeffs.pop(-1))  # Move smooth coefficients to the end

        # Stack coefficients along the spatial dimension
        # result = torch.stack(coeffs, dim=2)  # (B, C, S, T)
        result = torch.stack([coeffs[i] for i in self.selected_Levels], dim=2)
        if self.aggregate_levels:
            result = result.sum(dim=2, keepdim=True)  # Aggregate levels

        return result

    def _modwt(self, x):
        """Compute MODWT for the input tensor."""
        coeffs = []
        current_signal = x

        for level in range(self.level):
            lowpass_output = self._convolve(current_signal, self.lowpass_filter, self.boundary)
            highpass_output = self._convolve(current_signal, self.highpass_filter, self.boundary)

            # Trim outputs to match input size
            min_len = min(lowpass_output.size(-1), x.size(-1))
            lowpass_output = lowpass_output[..., :min_len]
            highpass_output = highpass_output[..., :min_len]

            coeffs.append(highpass_output)  # Add detail coefficients
            current_signal = lowpass_output  # Continue with lowpass for next level

        # Trim final smooth coefficients to match input size
        min_len = min(current_signal.size(-1), x.size(-1))
        current_signal = current_signal[..., :min_len]

        coeffs.append(current_signal)  # Add final smooth coefficients
        return coeffs


    def _convolve(self, x, filter_coeffs, boundary):
        """Apply convolution with the specified boundary handling."""
        if boundary == "periodic":
            padding = len(filter_coeffs) // 2
            x = torch.nn.functional.pad(x, (padding, padding), mode="circular")
        elif boundary == "reflection":
            padding = len(filter_coeffs) // 2
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
        else:
            raise ValueError(f"Unsupported boundary type: {boundary}")

        filter_tensor = torch.tensor(filter_coeffs, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.conv1d(x, filter_tensor, groups=x.size(1))

    def get_wavelet_filters(self, wavelet_name):
        """
        Returns the low-pass (dec_lo) and high-pass (dec_hi) decomposition filters
        for the specified wavelet without using PyWavelets.
        
        Args:
            wavelet_name (str): Name of the wavelet, such as:
                - "haar" (equivalent to "db1")
                - "db2", "db3", ... (Daubechies)
                - "sym2", "sym3", "sym4", ... (Symlets)

        Returns:
            tuple (dec_lo, dec_hi): low-pass and high-pass decomposition filters as lists.
        """
        wavelets = {
            "haar": {
                "dec_lo": [0.7071067811865476, 0.7071067811865476],
                "dec_hi": [-0.7071067811865476, 0.7071067811865476],
            },
            "db1": {
                "dec_lo": [0.7071067811865476, 0.7071067811865476],
                "dec_hi": [-0.7071067811865476, 0.7071067811865476],
            },
            "db2": {
                "dec_lo": [
                    -0.12940952255092145, 0.22414386804185735,
                    0.8365163037378079, 0.48296291314469025
                ],
                "dec_hi": [
                    -0.48296291314469025, 0.8365163037378079,
                    -0.22414386804185735, -0.12940952255092145
                ],
            },
            "db3": {
                "dec_lo": [
                    0.035226291885709536, -0.08544127388224149,
                    -0.13501102001025458, 0.4598775021193313,
                    0.8068915093133388, 0.3326705529509569
                ],
                "dec_hi": [
                    -0.3326705529509569, 0.8068915093133388,
                    -0.4598775021193313, -0.13501102001025458,
                    0.08544127388224149, 0.035226291885709536
                ],
            },
            "db4": {
                "dec_lo": [
                    -0.0105974017850021, 0.032883011666982945,
                    0.030841381835986965, -0.18703481171888114,
                    -0.02798376941698385, 0.6308807679295904,
                    0.7148465705529154, 0.23037781330885523
                ],
                "dec_hi": [
                    -0.23037781330885523, 0.7148465705529154,
                    -0.6308807679295904, -0.02798376941698385,
                    0.18703481171888114, 0.030841381835986965,
                    -0.032883011666982945, -0.0105974017850021
                ],
            },
            "sym2": {
                "dec_lo": [
                    -0.4829629131445341, 0.8365163037378077,
                    0.22414386804185735, -0.12940952255126034
                ],
                "dec_hi": [
                    0.12940952255126034, 0.22414386804185735,
                    -0.8365163037378077, -0.4829629131445341
                ],
            },
            "sym3": {
                "dec_lo": [
                    0.019538882735286728, -0.021101834024758855,
                    -0.17532808990845047, 0.01660210576452232,
                    0.6339789634582119, 0.7234076904024206,
                    0.1993975339773936, -0.039134249302383094
                ],
                "dec_hi": [
                    0.039134249302383094, 0.1993975339773936,
                    -0.7234076904024206, 0.6339789634582119,
                    -0.01660210576452232, -0.17532808990845047,
                    0.021101834024758855, 0.019538882735286728
                ],
            },
            "sym4": {
                "dec_lo": [
                    -0.07576571478927333, -0.02963552764599851,
                    0.49761866763201545, 0.8037387518059161,
                    0.29785779560527736, -0.09921954357684722,
                    -0.012603967262037833, 0.032223100604071306
                ],
                "dec_hi": [
                    -0.032223100604071306, -0.012603967262037833,
                    0.09921954357684722, 0.29785779560527736,
                    -0.8037387518059161, 0.49761866763201545,
                    0.02963552764599851, -0.07576571478927333
                ],
            }
        }

        # Check if requested wavelet is available
        if wavelet_name not in wavelets:
            raise ValueError(f"Wavelet '{wavelet_name}' is not supported in the current dictionary.")

        dec_lo = wavelets[wavelet_name]["dec_lo"]
        dec_hi = wavelets[wavelet_name]["dec_hi"]
        return dec_lo, dec_hi

class Layer2_HumanHM(nn.Module):
    def __init__(self):
        super(Layer2_HumanHM, self).__init__()
        
        # Sequence Input Layer: Assuming input dimension = (batch_size, 1, 1024)
        
        self.conv1 = nn.Conv1d(1, 3, kernel_size=4, padding='same')
        self.layer_norm1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(3, 8, kernel_size=64, stride=8, padding=28)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(8)
        self.dropout2 = nn.Dropout(0.2)
        
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1,padding=1, dilation=2)
        
        self.conv3 = nn.Conv1d(8, 8, kernel_size=32, stride=4, padding=14)
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(8)
        
        # MaxPooling1d (kernel size 2, padding same)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1,padding=1, dilation=2)
        
        self.transposed_conv1 = nn.ConvTranspose1d(8, 8, kernel_size=32, stride=4,padding=14)
        self.relu4 = nn.ReLU()
        
        self.transposed_conv2 = nn.ConvTranspose1d(8, 8, kernel_size=64, stride=8, padding=28)
        
        # BiLSTM Layer (8 units)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16, 4)  
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, 1024)
        
        x = self.conv1(x)
        
        x = self.layer_norm1(x)  # LayerNorm expects (batch_size, seq_len, num_features)
        
        x = self.dropout1(x)
        
        
        x = self.conv2(x)
        
        x = self.relu2(x)
        
        x = self.batch_norm2(x)
        
        x = self.dropout2(x)
        

        x = self.maxpool1(x)
        
        
        x = self.conv3(x)
        
        x = self.relu3(x)
        
        x = self.batch_norm3(x)
        

        x = self.maxpool2(x)
        
        
        x = self.transposed_conv1(x)
        
        x = self.relu4(x)
        
        
        x = self.transposed_conv2(x)
        

        x = x.permute(0, 2, 1)  # LSTM expects (batch_size, seq_len, features)
        x, _ = self.bilstm(x)
        # x = x.permute(0, 2, 1)
        
        
        # Fully Connected Layers
        x = self.fc1(x)  # Take the output of the last time step
        
        x = self.fc2(x)
        
        x = self.fc3(x)
        
        x = x.permute(0, 2, 1)
        

        return x

# ----------------------------
# RadarMisoCNNApp GUI Class
# ----------------------------
class RadarHumanHMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Health Monitoring Using Continuous Wave Radar and Deep Learning")
        self.default_folder = os.path.join(ssp.config.temp_folder, "datasets", "SynchronizedRadarECGData")
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
        zip_path = "SynchronizedRadarECGData.zip"
        url = "https://ssd.mathworks.com/supportfiles/SPT/data/SynchronizedRadarECGData.zip"
        zip_folder = self.default_folder

        # Check if the folder exists
        if not os.path.exists(zip_folder):

            # Download the ZIP file
            print("Downloading the ZIP file...")
            response = requests.get(url, stream=True)
            datasets_path = os.path.join(ssp.config.temp_folder, "datasets")
            if not os.path.exists(datasets_path):
                os.makedirs(datasets_path)
            zip_path = os.path.join(ssp.config.temp_folder, "datasets", "SynchronizedRadarECGData.zip")
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
            
        dataset_folder = zip_folder
        trainVal_radar_dir = os.path.join(dataset_folder, "trainVal", "radar")
        trainVal_ecg_dir = os.path.join(dataset_folder, "trainVal", "ecg")
        test_radar_dir = os.path.join(dataset_folder, "test", "radar")
        test_ecg_dir = os.path.join(dataset_folder, "test", "ecg")

        self.trainVal_dataset = RadarECGDataset(trainVal_radar_dir, trainVal_ecg_dir, transform_ecg=True)
        test_dataset = RadarECGDataset(test_radar_dir, test_ecg_dir, transform_ecg=True)
        # print(trainVal_dataset[0][0].shape,trainVal_dataset[0][1].shape)
        # torch.Size([1, 1024]) torch.Size([1, 1024])
        train_ratio = self.split_train_input.value()
        train_size = int(train_ratio * len(self.trainVal_dataset))
        val_size = len(self.trainVal_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.trainVal_dataset, [train_size, val_size])
        
        batch_size = self.batch_size_input.value() #*0+1
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # clutter_removal = self.clutter_removal_checkbox.isChecked() 
        # self.dataset = RadarGestureDataset(data_folder=self.default_folder, clutter_removal=clutter_removal,max_folder_number=self.folderN_input.value(),PercentBar=True)
        
        
        # self.combobox.clear()
        # self.combobox.addItems(self.dataset.gestureVocabulary)

        
        # total_samples = len(self.dataset)  # total number of gesture samples
        # train_ratio = self.split_train_input.value()
        # val_ratio = self.split_val_input.value()

        # train_size = int(train_ratio * total_samples)
        # val_size = int(val_ratio * total_samples)
        # test_size = total_samples - train_size - val_size

        # train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        # batch_size = self.batch_size_input.value() 
        # self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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
        self.status_label.setText(f"Status: Dataset is loaded {len(self.train_dataset)}")
        
    def def_model(self):
        # self.model = MultiInputModel(num_classes=self.num_classes)
        1
        
    def train_model(self):
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
            
        self.criterion = nn.MSELoss()
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
        
        running_loss = []
            
        for epoch in range(num_epochs):
            val_loader_iter = iter(self.val_loader)
        # ---- Training ----
            self.model.train()
            for inputs, labels in self.train_loader:
                # print(inputs.mean(),labels.mean())
                inputs = inputs.to(self.device)  # shape [B,3,90,189]
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                # printnnshape(inputs)
                outputs = self.model(inputs)  # shape [B, num_classes]
                # print((outputs-labels).mean())
                
                # loss1 = self.criterion(outputs, labels)
                mse = F.mse_loss(outputs, labels, reduction='none')  # Compute element-wise MSE
                loss = torch.sum(mse,dim=2)
                loss = torch.mean(loss)
                 
                loss.backward()
                self.optimizer.step()

                running_loss0 = loss.item()
                running_loss.append(running_loss0)
                print(epoch,running_loss0)
                # Validation_count+=1
                # if Validation_count==1 or Validation_count>Validation_frequency:
                #     Validation_count=1
                #     self.model.eval()
                #     with torch.no_grad():
                #         inputs, labels = next(val_loader_iter)
                #         inputs = inputs.to(self.device)
                #         labels = labels.to(self.device)

                #         outputs = self.model(inputs)
                #         loss = self.criterion(outputs, labels)

                #         _, preds = torch.max(outputs, 1)
                        
                #         matlab_acc = 100.0 * torch.sum(preds == labels).item() / labels.size(0) 
                #         matlab_loss = loss.item()
                #         matlab_acc_v.append(matlab_acc)
                #         matlab_loss_v.append(matlab_loss)
                #         matlab_v_index.append(len(matlab_acc_t))
                #         if len(matlab_loss_v)==1:
                #             best_loss=matlab_loss+1


                #         if matlab_loss < best_loss:
                #             best_loss = matlab_loss
                #             torch.save(self.model.state_dict(), self.savebestmodelpath)
                #     self.model.train()
                
                
                ax[0].clear()
                ax[0].plot(running_loss)
                # ax[0].plot(matlab_v_index,matlab_acc_v,'--o')
                # ax[0].set_title(f"Epoch {epoch + 1}")
                # ax[0].set_xlabel("Iteration")
                # ax[0].set_ylabel("Accuracy")
                # ax[0].grid(True) 
                # ax[1].clear()
                # ax[1].plot(matlab_t_index,matlab_loss_t)
                # ax[1].plot(matlab_v_index,matlab_loss_v,'--o')
                # ax[1].set_xlabel("Iteration")
                # ax[1].set_ylabel("Loss")
                # ax[1].grid(True) 
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
def runradarConvAEBiLSTMapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    appSTYLESHEET = ssp.config.appSTYLESHEET
    app.setStyleSheet(appSTYLESHEET)  # Replace with your desired stylesheet if any
    
    window = RadarHumanHMApp()
    window.show()
    app.exec_()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    runradarConvAEBiLSTMapp()

# def runradarConvAEBiLSTMapp():

#     import os
#     import glob
#     import requests
#     import zipfile
#     import numpy as np
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     from torch.utils.data import Dataset, DataLoader, random_split
#     from scipy.io import loadmat
#     import matplotlib.pyplot as plt
#     import sensingsp as ssp
#     import torch.nn.functional as F

# # trainCats: 830×1 categorical

# #      GDN0001_Resting        59 
# #      GDN0001_Valsalva       97 
# #      GDN0002_Resting        60 
# #      GDN0002_Valsalva       97 
# #      GDN0003_Resting        58 
# #      GDN0003_Valsalva      103 
# #      GDN0004_Apnea          14 
# #      GDN0004_Resting        58 
# #      GDN0004_Valsalva      106 
# #      GDN0005_Apnea          14 
# #      GDN0005_Resting        59 
# #      GDN0005_Valsalva      105 
# #      <undefined>             0 
# # testCats: 200×1 categorical

# #      GDN0006_Apnea          14 
# #      GDN0006_Resting        59 
# #      GDN0006_Valsalva      127 
# #      <undefined>             0 
#     if 0:
#         # Example usage equivalent to MATLAB
#         # Create MODWT layer with specified parameters
#         layer = MODWTLayer(wavelet="sym4", level=5, include_lowpass=False, aggregate_levels=False)

#         # Create a 1x1x64 input tensor: 1 channel, batch size 1, length 64
#         x = torch.randn(1, 1, 64)

#         # Run the forward pass through the MODWT layer
#         output = layer(x)

#         # Select levels 3 to 5 from the output
#         selected_levels = [2, 3, 4]  # Python is zero-indexed, so level 3 maps to index 2
#         output_selected = output[:, :, selected_levels, :]

#         print("Output shape:", output_selected.shape)
#         # Output shape: torch.Size([1, 1, 3, 64])


#     # Download and Extract the Data
#     # ==============================
#     zip_path = "SynchronizedRadarECGData.zip"
#     url = "https://ssd.mathworks.com/supportfiles/SPT/data/SynchronizedRadarECGData.zip"
#     zip_folder = os.path.join(ssp.config.temp_folder, "datasets", "SynchronizedRadarECGData")

#     # Check if the folder exists
#     if not os.path.exists(zip_folder):

#         # Download the ZIP file
#         print("Downloading the ZIP file...")
#         response = requests.get(url, stream=True)
#         datasets_path = os.path.join(ssp.config.temp_folder, "datasets")
#         if not os.path.exists(datasets_path):
#             os.makedirs(datasets_path)
#         zip_path = os.path.join(ssp.config.temp_folder, "datasets", "SynchronizedRadarECGData.zip")

#         # Save the ZIP file locally
#         with open(zip_path, "wb") as f:
#             f.write(response.content)
#         print("Download complete!")

#         # Extract the ZIP file
#         print("Extracting the ZIP file...")
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(zip_folder)
#         print(f"Data extracted to {zip_folder}")

#         # Clean up: Remove the ZIP file
#         os.remove(zip_path)
#         print("Temporary ZIP file removed.")

#     else:
#         print(f"The folder {zip_folder} already exists. No action taken.")
    
#     # ==============================
#     # Device Setup
#     # ==============================
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     # ==============================
#     # Normalization and Dataset
#     # ==============================
#     def normalize_ecg(ecg_signal):
#         ecg_signal = ecg_signal - np.median(ecg_signal)
#         max_val = np.max(np.abs(ecg_signal))
#         if max_val < 1e-12:
#             max_val = 1.0
#         ecg_signal = ecg_signal / max_val
#         return ecg_signal

#     class RadarECGDataset(Dataset):
#         def __init__(self, radar_dir, ecg_dir, transform_ecg=True):
#             self.radar_files = sorted(glob.glob(os.path.join(radar_dir, '*.mat')))
#             self.ecg_files = sorted(glob.glob(os.path.join(ecg_dir, '*.mat')))
#             assert len(self.radar_files) == len(self.ecg_files), "Mismatch in number of radar and ECG files."
#             self.transform_ecg = transform_ecg

#         def __len__(self):
#             return len(self.radar_files)

#         def __getitem__(self, idx):
#             radar_mat = loadmat(self.radar_files[idx])
#             ecg_mat = loadmat(self.ecg_files[idx])

#             # Adjust the key if necessary. Assuming 'signal' is correct.
#             radar_signal = radar_mat['radar_l'].squeeze() # (1024,)
#             ecg_signal = ecg_mat['ecg_l'].squeeze() # (1024,)

#             if self.transform_ecg:
#                 ecg_signal = normalize_ecg(ecg_signal)

#             radar_signal = torch.tensor(radar_signal, dtype=torch.float32).unsqueeze(0)  # (1, 1024)
#             ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)      # (1, 1024)

#             return radar_signal, ecg_signal

#     # ==============================
#     # Create Datasets and Loaders
#     # ==============================
#     dataset_folder = zip_folder
#     trainVal_radar_dir = os.path.join(dataset_folder, "trainVal", "radar")
#     trainVal_ecg_dir = os.path.join(dataset_folder, "trainVal", "ecg")
#     test_radar_dir = os.path.join(dataset_folder, "test", "radar")
#     test_ecg_dir = os.path.join(dataset_folder, "test", "ecg")

#     trainVal_dataset = RadarECGDataset(trainVal_radar_dir, trainVal_ecg_dir, transform_ecg=True)
#     test_dataset = RadarECGDataset(test_radar_dir, test_ecg_dir, transform_ecg=True)
#     # print(trainVal_dataset[0][0].shape,trainVal_dataset[0][1].shape)
#     # torch.Size([1, 1024]) torch.Size([1, 1024])
#     train_size = int(0.85 * len(trainVal_dataset))
#     val_size = len(trainVal_dataset) - train_size
#     train_dataset, val_dataset = random_split(trainVal_dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     # ==============================
#     # Model Definition
#     # ==============================
#     # class ECGReconstructionModel(nn.Module):
#     #     def __init__(self):
#     #         super(ECGReconstructionModel, self).__init__()
            
#     #         self.modwt_layer = MODWTLayer(wavelet="sym4", level=5, include_lowpass=False)
#     #         self.layer_norm = nn.LayerNorm(8, eps=1e-5)
#     #         self.dropout1 = nn.Dropout(0.2)
#     #         self.conv1d_1 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32)
#     #         self.batchnorm1 = nn.BatchNorm1d(num_features=8, eps=1e-5)
#     #         self.maxpool1d_1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

#     #         self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
#     #         self.batchnorm2 = nn.BatchNorm1d(num_features=8, eps=1e-5)
#     #         self.maxpool1d_2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

#     #         self.transposed_conv1d_1 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
#     #         self.transposed_conv1d_2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32)

#     #         self.bilstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, bidirectional=True, batch_first=True)

#     #         self.fc1 = nn.Linear(16, 4)  # Bidirectional LSTM doubles the feature size
#     #         self.fc2 = nn.Linear(4, 2)
#     #         self.fc3 = nn.Linear(2, 1)

#     #     def forward(self, x):
#     #         x = self.modwt_layer(x)
#     #         x = x.permute(0, 2, 1) 
#     #         x = x.flatten(start_dim=2)  # Equivalent to TimeDistributed(Flatten())
#     #         x = self.layer_norm(x)
#     #         x = self.dropout1(x)

#     #         x = x.transpose(1, 2)  
#     #         x = self.conv1d_1(x)
#     #         x = F.relu(x)
#     #         x = self.batchnorm1(x)
#     #         x = self.maxpool1d_1(x)

#     #         x = self.conv1d_2(x)
#     #         x = F.relu(x)
#     #         x = self.batchnorm2(x)
#     #         x = self.maxpool1d_2(x)

#     #         x = self.transposed_conv1d_1(x)
#     #         x = F.relu(x)
#     #         x = self.transposed_conv1d_2(x)

#     #         x = x.transpose(1, 2)  # Back to (B, T, C) for LSTM
#     #         x, _ = self.bilstm(x)

#     #         x = self.fc1(x)
#     #         x = self.fc2(x)
#     #         x = self.fc3(x)
#     #         return x

#     # class MODWTLayer(nn.Module):
#     #     def __init__(self):
#     #         super(MODWTLayer, self).__init__()
#     #         # Placeholder for the actual MODWT operation.
#     #         # In practice, implement the wavelet decomposition here.
#     #         pass
        
#     #     def forward(self, x):
#     #         # x: (batch, time, 1)
#     #         batch, time, features = x.shape
#     #         # According to the given shape from the explanation: output (batch, time, 3, 8)
#     #         # We'll just create a dummy output for demonstration.
#     #         # In practice, this would be replaced by the actual MODWT computation.
#     #         # For example: expand to (batch, time, 3, 8)
#     #         out = x.new_zeros(batch, time, 3, 8)
#     #         return out

#     class ECGReconstructionModel(nn.Module):
#         def __init__(self):
#             super(ECGReconstructionModel, self).__init__()
#             self.modwt = MODWTLayer(wavelet="sym4", level=5, include_lowpass=False)
            
#             # After modwt: (batch, 1024, 3, 8)
#             # Permute last two dims: (batch, 1024, 8, 3)
#             # Flatten last two dims: (batch, 1024, 24)
            
#             # LayerNorm over the feature dimension
#             # normalized_shape should be the size of the last dimension after flatten = 24
#             self.layernorm = nn.LayerNorm(24, eps=1e-5)
            
#             self.dropout1 = nn.Dropout(0.2)
            
#             # Conv1D in PyTorch: input shape (batch, channels, time)
#             # Currently we have (batch, 1024, 24). We'll transpose to (batch, 24, 1024).
#             # Conv1D with stride=8 and same padding -> output time = 128
#             # input_channels=24, out_channels=8
#             self.conv1 = nn.Conv1d(in_channels=24, out_channels=8, kernel_size=64, stride=8, padding=32) 
#             self.bn1 = nn.BatchNorm1d(8)
#             self.dropout2 = nn.Dropout(0.2)
            
#             # MaxPool1D(pool_size=2, stride=1, padding='same')
#             # In PyTorch, 'same' padding is available in newer versions, or we can manually set.
#             # To mimic Keras 'same' pooling with kernel=2, stride=1: 
#             # We can use padding=0 because maxpool with stride=1 and kernel=2 reduces length by 1 unless we pad.
#             # However, Keras 'same' ensures the time dimension remains the same.
#             # There's no native 'same' for MaxPool in PyTorch < 1.9. We can approximate by padding the input.
#             # For simplicity, assume PyTorch 1.9+ which supports 'same' padding in pooling:
#             self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1) 
            
#             # Another Conv1D with stride=4, same padding
#             # input_channels=8, out_channels=8
#             # input time=128, output time=128/4=32
#             self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
#             self.bn2 = nn.BatchNorm1d(8)
            
#             self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            
#             # ConvTranspose1D with stride=4 to upsample from 32 -> 128
#             self.convT1 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16, output_padding=0)
            
#             # ConvTranspose1D with stride=8 to upsample from 128 -> 1024
#             self.convT2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32, output_padding=0)
            
#             # BiLSTM: input_dim=8, hidden_dim=8, bidirectional → output_dim=16
#             # PyTorch LSTM expects (batch, time, features), which matches our final format after conv-transpose ops if we permute back.
#             self.biLSTM = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True)
            
#             # Dense layers: fc_1: 16 -> 4, fc_2: 4 -> 2, fc_3: 2 -> 1
#             self.fc_1 = nn.Linear(16, 4)
#             self.fc_2 = nn.Linear(4, 2)
#             self.fc_3 = nn.Linear(2, 1)

#         def forward(self, x):
#             # x: (batch, time, 1) e.g. (batch, 1024, 1)
            
#             # modwtLayer
#             x = self.modwt(x)  # (batch, 1024, 3, 8)
            
#             # Permute last two dims: (batch, time, features1, features2) -> (batch, 1024, 8, 3)
#             x = x.permute(0, 1, 3, 2)  # swapping the last two dims (3,8) -> (8,3)
            
#             # Flatten last two dims: (batch, 1024, 8, 3) -> (batch, 1024, 24)
#             # Flatten only the last two dims:
#             batch, time, d1, d2 = x.shape
#             x = x.reshape(batch, time, d1 * d2)  # (batch, 1024, 24)
            
#             # LayerNorm
#             x = self.layernorm(x)  # (batch, 1024, 24)
            
#             # Dropout #1
#             x = self.dropout1(x)  # (batch, 1024, 24)
            
#             # Conv1D #1 expects (batch, channels, time)
#             # currently (batch, time, features) = (batch, 1024, 24)
#             x = x.permute(0, 2, 1)  # (batch, 24, 1024)
#             x = self.conv1(x)       # (batch, 8, 128) after stride=8
#             x = F.relu(x)
#             x = self.bn1(x)
#             x = self.dropout2(x)
            
#             # MaxPool1D (same)
#             x = self.maxpool1(x)    # (batch, 8, 128)
            
#             # Conv1D #2
#             x = self.conv2(x)       # (batch, 8, 32) after stride=4
#             x = F.relu(x)
#             x = self.bn2(x)
            
#             # MaxPool1D #2 (same)
#             x = self.maxpool2(x)    # (batch, 8, 32)
            
#             # ConvTranspose1D #1: upsample from 32 -> 128
#             x = self.convT1(x)      # (batch, 8, 128)
#             x = F.relu(x)
            
#             # ConvTranspose1D #2: upsample from 128 -> 1024
#             x = self.convT2(x)      # (batch, 8, 1024)
            
#             # Before LSTM, permute to (batch, time, features)
#             x = x.permute(0, 2, 1)  # (batch, 1024, 8)
            
#             # Bidirectional LSTM
#             # output: (batch, time, hidden*2) = (batch, 1024, 16)
#             x, _ = self.biLSTM(x)
            
#             # Dense layers
#             x = self.fc_1(x)  # (batch, 1024, 4)
#             x = self.fc_2(x)  # (batch, 1024, 2)
#             x = self.fc_3(x)  # (batch, 1024, 1)
            
#             return x

    

#     # ==============================
#     # Training Setup
#     # ==============================
#     model = Layer1_HumanHM().to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     max_epochs = 5

#     # ==============================
#     # Training Loop
#     # ==============================
#     for epoch in range(max_epochs):
#         model.train()
#         train_loss = 0.0
#         for radar_batch, ecg_batch in train_loader:
#             radar_batch = radar_batch.to(device)   #torch.Size([64, 1, 1024])
#             ecg_batch = ecg_batch.to(device)       #torch.Size([64, 1, 1024])

#             optimizer.zero_grad()
#             outputs = model(radar_batch)
#             loss = criterion(outputs, ecg_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * radar_batch.size(0)
#         train_loss /= len(train_loader.dataset)

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for radar_batch, ecg_batch in val_loader:
#                 radar_batch = radar_batch.to(device)
#                 ecg_batch = ecg_batch.to(device)
#                 outputs = model(radar_batch)
#                 loss = criterion(outputs, ecg_batch)
#                 val_loss += loss.item() * radar_batch.size(0)
#         val_loss /= len(val_loader.dataset)

#         print(f"Epoch [{epoch+1}/{max_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     # ==============================
#     # Testing
#     # ==============================
#     model.eval()
#     test_losses = []
#     with torch.no_grad():
#         for radar_batch, ecg_batch in test_loader:
#             radar_batch = radar_batch.to(device)
#             ecg_batch = ecg_batch.to(device)
#             outputs = model(radar_batch)
#             loss = criterion(outputs, ecg_batch)
#             test_losses.append(loss.item())

#     test_loss_mean = np.mean(test_losses)
#     print("Test Loss:", test_loss_mean)

#     # ==============================
#     # Visualization
#     # ==============================
#     model.eval()
#     with torch.no_grad():
#         radar_batch, ecg_batch = next(iter(test_loader))
#         radar_batch = radar_batch.to(device)
#         outputs = model(radar_batch)
#         outputs = outputs.squeeze().cpu().numpy()
#         ecg_ref = ecg_batch.squeeze().numpy()

#     plt.figure(figsize=(10,4))
#     plt.plot(ecg_ref, label='Measured ECG')
#     plt.plot(outputs, label='Reconstructed ECG')
#     plt.title("ECG Reconstruction Sample")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# # ----------------------------
# # Main Execution
# # ----------------------------
# if __name__ == "__main__":
#     runradarConvAEBiLSTMapp()







# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # # Define the custom modwtLayer
# # class ModwtLayer(nn.Module):
# #     def __init__(self, weights_shape):
# #         super(ModwtLayer, self).__init__()
# #         self.weights = nn.Parameter(torch.zeros(weights_shape), requires_grad=True)

# #     def forward(self, x):
# #         # Implementing a simple forward pass using weights
# #         batch_size, seq_len, channels = x.shape
# #         weights = self.weights.unsqueeze(0).repeat(batch_size, 1, 1)  # Expand weights for batch processing
# #         output = torch.matmul(x, weights)  # Example operation (modify as needed)
# #         return output

# # # Define the PyTorch model
# # class MyModel(nn.Module):
# #     def __init__(self):
# #         super(MyModel, self).__init__()

# #         self.modwt_layer = ModwtLayer((1, 8))
# #         self.layer_norm = nn.LayerNorm(8, eps=1e-5)
# #         self.dropout1 = nn.Dropout(0.2)
# #         self.conv1d_1 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32)
# #         self.batchnorm1 = nn.BatchNorm1d(num_features=8, eps=1e-5)
# #         self.maxpool1d_1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

# #         self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
# #         self.batchnorm2 = nn.BatchNorm1d(num_features=8, eps=1e-5)
# #         self.maxpool1d_2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

# #         self.transposed_conv1d_1 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
# #         self.transposed_conv1d_2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32)

# #         self.bilstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, bidirectional=True, batch_first=True)

# #         self.fc1 = nn.Linear(16, 4)  # Bidirectional LSTM doubles the feature size
# #         self.fc2 = nn.Linear(4, 2)
# #         self.fc3 = nn.Linear(2, 1)

# #     def forward(self, x):
# #         x = self.modwt_layer(x)
# #         x = x.permute(0, 2, 1)  # Equivalent to TimeDistributed(Permute(2,1))
# #         x = x.flatten(start_dim=2)  # Equivalent to TimeDistributed(Flatten())
# #         x = self.layer_norm(x)
# #         x = self.dropout1(x)

# #         x = x.transpose(1, 2)  # Switch to (B, C, T) for Conv1d
# #         x = self.conv1d_1(x)
# #         x = F.relu(x)
# #         x = self.batchnorm1(x)
# #         x = self.maxpool1d_1(x)

# #         x = self.conv1d_2(x)
# #         x = F.relu(x)
# #         x = self.batchnorm2(x)
# #         x = self.maxpool1d_2(x)

# #         x = self.transposed_conv1d_1(x)
# #         x = F.relu(x)
# #         x = self.transposed_conv1d_2(x)

# #         x = x.transpose(1, 2)  # Back to (B, T, C) for LSTM
# #         x, _ = self.bilstm(x)

# #         x = self.fc1(x)
# #         x = self.fc2(x)
# #         x = self.fc3(x)
# #         return x

# # # Instantiate the model
# # model = MyModel()
# # print(model)



# # import tensorflow as tf
# # from tensorflow import keras
# # from tensorflow.keras import layers
# # from myModel.customLayers.modwtLayer import modwtLayer

# # def create_model():
# #     sequenceinput = keras.Input(shape=(None,1))
# #     layer = modwtLayer(name="layer_", Weights_Shape_=(2,8))(sequenceinput)
# #     layerperm = layers.TimeDistributed(layers.Permute((2,1)))(layer)
# #     flatten = layers.TimeDistributed(layers.Flatten())(layerperm)
# #     layernorm = layers.LayerNormalization(axis=-1, epsilon=0.000010, name="layernorm_")(flatten)
# #     dropout_1 = layers.Dropout(0.200000)(layernorm)
# #     conv1d_1 = layers.Conv1D(8, 64, strides=(8), padding="same", name="conv1d_1_")(dropout_1)
# #     relu_1 = layers.ReLU()(conv1d_1)
# #     batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(relu_1)
# #     dropout_2 = layers.Dropout(0.200000)(batchnorm_1)
# #     maxpool1d_1 = layers.MaxPool1D(pool_size=2, strides=1, padding="same")(dropout_2)
# #     conv1d_2 = layers.Conv1D(8, 32, strides=(4), padding="same", name="conv1d_2_")(maxpool1d_1)
# #     relu_2 = layers.ReLU()(conv1d_2)
# #     batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(relu_2)
# #     maxpool1d_2 = layers.MaxPool1D(pool_size=2, strides=1, padding="same")(batchnorm_2)
# #     transposed_conv1d_1 = layers.Conv1DTranspose(8, 32, strides=4, padding="same", name="transposed_conv1d_1_")(maxpool1d_2)
# #     relu_3 = layers.ReLU()(transposed_conv1d_1)
# #     transposed_conv1d_2 = layers.Conv1DTranspose(8, 64, strides=8, padding="same", name="transposed_conv1d_2_")(relu_3)
# #     transposed_conv1d_2_bilstm_input = transposed_conv1d_2
# #     biLSTM = layers.Bidirectional(layers.LSTM(8, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=False), name="biLSTM_")(transposed_conv1d_2_bilstm_input)
# #     fc_1 = layers.Dense(4, name="fc_1_")(biLSTM)
# #     fc_2 = layers.Dense(2, name="fc_2_")(fc_1)
# #     fc_3 = layers.Dense(1, name="fc_3_")(fc_2)

# #     model = keras.Model(inputs=[sequenceinput], outputs=[fc_3])
# #     return model