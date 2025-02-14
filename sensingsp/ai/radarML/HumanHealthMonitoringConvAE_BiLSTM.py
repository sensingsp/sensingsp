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
from scipy.signal import convolve
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

def load_sample(matfile,matfileecg):
    radar_mat = loadmat(matfile)
    ecg_mat = loadmat(matfileecg)
    radar_signal = radar_mat['radar_l'].squeeze() 
    ecg_signal = ecg_mat['ecg_l'].squeeze() 

    radar_signal = torch.tensor(radar_signal, dtype=torch.float32).unsqueeze(0)  # (1, 1024)
    ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)      # (1, 1024)
    return radar_signal, ecg_signal

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

class MODWTLayer_fix(nn.Module):
    def __init__(self, wavelet='sym4', level=6, selected_levels=range(3, 5), trim_approx=False):
        """
        Modified Discrete Wavelet Transform (MODWT) as a PyTorch Layer.

        Args:
            wavelet (str): Wavelet type ('sym4' supported).
            level (int): Number of decomposition levels.
            selected_levels (iterable): Levels of detail coefficients to return.
            trim_approx (bool): Whether to trim the final approximation coefficients.
        """
        super(MODWTLayer_fix, self).__init__()

        # Define wavelet filters for sym4
        if wavelet == 'sym4':
            self.h0 = torch.tensor([-0.075765714789273, -0.029635527645998, 0.497618667632015, 
                                    0.803738751805216, 0.297857795605542, -0.099219543576847, 
                                    -0.012603967262261, 0.032223100604042], dtype=torch.float32)
            self.h1 = torch.tensor([-0.032223100604042, -0.012603967262261, 0.099219543576847, 
                                    0.297857795605542, -0.803738751805216, 0.497618667632015, 
                                    0.029635527645998, -0.075765714789273], dtype=torch.float32)
        else:
            raise ValueError(f"Wavelet {wavelet} not supported. Implement other wavelets as needed.")

        self.level = level
        self.selected_levels = list(selected_levels)
        self.trim_approx = trim_approx

    def forward(self, signal):
        """
        Perform MODWT on the input signal.

        Args:
            signal (torch.Tensor): Input signal tensor of shape (batch, channel, features).
        
        Returns:
            mra_selected (list): Detail coefficients for selected levels.
            scaling_coeffs (torch.Tensor): Approximation coefficients for the lowest frequency.
        """
        batch_size, channels, signal_length = signal.shape

        # Initialize filters and storage for coefficients
        coeffs = []
        approx = signal.clone()

        # Perform MODWT across all levels
        for lvl in range(self.level):
            # Upsample filters by 2^lvl (insert zeros between filter coefficients)
            upsampling_factor = 2 ** lvl
            lp_filter = torch.zeros((len(self.h0) * upsampling_factor,), dtype=torch.float32, device=signal.device)
            hp_filter = torch.zeros((len(self.h1) * upsampling_factor,), dtype=torch.float32, device=signal.device)
            lp_filter[::upsampling_factor] = self.h0
            hp_filter[::upsampling_factor] = self.h1

            # Apply convolutions
            detail = self._apply_convolution(approx, hp_filter)
            approx = self._apply_convolution(approx, lp_filter)

            # Append coefficients
            coeffs.append((approx.clone(), detail.clone()))

        # Optionally trim the final approximation coefficients
        if self.trim_approx:
            coeffs[-1] = (None, coeffs[-1][1])

        # Prepare output
        coeffs = coeffs[::-1]
        mra = [c[1] for c in reversed(coeffs)]  # Reversed detail coefficients

        # Select specified levels
        mra_selected = [mra[i-1] for i in self.selected_levels]
        
        o = torch.stack(mra_selected, dim=2)
        # return mra_selected, scaling_coeffs
        return o

    def _apply_convolution(self, signal, filter_tensor):
        """
        Apply convolution to the input signal with a specified filter.

        Args:
            signal (torch.Tensor): Input signal tensor of shape (batch, channel, features).
            filter_tensor (torch.Tensor): Filter tensor.
        
        Returns:
            torch.Tensor: Convolved signal.
        """
        # Reshape filter for 1D convolution
        filter_tensor = filter_tensor.view(1, 1, -1)
        
        conv_result = torch.nn.functional.conv1d(signal, filter_tensor, padding='same', groups=signal.shape[1])
        
        return conv_result


class MODWTLayer(nn.Module):
    def __init__(self, wavelet='sym4', level=6, selected_levels=range(3, 5), trim_approx=False):
        super(MODWTLayer, self).__init__()

        # Define and initialize wavelet filters for sym4
        if wavelet == 'sym4':
            h0 = [-0.075765714789273, -0.029635527645998, 0.497618667632015, 
                  0.803738751805216, 0.297857795605542, -0.099219543576847, 
                  -0.012603967262261, 0.032223100604042]
            h1 = [-0.032223100604042, -0.012603967262261, 0.099219543576847, 
                  0.297857795605542, -0.803738751805216, 0.497618667632015, 
                  0.029635527645998, -0.075765714789273]
        else:
            raise ValueError(f"Wavelet {wavelet} not supported. Implement other wavelets as needed.")

        # Register learnable parameters
        self.h0 = nn.Parameter(torch.tensor(h0, dtype=torch.float32))
        self.h1 = nn.Parameter(torch.tensor(h1, dtype=torch.float32))

        self.level = level
        self.selected_levels = list(selected_levels)
        self.trim_approx = trim_approx

    def forward(self, signal):
        """
        Perform MODWT on the input signal.

        Args:
            signal (torch.Tensor): Input signal tensor of shape (batch, channel, features).
        
        Returns:
            torch.Tensor: Stacked detail coefficients for selected levels (batch, channel, selected_levels, features).
        """
        batch_size, channels, signal_length = signal.shape

        # Initialize filters and storage for coefficients
        coeffs = []
        approx = signal.clone()

        # Perform MODWT across all levels
        for lvl in range(self.level):
            # Upsample filters by 2^lvl (insert zeros between filter coefficients)
            upsampling_factor = 2 ** lvl
            lp_filter = torch.zeros((len(self.h0) * upsampling_factor,), dtype=torch.float32, device=signal.device)
            hp_filter = torch.zeros((len(self.h1) * upsampling_factor,), dtype=torch.float32, device=signal.device)
            lp_filter[::upsampling_factor] = self.h0
            hp_filter[::upsampling_factor] = self.h1

            # Apply convolutions
            detail = self._apply_convolution(approx, hp_filter)
            approx = self._apply_convolution(approx, lp_filter)

            # Append coefficients
            coeffs.append((approx.clone(), detail.clone()))

        # Optionally trim the final approximation coefficients
        if self.trim_approx:
            coeffs[-1] = (None, coeffs[-1][1])

        # Prepare output
        coeffs = coeffs[::-1]
        mra = [c[1] for c in reversed(coeffs)]  # Reversed detail coefficients

        # Select specified levels (adjust for zero-based indexing)
        mra_selected = [mra[i - 1] for i in self.selected_levels]
        
        # Stack the selected detail coefficients along a new dimension
        o = torch.stack(mra_selected, dim=2)

        return o

    def _apply_convolution(self, signal, filter_tensor):
        """
        Apply convolution to the input signal with a specified filter.

        Args:
            signal (torch.Tensor): Input signal tensor of shape (batch, channel, features).
            filter_tensor (torch.Tensor): Filter tensor.
        
        Returns:
            torch.Tensor: Convolved signal.
        """
        # Reshape filter for 1D convolution
        filter_tensor = filter_tensor.view(1, 1, -1)
        
        # Perform convolution (padding='same' equivalent)
        conv_result = torch.nn.functional.conv1d(signal, filter_tensor, padding='same', groups=signal.shape[1])
        
        return conv_result

class Layer2_HumanHM(nn.Module):
    def __init__(self):
        super(Layer2_HumanHM, self).__init__()
        
        # Sequence Input Layer: Assuming input dimension = (batch_size, 1, 1024)
        
        self.modwt_layer = MODWTLayer(wavelet='sym4', level=5, selected_levels=range(3, 5+1))
        
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
        x = self.modwt_layer(x)
        printnnshape(x,"modwt_layer")
        x = torch.flatten(x, start_dim=1, end_dim=2)
        printnnshape(x,"flatten")
        
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
        self.train_button = QPushButton("Train")
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
        
        
        self.modwt_button = QPushButton("MODWT")
        self.modwt_button.clicked.connect(self.modwt_but)
        main_layout.addWidget(self.modwt_button)
        self.testpretrained_button = QPushButton("Test Pretrained NN")
        self.testpretrained_button.clicked.connect(self.testpretrained_but)
        main_layout.addWidget(self.testpretrained_button)
        

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
                loss = torch.mean(mse,dim=2)
                loss = torch.sum(loss)
                 
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
            
        torch.save(self.model.state_dict(), self.savebestmodelpath)
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
        # self.model.load_state_dict(torch.load(self.savebestmodelpath))
        # self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []
        fig, axs = plt.subplots(3, 1)
        ax = axs.flatten()
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                ax[0].clear()
                ax[1].clear()
                ax[2].clear()
                ax[0].plot(inputs[0,0,:].cpu().numpy())
                ax[1].plot(labels[0,0,:].cpu().numpy())
                ax[2].plot(outputs[0,0,:].cpu().numpy())
                plt.draw()
                plt.pause(0.001)
                plt.gcf().canvas.flush_events()
                
                # break
            

        # Add actual training logic here
        plt.show()

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
        
        for inputs, labels in self.trainVal_dataset:#self.train_loader:
            xax = np.arange(xv,xv+inputs.shape[1])
            xv+=inputs.shape[1]
            print(xv,np.mean(inputs.cpu().numpy()**2),np.mean(labels.cpu().numpy()**2))
            i=0
            # axs[i].clear()
            axs[i].plot(xax,inputs[0,:])
            # axs[i].set_aspect('equal', adjustable='box')
            i=1
            # axs[i].clear()
            axs[i].plot(xax,labels[0,:])
            # axs[i].set_aspect('equal', adjustable='box')
            plt.draw()
            plt.tight_layout()
            plt.gcf().canvas.flush_events()
            plt.pause(0.001)
            kd +=1
            if kd >= MaxVis:
                break

        plt.show()
    def testpretrained_but(self):
        ssp.utils.research.algorithms.MLRadarHumanHealthMonitoring.runSimpleScenario(trained_Model_index = 1, health_state_index = 1, sample_index = 20,sim=False)
    def modwt_but(self):
        fig, axs = plt.subplots(8, 1)
        ax = axs.flatten()
        module = MODWTLayer(wavelet='sym4', level=6, selected_levels=range(1, 6+1))
        
        idx = range(100, 800)
        k=0
        for inputs, labels in self.trainVal_dataset:
            k+=1
            
            modwt = module(inputs.unsqueeze(0))
            modwt = modwt.cpu().detach().numpy()
            x=inputs.cpu().numpy()
            y=labels.cpu().numpy()
            for i in range(8):
                ax[i].clear()
            ax[0].plot(x[0,idx])
            ax[1].plot(y[0,idx])
            for i in range(6):
                ax[i+2].plot(modwt[0,0,i,idx])
            plt.draw()
            plt.pause(0.001)
            plt.gcf().canvas.flush_events()
            if k>10:
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
