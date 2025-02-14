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

def initialize_weights_scaled_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def initialize_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def initialize_weights_lecun(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1 / (m.weight.size(1) ** 0.5))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def normalize_along_rows(X):
    min_vals = np.min(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)
    denom = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))
    X_norm = (X - min_vals) / denom
    return X_norm.astype(np.float32)

def make_sample(left,top,right):
    b = [1, -1]
    a = [1, -0.9]
    left = lfilter(b, a, left, axis=0)
    top = lfilter(b, a, top, axis=0)
    right = lfilter(b, a, right, axis=0)
    left = normalize_along_rows(left)
    top = normalize_along_rows(top)
    right = normalize_along_rows(right)
    slow_time_per_sample = 90
    B = int(left.shape[0]/slow_time_per_sample)
    o=[]
    for i in range(B):
        x1 = left[i*slow_time_per_sample:(i+1)*slow_time_per_sample]
        x2 = top[i*slow_time_per_sample:(i+1)*slow_time_per_sample]
        x3 = right[i*slow_time_per_sample:(i+1)*slow_time_per_sample]
        radar_tensor = np.stack([x1, x2, x3], axis=0)  # shape: (3,90,189)
        radar_tensor = torch.tensor(radar_tensor, dtype=torch.float32).unsqueeze(0)
        o.append(radar_tensor)
    return o
def load_sample(matfile):
    mat_data = loadmat(matfile)
    
    left = mat_data.get("Left")
    top = mat_data.get("Top")
    right = mat_data.get("Right")
    return make_sample(left,top,right)
class RadarGestureDataset(Dataset):
    def __init__(self, data_folder, max_folder_number=1e6, clutter_removal=True,PercentBar=False):
        self.data_folder = data_folder
        self.folders_data_classes = []  # Stores (left, top, right, gesture_number)
        # self.classes = {}  # Map gesture_number to class_idx
        # self.index2gestureCodes = {}

        self.data_samples_per_file = 100
        self.slow_time_per_sample = 90
        self.gestureCodes = ["G1","G2","G3","G4",
                             "G5","G6","G7","G8",
                             "G9","G10","G11","G12"]
        self.gestureVocabulary = ["L-R swipe", "R-L swipe", "U-D swipe", "D-U swipe",
                                  "Diag-LR-UD swipe", "Diag-LR-DU swipe", "Diag-RL-UD swipe", "Diag-RL-DU swipe",
                                  "clockwise", "counterclockwise", "inward push", "empty"]
        # self.gestureCodes2Vocabulary = {g: v for g, v in zip(self.gestureCodes, self.gestureVocabulary)}

        self.clutter_removal = clutter_removal
        self.max_folder_number = int(max_folder_number)
        NValid = self.data_samples_per_file * self.slow_time_per_sample

        class_idx = 0
        folder_number = 0
        self.Max_folder_available = 0
        if data_folder=='':
            return
        for subject_folder in sorted(os.listdir(data_folder)):
            subject_path = os.path.join(data_folder, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            self.Max_folder_available += 1
        if PercentBar:
            progressBar = QProgressBar()
            progressBar.setMinimum(0)
            progressBar.setMaximum(min(self.max_folder_number,self.Max_folder_available))
            progressBar.setWindowTitle("Loading dataset SensingSP")
            progressBar.show()
        for subject_folder in sorted(os.listdir(data_folder)):
            subject_path = os.path.join(data_folder, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            folder_number += 1
            if PercentBar:
                progressBar.setValue(folder_number)
                QApplication.processEvents()
            if folder_number > self.max_folder_number:
                break

            for mat_file in sorted(os.listdir(subject_path)):
                if mat_file.endswith(".mat"):
                    file_path = os.path.join(subject_path, mat_file)
                    gesture_name = mat_file.split("_")[2]
                    mat_data = loadmat(file_path)

                    left = mat_data.get("Left")
                    top = mat_data.get("Top")
                    right = mat_data.get("Right")

                    if left is None or top is None or right is None:
                        continue

                    # Check shape consistency
                    if (left.shape[0] != NValid or top.shape[0] != NValid or right.shape[0] != NValid):
                        continue

                    # Clutter removal
                    if self.clutter_removal:
                        b = [1, -1]
                        a = [1, -0.9]
                        left = lfilter(b, a, left, axis=0)
                        top = lfilter(b, a, top, axis=0)
                        right = lfilter(b, a, right, axis=0)

                    # Normalize each row to [0,1]
                    # Similar to MATLAB's normalize(...,2,"range",[0,1]), 
                    # which normalizes along the second dimension (fast-time)
                    left = self.normalize_along_rows(left)
                    top = self.normalize_along_rows(top)
                    right = self.normalize_along_rows(right)

                    # Extract gesture number
                    gesture_number = int(re.sub(r'\D', '', gesture_name))
                    # if gesture_number not in self.classes:
                    #     self.classes[gesture_number] = class_idx
                    #     self.index2gestureCodes[class_idx] = gesture_name
                    #     class_idx += 1

                    self.folders_data_classes.append([left, top, right, gesture_number-1])

    def normalize_along_rows(self, X):
        # Normalize each row of X to [0, 1]
        min_vals = np.min(X, axis=1, keepdims=True)
        max_vals = np.max(X, axis=1, keepdims=True)
        # Avoid division by zero if max_vals == min_vals
        denom = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))
        X_norm = (X - min_vals) / denom
        return X_norm.astype(np.float32)

    def __len__(self):
        return len(self.folders_data_classes) * self.data_samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.data_samples_per_file
        sample_idx = idx % self.data_samples_per_file
        left, top, right, gesture_number = self.folders_data_classes[file_idx]

        start_idx = sample_idx * self.slow_time_per_sample
        end_idx = start_idx + self.slow_time_per_sample

        left_sample = left[start_idx:end_idx, :]
        top_sample = top[start_idx:end_idx, :]
        right_sample = right[start_idx:end_idx, :]

        # radar_tensor: shape [3, slow_time, fast_time]
        radar_tensor = np.stack([left_sample, top_sample, right_sample], axis=0)  # shape: (3,90,189)
        radar_tensor = torch.tensor(radar_tensor, dtype=torch.float32)

        label = torch.tensor(gesture_number, dtype=torch.long)
        return radar_tensor, label

class RadarBranch(nn.Module):
    """A single radar branch mirroring the MATLAB CNN layers."""
    def __init__(self):
        super(RadarBranch, self).__init__()
        # Input: (1, 90, 189)
        # We'll treat each radar as a single-channel "image" of shape [90 x 189]
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After these layers, the feature map shape should be reduced accordingly.

    def forward(self, x):
        # x: shape (N, 1, 90, 189)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # => (N,8,45,94)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # => (N,16,22,47)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # => (N,32,11,23)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # => (N,64,5,11) approximately
        return x

class MultiInputModel(nn.Module):
    def __init__(self, num_classes=12):
        super(MultiInputModel, self).__init__()
        self.branch_left = RadarBranch()
        self.branch_top = RadarBranch()
        self.branch_right = RadarBranch()

        # After each branch, we have a feature map of shape (N, 64, 5, 11)
        # We'll add them element-wise (as in the MATLAB code additionLayer)
        # Ensure that all shapes match, which they do by construction.

        # The final FC layers: flatten the feature maps and pass through FC -> softmax
        self.fc = nn.Linear(64*5*11, num_classes)  # Adjust if final dimensions differ

    def forward(self, x):
        # x: shape (N, 3, 90, 189)
        # Split along radar dimension:
        left = x[:, 0:1, :, :]  # (N,1,90,189)
        top = x[:, 1:2, :, :]   # (N,1,90,189)
        right = x[:, 2:3, :, :] # (N,1,90,189)

        left_features = self.branch_left(left)
        top_features = self.branch_top(top)
        right_features = self.branch_right(right)

        # Add feature maps element-wise
        combined = left_features + top_features + right_features
        # Flatten for FC
        combined = combined.view(combined.size(0), -1)  # (N,64*5*11)
        out = self.fc(combined)
        # out = nn.functional.softmax(out, dim=1)
        return out

# ----------------------------
# Run Application Function
# ----------------------------
def runradarmisoCNNapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    appSTYLESHEET = ssp.config.appSTYLESHEET
    app.setStyleSheet(appSTYLESHEET)  # Replace with your desired stylesheet if any
    
    window = RadarMisoCNNApp()
    window.show()
    app.exec_()


from scipy.optimize import minimize

# ----------------------------
# RadarMisoCNNApp GUI Class
# ----------------------------
class RadarMisoCNNApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Classification Using Radar Signals and Deep Learning")
        self.default_folder = "/home/moein/Documents/MATLAB/Examples/R2024a/supportfiles/SPT/data/uwb-gestures"
        self.default_folder = "C:/Users/moein.ahmadi/OneDrive - University of Luxembourg/ThingsToDo/Max/HG Nature/data/uwb-gestures"
        # self.default_folder = "/home/moein/Downloads/uwb-gestures"  # Update with the path to your dataset
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

        self.folderN_input = QSpinBox(self)
        self.folderN_input.setMinimum(1)
        self.folderN_input.setValue(8)
        form_layout.addRow("Folder read num:", self.folderN_input)
        self.epochs_input = QSpinBox(self)
        self.epochs_input.setMinimum(1)
        self.epochs_input.setValue(3)
        form_layout.addRow("Epochs:", self.epochs_input)

        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setMinimum(1)
        self.batch_size_input.setValue(32)
        form_layout.addRow("Batch Size:", self.batch_size_input)

        self.learning_rate_input = QDoubleSpinBox(self)
        self.learning_rate_input.setMinimum(0.0001)
        self.learning_rate_input.setValue(0.5)
        self.learning_rate_input.setDecimals(4)
        form_layout.addRow("Learning Rate (x*1e-3):", self.learning_rate_input)

        self.split_train_input = QDoubleSpinBox(self)
        self.split_train_input.setMinimum(0.1)
        self.split_train_input.setMaximum(0.9)
        self.split_train_input.setSingleStep(0.05)
        self.split_train_input.setValue(0.7)
        form_layout.addRow("Train Split Ratio:", self.split_train_input)

        self.split_val_input = QDoubleSpinBox(self)
        self.split_val_input.setMinimum(0.05)
        self.split_val_input.setMaximum(0.9)
        self.split_val_input.setSingleStep(0.05)
        self.split_val_input.setValue(0.15)
        form_layout.addRow("Validation Split Ratio:", self.split_val_input)

        # Add "Clutter Removal" checkbox
        self.clutter_removal_checkbox = QCheckBox("Enable Clutter Removal", self)
        self.clutter_removal_checkbox.setChecked(True)
        form_layout.addRow(self.clutter_removal_checkbox)

        main_layout.addLayout(form_layout)

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
        row2_layout.addWidget(self.combobox)
        
        
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

        self.testfile_input = QLineEdit(self)
        self.testfile_input.setText(os.path.join(ssp.config.temp_folder,'HandG.mat'))  # Set default folder
        self.testfile_browse_button = QPushButton("Browse")
        self.testfile_browse_button.clicked.connect(self.browse_testfile)
        testfile_layout = QHBoxLayout()
        
        testfile_layout.addWidget(self.testfile_input)
        testfile_layout.addWidget(self.testfile_browse_button)
        # form_layout.addRow("Test File:", testfile_layout)

        row4_layout = QHBoxLayout()
        self.testfile_button = QPushButton("Test input file")
        self.testfile_button.clicked.connect(self.testinput_model)
        row4_layout.addLayout(testfile_layout)
        row4_layout.addWidget(self.testfile_button)
        main_layout.addLayout(row4_layout)

        
        self.combobox_hub = QComboBox()
        available_files = ssp.utils.hub.available_files()
        all = []
        for category, files in available_files.items():
            if files:
                for file_info in files:
                    all.append(f'{category}/{file_info["name"]}')
        self.combobox_hub.addItems(all)
        self.loadhub_button = QPushButton("Load from Hub")
        self.loadhub_button.clicked.connect(self.loadfromhub)
        main_layout.addWidget(self.combobox_hub)
        main_layout.addWidget(self.loadhub_button)

        # Status display
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

    def initNet(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.num_classes = 12
        self.savebestmodelpath = os.path.join(ssp.config.temp_folder,"best_model.pth")
        
        # dataset = RadarGestureDataset(data_folder=self.default_folder, clutter_removal=True)

    def initdataset(self):
        clutter_removal = self.clutter_removal_checkbox.isChecked() 
        self.dataset = RadarGestureDataset(data_folder=self.default_folder, clutter_removal=clutter_removal,max_folder_number=self.folderN_input.value(),PercentBar=True)
        
        
        self.combobox.clear()
        self.combobox.addItems(self.dataset.gestureVocabulary)

        
        total_samples = len(self.dataset)  # total number of gesture samples
        train_ratio = self.split_train_input.value()
        val_ratio = self.split_val_input.value()

        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        batch_size = self.batch_size_input.value() 
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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
                    # Apply softmax to compute probabilities
                    probabilities = nn.functional.softmax(outputs, dim=1)
                    print("Class probabilities:")
                    classes,softmax_probs=[],[]
                    for idx, prob in enumerate(probabilities[0]):
                        class_name = self.dataset.gestureVocabulary[idx]
                        print(f"Class {class_name}: {prob:.4f} : {outputs[0,idx]:.4f}")
                        classes.append(class_name)
                        softmax_probs.append(prob)
                    print("Class probabilities (visualized):")
                    for idx, prob in enumerate(softmax_probs):
                        bar_length = int(prob * 50)  # Scale probabilities to a length of 50
                        print(f"Class {classes[idx]:<20}: {prob:.2f}: {'#' * bar_length}")
                    plt.barh(classes, softmax_probs, color='skyblue')
                    plt.title("Class Probabilities (Softmax)")
                    plt.xlabel("Probability")
                    plt.tight_layout()
                    plt.show()
                
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
        self.status_label.setText(f"Status: Dataset is loaded, {self.dataset.Max_folder_available} folders available...")
        
    def def_model(self):
        # self.model = MultiInputModel(num_classes=self.num_classes)
        1
        
    def train_model(self):
        # Placeholder for training logic
        self.status_label.setText("Status: Training started...")
        # -----------------------------
        # Model, Loss, Optimizer
        # -----------------------------
        self.model = MultiInputModel(num_classes=self.num_classes)
        self.model.apply(initialize_weights_xavier)
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
    def loadfromhub(self):
        file = self.combobox_hub.currentText()
        hubcategory,hubname=file.split('/')
        # ssp.utils.initialize_environment()
        hand_file_path = ssp.utils.hub.fetch_file(category=hubcategory,name=hubname)
        ssp.environment.add_blenderfileobjects(hand_file_path,RCS0=1,translation=(.55,.3,-.1),rotation=(0,0,np.pi/2))

        
    def visualize_samples(self):
        # Placeholder for sample visualization logic
        self.status_label.setText("Status: Visualizing samples...")
        g = self.combobox.currentText()
        fig, axs = plt.subplots(1, 4, figsize=(15, 6))
        kd = 0
        radar=["Left","Top","Right"]
        MaxVis = self.sampleVisN_input.value()
        for d,l in self.dataset:
            
            gi = self.dataset.gestureVocabulary[l.item()] 
            if gi == g:

                SourceLocations = np.array([[0, 0], [0.5, 0.5], [1, 0]])
                results= []
                data = d.numpy()
                max_indices = np.argmax(np.abs(data), axis=2)
                for i in range(max_indices.shape[1]):
                    ranges = 1.2/189 * max_indices[:,i]
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
                for i in range(3):
                    axs[i].clear()
                    axs[i].imshow(data[i,:,:], aspect='auto', cmap='viridis')
                    axs[i].set_title(f"{g} : {radar[i]}")
                i=3
                axs[i].clear()
                axs[i].plot(results[:,0],results[:,1],'.')
                axs[i].set_aspect('equal', adjustable='box')

                axs[i].set_title(f"{g}")
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
# Main Execution
# ----------------------------
if __name__ == "__main__":
    runradarmisoCNNapp()