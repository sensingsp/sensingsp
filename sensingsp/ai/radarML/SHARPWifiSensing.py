import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QSpinBox, QFileDialog, QGroupBox, QComboBox, QMessageBox
)
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

class SHARPPipelineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHARP Signal Processing Pipeline UI")

        self.dataset_folder = "C:/Users/moein.ahmadi/OneDrive - University of Luxembourg/ThingsToDo/MERL ICASSP2025/wifisensing/WiFisensing/dataset/"
        self.python_code_folder = "C:/Users/moein.ahmadi/OneDrive - University of Luxembourg/ThingsToDo/MERL ICASSP2025/wifisensing/WiFisensing/SHARP/Python_code/"
        self.phase_output_folder = self.python_code_folder + "processed_phase/"
        self.doppler_output_folder = self.python_code_folder + "doppler_traces/"
        self.subdirs = "AR-1a"
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        sharp_group = QGroupBox("SHARP Signal Processing Pipeline")
        sharp_layout = QVBoxLayout()
        sharp_group.setLayout(sharp_layout)

        # Dataset folder
        dataset_row = QHBoxLayout()
        self.dataset_folder_input = QLineEdit(self)
        self.dataset_folder_input.setText(self.dataset_folder)
        dataset_browse = QPushButton("Browse")
        dataset_browse.clicked.connect(self.browse_dataset_folder)
        dataset_row.addWidget(self.dataset_folder_input)
        dataset_row.addWidget(dataset_browse)
        sharp_layout.addWidget(QLabel("CSI Dataset Folder:"))
        sharp_layout.addLayout(dataset_row)

        # Python code folder
        code_row = QHBoxLayout()
        self.python_code_folder_input = QLineEdit(self)
        self.python_code_folder_input.setText(self.python_code_folder)
        code_browse = QPushButton("Browse")
        code_browse.clicked.connect(self.browse_code_folder)
        code_row.addWidget(self.python_code_folder_input)
        code_row.addWidget(code_browse)
        sharp_layout.addWidget(QLabel("SHARP Python Code Folder:"))
        sharp_layout.addLayout(code_row)

        # Subdirectories
        self.subdirs_input = QLineEdit(self)
        self.subdirs_input.setText(self.subdirs)
        sharp_layout.addWidget(QLabel("Subdirectories (comma separated):"))
        sharp_layout.addWidget(self.subdirs_input)

        # ---- New: File browser/visualizer row ----
        filelist_row = QHBoxLayout()
        self.get_files_button = QPushButton("Get Files")
        self.get_files_button.clicked.connect(self.get_files_in_subdir)
        self.file_combo = QComboBox()
        self.visualize_button = QPushButton("Visualize / Show Info")
        self.visualize_button.clicked.connect(self.visualize_selected_file)
        filelist_row.addWidget(self.get_files_button)
        filelist_row.addWidget(self.file_combo)
        filelist_row.addWidget(self.visualize_button)
        sharp_layout.addLayout(filelist_row)

        # ------------------ PHASE SANITIZATION (ALL INPUTS) -------------------
        sharp_layout.addWidget(QLabel("Phase Sanitization Inputs (Script 1/2/3):"))

        # --- For ALL phase scripts ---
        self.phase_proc_all_files = QSpinBox()
        self.phase_proc_all_files.setMinimum(0)
        self.phase_proc_all_files.setMaximum(1)
        self.phase_proc_all_files.setValue(1)
        sharp_layout.addWidget(QLabel("Process all files in subdirectories (1) or not (0):"))
        sharp_layout.addWidget(self.phase_proc_all_files)

        self.phase_file_name = QLineEdit()
        self.phase_file_name.setText("-")
        sharp_layout.addWidget(QLabel("File name to process (use '-' if processing all):"))
        sharp_layout.addWidget(self.phase_file_name)

        self.phase_streams = QSpinBox()
        self.phase_streams.setMinimum(1)
        self.phase_streams.setValue(1)
        sharp_layout.addWidget(QLabel("Number of spatial streams:"))
        sharp_layout.addWidget(self.phase_streams)

        self.phase_cores = QSpinBox()
        self.phase_cores.setMinimum(1)
        self.phase_cores.setValue(4)
        sharp_layout.addWidget(QLabel("Number of cores:"))
        sharp_layout.addWidget(self.phase_cores)

        self.phase_start_idx = QSpinBox()
        self.phase_start_idx.setMinimum(0)
        self.phase_start_idx.setMaximum(100000)
        self.phase_start_idx.setValue(0)
        sharp_layout.addWidget(QLabel("Index where to start processing for each stream:"))
        sharp_layout.addWidget(self.phase_start_idx)

        self.phase_stop_idx = QSpinBox()
        self.phase_stop_idx.setMinimum(-1)
        self.phase_stop_idx.setMaximum(100000)
        self.phase_stop_idx.setValue(-1)
        sharp_layout.addWidget(QLabel("Index where to stop processing for each stream (-1 for all):"))
        sharp_layout.addWidget(self.phase_stop_idx)

        self.phase_processed_dir = QLineEdit()
        self.phase_processed_dir.setText(self.python_code_folder + "phase_processing/")
        sharp_layout.addWidget(QLabel("Directory of processed data (for reconstruction step):"))
        sharp_layout.addWidget(self.phase_processed_dir)

        self.phase_out_dir = QLineEdit()
        self.phase_out_dir.setText(self.phase_output_folder)
        sharp_layout.addWidget(QLabel("Directory to save reconstructed data (for reconstruction step):"))
        sharp_layout.addWidget(self.phase_out_dir)

        # Status Label
        self.status_label = QLabel("Status: Ready.")
        sharp_layout.addWidget(self.status_label)

        # --- Buttons ---
        self.phase_button = QPushButton("1. Run Phase Sanitization (all scripts)")
        self.phase_button.clicked.connect(self.run_phase_sanitization)
        sharp_layout.addWidget(self.phase_button)

        # (Other steps below unchanged, keep simple for brevity)
        self.doppler_button = QPushButton("2. Run Doppler Computation")
        self.doppler_button.clicked.connect(self.run_doppler_computation)
        sharp_layout.addWidget(self.doppler_button)

        self.dataset_button = QPushButton("3. Create Dataset (Train)")
        self.dataset_button.clicked.connect(self.run_create_datasets)
        sharp_layout.addWidget(self.dataset_button)

        self.train_button = QPushButton("4. Train HAR Model")
        self.train_button.clicked.connect(self.run_train_model)
        sharp_layout.addWidget(self.train_button)

        main_layout.addWidget(sharp_group)

    # ---- Folder browse handlers ----
    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.dataset_folder_input.text())
        if folder:
            self.dataset_folder_input.setText(folder)
    def browse_code_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Python Code Folder", self.python_code_folder_input.text())
        if folder:
            self.python_code_folder_input.setText(folder)

    # ---- File browser/visualizer ----
    def get_files_in_subdir(self):
        dataset_base = self.dataset_folder_input.text()
        subdir = self.subdirs_input.text().split(",")[0].strip()
        subdir_path = os.path.join(dataset_base, subdir)
        self.file_combo.clear()
        if not os.path.exists(subdir_path):
            QMessageBox.warning(self, "Not found", f"Folder not found:\n{subdir_path}")
            return
        files = [f for f in os.listdir(subdir_path) if f.endswith('.mat')]
        files.sort()
        if not files:
            QMessageBox.information(self, "No Files", "No .mat files found in selected subdir.")
        self.file_combo.addItems(files)
        self.status_label.setText(f"Found {len(files)} .mat files in {subdir}")

    def visualize_selected_file(self):
        dataset_base = self.dataset_folder_input.text()
        subdir = self.subdirs_input.text().split(",")[0].strip()
        fname = self.file_combo.currentText()
        subdir_path = os.path.join(dataset_base, subdir)
        fpath = os.path.join(subdir_path, fname)
        if not os.path.exists(fpath):
            QMessageBox.warning(self, "Not found", f"File not found:\n{fpath}")
            return
        try:
            mat = loadmat(fpath)
            if "csi_buff" not in mat:
                raise KeyError("Key 'csi_buff' not found in .mat file.")

            csi_buff = mat["csi_buff"]
            N = csi_buff.shape[0]
            CSI_Rate = 200  # Hz
            Plot_Time = 3  # seconds
            PlotN = int(CSI_Rate * Plot_Time)
            loopN = N // PlotN
            loopN = min(loopN, 10)  # Limit to 10 plots for performance
            # Set up the figure once
            fig, ax = plt.subplots()
            img = ax.imshow(np.abs(csi_buff[0:PlotN, :]), aspect='auto', cmap='jet')
            plt.title("CSI Plot")
            plt.colorbar(img, ax=ax, label='Amplitude')
            plt.xlabel("Subcarriers")
            plt.ylabel("Time (samples)")
            plt.tight_layout()

            for plot_i in range(loopN):
                csi_buff_plot = np.abs(csi_buff[plot_i*PlotN:(plot_i+1)*PlotN, :])
                img.set_data(csi_buff_plot)
                ax.set_title(f"CSI Frame {plot_i}")
                ax.set_ylim(0, csi_buff_plot.shape[0])
                ax.set_xlim(0, csi_buff_plot.shape[1])
                plt.pause(0.1)

            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading {fname}:\n{str(e)}")
    # ---- Phase Sanitization step, with all parameters ----
    def run_phase_sanitization(self):
        data_dir = self.dataset_folder_input.text()
        py_folder = self.python_code_folder_input.text()
        process_all = self.phase_proc_all_files.value()
        file_name = self.phase_file_name.text()
        streams = self.phase_streams.value()
        cores = self.phase_cores.value()
        start_idx = self.phase_start_idx.value()
        stop_idx = self.phase_stop_idx.value()
        processed_dir = self.phase_processed_dir.text()
        out_dir = self.phase_out_dir.text()

        # 1. Signal Preprocessing
        cmd1 = f'python "{os.path.join(py_folder,"CSI_phase_sanitization_signal_preprocessing.py")}" "{data_dir}" {process_all} {file_name} {streams} {cores} {start_idx}'
        # 2. H Estimation
        cmd2 = f'python "{os.path.join(py_folder,"CSI_phase_sanitization_H_estimation.py")}" "{data_dir}" {process_all} {file_name} {streams} {cores} {start_idx} {stop_idx}'
        # 3. Signal Reconstruction
        cmd3 = f'python "{os.path.join(py_folder,"CSI_phase_sanitization_signal_reconstruction.py")}" "{processed_dir}" "{out_dir}" {streams} {cores} {start_idx} {stop_idx}'
        self.status_label.setText("Running Phase Sanitization...")
        for cmd in [cmd1, cmd2, cmd3]:
            subprocess.call(cmd, shell=True)
        self.status_label.setText("Phase Sanitization Finished.")

    # ---- Doppler and rest (unchanged for brevity, can be expanded in same way) ----
    def run_doppler_computation(self):
        py_folder = self.python_code_folder_input.text()
        phase_out = self.phase_out_dir.text()
        doppler_out = self.doppler_output_folder
        subdirs = self.subdirs_input.text()
        cmd = f'python "{os.path.join(py_folder,"CSI_doppler_computation.py")}" "{phase_out}" "{subdirs}" "{doppler_out}" 800 800 31 1 -1.2'
        self.status_label.setText("Running Doppler Computation...")
        subprocess.call(cmd, shell=True)
        self.status_label.setText("Doppler Computation Finished.")

    def run_create_datasets(self):
        py_folder = self.python_code_folder_input.text()
        doppler_out = self.doppler_output_folder
        subdirs = self.subdirs_input.text()
        cmd = f'python "{os.path.join(py_folder,"CSI_doppler_create_dataset_train.py")}" "{doppler_out}" "{subdirs}" 31 1 340 30 E,L,W,R,J 4'
        self.status_label.setText("Creating Dataset...")
        subprocess.call(cmd, shell=True)
        self.status_label.setText("Dataset Creation Finished.")

    def run_train_model(self):
        py_folder = self.python_code_folder_input.text()
        doppler_out = self.doppler_output_folder
        subdirs = self.subdirs_input.text()
        cmd = f'python "{os.path.join(py_folder,"CSI_network.py")}" "{doppler_out}" "{subdirs}" 100 340 1 32 4 single_ant E,L,W,R,J'
        self.status_label.setText("Training HAR Model...")
        subprocess.call(cmd, shell=True)
        self.status_label.setText("Training Finished.")

import sys
sys.path.append("C:/Users/moein.ahmadi/OneDrive - University of Luxembourg/SensingSP/sensingsp-main/sensingsp")
sys.path.append("/Users/moeinahmadi/Library/CloudStorage/OneDrive-UniversityofLuxembourg/SensingSP/sensingsp-main/sensingsp")
import sensingsp as ssp

def runradarSHARPWifiSensingapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    appSTYLESHEET = ssp.config.appSTYLESHEET
    app.setStyleSheet(appSTYLESHEET)  # Replace with your desired stylesheet if any
    
    window = SHARPPipelineApp()
    window.show()
    app.exec_()


if __name__ == "__main__":
    runradarSHARPWifiSensingapp()