from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton, QWidget,
                             QLabel, QTextEdit,QHeaderView, QComboBox, QCheckBox, QFileDialog, QLineEdit)
from PyQt5.QtCore import QTimer
from PyQt5.QtSerialPort import QSerialPortInfo
import sys
import os

import sensingsp as ssp

import numpy as np
import serial.tools.list_ports

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RadarPlotCanvas(FigureCanvas):
    """
    A Matplotlib canvas for displaying the radar data plot.
    """
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

        # Initial plot configuration
        self.axes.set_title("Radar Data")
        self.axes.set_xlabel("Samples")
        self.axes.set_ylabel("Amplitude")

        # Create lines for range and noise profiles
        self.range_line, = self.axes.plot([], [], lw=2, label="Range Profile")
        self.noise_line, = self.axes.plot([], [], lw=2, label="Noise Profile")

        self.axes.legend()

    def update_plot(self, range_profile, noise_profile):
        """
        Update the plot with new range and noise profile data.
        """
        self.range_line.set_data(range(len(range_profile)), range_profile)
        self.noise_line.set_data(range(len(noise_profile)), noise_profile)

        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()



def list_available_ports():
    """List all available serial ports."""
    ports = serial.tools.list_ports.comports()
    available_ports = [[port.device,port.description] for port in ports if "USB" in port.device or "COM" in port.device or "ACM" in port.device]
    return available_ports

class GeneralDevice():
    Name = "TI IWR6843ISK"
    connected = False
    ConfigPortName = ""
    DataPortName = ""
    ConfigFile = ""
    avalaibeDevices=["TI IWR6843ISK",
                               "TI AWR6843ISK" ,
                                "TI IWR6843AOPEVM",
                                  "TI AWR6843AOPEVM" ,
                                    "TI AWR1843AOPEVM" ,
                                      "TI AWR1642BOOST" ,
                                        "TI IWR1642BOOST" ,
                                          "TI IWRL6432BOOST",
                                          "Xhetru X4"]
        


class RadarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Device Manager")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.devices = []  # List of devices
        # Main Layout
        main_layout = QVBoxLayout()

        # Device Properties Table
        self.device_table = QTableWidget()
        self.device_table.setColumnCount(5)
        self.device_table.setHorizontalHeaderLabels([
            "Device", "Config Port", "Data Port", "Connected", "Config File"
        ])
        # Set the height of rows
        for row in range(self.device_table.rowCount()):
            self.device_table.setRowHeight(row, 40)  # Adjust the height value as needed

        # Stretch columns to fit the available width
        header = self.device_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Optional: Enable vertical header resizing if you dynamically add rows
        self.device_table.verticalHeader().setDefaultSectionSize(40)  # Adjust as needed
        self.device_table.setRowCount(0)
        main_layout.addWidget(self.device_table)

        # Device Actions
        action_layout = QHBoxLayout()
        self.add_device_btn = QPushButton("Add Device")
        self.add_device_btn.clicked.connect(self.add_device)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_device)

        action_layout.addWidget(self.add_device_btn)
        action_layout.addWidget(self.connect_btn)
        main_layout.addLayout(action_layout)

    # Set up the Matplotlib canvas
        self.canvas = RadarPlotCanvas(central_widget)
        main_layout.addWidget(self.canvas)

        # Timer for updating the plot
        # self.timer = Timer(0.1, self.update_plot)  # Update every 100 ms
        # self.timer.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)
        # Console
        self.console = QTextEdit(self)
        self.console.setText(
            """ TI mmWave Hint: for Linux do:
            sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1
            sudo usermod -a -G dialout $USER

            go to:
            https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/
            and save:
            "/tmp/SensingSP/profile_2024_12_24T15_12_37_649.cfg"
            ______________________________________________
            """
        )
        self.console.setReadOnly(True)
        main_layout.addWidget(self.console)

        # Status Bar
        self.status_bar = QLabel("Status: Ready")
        main_layout.addWidget(self.status_bar)

        # Set Main Layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.devices_comm  = [] 
        # # Timer for UI updates
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_ui)
        # self.timer.start(500)  # Check every 500ms

    def add_device(self):
        d = GeneralDevice()
        cfg_file = next((file for file in os.listdir(ssp.config.temp_folder) if file.endswith(".cfg")), None)
        if cfg_file:
            d.ConfigFile=ssp.utils.file_in_tmpfolder(cfg_file)
        self.devices.append(d)
        self.update_device_table()
        self.status_bar.setText(f"Added")

    def update_device_table(self):
        self.device_table.setRowCount(len(self.devices))
        for row, device in enumerate(self.devices):
            k = 0
            device_type_combo = QComboBox()
            device_type_combo.addItems(device.avalaibeDevices)
            self.device_table.setCellWidget(row, k, device_type_combo)
            k += 1

            config_port_combo = QComboBox()
            data_port_combo = QComboBox()
            available_ports = list_available_ports()
            for port,desc in available_ports:
                config_port_combo.addItem(port)
                data_port_combo.addItem(port)
                if "enhanced" in desc.lower():
                    config_port_combo.setCurrentIndex(config_port_combo.count()-1)
                if "standard" in desc.lower():
                    data_port_combo.setCurrentIndex(data_port_combo.count()-1)
            self.device_table.setCellWidget(row, k, config_port_combo)
            k += 1
            self.device_table.setCellWidget(row, k, data_port_combo)
            k += 1

            is_connected_checkbox = QCheckBox()
            is_connected_checkbox.setChecked(device.connected)
            is_connected_checkbox.setEnabled(False)
            self.device_table.setCellWidget(row, k, is_connected_checkbox)
            k += 1

            # self.device_table.setItem(row, k, QTableWidgetItem(device["SDKVersion"]))
            # k += 1
            # self.device_table.setItem(row, k, QTableWidgetItem(str(device["UpdateRate"])))
            # k += 1
            # self.device_table.setItem(row, k, QTableWidgetItem(str(device["MaximumRange"])))
            # k += 1
            # self.device_table.setItem(row, k, QTableWidgetItem(str(device["AzimuthResolution"])))
            # k += 1
            
            config_file_layout = QWidget()
            config_file_layout_layout = QHBoxLayout()
            config_file_button = QPushButton("Browse")
            config_file_text = QLineEdit(device.ConfigFile)
                    
            def browse_file():
                file_path, _ = QFileDialog.getOpenFileName(self, "Select Config File", ssp.config.temp_folder, "Config Files (*.cfg)")
                if file_path:
                    config_file_text.setText(file_path)
                    device.ConfigFile = file_path

            config_file_button.clicked.connect(browse_file)

            config_file_layout_layout.addWidget(config_file_text)
            config_file_layout_layout.addWidget(config_file_button)
            config_file_layout_layout.setContentsMargins(0, 0, 0, 0)
            config_file_layout.setLayout(config_file_layout_layout)
            self.device_table.setCellWidget(row, k, config_file_layout)

    
    def connect_device(self):
        self.devices.clear()
        for row in range(self.device_table.rowCount()):
            self.devices.append(GeneralDevice())
            self.devices[-1].Name = self.device_table.cellWidget(row, 0).currentText()
            self.devices[-1].ConfigPortName = self.device_table.cellWidget(row, 1).currentText()
            self.devices[-1].DataPortName = self.device_table.cellWidget(row, 2).currentText()
            self.devices[-1].connected = self.device_table.cellWidget(row, 3).isChecked()
            self.devices[-1].ConfigFile = self.device_table.cellWidget(row, 4).layout().itemAt(0).widget().text()
        row=-1
        for d in self.devices:
            row+=1
            if not d.connected:
                if d.Name.startswith("TI "):
                    ti = ssp.hardware.radar.TImmWave.MMWaveDevice(config_port=d.ConfigPortName, data_port=d.DataPortName)
                    ti.connect()
                    if ti.connected:
                        self.device_table.cellWidget(row, 3).setChecked(ti.connected)
                        ti.send_config_file(d.ConfigFile)
                        self.devices_comm.append(["TI",d.ConfigPortName,d.DataPortName,ti])
                elif d.Name.startswith("Xhetru "):
                    x4 = ssp.hardware.radar.XeThru.XeThruDevice(port=d.ConfigPortName)
                    x4.connect()
                    if x4.connected:
                        self.devices_comm.append(["Xhetru",d.ConfigPortName,"",x4])
        # self.devices_comm[-1].decoded

    def update_plot(self):
        """
        Update the plot with the latest data from devices_comm.
        """
        if len(self.devices_comm) > 0:
            # Get the latest device communication object
            latest_device = self.devices_comm[-1][3]

            if len(latest_device.decoded) > 0:
                if self.devices_comm[-1][0]=="TI":
                    decoded_data = latest_device.decoded[-1][0]
                    range_profile = decoded_data.get('range_profile', [])
                    noise_profile = decoded_data.get('noise_profile', [])

                    # Update the plot with the new data
                    self.canvas.update_plot(range_profile, noise_profile)
                elif self.devices_comm[-1][0]=="Xhetru":
                    decoded_data = latest_device.decoded[-1][0]
                    I = decoded_data.real
                    Q = decoded_data.imag
                    # phase = np.arctan2(Q, I)/np.pi*180
                    # self.canvas.update_plot(phase, phase)
                    self.canvas.update_plot(I, Q)
                    
        # Restart the timer
        # self.timer = Timer(0.1, self.update_plot)
        # self.timer.start()

    def closeEvent(self, event):
        # self.radar_device.disconnect()  # Disconnect the radar device
        self.timer.stop()  # Stop the timer
        event.accept()
def runapp():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)  # Replace with your desired stylesheet
    window = RadarApp()
    window.show()
    app.exec_()

