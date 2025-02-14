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

from scipy.signal import lfilter
import copy

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

        self.rangePercentage = 100
        self.slowtimeremoveindex = 0
        central_widget = QWidget()
        self.devices = []  # List of devices
        # Main Layout
        main_layout = QVBoxLayout()

        
        # Device Actions
        action_layout = QHBoxLayout()
        self.LineEdit_port = QLineEdit()
        available_ports = list_available_ports()
        for port,desc in available_ports:
            self.LineEdit_port.setText(port)
            break
        self.add_device_btn = QPushButton("Add Device")
        self.add_device_btn.clicked.connect(self.add_device)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_device)

        # action_layout.addWidget(self.add_device_btn)
        action_layout.addWidget(self.LineEdit_port)
        action_layout.addWidget(self.connect_btn)
        main_layout.addLayout(action_layout)

        action_layout2 = QHBoxLayout()
        self.LineEdit_input = QLineEdit()
        self.LineEdit_input1D = QLineEdit()
        self.Combobox_input = QComboBox()
        self.Combobox_input.addItems(["fps 10","fps 50","fps 90","fps 0",
                                      "fps 120","fps 150","fps 180","fps 200","fps 300","fps 500",
                                      "stop x4","start x4",
                                      "RF mode","IQ baseband",
                                      "set_range_0to9","set_range_2to11",
                                      "set_range_5to14","set_range_8to17",
                                      "set_range_21to30","set_range_0to1_2",
                                      "set_range txt1-txt2","set max buffer (CPI)",
                                      "Range Percentage","slow time remove first index",
                                      "toggle timer"])
        self.send_btn = QPushButton("Set")
        self.send_btn.clicked.connect(self.send_fcn)
        action_layout2.addWidget(self.LineEdit_input)
        action_layout2.addWidget(self.LineEdit_input1D)
        self.LineEdit_input.setText("0:3")
        self.LineEdit_input1D.setText("20")
        action_layout2.addWidget(self.Combobox_input)
        action_layout2.addWidget(self.send_btn)
        main_layout.addLayout(action_layout2)

        self.Combobox_input2 = QComboBox()
        self.Combobox_input2.addItems(["fast time","fast-slow","fast-doppler","phase -> fast0"])
        main_layout.addWidget(self.Combobox_input2)
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

    def send_fcn(self):

        if len(self.devices_comm) == 0:
            return
        # ssp.hardware.radar.XeThru.XeThruDevice().cmd_set_downconversion_0 
        # latest_device = self.devices_comm[-1][3]
        latest_device: ssp.hardware.radar.XeThru.XeThruDevice = self.devices_comm[-1][3]
        if self.Combobox_input.currentText() == "fps 10":
            latest_device.serial.write(latest_device.cmd_set_fps_to_10)
        elif self.Combobox_input.currentText() == "fps 50":
            latest_device.serial.write(latest_device.cmd_set_fps_to_50)
        elif self.Combobox_input.currentText() == "fps 90":
            latest_device.serial.write(latest_device.cmd_set_fps_to_90)
        elif self.Combobox_input.currentText() == "fps 0":
            latest_device.serial.write(latest_device.cmd_set_fps_to_0)  
        # elif self.Combobox_input.currentText() == "fps 120":
        #     latest_device.serial.write(latest_device.cmd_set_fps_to_120)
        # elif self.Combobox_input.currentText() == "fps 150":
        #     latest_device.serial.write(latest_device.cmd_set_fps_to_150)
        # elif self.Combobox_input.currentText() == "fps 180":
        #     latest_device.serial.write(latest_device.cmd_set_fps_to_180)
        # elif self.Combobox_input.currentText() == "fps 200":
        #     latest_device.serial.write(latest_device.cmd_set_fps_to_200)
        # elif self.Combobox_input.currentText() == "fps 300":
        #     latest_device.serial.write(latest_device.cmd_set_fps_to_300)
        # elif self.Combobox_input.currentText() == "fps 500":
        #     latest_device.serial.write(latest_device.cmd_set_fps_to_500)
        elif self.Combobox_input.currentText() == "RF mode":
            latest_device.set_RF_mode(1)
        elif self.Combobox_input.currentText() == "IQ baseband":
            latest_device.set_RF_mode(0)
        elif self.Combobox_input.currentText() == "set_range_0to9":
            latest_device.serial.write(latest_device.cmd_set_range_0to9)
        elif self.Combobox_input.currentText() == "set_range_2to11":
            latest_device.serial.write(latest_device.cmd_set_range_2to11)
        elif self.Combobox_input.currentText() == "set_range_5to14":
            latest_device.serial.write(latest_device.cmd_set_range_5to14)
        elif self.Combobox_input.currentText() == "set_range_8to17":
            latest_device.serial.write(latest_device.cmd_set_range_8to17)
        elif self.Combobox_input.currentText() == "set_range_21to30":
            latest_device.serial.write(latest_device.cmd_set_range_21to30)
        elif self.Combobox_input.currentText() == "set_range_0to1_2":
            latest_device.serial.write(latest_device.cmd_set_range_0to1_2)
        elif self.Combobox_input.currentText() == "set_range txt1-txt2":
            r1r2=self.LineEdit_input.text().split(":")
            if len(r1r2) == 2:
                start_range, end_range = float(r1r2[0]), float(r1r2[1])   
                m = ssp.hardware.radar.XeThru.set_detection_zone_command(start_range, end_range)
                latest_device.serial.write(m)   
        elif self.Combobox_input.currentText() == "stop x4":
            latest_device.serial.write(latest_device.sensor_stop)
        elif self.Combobox_input.currentText() == "start x4":
            latest_device.serial.write(latest_device.sensor_start)
        elif self.Combobox_input.currentText() == "set max buffer (CPI)":
            N=int(self.LineEdit_input1D.text())
            if N>0:
                latest_device.MaxBufferSize = N
        elif self.Combobox_input.currentText() == "Range Percentage":
            N=int(self.LineEdit_input1D.text())
            if N>0:
                self.rangePercentage = N
        elif self.Combobox_input.currentText() == "slow time remove first index":
            N=int(self.LineEdit_input1D.text())
            if N>0:
                self.slowtimeremoveindex = N
        elif self.Combobox_input.currentText() == "toggle timer":
            if self.timer.isActive():
                self.timer.stop()
            else:
                self.timer.start(100)
    
    def connect_device(self):
        portname = self.LineEdit_port.text()
        if len(portname) == 0:
            return
        x4 = ssp.hardware.radar.XeThru.XeThruDevice(port=portname)
        x4.connect()
        self.devices_comm.append(["Xhetru",x4.connected,x4.port,x4])
        
    def update_plot(self):
        """
        Update the plot with the latest data from devices_comm.
        """
        QApplication.processEvents()
        if len(self.devices_comm) > 0:
                
            # Get the latest device communication object
            # latest_device = self.devices_comm[-1][3]
            latest_device: ssp.hardware.radar.XeThru.XeThruDevice = self.devices_comm[-1][3]
            devicedata = copy.deepcopy(latest_device.decoded)

            self.console.append(str(len(devicedata)))
            if len(devicedata) > 0:
                if self.Combobox_input2.currentText() == "fast time":
                    if latest_device.RF_mode == 0:
                        X=np.array(devicedata[-1][0])
                        x=np.real(X[:int(X.shape[0]*self.rangePercentage/100)])
                        y=np.imag(X[:int(X.shape[0]*self.rangePercentage/100)])
                        self.canvas.update_plot(x,y)
                    else:
                        x=(devicedata[-1][0])
                        self.canvas.update_plot(x,x)
                elif self.Combobox_input2.currentText() == "fast-slow":
                    # self.canvas.axes.relim()
                    # self.canvas.axes.autoscale_view()
                    self.canvas.axes.clear()
                    x=np.array([np.real(d) for d,time in devicedata])

                    x = x[:,:int(x.shape[1]*self.rangePercentage/100)]
                    b = [1, -1]
                    a = 1#[1, -0.9]
                    x = lfilter(b, a, x, axis=0)
                    x = x[self.slowtimeremoveindex:,:]

                    self.canvas.axes.imshow(x.T, aspect='auto', cmap='viridis')
                    self.canvas.draw()

                elif self.Combobox_input2.currentText() == "fast-doppler":
                    1
                elif self.Combobox_input2.currentText() == "phase -> fast0":
                    x=np.array([d for d,time in devicedata])
                    x = x[:,int(x.shape[1]*self.rangePercentage/100)]
                    x = np.angle(x)
                    x=np.unwrap(x)

                    self.canvas.update_plot(x,x)

                
                
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

