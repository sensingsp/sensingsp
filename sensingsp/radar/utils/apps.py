import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton, QHBoxLayout, QScrollArea, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
import json

import numpy as np
# from pyqtgraph.Qt import QtWidgets
# import pyqtgraph.opengl as gl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sensingsp as ssp

# class PyQtGraph3DApp(QtWidgets.QMainWindow):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data  # The input data (list of frames with vertices)
#         self.current_frame = 0  # Start at frame 0

#         # Set up the window
#         self.setWindowTitle("3D Mesh Visualization")
#         self.resize(800, 600)

#         # Create a GL view widget
#         self.gl_widget = gl.GLViewWidget()
#         self.setCentralWidget(self.gl_widget)
#         self.gl_widget.setCameraPosition(distance=10)

#         # Add a grid for reference
#         self.grid = gl.GLGridItem()
#         self.gl_widget.addItem(self.grid)

#         # Initialize the scatter plot for vertices
#         self.vertex_plot = gl.GLScatterPlotItem()
#         self.gl_widget.addItem(self.vertex_plot)

#         # Timer for frame updates
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(100)  # Update every 100 ms

#         # Initialize the first frame
#         self.update_frame()

#     def update_frame(self):
#         """Update the plot with the current frame's data."""
#         if self.current_frame >= len(self.data):
#             self.timer.stop()  # Stop the timer if all frames are displayed
#             return

#         # Extract vertices for the current frame
#         frame_data = self.data[self.current_frame]

#         # Flatten the vertex list and convert to a numpy array
#         vertices = np.array([v for face in frame_data for v in face])

#         # Update the scatter plot with the new vertices
#         self.vertex_plot.setData(pos=vertices, size=5, color=(1, 1, 1, 1))

#         # Move to the next frame
#         self.current_frame += 1


def pyqtgraph3DApp(input_x):
    """Visualize 3D data frame by frame using Matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # print("input:",input_x)
        
    for i, xi in enumerate(input_x):  # Iterate over frames
        ax.clear()  # Clear the previous frame's data
        # print(xi,len(xi))
        # print("_________________________________________________")
        for f in xi:  # Iterate over faces in the current frame
            xv, yv, zv = [], [], []
            for v in f:  # Iterate over vertices in the current face
                xv.append(v[0])  # Append X coordinate
                yv.append(v[1])  # Append Y coordinate
                zv.append(v[2])  # Append Z coordinate

            # Close the face by adding the first vertex to the end
            xv.append(f[0][0])
            yv.append(f[0][1])
            zv.append(f[0][2])

            # Plot the face in 3D
            ax.plot(xv, yv, zv)

        # Update the title with the frame number
        ax.set_title(f"Frame {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ssp.utils.set_axes_equal(ax)
        # Render the updated frame
        plt.draw()
        plt.pause(0.1)  # Pause for smooth animation
        plt.gcf().canvas.flush_events()

    plt.show()
    return
    app = QtWidgets.QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QtWidgets.QApplication(sys.argv)
    
    window = PyQtGraph3DApp(input_x)
    window.show()
    app.exec_()



def FMCW_Chirp_Parameters(rangeResolution, N_ADC, ChirpTimeMax, radialVelocityResolution, CentralFreq):
    results = []
    B = 299792458.0 / (2 * rangeResolution)
    results.append(f"FMCW Bandwidth (GHz) = {B / 1e9:.3f}")
    results.append(f"With {N_ADC} samples and {ChirpTimeMax * 1e6:.2f} us chirp ADC time, "
                   f"ADC Sampling rate is {N_ADC / ChirpTimeMax / 1e6:.2f} MSps")
    results.append(f"Chirp Slope = {B * 1e-6 / (ChirpTimeMax * 1e6):.2f} MHz/us")
    results.append(f"Max Range = {N_ADC * rangeResolution:.2f} m")
    WaveLength = 299792458.0 / CentralFreq
    CPI_Time = WaveLength / (2 * radialVelocityResolution)
    results.append(f"With radial Velocity Resolution {radialVelocityResolution} m/s, "
                   f"CPI Time = {CPI_Time * 1e3:.2f} msec")
    PRI = ChirpTimeMax
    results.append(f"Pulse Number in CPI < {int(CPI_Time / PRI)}")
    return results

class FMCWApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMCW Chirp Parameters Calculator")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Input form
        form_layout = QFormLayout()

        # Input fields with default values
        self.range_resolution_input = QLineEdit("0.039")
        self.n_adc_input = QLineEdit("256")
        self.chirp_time_input = QLineEdit("60e-6")
        self.radial_velocity_input = QLineEdit("0.13")
        self.central_freq_input = QLineEdit("76e9")

        form_layout.addRow("Range Resolution (m):", self.range_resolution_input)
        form_layout.addRow("Number of ADC Samples:", self.n_adc_input)
        form_layout.addRow("Chirp Time Max (s):", self.chirp_time_input)
        form_layout.addRow("Radial Velocity Resolution (m/s):", self.radial_velocity_input)
        form_layout.addRow("Central Frequency (Hz):", self.central_freq_input)

        main_layout.addLayout(form_layout)

        # Output display
        self.output_label = QLabel("Results will be shown here.")
        self.output_label.setAlignment(Qt.AlignTop)
        self.output_label.setWordWrap(True)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.output_label)
        main_layout.addWidget(scroll_area)

        # Buttons
        button_layout = QHBoxLayout()
        calculate_button = QPushButton("Calculate")
        calculate_button.clicked.connect(self.calculate_parameters)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_inputs)
        button_layout.addWidget(calculate_button)
        button_layout.addWidget(clear_button)
        main_layout.addLayout(button_layout)

    def calculate_parameters(self):
        try:
            # Read input values
            range_resolution = float(self.range_resolution_input.text())
            n_adc = int(self.n_adc_input.text())
            chirp_time_max = float(self.chirp_time_input.text())
            radial_velocity_resolution = float(self.radial_velocity_input.text())
            central_freq = float(self.central_freq_input.text())

            # Calculate parameters
            results = FMCW_Chirp_Parameters(range_resolution, n_adc, chirp_time_max,
                                            radial_velocity_resolution, central_freq)

            # Display results
            self.output_label.setText("\n".join(results))
        except ValueError:
            self.output_label.setText("Please enter valid numeric values in all fields.")

    def clear_inputs(self):
        # Reset fields to default values
        self.range_resolution_input.setText("0.039")
        self.n_adc_input.setText("256")
        self.chirp_time_input.setText("60e-6")
        self.radial_velocity_input.setText("0.13")
        self.central_freq_input.setText("76e9")
        self.output_label.setText("Results will be shown here.")

class RadarConfigApp(QMainWindow):
    def __init__(self):
        super(RadarConfigApp, self).__init__()

        # Setting up the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Radar Parameters (Not Impelemented Yet)")
        
        # Scroll Area Setup
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QFormLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        # Adding widgets for radar parameters in a form layout
        self.add_parameter_widgets()

        # Save and Load buttons in a horizontal layout
        self.button_layout = QHBoxLayout()
        self.saveButton = QPushButton("Save to JSON")
        self.saveButton.clicked.connect(self.save_to_json)
        self.button_layout.addWidget(self.saveButton)

        self.loadButton = QPushButton("Load from JSON")
        self.loadButton.clicked.connect(self.load_from_json)
        self.button_layout.addWidget(self.loadButton)

        self.scroll_layout.addRow(self.button_layout)

        # Set the scroll area as the central widget
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.scroll_area)

        # Example: Radar parameters
        self.radar_parameters = {
            "Transmit_Power_dBm": 12,
            "Transmit_Antenna_Element_Gain_db": 3,
            "Transmit_Antenna_Element_Azimuth_BeamWidth_deg": 120,
            "Transmit_Antenna_Element_Elevation_BeamWidth_deg": 120,
            "Receive_Antenna_Element_Gain_db": 0,
            "Receive_Antenna_Element_Azimuth_BeamWidth_deg": 120,
            "Receive_Antenna_Element_Elevation_BeamWidth_deg": 120,
            "Center_Frequency_GHz": 76.0,
            "PRI_us": 70,
            "Fs_MHz": 5,
            "NPulse": 3 * 64,
            "N_ADC": 256,
            "RangeWindow": "Hamming",
            "DopplerWindow": "Hamming",
            "FMCW_ChirpTime_us": 60,
            "FMCW_Bandwidth_GHz": 1,
            "Temperature_K": 290,
            "Range_Start": 0,
            "Range_End": 100,
            "CFAR_RD_guard_cells": 2,
            "CFAR_RD_training_cells": 10,
            "CFAR_RD_false_alarm_rate": 1e-3,
            "STC_Enabled": False,
            "MTI_Enabled": False,
            "DopplerFFT_OverNextP2": 3,
            "AzFFT_OverNextP2": 2,
            "ElFFT_OverNextP2": 3,
            "CFAR_Angle_guard_cells": 1,
            "CFAR_Angle_training_cells": 3,
            "CFAR_Angle_false_alarm_rate": 0.1,
            "CFAR_RD_alpha": 30,
            "CFAR_Angle_alpha": 5,
            "FMCW": True,
            "ADC_peak2peak": 2,
            "ADC_levels": 256,
            "ADC_ImpedanceFactor": 300,
            "ADC_LNA_Gain_dB": 50,
            "RF_NoiseFiguredB": 5,
            "RF_AnalogNoiseFilter_Bandwidth_MHz": 10,
            "ADC_SaturationEnabled": False,
            "RadarMode": "FMCW",
            "PulseWaveform": "WaveformFile.txt",
            "t_start_radar": 0,
            "MaxRangeScatter": 1e12,
            "SaveSignalGenerationTime": True,
            "continuousCPIsTrue_oneCPIpeerFrameFalse": False
        }

    def add_parameter_widgets(self):
        # Adding widgets using a form layout for better alignment
        self.transmitPowerSpinBox = QSpinBox()
        self.transmitPowerSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("Transmit Power (dBm):", self.transmitPowerSpinBox)

        self.transmitAntennaGainSpinBox = QSpinBox()
        self.transmitAntennaGainSpinBox.setRange(0, 50)
        self.scroll_layout.addRow("Transmit Antenna Gain (dB):", self.transmitAntennaGainSpinBox)

        self.transmitAzimuthBeamWidthSpinBox = QSpinBox()
        self.transmitAzimuthBeamWidthSpinBox.setRange(0, 360)
        self.scroll_layout.addRow("Transmit Azimuth BeamWidth (deg):", self.transmitAzimuthBeamWidthSpinBox)

        self.transmitElevationBeamWidthSpinBox = QSpinBox()
        self.transmitElevationBeamWidthSpinBox.setRange(0, 360)
        self.scroll_layout.addRow("Transmit Elevation BeamWidth (deg):", self.transmitElevationBeamWidthSpinBox)

        self.receiveAntennaGainSpinBox = QSpinBox()
        self.receiveAntennaGainSpinBox.setRange(0, 50)
        self.scroll_layout.addRow("Receive Antenna Gain (dB):", self.receiveAntennaGainSpinBox)

        self.receiveAzimuthBeamWidthSpinBox = QSpinBox()
        self.receiveAzimuthBeamWidthSpinBox.setRange(0, 360)
        self.scroll_layout.addRow("Receive Azimuth BeamWidth (deg):", self.receiveAzimuthBeamWidthSpinBox)

        self.receiveElevationBeamWidthSpinBox = QSpinBox()
        self.receiveElevationBeamWidthSpinBox.setRange(0, 360)
        self.scroll_layout.addRow("Receive Elevation BeamWidth (deg):", self.receiveElevationBeamWidthSpinBox)

        self.centerFrequencySpinBox = QDoubleSpinBox()
        self.centerFrequencySpinBox.setRange(0, 100)
        self.centerFrequencySpinBox.setDecimals(2)
        self.scroll_layout.addRow("Center Frequency (GHz):", self.centerFrequencySpinBox)

        self.priSpinBox = QDoubleSpinBox()
        self.priSpinBox.setRange(0, 10000)
        self.scroll_layout.addRow("PRI (us):", self.priSpinBox)

        self.fsSpinBox = QDoubleSpinBox()
        self.fsSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("Sampling Frequency (MHz):", self.fsSpinBox)

        self.nPulseSpinBox = QSpinBox()
        self.nPulseSpinBox.setRange(0, 10000)
        self.scroll_layout.addRow("Number of Pulses:", self.nPulseSpinBox)

        self.nADCSpinBox = QSpinBox()
        self.nADCSpinBox.setRange(0, 10000)
        self.scroll_layout.addRow("Number of ADC Samples:", self.nADCSpinBox)

        self.rangeWindowComboBox = QComboBox()
        self.rangeWindowComboBox.addItems(["Hamming", "Hann", "Rectangular"])
        self.scroll_layout.addRow("Range Window:", self.rangeWindowComboBox)

        self.dopplerWindowComboBox = QComboBox()
        self.dopplerWindowComboBox.addItems(["Hamming", "Hann", "Rectangular"])
        self.scroll_layout.addRow("Doppler Window:", self.dopplerWindowComboBox)

        self.chirpTimeSpinBox = QDoubleSpinBox()
        self.chirpTimeSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("Chirp Time (us):", self.chirpTimeSpinBox)

        self.bandwidthSpinBox = QDoubleSpinBox()
        self.bandwidthSpinBox.setRange(0, 10)
        self.bandwidthSpinBox.setDecimals(2)
        self.scroll_layout.addRow("Bandwidth (GHz):", self.bandwidthSpinBox)

        self.temperatureSpinBox = QDoubleSpinBox()
        self.temperatureSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("Temperature (K):", self.temperatureSpinBox)

        self.rangeStartSpinBox = QDoubleSpinBox()
        self.rangeStartSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("Range Start (m):", self.rangeStartSpinBox)

        self.rangeEndSpinBox = QDoubleSpinBox()
        self.rangeEndSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("Range End (m):", self.rangeEndSpinBox)

        self.cfarRdGuardCellsSpinBox = QSpinBox()
        self.cfarRdGuardCellsSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("CFAR RD Guard Cells:", self.cfarRdGuardCellsSpinBox)

        self.cfarRdTrainingCellsSpinBox = QSpinBox()
        self.cfarRdTrainingCellsSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("CFAR RD Training Cells:", self.cfarRdTrainingCellsSpinBox)

        self.cfarRdFalseAlarmRateSpinBox = QDoubleSpinBox()
        self.cfarRdFalseAlarmRateSpinBox.setRange(0, 1)
        self.cfarRdFalseAlarmRateSpinBox.setDecimals(6)
        self.scroll_layout.addRow("CFAR RD False Alarm Rate:", self.cfarRdFalseAlarmRateSpinBox)

        self.stcEnabledCheckBox = QCheckBox("STC Enabled")
        self.scroll_layout.addRow(self.stcEnabledCheckBox)

        self.mtiEnabledCheckBox = QCheckBox("MTI Enabled")
        self.scroll_layout.addRow(self.mtiEnabledCheckBox)

        self.dopplerFftSpinBox = QSpinBox()
        self.dopplerFftSpinBox.setRange(0, 10)
        self.scroll_layout.addRow("Doppler FFT Over Next Power of 2:", self.dopplerFftSpinBox)

        self.azFftSpinBox = QSpinBox()
        self.azFftSpinBox.setRange(0, 10)
        self.scroll_layout.addRow("Azimuth FFT Over Next Power of 2:", self.azFftSpinBox)

        self.elFftSpinBox = QSpinBox()
        self.elFftSpinBox.setRange(0, 10)
        self.scroll_layout.addRow("Elevation FFT Over Next Power of 2:", self.elFftSpinBox)

        self.cfarAngleGuardCellsSpinBox = QSpinBox()
        self.cfarAngleGuardCellsSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("CFAR Angle Guard Cells:", self.cfarAngleGuardCellsSpinBox)

        self.cfarAngleTrainingCellsSpinBox = QSpinBox()
        self.cfarAngleTrainingCellsSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("CFAR Angle Training Cells:", self.cfarAngleTrainingCellsSpinBox)

        self.cfarAngleFalseAlarmRateSpinBox = QDoubleSpinBox()
        self.cfarAngleFalseAlarmRateSpinBox.setRange(0, 1)
        self.cfarAngleFalseAlarmRateSpinBox.setDecimals(6)
        self.scroll_layout.addRow("CFAR Angle False Alarm Rate:", self.cfarAngleFalseAlarmRateSpinBox)

        self.cfarRdAlphaSpinBox = QDoubleSpinBox()
        self.cfarRdAlphaSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("CFAR RD Alpha:", self.cfarRdAlphaSpinBox)

        self.cfarAngleAlphaSpinBox = QDoubleSpinBox()
        self.cfarAngleAlphaSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("CFAR Angle Alpha:", self.cfarAngleAlphaSpinBox)

        self.fmcwCheckBox = QCheckBox("FMCW Enabled")
        self.scroll_layout.addRow(self.fmcwCheckBox)

        self.adcPeakToPeakSpinBox = QDoubleSpinBox()
        self.adcPeakToPeakSpinBox.setRange(0, 10)
        self.scroll_layout.addRow("ADC Peak to Peak Voltage:", self.adcPeakToPeakSpinBox)

        self.adcLevelsSpinBox = QSpinBox()
        self.adcLevelsSpinBox.setRange(0, 4096)
        self.scroll_layout.addRow("ADC Levels:", self.adcLevelsSpinBox)

        self.adcImpedanceFactorSpinBox = QDoubleSpinBox()
        self.adcImpedanceFactorSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("ADC Impedance Factor:", self.adcImpedanceFactorSpinBox)

        self.adcLnaGainSpinBox = QDoubleSpinBox()
        self.adcLnaGainSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("ADC LNA Gain (dB):", self.adcLnaGainSpinBox)

        self.rfNoiseFigureSpinBox = QDoubleSpinBox()
        self.rfNoiseFigureSpinBox.setRange(0, 100)
        self.scroll_layout.addRow("RF Noise Figure (dB):", self.rfNoiseFigureSpinBox)

        self.rfAnalogNoiseFilterSpinBox = QDoubleSpinBox()
        self.rfAnalogNoiseFilterSpinBox.setRange(0, 1000)
        self.scroll_layout.addRow("RF Analog Noise Filter Bandwidth (MHz):", self.rfAnalogNoiseFilterSpinBox)

        self.adcSaturationCheckBox = QCheckBox("ADC Saturation Enabled")
        self.scroll_layout.addRow(self.adcSaturationCheckBox)

        self.radarModeComboBox = QComboBox()
        self.radarModeComboBox.addItems(["FMCW", "Pulse", "CW"])
        self.scroll_layout.addRow("Radar Mode:", self.radarModeComboBox)

        self.pulseWaveformLineEdit = QLineEdit()
        self.scroll_layout.addRow("Pulse Waveform File:", self.pulseWaveformLineEdit)

        self.tStartRadarSpinBox = QDoubleSpinBox()
        self.tStartRadarSpinBox.setRange(0, 10000)
        self.scroll_layout.addRow("Radar Start Time (s):", self.tStartRadarSpinBox)

        self.maxRangeScatterSpinBox = QDoubleSpinBox()
        self.maxRangeScatterSpinBox.setRange(0, 1e12)
        self.scroll_layout.addRow("Max Range Scatter:", self.maxRangeScatterSpinBox)

        self.saveSignalGenerationTimeCheckBox = QCheckBox("Save Signal Generation Time")
        self.scroll_layout.addRow(self.saveSignalGenerationTimeCheckBox)

        self.continuousCpiCheckBox = QCheckBox("Continuous CPI")
        self.scroll_layout.addRow(self.continuousCpiCheckBox)

    def create_labeled_widget(self, label_text, widget):
        """Utility function to create a labeled widget."""
        container = QWidget()
        layout = QHBoxLayout()
        label = QLabel(label_text)
        layout.addWidget(label)
        layout.addWidget(widget)
        container.setLayout(layout)
        return container

    def save_to_json(self):
        # Get filename to save JSON
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            try:
                # Get updated radar parameters from the UI
                self.update_parameters_from_ui()
                # Write the parameters to JSON file
                with open(fileName, 'w') as json_file:
                    json.dump(self.radar_parameters, json_file, indent=4)
                QMessageBox.information(self, "Success", "Radar parameters saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def load_from_json(self):
        # Get filename to load JSON
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            try:
                # Read the parameters from JSON file
                with open(fileName, 'r') as json_file:
                    self.radar_parameters = json.load(json_file)
                # Update UI with the loaded parameters
                self.update_ui_from_parameters()
                QMessageBox.information(self, "Success", "Radar parameters loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def update_parameters_from_ui(self):
        # Update parameters from UI elements
        self.radar_parameters["Transmit_Power_dBm"] = int(self.transmitPowerSpinBox.value())
        self.radar_parameters["Transmit_Antenna_Element_Gain_db"] = int(self.transmitAntennaGainSpinBox.value())
        self.radar_parameters["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = int(self.transmitAzimuthBeamWidthSpinBox.value())
        self.radar_parameters["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = int(self.transmitElevationBeamWidthSpinBox.value())
        self.radar_parameters["Receive_Antenna_Element_Gain_db"] = int(self.receiveAntennaGainSpinBox.value())
        self.radar_parameters["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = int(self.receiveAzimuthBeamWidthSpinBox.value())
        self.radar_parameters["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = int(self.receiveElevationBeamWidthSpinBox.value())
        self.radar_parameters["Center_Frequency_GHz"] = float(self.centerFrequencySpinBox.value())
        self.radar_parameters["PRI_us"] = float(self.priSpinBox.value())
        self.radar_parameters["Fs_MHz"] = float(self.fsSpinBox.value())
        self.radar_parameters["NPulse"] = int(self.nPulseSpinBox.value())
        self.radar_parameters["N_ADC"] = int(self.nADCSpinBox.value())
        self.radar_parameters["RangeWindow"] = self.rangeWindowComboBox.currentText()
        self.radar_parameters["DopplerWindow"] = self.dopplerWindowComboBox.currentText()
        self.radar_parameters["FMCW_ChirpTime_us"] = float(self.chirpTimeSpinBox.value())
        self.radar_parameters["FMCW_Bandwidth_GHz"] = float(self.bandwidthSpinBox.value())
        self.radar_parameters["Temperature_K"] = float(self.temperatureSpinBox.value())
        self.radar_parameters["Range_Start"] = float(self.rangeStartSpinBox.value())
        self.radar_parameters["Range_End"] = float(self.rangeEndSpinBox.value())
        self.radar_parameters["CFAR_RD_guard_cells"] = int(self.cfarRdGuardCellsSpinBox.value())
        self.radar_parameters["CFAR_RD_training_cells"] = int(self.cfarRdTrainingCellsSpinBox.value())
        self.radar_parameters["CFAR_RD_false_alarm_rate"] = float(self.cfarRdFalseAlarmRateSpinBox.value())
        self.radar_parameters["STC_Enabled"] = self.stcEnabledCheckBox.isChecked()
        self.radar_parameters["MTI_Enabled"] = self.mtiEnabledCheckBox.isChecked()
        self.radar_parameters["DopplerFFT_OverNextP2"] = int(self.dopplerFftSpinBox.value())
        self.radar_parameters["AzFFT_OverNextP2"] = int(self.azFftSpinBox.value())
        self.radar_parameters["ElFFT_OverNextP2"] = int(self.elFftSpinBox.value())
        self.radar_parameters["CFAR_Angle_guard_cells"] = int(self.cfarAngleGuardCellsSpinBox.value())
        self.radar_parameters["CFAR_Angle_training_cells"] = int(self.cfarAngleTrainingCellsSpinBox.value())
        self.radar_parameters["CFAR_Angle_false_alarm_rate"] = float(self.cfarAngleFalseAlarmRateSpinBox.value())
        self.radar_parameters["CFAR_RD_alpha"] = float(self.cfarRdAlphaSpinBox.value())
        self.radar_parameters["CFAR_Angle_alpha"] = float(self.cfarAngleAlphaSpinBox.value())
        self.radar_parameters["FMCW"] = self.fmcwCheckBox.isChecked()
        self.radar_parameters["ADC_peak2peak"] = float(self.adcPeakToPeakSpinBox.value())
        self.radar_parameters["ADC_levels"] = int(self.adcLevelsSpinBox.value())
        self.radar_parameters["ADC_ImpedanceFactor"] = float(self.adcImpedanceFactorSpinBox.value())
        self.radar_parameters["ADC_LNA_Gain_dB"] = float(self.adcLnaGainSpinBox.value())
        self.radar_parameters["RF_NoiseFiguredB"] = float(self.rfNoiseFigureSpinBox.value())
        self.radar_parameters["RF_AnalogNoiseFilter_Bandwidth_MHz"] = float(self.rfAnalogNoiseFilterSpinBox.value())
        self.radar_parameters["ADC_SaturationEnabled"] = self.adcSaturationCheckBox.isChecked()
        self.radar_parameters["RadarMode"] = self.radarModeComboBox.currentText()
        self.radar_parameters["PulseWaveform"] = self.pulseWaveformLineEdit.text()
        self.radar_parameters["t_start_radar"] = float(self.tStartRadarSpinBox.value())
        self.radar_parameters["MaxRangeScatter"] = float(self.maxRangeScatterSpinBox.value())
        self.radar_parameters["SaveSignalGenerationTime"] = self.saveSignalGenerationTimeCheckBox.isChecked()
        self.radar_parameters["continuousCPIsTrue_oneCPIpeerFrameFalse"] = self.continuousCpiCheckBox.isChecked()

    def update_ui_from_parameters(self):
        # Update UI elements from parameters
        self.transmitPowerSpinBox.setValue(self.radar_parameters["Transmit_Power_dBm"])
        self.transmitAntennaGainSpinBox.setValue(self.radar_parameters["Transmit_Antenna_Element_Gain_db"])
        self.transmitAzimuthBeamWidthSpinBox.setValue(self.radar_parameters["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"])
        self.transmitElevationBeamWidthSpinBox.setValue(self.radar_parameters["Transmit_Antenna_Element_Elevation_BeamWidth_deg"])
        self.receiveAntennaGainSpinBox.setValue(self.radar_parameters["Receive_Antenna_Element_Gain_db"])
        self.receiveAzimuthBeamWidthSpinBox.setValue(self.radar_parameters["Receive_Antenna_Element_Azimuth_BeamWidth_deg"])
        self.receiveElevationBeamWidthSpinBox.setValue(self.radar_parameters["Receive_Antenna_Element_Elevation_BeamWidth_deg"])
        self.centerFrequencySpinBox.setValue(self.radar_parameters["Center_Frequency_GHz"])
        self.priSpinBox.setValue(self.radar_parameters["PRI_us"])
        self.fsSpinBox.setValue(self.radar_parameters["Fs_MHz"])
        self.nPulseSpinBox.setValue(self.radar_parameters["NPulse"])
        self.nADCSpinBox.setValue(self.radar_parameters["N_ADC"])
        self.rangeWindowComboBox.setCurrentText(self.radar_parameters["RangeWindow"])
        self.dopplerWindowComboBox.setCurrentText(self.radar_parameters["DopplerWindow"])
        self.chirpTimeSpinBox.setValue(self.radar_parameters["FMCW_ChirpTime_us"])
        self.bandwidthSpinBox.setValue(self.radar_parameters["FMCW_Bandwidth_GHz"])
        self.temperatureSpinBox.setValue(self.radar_parameters["Temperature_K"])
        self.rangeStartSpinBox.setValue(self.radar_parameters["Range_Start"])
        self.rangeEndSpinBox.setValue(self.radar_parameters["Range_End"])
        self.cfarRdGuardCellsSpinBox.setValue(self.radar_parameters["CFAR_RD_guard_cells"])
        self.cfarRdTrainingCellsSpinBox.setValue(self.radar_parameters["CFAR_RD_training_cells"])
        self.cfarRdFalseAlarmRateSpinBox.setValue(self.radar_parameters["CFAR_RD_false_alarm_rate"])
        self.stcEnabledCheckBox.setChecked(self.radar_parameters["STC_Enabled"])
        self.mtiEnabledCheckBox.setChecked(self.radar_parameters["MTI_Enabled"])
        self.dopplerFftSpinBox.setValue(self.radar_parameters["DopplerFFT_OverNextP2"])
        self.azFftSpinBox.setValue(self.radar_parameters["AzFFT_OverNextP2"])
        self.elFftSpinBox.setValue(self.radar_parameters["ElFFT_OverNextP2"])
        self.cfarAngleGuardCellsSpinBox.setValue(self.radar_parameters["CFAR_Angle_guard_cells"])
        self.cfarAngleTrainingCellsSpinBox.setValue(self.radar_parameters["CFAR_Angle_training_cells"])
        self.cfarAngleFalseAlarmRateSpinBox.setValue(self.radar_parameters["CFAR_Angle_false_alarm_rate"])
        self.cfarRdAlphaSpinBox.setValue(self.radar_parameters["CFAR_RD_alpha"])
        self.cfarAngleAlphaSpinBox.setValue(self.radar_parameters["CFAR_Angle_alpha"])
        self.fmcwCheckBox.setChecked(self.radar_parameters["FMCW"])
        self.adcPeakToPeakSpinBox.setValue(self.radar_parameters["ADC_peak2peak"])
        self.adcLevelsSpinBox.setValue(self.radar_parameters["ADC_levels"])
        self.adcImpedanceFactorSpinBox.setValue(self.radar_parameters["ADC_ImpedanceFactor"])
        self.adcLnaGainSpinBox.setValue(self.radar_parameters["ADC_LNA_Gain_dB"])
        self.rfNoiseFigureSpinBox.setValue(self.radar_parameters["RF_NoiseFiguredB"])
        self.rfAnalogNoiseFilterSpinBox.setValue(self.radar_parameters["RF_AnalogNoiseFilter_Bandwidth_MHz"])
        self.adcSaturationCheckBox.setChecked(self.radar_parameters["ADC_SaturationEnabled"])
        self.radarModeComboBox.setCurrentText(self.radar_parameters["RadarMode"])
        self.pulseWaveformLineEdit.setText(self.radar_parameters["PulseWaveform"])
        self.tStartRadarSpinBox.setValue(self.radar_parameters["t_start_radar"])
        self.maxRangeScatterSpinBox.setValue(self.radar_parameters["MaxRangeScatter"])
        self.saveSignalGenerationTimeCheckBox.setChecked(self.radar_parameters["SaveSignalGenerationTime"])
        self.continuousCpiCheckBox.setChecked(self.radar_parameters["continuousCPIsTrue_oneCPIpeerFrameFalse"])

def appselect(st):
    if st=="FMCW Chirp Parameters Calculator":
        runfmcwchirpapp()
    if st=="Radar Parameters":
        runradarconfigapp()       
    if st=="Hand Gesture MisoCNN":
        ssp.ai.radarML.HandGestureMisoCNN.runradarmisoCNNapp() 
    if st=="Human Health Monitoring":
        ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.runradarConvAEBiLSTMapp() 
    if st=="GAN Radar Waveforms":
        ssp.ai.radarML.GANWaveforms.runradarWaveformapp()
        # ssp.ai.radarML.H
        
        
def runfmcwchirpapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = FMCWApp()
    window.show()
    app.exec_()  # Do not use sys.exit()

def runradarconfigapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = RadarConfigApp()
    window.show()
    app.exec_()  # Do not use sys.exit()




