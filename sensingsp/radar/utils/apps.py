import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton, QHBoxLayout, QScrollArea, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen,QTransform
from PyQt5.QtWidgets import (
    QPushButton, QTableWidget, QTableWidgetItem, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QSizePolicy,QGraphicsLineItem
)
from PyQt5.QtCore import QRectF
import json

import numpy as np
# from pyqtgraph.Qt import QtWidgets
# import pyqtgraph.opengl as gl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import os
import sensingsp as ssp

# import numpy as np
# from PyQt5.QtWidgets import (
#     QMainWindow, QApplication, QWidget, QVBoxLayout,
#     QPushButton, QFileDialog, QLabel, QHBoxLayout
# )
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# from mpl_toolkits.mplot3d import Axes3D
# import sys

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


class RISAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIS Analysis")
        self.setGeometry(100, 100, 1000, 800)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.vbox = QVBoxLayout(main_widget)
        
        self.linedit = QLineEdit()
        self.linedit.setText(os.path.join(ssp.config.temp_folder, "RIS_data.mat"))
        self.vbox.addWidget(self.linedit)
        
        self.label = QLabel("Freq (GHz):")
        self.vbox.addWidget(self.label)
        self.f0_lineedit = QLineEdit()
        self.f0_lineedit.setText("62")
        self.vbox.addWidget(self.f0_lineedit)
        self.label2 = QLabel("Nx,Ny,dx,dy")
        self.vbox.addWidget(self.label2)
        self.n_lineedit = QLineEdit()
        self.n_lineedit.setText("100,100,0.5,0.5")
        self.vbox.addWidget(self.n_lineedit)
        self.labelradarpos = QLabel("Radar Position (x,y,z):")
        self.vbox.addWidget(self.labelradarpos)
        self.radarpos_lineedit = QLineEdit()
        self.radarpos_lineedit.setText("-0.5,0,-.5")
        self.vbox.addWidget(self.radarpos_lineedit)
        
        self.labeltargetpos = QLabel("Target Position (x,y,z):")    
        self.vbox.addWidget(self.labeltargetpos)
        self.targetpos_lineedit = QLineEdit()
        self.targetpos_lineedit.setText("0.5,0,-0.5")
        self.vbox.addWidget(self.targetpos_lineedit)
        
        self.generate_button = QPushButton("Generate RIS Scenario 1")
        self.generate_button.clicked.connect(self.generate_ris_data)
        self.vbox.addWidget(self.generate_button)
        self.generate_button2 = QPushButton("Generate RIS Scenario 2")
        self.generate_button2.clicked.connect(self.generate_ris_data2)
        self.vbox.addWidget(self.generate_button2)
        
        self.vbox.addStretch()
        
        # self.generate_button3 = QPushButton("Generate RIS Scenario 3")
        # self.generate_button3.clicked.connect(self.generate_ris_data3)
        # self.vbox.addWidget(self.generate_button3)

        # self.load_button = QPushButton("Load RIS, Radar, and Target Positions")
        # self.load_button.clicked.connect(self.load_data)
        # self.vbox.addWidget(self.load_button)

        self.label = QLabel("N Grid")
        self.vbox.addWidget(self.label)
        self.ng_lineedit = QLineEdit()
        self.ng_lineedit.setText("40")
        self.vbox.addWidget(self.ng_lineedit)
        self.label = QLabel("d(Wavelength),Arc(deg) Grid")
        self.vbox.addWidget(self.label)
        self.arc_lineedit = QLineEdit()
        self.arc_lineedit.setText("2,40")
        self.vbox.addWidget(self.arc_lineedit)
        self.label = QLabel("Dynamic Range (dB)")
        self.vbox.addWidget(self.label)
        self.dynr_lineedit = QLineEdit()
        self.dynr_lineedit.setText("100")
        self.vbox.addWidget(self.dynr_lineedit)
        
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_analysis)
        self.vbox.addWidget(self.run_button)
        self.vbox.addStretch()
        self.blender_button = QPushButton("Blender scenario BR 18")
        self.blender_button.clicked.connect(self.run_blender_button)
        self.vbox.addWidget(self.blender_button)
        self.blender_button2 = QPushButton("Blender scenario BR 18+HR 70")
        self.blender_button2.clicked.connect(self.run_blender_button2)
        self.vbox.addWidget(self.blender_button2)
        self.blenderpath_button = QPushButton("Path sim")
        self.blenderpath_button.clicked.connect(self.run_blenderpath_button)
        self.vbox.addWidget(self.blenderpath_button)

        # self.status_label = QLabel("Please generate or load data.")
        # self.vbox.addWidget(self.status_label)

        # self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        # self.vbox.addWidget(self.canvas)
        # self.ax = self.canvas.figure.add_subplot(111, projection='3d')

    def generate_ris_data(self):
        f0 = self.f0_lineedit.text()
        f0 = float(f0)
        f0 = f0 * 1e9
        Lambda = 3e8 / f0
        nxnyLxLy = self.n_lineedit.text().split(",")
        if len(nxnyLxLy) != 4:
            QMessageBox.warning(self, "Input Error", "Please enter Nx, Ny, dx, dy in the format: Nx,Ny,dx,dy")
            return
        radarposLineedit = self.radarpos_lineedit.text().split(",")
        if len(radarposLineedit) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter Radar Position (x,y,z) in the format: x,y,z")
            return
        radar_pos = np.array(radarposLineedit, dtype=float)
        # target_pos = np.array([.5, 0, .5])
        targetposLineedit = self.targetpos_lineedit.text().split(",")
        if len(targetposLineedit) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter Target Position (x,y,z) in the format: x,y,z")
            return 
        target_pos = np.array(targetposLineedit, dtype=float)
        # target_pos = np.array([.5, 0, .5])

        pos = []
        Nx, Ny = int(nxnyLxLy[0]), int(nxnyLxLy[1])
        dx, dy = float(nxnyLxLy[2])*Lambda, float(nxnyLxLy[3])*Lambda
        for i in range(Nx):
            for j in range(Ny):
                x = (i-Nx/2) * dx
                y = (j-Ny/2) * dy
                z = 0
                pos.append((x, y, z))
        pos = np.array(pos)
        N=0
        d1 = np.linalg.norm(radar_pos - pos[N])
        d2 = np.linalg.norm(target_pos - pos[N])
        d0 = d1 + d2
        d1 = np.linalg.norm(radar_pos - pos, axis=1)
        d2 = np.linalg.norm(target_pos - pos, axis=1)
        di = d1 + d2
        d = di - d0
        phase = 2 * np.pi / Lambda * d
        a = 1 * np.exp(1j * phase)
        
        
        scipy.io.savemat(self.linedit.text(), {'pos': pos,'a': a, 'radar_pos': radar_pos, 'target_pos': target_pos})
        # Plotting
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot radar and target
        ax.plot([radar_pos[0]], [radar_pos[1]], [radar_pos[2]], 'ro', markersize=10, label='Radar Position')
        ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'bo', markersize=10, label='Target Position')

        # Plot antennas
        ax.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]), c='g', s=10, label='Antenna Elements')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        ax.set_title('Radar, Target, and Antenna Elements')
        # Equal axis function
        self.set_axes_equal(ax)

        phases = np.degrees(np.angle(a))

        # Plot antennas colored by phase
        fig2 = plt.figure(figsize=(10,10))
        ax2 = fig2.add_subplot(111, projection='3d')
        # scatter with phase colormap
        sc = ax2.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]),
                        c=phases, cmap='hsv', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        cbar.set_label('Phase (radians)')

        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_zlabel('Z [m]')
        ax2.set_title('Antenna Element Phases')
        ax2.legend()
        self.set_axes_equal(ax2)

        plt.show()
    def generate_ris_data2(self):
        f0 = self.f0_lineedit.text()
        f0 = float(f0)
        f0 = f0 * 1e9
        Lambda = 3e8 / f0
        nxnyLxLy = self.n_lineedit.text().split(",")
        if len(nxnyLxLy) != 4:
            QMessageBox.warning(self, "Input Error", "Please enter Nx, Ny, dx, dy in the format: Nx,Ny,dx,dy")
            return
        radarposLineedit = self.radarpos_lineedit.text().split(",")
        if len(radarposLineedit) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter Radar Position (x,y,z) in the format: x,y,z")
            return
        radar_pos = np.array(radarposLineedit, dtype=float)
        # target_pos = np.array([.5, 0, .5])
        targetposLineedit = self.targetpos_lineedit.text().split(",")
        if len(targetposLineedit) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter Target Position (x,y,z) in the format: x,y,z")
            return 
        target_pos = np.array(targetposLineedit, dtype=float)
        # target_pos = np.array([.5, 0, .5])

        pos = []
        Nx, Ny = int(nxnyLxLy[0]), int(nxnyLxLy[1])
        dx, dy = float(nxnyLxLy[2])*Lambda, float(nxnyLxLy[3])*Lambda
        for i in range(Nx):
            for j in range(Ny):
                x = (i-Nx/2) * dx
                y = (j-Ny/2) * dy
                z = 0
                pos.append((x, y, z))
        pos = np.array(pos)
        N=0
        d1 = np.linalg.norm(radar_pos - pos[N])
        d2 = np.linalg.norm(target_pos - pos[N])
        d0 = d1 + d2
        d1 = np.linalg.norm(radar_pos - pos, axis=1)
        d2 = np.linalg.norm(target_pos - pos, axis=1)
        di = d1 + d2
        d = di - d0
        phase = 2 * np.pi / Lambda * d
        phase = phase * 0
        a = 1 * np.exp(1j * phase)
        
        
        scipy.io.savemat(self.linedit.text(), {'pos': pos,'a': a, 'radar_pos': radar_pos, 'target_pos': target_pos})
        # Plotting
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot radar and target
        ax.plot([radar_pos[0]], [radar_pos[1]], [radar_pos[2]], 'ro', markersize=10, label='Radar Position')
        ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'bo', markersize=10, label='Target Position')

        # Plot antennas
        ax.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]), c='g', s=10, label='Antenna Elements')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        ax.set_title('Radar, Target, and Antenna Elements')
        # Equal axis function
        self.set_axes_equal(ax)

        phases = np.degrees(np.angle(a))

        # Plot antennas colored by phase
        fig2 = plt.figure(figsize=(10,10))
        ax2 = fig2.add_subplot(111, projection='3d')
        # scatter with phase colormap
        sc = ax2.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]),
                        c=phases, cmap='hsv', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        cbar.set_label('Phase (radians)')

        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_zlabel('Z [m]')
        ax2.set_title('Antenna Element Phases')
        ax2.legend()
        self.set_axes_equal(ax2)

        plt.show()
    def run_blenderpath_button(self):
        all = []
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))
        all_signals = []
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            
            all.append( path_d_drate_amp[0][0][0][0][0][0] )
            N=0
            for i, a in enumerate(all):
                if len(a)>N:
                    N = len(a)
            dva_path_frame = np.zeros((len(all),N, 3))
            for i, a in enumerate(all):
                for p,ap in enumerate(a):
                    dva_path_frame[i,p,:] = ap[:3]
            axs[0,0].cla()
            axs[0,0].plot(dva_path_frame[:, :, 0])
            axs[0,0].set_title("distance")
            axs[0,0].set_ylabel("Distance (m)")
            axs[0,0].set_xlabel("Frame")
            axs[0,1].cla()
            axs[0,1].plot(dva_path_frame[:, :, 1])
            axs[0,1].set_title("Doppler")
            axs[0,1].set_ylabel("Distance Rate")
            axs[0,1].set_xlabel("Frame")
            axs[1,0].cla()
            axs[1,0].plot(20*np.log10(np.abs(dva_path_frame[:, :, 2])))
            axs[1,0].set_title("Amplitude")
            axs[1,0].set_ylabel("Amplitude (dB)")
            axs[1,0].set_xlabel("Frame")
            
            Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
            for XRadar, timeX in Signals[0]['radars'][0]:
                fast_time_window = ssp.radar.utils.hamming(XRadar.shape[0])
                X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]
                NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0])) + 1))
                d_fft = (np.arange(NFFT_Range) * ssp.LightSpeed / 2 /
                         ssp.RadarSpecifications[0][0]['FMCW_ChirpSlobe'] / NFFT_Range / ssp.RadarSpecifications[0][0]['Ts'])
                X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range) / XRadar.shape[0]
                axs[1,1].cla()
                amplitude_db = 20 * np.log10(np.abs(X_fft_fast[:, 0, 0]))
                axs[1,1].plot(d_fft, amplitude_db)
                axs[1,1].set_title("Range FFT")
                axs[1,1].set_ylabel("Amplitude (dB)")
                axs[1,1].set_xlabel("Range (m)")
                axs[1,1].set_xlim(0, 5)
                range_target = 1.55
                idx = np.argmin(np.abs(d_fft - range_target))
                value_at_1_8 = amplitude_db[idx]
                axs[1,1].vlines(range_target, value_at_1_8, -30, color='red', linestyle='--', linewidth=2)
                all_signals.append([X_fft_fast[idx, 0, 0], timeX[0]])
                sig = np.array(all_signals)
                uw = np.unwrap(np.angle(sig[:,0]))
                
                axs[0,2].cla()
                axs[0,2].plot(np.real(sig[:,1]), uw)
                axs[0,2].set_title("Unwrapped Phase")
                axs[0,2].set_ylabel("Phase (rad)")
                axs[0,2].set_xlabel("Time (s)")
                axs[1,2].cla()
                T = np.mean(np.diff(np.real(sig[:,1])))
                NFFT = int(2 ** (3+np.ceil(np.log2(len(uw)))))
                fftuw = np.fft.fft(uw, n=NFFT) / len(uw)
                uwdc = uw - np.mean(uw)
                fftuwdc = np.fft.fft(uwdc, n=NFFT) / len(uwdc)
                N = int(len(fftuw)/2)
                frpm = np.arange(N) / NFFT / T * 60
                axs[1,2].plot(frpm[1:N], 20*np.log10(np.abs(fftuw[1:N])))
                axs[1,2].plot(frpm[1:N], 20*np.log10(np.abs(fftuwdc[1:N])), linestyle='--')
                axs[1,2].set_title("FFT of Unwrapped Phase")
                axs[1,2].set_ylabel("Amplitude (dB)")
                axs[1,2].set_xlabel("Frequency (RPM)")
                
                
            
                
                
            
                        
            
            
            for ax in axs.flat:
                ax.grid()
            plt.draw()
            plt.pause(0.1)
            print(f'Processed frame = {ssp.config.CurrentFrame}')
            ssp.utils.increaseCurrentFrame()
        plt.show()
    def run_blender_button(self):
        ssp.environment.scenarios.ris_analysis_app_scenario()
    def run_blender_button2(self):
        ssp.environment.scenarios.ris_analysis_app_scenario2()
    def run_analysis(self):
        f0 = self.f0_lineedit.text()
        f0 = float(f0)
        f0 = f0 * 1e9
        Lambda = 3e8 / f0
        data = scipy.io.loadmat(self.linedit.text())
        pos = data['pos']              # shape: (N, 3) if saved like that
        a = data['a']                  # shape: (N,) or (N, 1)
        radar_pos = data['radar_pos'] # shape: (1, 3) or (3,)
        target_pos = data['target_pos'] # shape: (1, 3) or (3,)
        d1=radar_pos - pos
        d10=np.linalg.norm(d1, axis=1)
        Ng = int(self.ng_lineedit.text())
        arc = self.arc_lineedit.text().split(",")
        if len(arc) != 2:
            QMessageBox.warning(self, "Input Error", "Please enter d(Wavelength),Arc(deg) in the format: d,Arc")
            return
        d_grid = float(arc[0]) * Lambda
        arc = float(arc[1])
        resg = d_grid * Lambda
        pos_o = []
        for i in range(-Ng,Ng):
            for j in range(-Ng,Ng):
                x = i * resg
                y = j * resg
                z = 0
                target_pos_i = target_pos + np.array([x, y, z])
                d2=target_pos_i - pos
                d = d10 + np.linalg.norm(d2, axis=1)
                phase = -2 * np.pi / Lambda * d
                a0 =  np.exp(1j * phase)
                pc = a * a0
                o = np.abs(np.sum(pc))
                pos_o.append((target_pos_i[0,0], target_pos_i[0,1], target_pos_i[0,2], o))
        for i in range(-Ng,Ng):
            for j in range(-Ng,Ng):
                x = i * resg
                z = j * resg
                y = 0
                target_pos_i = target_pos + np.array([x, y, z])
                d2=target_pos_i - pos
                d = d10 + np.linalg.norm(d2, axis=1)
                phase = -2 * np.pi / Lambda * d
                a0 =  np.exp(1j * phase)
                pc = a * a0
                o = np.abs(np.sum(pc))
                pos_o.append((target_pos_i[0,0], target_pos_i[0,1], target_pos_i[0,2], o))
        for i in range(-Ng,Ng):
            for j in range(-Ng,Ng):
                z = i * resg
                y = j * resg
                x = 0
                target_pos_i = target_pos + np.array([x, y, z])
                d2=target_pos_i - pos
                d = d10 + np.linalg.norm(d2, axis=1)
                phase = -2 * np.pi / Lambda * d
                a0 =  np.exp(1j * phase)
                pc = a * a0
                o = np.abs(np.sum(pc))
                pos_o.append((target_pos_i[0,0], target_pos_i[0,1], target_pos_i[0,2], o))
        pos_o = np.array(pos_o)

        # Plotting
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot radar and target
        ax.plot([radar_pos[0,0]], [radar_pos[0,1]], [radar_pos[0,2]], 'ro', markersize=10, label='Radar Position')
        ax.plot([target_pos[0,0]], [target_pos[0,1]], [target_pos[0,2]], 'bo', markersize=10, label='Target Position')

        # Plot antennas
        ax.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]), c='g', s=10, label='Antenna Elements')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        ax.set_title('Radar, Target, and Antenna Elements')
        self.set_axes_equal(ax)

        phases = np.degrees(np.angle(a))

        # Plot antennas colored by phase
        fig2 = plt.figure(figsize=(10,10))
        ax2 = fig2.add_subplot(111, projection='3d')
        # scatter with phase colormap
        sc = ax2.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]),
                        c=phases, cmap='hsv', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        cbar.set_label('Phase (radians)')

        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_zlabel('Z [m]')
        ax2.set_title('Antenna Element Phases')
        ax2.legend()
        self.set_axes_equal(ax2)

        fig3 = plt.figure(figsize=(10,10))
        ax3 = fig3.add_subplot(111, projection='3d')
        # scatter with phase colormap
        dynr = float(self.dynr_lineedit.text())
        co = 40*np.log10(pos_o[:, 3])
        lim = co.max() - dynr
        co[co < lim] = lim
        sc = ax3.scatter(np.real(pos_o[:, 0]), np.real(pos_o[:, 1]), np.real(pos_o[:, 2]),
                        c=co, cmap='viridis', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        # cbar.set_label('Phase (radians)')

        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_zlabel('Z [m]')
        # ax2.set_title('Antenna Element Phases')
        # ax2.legend()
        self.set_axes_equal(ax3)


        fig4 = plt.figure(figsize=(10,10))
        ax4 = fig4.add_subplot(111, projection='3d')
        # scatter with phase colormap
        sc = ax4.scatter(np.real(pos_o[:, 0]), np.real(pos_o[:, 1]), np.real(pos_o[:, 2]),
                        c=co, cmap='viridis', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        # cbar.set_label('Phase (radians)')
        ax4.plot([radar_pos[0,0]], [radar_pos[0,1]], [radar_pos[0,2]], 'ro', markersize=10, label='Radar Position')
        ax4.plot([target_pos[0,0]], [target_pos[0,1]], [target_pos[0,2]], 'bo', markersize=10, label='Target Position')

        ax4.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]), c='g', s=10, label='Antenna Elements')

        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_zlabel('Z [m]')
        # ax2.set_title('Antenna Element Phases')
        # ax2.legend()
        self.set_axes_equal(ax4)
        
        posMean = np.mean(pos, axis=0).reshape(1, 3)
        dir = target_pos - posMean
        radius, azimuth, elevation = ssp.utils.cart2sph(dir[0,0], dir[0,1], dir[0,2])
        resg = np.deg2rad(arc)/Ng
        pos_o = []
        for i in range(-Ng,Ng):
            for j in range(-Ng,Ng):
                azimuth_i = i * resg + azimuth
                elevation_j = j * resg + elevation
                x,y,z=ssp.utils.sph2cart(radius, azimuth_i, elevation_j)
                target_pos_i = posMean + np.array([x, y, z])
                d2=target_pos_i - pos
                d = d10 + np.linalg.norm(d2, axis=1)
                phase = -2 * np.pi / Lambda * d
                a0 =  np.exp(1j * phase)
                pc = a * a0
                o = np.abs(np.sum(pc))
                pos_o.append((target_pos_i[0,0], target_pos_i[0,1], target_pos_i[0,2], o))
        pos_o = np.array(pos_o)
        fig3 = plt.figure(figsize=(10,10))
        ax3 = fig3.add_subplot(111, projection='3d')
        # scatter with phase colormap
        
        co = 40*np.log10(pos_o[:, 3])
        lim = co.max() - dynr
        co[co < lim] = lim
        sc = ax3.scatter(np.real(pos_o[:, 0]), np.real(pos_o[:, 1]), np.real(pos_o[:, 2]),
                        c=co, cmap='viridis', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        # cbar.set_label('Phase (radians)')

        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_zlabel('Z [m]')
        # ax2.set_title('Antenna Element Phases')
        # ax2.legend()
        self.set_axes_equal(ax3)


        fig4 = plt.figure(figsize=(10,10))
        ax4 = fig4.add_subplot(111, projection='3d')
        # scatter with phase colormap
        sc = ax4.scatter(np.real(pos_o[:, 0]), np.real(pos_o[:, 1]), np.real(pos_o[:, 2]),
                        c=co, cmap='viridis', s=20)

        # colorbar to show mapping from phase to color
        cbar = plt.colorbar(sc, pad=0.1, shrink=0.7)
        # cbar.set_label('Phase (radians)')
        ax4.plot([radar_pos[0,0]], [radar_pos[0,1]], [radar_pos[0,2]], 'ro', markersize=10, label='Radar Position')
        ax4.plot([target_pos[0,0]], [target_pos[0,1]], [target_pos[0,2]], 'bo', markersize=10, label='Target Position')

        ax4.scatter(np.real(pos[:, 0]), np.real(pos[:, 1]), np.real(pos[:, 2]), c='g', s=10, label='Antenna Elements')

        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_zlabel('Z [m]')
        # ax2.set_title('Antenna Element Phases')
        # ax2.legend()
        self.set_axes_equal(ax4)
        
        plt.show()
    
    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


        
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

class PatchAntennaPatternApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patch Antenna Segment Visualizer")
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # --- Table + Buttons ---
        hbox = QHBoxLayout()
        vbox.addLayout(hbox)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["W", "H", "d"])
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        hbox.addWidget(self.table)

        
        btn_layout = QVBoxLayout()
        hbox.addLayout(btn_layout)
        add_btn = QPushButton("Add Segment")
        add_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_btn)

        del_btn = QPushButton("Remove Selected")
        del_btn.clicked.connect(self.remove_selected_rows)
        btn_layout.addWidget(del_btn)

        draw_btn = QPushButton("Draw")
        draw_btn.clicked.connect(self.draw_segments)
        btn_layout.addWidget(draw_btn)
        def1_btn = QPushButton("Patch 10 Segments")
        def1_btn.clicked.connect(self.def1)
        btn_layout.addWidget(def1_btn)
        def2_btn = QPushButton("Patch 3 Segments IWR6843ISK")
        def2_btn.clicked.connect(self.def2)
        btn_layout.addWidget(def2_btn)
        def3_btn = QPushButton("Patch single Segment")
        def3_btn.clicked.connect(self.def3)
        btn_layout.addWidget(def3_btn)
        btn_layout.addStretch()

        freq_layout = QHBoxLayout()
        freq_label = QLabel("Center Frequency:")
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(1, 1000)            # 1 GHz to 1000 GHz
        self.freq_input.setValue(77)                  # default 77 GHz
        self.freq_input.setSuffix(" GHz")
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        vbox.addLayout(freq_layout)

        # --- Graphics View ---
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        vbox.addWidget(self.view, 1)
        
        # self.def1()  # Initialize with default values
    def def1(self):
        # Clear the table and add default values
        self.freq_input.setValue(77)
        self.table.setRowCount(0)
        for _ in range(10):
            self.add_row()
        # Set default values for the first row
        self.table.setItem(0, 0, QTableWidgetItem("1"))
        self.table.setItem(0, 1, QTableWidgetItem("3"))
        self.table.setItem(0, 2, QTableWidgetItem("7.79"))

        self.table.setItem(1, 0, QTableWidgetItem("2"))
        self.table.setItem(1, 1, QTableWidgetItem("3"))
        self.table.setItem(1, 2, QTableWidgetItem("7.79"))

        self.table.setItem(2, 0, QTableWidgetItem("4"))
        self.table.setItem(2, 1, QTableWidgetItem("3"))
        self.table.setItem(2, 2, QTableWidgetItem("7.79"))

        self.table.setItem(3, 0, QTableWidgetItem("5"))
        self.table.setItem(3, 1, QTableWidgetItem("3"))
        self.table.setItem(3, 2, QTableWidgetItem("7.79"))

        self.table.setItem(4, 0, QTableWidgetItem("6"))
        self.table.setItem(4, 1, QTableWidgetItem("4"))
        self.table.setItem(4, 2, QTableWidgetItem("7.79"))

        self.table.setItem(5, 0, QTableWidgetItem("6"))
        self.table.setItem(5, 1, QTableWidgetItem("3"))
        self.table.setItem(5, 2, QTableWidgetItem("7.79"))

        self.table.setItem(6, 0, QTableWidgetItem("5"))
        self.table.setItem(6, 1, QTableWidgetItem("3"))
        self.table.setItem(6, 2, QTableWidgetItem("7.79"))

        self.table.setItem(7, 0, QTableWidgetItem("4"))
        self.table.setItem(7, 1, QTableWidgetItem("3"))
        self.table.setItem(7, 2, QTableWidgetItem("7.79"))

        self.table.setItem(8, 0, QTableWidgetItem("2"))
        self.table.setItem(8, 1, QTableWidgetItem("3"))
        self.table.setItem(8, 2, QTableWidgetItem("7.79"))

        self.table.setItem(9, 0, QTableWidgetItem("1"))
        self.table.setItem(9, 1, QTableWidgetItem("3"))
        self.table.setItem(9, 2, QTableWidgetItem("7.79"))
        
        scale_factor = 2.0
        self.view.setTransform(QTransform().scale(scale_factor, scale_factor))
        self.draw_segments()  # Draw the segments with default values
    def def2(self):
        self.freq_input.setValue(62)
        # Clear the table and add default values
        self.table.setRowCount(0)
        for _ in range(3):
            self.add_row()
        # Set default values for the first row
        self.table.setItem(0, 0, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(0, 1, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(0, 2, QTableWidgetItem(str(round(34/34.0 * 300/62 *.5,2))))
        
        self.table.setItem(1, 0, QTableWidgetItem(str(round(22/34.0 * 300/62 *.5,2))))
        self.table.setItem(1, 1, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(1, 2, QTableWidgetItem(str(round(34/34.0 * 300/62 *.5,2))))
        
        self.table.setItem(2, 0, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(2, 1, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(2, 2, QTableWidgetItem(str(round(34/34.0 * 300/62 *.5,2))))

        
        scale_factor = 16.0
        self.view.setTransform(QTransform().scale(scale_factor, scale_factor))
        self.draw_segments()  # Draw the segments with default values
    def def3(self):
        self.freq_input.setValue(62)
        # Clear the table and add default values
        self.table.setRowCount(0)
        for _ in range(1):
            self.add_row()
        # Set default values for the first row
        self.table.setItem(0, 0, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(0, 1, QTableWidgetItem(str(round(15/34.0 * 300/62 *.5,2))))
        self.table.setItem(0, 2, QTableWidgetItem(str(round(34/34.0 * 300/62 *.5,2))))
        
        
        scale_factor = 16.0
        self.view.setTransform(QTransform().scale(scale_factor, scale_factor))
        self.draw_segments()  # Draw the segments with default values
    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        # Optional: default values
        for col, default in enumerate(("10", "10", "2")):
            self.table.setItem(row, col, QTableWidgetItem(default))

    def remove_selected_rows(self):
        for idx in sorted({i.row() for i in self.table.selectedItems()}, reverse=True):
            self.table.removeRow(idx)

    def draw_segments(self):
        self.scene.clear()
        # read data
        segments = []
        for row in range(self.table.rowCount()):
            try:
                W = float(self.table.item(row, 0).text())
                H = float(self.table.item(row, 1).text())
                d = float(self.table.item(row, 2).text())
                segments.append((H, W, d))
            except Exception:
                continue

        # start drawing from y = 0
        y = 0
        f0 = self.freq_input.value()  # Get the center frequency, not used in drawing but can be used for further calculations
        Lambda = 3e8 / (f0 * 1e9)
        # find max width for centering
        maxW = max((W for _,W,_ in segments), default=0)
        ad = []
        pen = QPen(Qt.black)
        pen.setWidth(0)
        for H, W, d in segments:
            # center rectangle around x=0
            x = -W/2
            rect = QGraphicsRectItem(QRectF(x, y, W, H))
            rect.setPen(pen)  # default pen
            self.scene.addItem(rect)
            ad.append((np.sqrt(W*H),y/1000.0/Lambda,W/1000.0/Lambda,H/1000.0/Lambda))
            y += d
        
        spine = QGraphicsLineItem(0, 0, 0, y)

        spine.setPen(pen)
        self.scene.addItem(spine)

        # adjust scene rect so all items are visible
        self.scene.setSceneRect(-maxW, 0, 2*maxW, y + 10)

        elevation = np.linspace(-90, 90, 1000)
        # Calculate the antenna pattern
        pattern = np.zeros_like(elevation)
        ad = np.array(ad)
        for i,el in enumerate(elevation):
            pa = patch_pattern(0,el,ad[:,2],ad[:,3])
            # print(el,pa)
            sv = pa*ad[:, 0] * np.exp(-1j * 2 * np.pi * ad[:, 1] * np.sin(np.radians(el)) )
            pattern[i] = np.abs(np.sum(sv))**2

        import matplotlib
        matplotlib.use("Qt5Agg") 
        plt.figure(figsize=(10, 5))
        plt.plot(elevation, 10 * np.log10(pattern / np.max(pattern)), label='Antenna Pattern')
        plt.title('Antenna Pattern')
        plt.xlabel('Elevation Angle (degrees)')
        plt.ylabel('Normalized Power (dB)')
        plt.grid()
        plt.ylim(-30, 2)
        # plt.legend()
        # az = np.linspace(-90, 90, 1000)
        # pattern = np.zeros_like(az)
        # for i,a in enumerate(az):
        #     pa = patch_pattern(a,0,ad[:,2],ad[:,3])
        #     sv = pa
        #     pattern[i] = np.abs(np.sum(sv))**2
        # plt.figure(figsize=(10, 5))
        # plt.plot(az, 10 * np.log10(pattern / np.max(pattern)), label='Antenna Pattern')
        # plt.title('Antenna Pattern')
        # plt.xlabel('Azimuth Angle (degrees)')
        # plt.ylabel('Normalized Power (dB)')
        # plt.grid()
        # plt.ylim(-30, 2)
        
        
        plt.show()

def patch_pattern(azimuth_deg, elevation_deg, Waz_per_lambda, Hel_per_lambda):
    theta = np.radians(elevation_deg)
    phi = np.radians(azimuth_deg)

    # Argument for sinc: sin(x)/x where x = (π * W/λ) * sinθ sinφ
    x = np.pi * Waz_per_lambda * np.sin(theta) * np.sin(phi)
    sinc_term = np.sinc(x / np.pi)  # sinc(x) = sin(x)/x

    # Cosine factor from slot separation
    cos_term = np.cos(np.pi * Hel_per_lambda * np.sin(theta) * np.cos(phi))

    # Normalized field (magnitude only)
    E = sinc_term * cos_term
    E_norm = np.abs(E) / np.max(np.abs(E))  # Normalize to 1

    return E
def appselect(st):
    if st=="FMCW Chirp Parameters Calculator":
        runfmcwchirpapp()
    if st=="RIS Analysis":
        runRISAnalysisapp()
    if st=="Patch(microstrip) Antenna Pattern":
        runradarPatchAntennaPatternapp()
    if st=="Radar Parameters":
        runradarconfigapp()       
    if st=="Hand Gesture MisoCNN":
        ssp.ai.radarML.HandGestureMisoCNN.runradarmisoCNNapp() 
    if st=="Human Health Monitoring":
        ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.runradarConvAEBiLSTMapp() 
    if st=="GAN Radar Waveforms":
        ssp.ai.radarML.GANWaveforms.runradarWaveformapp()
        # ssp.ai.radarML.H
    if st=="SHARP Wifi Sensing":
        ssp.ai.radarML.SHARPWifiSensing.runradarSHARPWifiSensingapp() 
        
        
def runfmcwchirpapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = FMCWApp()
    window.show()
    app.exec_()  # Do not use sys.exit()
def runRISAnalysisapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = RISAnalysisApp()
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

def runradarPatchAntennaPatternapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = PatchAntennaPatternApp()
    window.show()
    app.exec_()  # Do not use sys.exit()


