import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton, QHBoxLayout, QScrollArea, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen,QTransform
from PyQt5.QtWidgets import (
    QPushButton, QTableWidget, QTableWidgetItem, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QSizePolicy,QGraphicsLineItem
)
from PyQt5.QtCore import QRectF
import json
import re
from typing import Dict, Tuple
import numpy as np
# from pyqtgraph.Qt import QtWidgets
# import pyqtgraph.opengl as gl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import os
import sensingsp as ssp
import bpy
from mathutils import Vector
# import numpy as np
# from PyQt5.QtWidgets import (
#     QMainWindow, QApplication, QWidget, QVBoxLayout,
#     QPushButton, QFileDialog, QLabel, QHBoxLayout
# )
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for Matplotlib
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

class AntennaElementPattern(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Visualization")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # ssp.utils.delete_all_objects()
        # ssp.utils.define_settings()
        # ssp.integratedSensorSuite.define_suite(0, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
        # radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
        # # radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843_az(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9)
        # # radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=0, iradar=0, location=Vector((0, 0,0)), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=70e9,LinearArray_TXPos=[0],LinearArray_RXPos=[i*3e8/70e9/2 for i in range(30)])
        # radar['CFAR_RD_training_cells']=20
        # radar['CFAR_RD_guard_cells']=10
        # radar['CFAR_RD_alpha']=5.0
        # radar['CFAR_Angle_training_cells']=10
        # radar['CFAR_Angle_guard_cells']=0
        # radar['CFAR_Angle_alpha']=2.0
        # radar['Transmit_Antenna_Element_Pattern']="Omni0"
        # radar['CFAR_Angle_alpha']=2.0
        
        # Main container widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout(main_widget)
        # Controls layout: Run button, Next button, ComboBox
        controls_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.next_button = QLineEdit("Note: Directional-Sinc")
        self.combo = QComboBox()
        # Add processing options here
        self.combo.addItems(["RD-Det-Ang-Det", "VitalSign"])
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.next_button)
        # controls_layout.addWidget(self.combo)
        vbox.addLayout(controls_layout)
        ssp.utils.trimUserInputs()
        self.combo2 = QComboBox()
        controls_layout.addWidget(self.combo2)
        for isuite,suiteobject in enumerate(ssp.suite_information):
            for iradar,radarobject in enumerate(suiteobject['Radar']):
                for itx,txobj in enumerate(radarobject['TX']):
                    for irx,rxobj in enumerate(radarobject['RX']):
                        self.combo2.addItem(f'{isuite},{iradar},tx:{itx},rx:{irx}')
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Range"))
        self.range0_lineedit = QLineEdit("10")
        controls_layout.addWidget(self.range0_lineedit)
        controls_layout.addWidget(QLabel("min,max,N Azimuth"))
        self.azlim_lineedit = QLineEdit("0,360,200")
        controls_layout.addWidget(self.azlim_lineedit)
        controls_layout.addWidget(QLabel("min,max,N Elevation"))
        self.ellim_lineedit = QLineEdit("-90,90,100")
        controls_layout.addWidget(self.ellim_lineedit)
        vbox.addLayout(controls_layout)

        self.figure = Figure(figsize=(12, 8), facecolor='#4e4e4e')  # dark background
        self.canvas = FigureCanvas(self.figure)
        vbox.addWidget(self.canvas)

        # Initialize axes
        self.axes = []
        for i in range(6):
            if i in [500]:
                ax = self.figure.add_subplot(2, 4, i + 1, projection='3d')
                ax.set_facecolor('#6e6e6e')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
            else:
                ax = self.figure.add_subplot(3, 2, i + 1)
                ax.set_facecolor('#6e6e6e')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')

                # Optional: Set titles for each subplot
                titles = [
                    "Raw ADC Data", "Real vs Imag", "Range-Time", "Range Profile",
                    "Range-Doppler", "", "", "Detected Points"
                ]
                # if i < len(titles):
                #     ax.set_title(titles[i], fontsize=10)

            self.axes.append(ax)

        self.figure.tight_layout(pad=3.0)

        self.run_button.clicked.connect(self.on_run)
        # self.next_button.clicked.connect(self.on_next)
    def on_run(self):
        obj_name = "PatternAnalysis_TestObject"
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.delete()
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        geocalculator = ssp.raytracing.BlenderGeometry()
        suite_information = ssp.raytracing.BlenderSuiteFinder().find_suite_information()
        Suite_Position, ScattersGeo, HashFaceIndex_ScattersGeo, ScattersGeoV = geocalculator.get_Position_Velocity(
            bpy.context.scene, suite_information, ssp.config.CurrentFrame, 1
        )
        self.next_button.setText("")
        if len(ScattersGeo)>0:
            self.next_button.setText("Remove all objects")
            return
        
        

        isuite = int(self.combo2.currentText().split(',')[0])
        iradar = int(self.combo2.currentText().split(',')[1])
        itx = int(self.combo2.currentText().split(',')[2].split(':')[1])
        irx = int(self.combo2.currentText().split(',')[3].split(':')[1])
        loc0=Suite_Position[isuite]['Radar'][iradar]['TX-Position'][itx]
        
        R=float(self.range0_lineedit.text())
        azmin =  float(self.azlim_lineedit.text().split(',')[0])
        azmax =  float(self.azlim_lineedit.text().split(',')[1])
        azN =  int(self.azlim_lineedit.text().split(',')[2])
        elmin =  float(self.ellim_lineedit.text().split(',')[0])
        elmax =  float(self.ellim_lineedit.text().split(',')[1])
        elN =  int(self.ellim_lineedit.text().split(',')[2])
        

        cube = ssp.environment.add_cube(location=Vector((R, 0, 0)), direction=Vector((1, 0, 0)), scale=Vector((.5, .5, .5)), subdivision=0)
        cube.name=obj_name
        cube["RCS0"]=1.0
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=loc0, rotation=Vector((0, 0, 0)), scale=(.01, .01, .01))
        empty = bpy.context.object
        cube.parent = empty
        empty.rotation_euler  = (0,0,np.deg2rad(azmin))
        empty.keyframe_insert(data_path="rotation_euler", frame=1)
        empty.rotation_euler  = (0,0,np.deg2rad(azmax))
        empty.keyframe_insert(data_path="rotation_euler", frame=azN)
        
        empty.rotation_euler  = (0,np.deg2rad(-elmin),0)
        empty.keyframe_insert(data_path="rotation_euler", frame=azN+1)
        empty.rotation_euler  = (0,np.deg2rad(-elmax),0)
        empty.keyframe_insert(data_path="rotation_euler", frame=azN+elN)
        
        for fcurve in empty.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
        ssp.config.restart()
        ssp.utils.set_frame_start_end(1,azN+elN+1)
        pat=[]
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            d = path_d_drate_amp[isuite][iradar][irx][isuite][iradar][itx]
            if len(d):
                pat.append([ssp.config.CurrentFrame,d[0][2]])
            ssp.utils.increaseCurrentFrame()
        pat = np.array(pat)
        azpat = pat[:azN,1]
        elpat = pat[azN:,1]
        az = np.linspace(azmin,azmax,azN)
        el = np.linspace(elmin,elmax,elN)

        self.axes[0].cla()
        self.axes[0].plot(az,azpat)
        self.axes[0].set_xlabel("Azimuth (deg)")
        self.axes[0].set_ylabel("Azimuth Beampatern")
        self.axes[0].set_title("Azimuth Beampatern")
        self.axes[0].set_ylim(0, 1.1*np.max(azpat))
        
        self.axes[1].cla()
        self.axes[2].cla()
        self.axes[1].plot(el,elpat)
        self.axes[1].set_xlabel("Elevation (deg)")
        self.axes[1].set_ylabel("Elevation Beampatern")
        self.axes[1].set_title("Elevation Beampatern")
        self.axes[1].set_ylim(0, 1.1*np.max(elpat))
        
        self.axes[2].plot(az,20*np.log10(azpat))
        self.axes[2].set_xlabel("Azimuth (deg)")
        self.axes[2].set_ylabel("Azimuth Beampatern (dB)")
        self.axes[3].cla()
        self.axes[4].cla()
        self.axes[5].cla()
        self.axes[3].plot(el,20*np.log10(elpat))
        self.axes[3].set_xlabel("Elevation (deg)")
        self.axes[3].set_ylabel("Elevation Beampatern (dB)")
        
        self.axes[4].plot(az,20*np.log10(azpat))
        self.axes[4].set_ylim(np.max(20*np.log10(azpat))-6, np.max(20*np.log10(azpat))+3)
        self.axes[4].set_xlabel("Azimuth (deg)")
        self.axes[4].set_ylabel("Azimuth Beampatern (dB)")
        self.axes[5].plot(el,20*np.log10(elpat))
        self.axes[5].set_xlabel("Elevation (deg)")
        self.axes[5].set_ylabel("Elevation Beampatern (dB)")
        self.axes[5].set_ylim(np.max(20*np.log10(elpat))-6, np.max(20*np.log10(elpat))+3)
        self.canvas.draw()
        self.canvas.flush_events()


class RadarVisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Visualization")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Main container widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout(main_widget)
        # Controls layout: Run button, Next button, ComboBox
        controls_layout = QHBoxLayout()
        self.run_button = QPushButton("Load")
        self.next_button = QPushButton("Next")
        self.combo = QComboBox()
        # Add processing options here
        self.combo.addItems(["RD-Det-Ang-Det", "VitalSign"])
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.combo)
        vbox.addLayout(controls_layout)
        ssp.utils.trimUserInputs()
        self.combo2 = QComboBox()
        controls_layout.addWidget(self.combo2)
        for isuite,suiteobject in enumerate(ssp.suite_information):
            for iradar,radarobject in enumerate(suiteobject['Radar']):
                self.combo2.addItem(f'{isuite},{iradar}')
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Point cloud colors"))
        self.combocolor = QComboBox()
        self.combocolor.addItems(["Amp", "Doppler"])
        controls_layout.addWidget(self.combocolor)
        
        
        controls_layout.addWidget(QLabel("Save Outputs"))
        self.combosave = QComboBox()
        self.combosave.addItems(["None","Raw DataCube", "Point Cloud"])
        controls_layout.addWidget(self.combosave)
        
        
        self.doCFARCB = QCheckBox()
        self.doCFARCB.setText("CFAR (Heavy)")
        self.doCFARCB.setChecked(True)
        controls_layout.addWidget(self.doCFARCB)
        self.FIXEXTHRCB = QCheckBox()
        self.FIXEXTHRCB.setText("Fix THR")
        controls_layout.addWidget(self.FIXEXTHRCB)
        self.thrCH = QCheckBox()
        self.thrCH.setText("Threshold Plot")
        controls_layout.addWidget(self.thrCH)
        vbox.addLayout(controls_layout)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Vital Sign Processing Time (Sec)"))
        self.vstime_lineedit = QLineEdit("6")
        controls_layout.addWidget(self.vstime_lineedit)
        controls_layout.addWidget(QLabel("Detection range limit"))
        self.vsrangelim_lineedit = QLineEdit("1.3,2")
        controls_layout.addWidget(self.vsrangelim_lineedit)
        controls_layout.addWidget(QLabel("RPM limit"))
        self.rpmlim_lineedit = QLineEdit("0,90")
        controls_layout.addWidget(self.rpmlim_lineedit)
        vbox.addLayout(controls_layout)

        self.figure = Figure(figsize=(12, 8), facecolor='#4e4e4e')  # dark background
        self.canvas = FigureCanvas(self.figure)
        vbox.addWidget(self.canvas)

        # Initialize axes
        self.axes = []
        for i in range(8):
            if i in [5, 6, 7]:
                ax = self.figure.add_subplot(2, 4, i + 1, projection='3d')
                ax.set_facecolor('#6e6e6e')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
            else:
                ax = self.figure.add_subplot(2, 4, i + 1)
                ax.set_facecolor('#6e6e6e')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')

                # Optional: Set titles for each subplot
                titles = [
                    "Raw ADC Data", "Real vs Imag", "Range-Time", "Range Profile",
                    "Range-Doppler", "", "", "Detected Points"
                ]
                if i < len(titles):
                    ax.set_title(titles[i], fontsize=10)

            self.axes.append(ax)

        self.figure.tight_layout(pad=3.0)

        # Connect button signals to handler methods
        self.run_button.clicked.connect(self.on_run)
        self.next_button.clicked.connect(self.on_next)

        
    def on_run(self):
        fn = os.path.join(ssp.config.temp_folder, "arrayFolder","radar_array.mat")
        ssp.environment.scenarios.predefine_Altos_Radar_file(fn)
        # ssp.environment.scenarios.predefine_movingcube_6843()
        # ssp.utils.initialize_environment()
        # radar = ssp.radar.utils.addRadar(
        #     radarSensor=ssp.radar.utils.RadarSensorsCategory.TI_IWR6843,
        #     location_xyz=(0, -2, 0)
        # )
        # radar.rotation_euler.z=0
        # radar['NPulse'] = 1
        # radar['PRI_us'] = 50000
        # radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=.1
        # respiration_cycles = 5
        # respiration_rpm = 12 # 12 - 20
        # respiration_time = 60.0/respiration_rpm
        # respiration_period = int( respiration_time * bpy.context.scene.render.fps /2)
        # ssp.environment.generate_chest_vibration_target(respiration_angle_amplitude=20,
        #                             bone_lengths=(0.07, 0.035, 0.035),
        #                             chest_dimensions=(0.175, 0.14, 1),
        #                             subdivisions=10,
        #                             respiration_cycles=respiration_cycles,
        #                             respiration_period=respiration_period)
        # ssp.utils.save_Blender()
        ssp.utils.trimUserInputs()
        self.combo2.clear()
        for isuite,suiteobject in enumerate(ssp.suite_information):
            for iradar,radarobject in enumerate(suiteobject['Radar']):
                self.combo2.addItem(f'{isuite},{iradar}')
    def style_axes(ax, title=None, xlabel=None, ylabel=None, zlabel=None):
        ax.set_facecolor('#2e2e2e')
        ax.tick_params(colors='white', which='both')
        if title:
            ax.set_title(title, color='white', fontsize=10)
        if xlabel:
            ax.set_xlabel(xlabel, color='white')
        if ylabel:
            ax.set_ylabel(ylabel, color='white')
        if hasattr(ax, 'set_zlabel') and zlabel:  # Only for 3D
            ax.set_zlabel(zlabel, color='white')
        ax.grid(True, linestyle='--', alpha=0.3)

    def pointcloud_axe(self):
        # Delete existing object named 'Detection Cloud' and its children
        if 'Detection Point Cloud' in bpy.data.objects:
            obj = bpy.data.objects['Detection Point Cloud']
            
            # Deselect all objects
            bpy.ops.object.select_all(action='DESELECT')
            
            # Select the object and its children
            obj.select_set(True)
            for child in list(obj.children):
                child.select_set(True)
            # Set active object
            bpy.context.view_layer.objects.active = obj
            
            # Delete selected objects
            bpy.ops.object.delete()

        # Add new empty object
        bpy.ops.object.empty_add(
            type='PLAIN_AXES',
            align='WORLD',
            location=(0, 0, 0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1)
        )

        empty = bpy.context.object
        empty.name = 'Detection Cloud'
        bpy.ops.object.select_all(action='DESELECT')
        return empty
    
    def vs_processing(self):
        vstime = float(self.vstime_lineedit.text())
        rmin = float(self.vsrangelim_lineedit.text().split(',')[0])
        rmax = float(self.vsrangelim_lineedit.text().split(',')[1]) 
        rpmmin = float(self.rpmlim_lineedit.text().split(',')[0])
        rpmmax = float(self.rpmlim_lineedit.text().split(',')[1]) 
        
        all_signals = []
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            
            
            Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
            for XRadar, timeX in Signals[0]['radars'][0]:
                fast_time_window = ssp.radar.utils.hamming(XRadar.shape[0])
                X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]
                NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0])) + 1))
                d_fft = (np.arange(NFFT_Range) * ssp.LightSpeed / 2 /
                         ssp.RadarSpecifications[0][0]['FMCW_ChirpSlobe'] / NFFT_Range / ssp.RadarSpecifications[0][0]['Ts'])
                X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range) / XRadar.shape[0]
                self.axes[0].cla()
                amplitude_db = 20 * np.log10(np.abs(X_fft_fast[:, 0, 0]))
                self.axes[0].plot(d_fft, amplitude_db)
                self.axes[0].set_title("Range FFT")
                self.axes[0].set_ylabel("Amplitude (dB)")
                self.axes[0].set_xlabel("Range (m)")
                self.axes[0].set_xlim(0, 5)
                range_target = rmin
                idx1 = np.argmin(np.abs(d_fft - range_target))
                range_target = rmax
                idx2 = np.argmin(np.abs(d_fft - range_target))
                idx=idx1+np.argmax(amplitude_db[idx1:idx2])
                value_at_1_8 = amplitude_db[idx]
                self.axes[0].vlines(range_target, value_at_1_8, -30, color='red', linestyle='--', linewidth=2)
                all_signals.append([X_fft_fast[idx, 0, 0], timeX[0]])
                sig = np.array(all_signals)
                uw = np.unwrap(np.angle(sig[:,0]))
                
                self.axes[1].cla()
                self.axes[1].plot(np.real(sig[:,1]), uw)
                self.axes[1].set_title("Unwrapped Phase")
                self.axes[1].set_ylabel("Phase (rad)")
                self.axes[1].set_xlabel("Time (s)")
                self.axes[2].cla()
                T = np.mean(np.diff(np.real(sig[:,1])))
                NFFT = int(2 ** (3+np.ceil(np.log2(len(uw)))))
                fftuw = np.fft.fft(uw, n=NFFT) / len(uw)
                uwdc = uw - np.mean(uw)
                fftuwdc = np.fft.fft(uwdc, n=NFFT) / len(uwdc)
                N = int(len(fftuw)/2)
                frpm = np.arange(N) / NFFT / T * 60
                self.axes[2].plot(frpm[1:N], 20*np.log10(np.abs(fftuw[1:N])))
                self.axes[2].plot(frpm[1:N], 20*np.log10(np.abs(fftuwdc[1:N])), linestyle='--')
                self.axes[2].set_title("FFT of Unwrapped Phase")
                self.axes[2].set_ylabel("Amplitude (dB)")
                self.axes[2].set_xlabel("Frequency (RPM)")
                self.axes[2].set_xlim(rpmmin, rpmmax)
                
            self.canvas.draw()
            self.canvas.flush_events()
            ssp.utils.increaseCurrentFrame()
       
    def on_next(self):
        ssp.utils.trimUserInputs()
        for a in self.axes:
            a.cla()
        if self.combo.currentText()==["RD-Det-Ang-Det", "VitalSign"][1]:
            self.vs_processing()
            return
        ssp.utils.trimUserInputs() 
        if self.combo2.count()==0:
            return
        
        isuite = int(self.combo2.currentText().split(',')[0])
        iradar = int(self.combo2.currentText().split(',')[1])
        specifications = ssp.RadarSpecifications[isuite][iradar]
        
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        
        Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
        pointcloud_axe_parent = self.pointcloud_axe()
        
        if len(Signals[isuite]['radars'][iradar])==0:
            return
        XRadar,t = Signals[isuite]['radars'][iradar][0]
        
        FMCW_ChirpSlobe = specifications['FMCW_ChirpSlobe']
        Ts = specifications['Ts']
        PRI = specifications['PRI']
        PrecodingMatrix = specifications['PrecodingMatrix']
        RadarMode = specifications['RadarMode']
        
        # specifications['CFAR_RD_guard_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_guard_cells']
        # specifications['CFAR_RD_training_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_training_cells']
        # specifications['CFAR_RD_false_alarm_rate'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_false_alarm_rate']
        # specifications['STC_Enabled'] = radarobject['GeneralRadarSpec_Object']['STC_Enabled']
        # specifications['MTI_Enabled'] = radarobject['GeneralRadarSpec_Object']['MTI_Enabled']
        # specifications['CFAR_Angle_guard_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_guard_cells']
        # specifications['CFAR_Angle_training_cells'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_training_cells']
        # specifications['CFAR_Angle_false_alarm_rate'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_false_alarm_rate']
        # specifications['CFAR_RD_alpha'] = radarobject['GeneralRadarSpec_Object']['CFAR_RD_alpha']
        # specifications['CFAR_Angle_alpha'] = radarobject['GeneralRadarSpec_Object']['CFAR_Angle_alpha']
        
        self.axes[1].plot(np.real(XRadar[:, 0, 0]), label="Real Part")
        self.axes[1].plot(np.imag(XRadar[:, 0, 0]), label="Imaginary Part")
        self.axes[1].set_xlabel("ADC Samples")
        self.axes[1].set_ylabel("ADC Output Level")
        self.axes[1].legend(loc='upper right')
        self.axes[0].imshow(np.abs(XRadar[:, :, 0]),extent=[1*PRI, XRadar.shape[1]*PRI, 1, XRadar.shape[0]],
                            aspect='auto', origin='lower')
        self.axes[0].set_xlabel("Slow time")
        self.axes[0].set_ylabel("ADC Samples")
        
        fast_time_window = scipy.signal.windows.hamming(XRadar.shape[0])
        X_windowed_fast = XRadar * fast_time_window[:, np.newaxis, np.newaxis]

        NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
        NFFT_Range = int(2 ** (np.ceil(np.log2(XRadar.shape[0]))+NFFT_Range_OverNextPow2))
        X_fft_fast = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
        d_fft = np.arange(NFFT_Range) * ssp.LightSpeed / 2 / FMCW_ChirpSlobe / NFFT_Range / Ts
        Range_Start = specifications['Range_Start']
        Range_End = specifications['Range_End']
        d1i = int(X_fft_fast.shape[0]*Range_Start/100.0)
        d2i = int(X_fft_fast.shape[0]*Range_End/100.0)
        d_fft = d_fft[d1i:d2i]
        X_fft_fast = X_fft_fast[d1i:d2i,:,:] 
        
        self.axes[3].plot(d_fft, np.abs(X_fft_fast[:, 0, 0]))
        self.axes[3].set_xlabel("Range (m)")
        self.axes[3].set_ylabel("Range Profile")
        self.axes[2].imshow(np.abs(X_fft_fast[:, :, 0]),
                            extent=[1*PRI, XRadar.shape[1]*PRI, d_fft[0], d_fft[-1]],
                            aspect='auto', origin='lower')
        self.axes[2].set_xlabel("Slow time")
        self.axes[2].set_ylabel("Range")
        
        test1,test2 = np.min(np.abs(X_fft_fast)),np.max(np.abs(X_fft_fast))
        M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
        L = X_fft_fast.shape[1]
        Leff = int(L/M_TX)
        if Leff == 0:
            self.canvas.draw()
            self.canvas.flush_events()
            return
        rangePulseTXRX = np.zeros((X_fft_fast.shape[0], Leff, M_TX, X_fft_fast.shape[2]),dtype=complex)
        for ipulse in range(Leff):
            ind = ipulse*M_TX
            rangePulseTXRX[:,ipulse,:,:]=X_fft_fast[:,ind:ind+M_TX,:]
        NFFT_Doppler_OverNextPow2=  specifications['DopplerFFT_OverNextP2']
        NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
        rangeDopplerTXRX = np.fft.fft(rangePulseTXRX, axis=1, n=NFFT_Doppler)
        rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
        f_Doppler = np.linspace(0,1/PRI/M_TX,NFFT_Doppler)-1/PRI/M_TX/2
        
        f_Doppler = np.fft.fftshift(np.fft.fftfreq(NFFT_Doppler))/PRI/M_TX
            
        
        im = self.axes[4].imshow(np.abs(rangeDopplerTXRX[:, :, 0, 0]),
                                extent=[f_Doppler[0], f_Doppler[-1], d_fft[0], d_fft[-1]],
                                aspect='auto', origin='lower')
        self.axes[4].set_xlabel("Doppler Frequency (Hz)")
        self.axes[4].set_ylabel("Range (m)")
        
        
        test1_1,test2_1 = np.min(np.abs(rangeDopplerTXRX)),np.max(np.abs(rangeDopplerTXRX))
        
        rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
        rangeDoppler4CFAR = np.mean(np.abs(rangeDopplerTXRX),axis=(2,3))
        rangeDoppler4CFAR -= np.min(rangeDoppler4CFAR)
        
        distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
        elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
        X, Y = np.meshgrid(elevation, distance)
        # FigsAxes[1,2].plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
        self.axes[5].plot_surface(X, Y, (rangeDoppler4CFAR), cmap='viridis', alpha=1)
        self.axes[5].set_xlabel('Doppler (Hz)')
        self.axes[5].set_ylabel('Distance (m)')
        self.axes[5].set_zlabel('Magnitude (normalized, dB)')
        if self.doCFARCB.isChecked() == False:
            self.canvas.draw()
            self.canvas.flush_events()
            ssp.utils.increaseCurrentFrame()
            return
            
        # rangeDoppler4CFAR = np.abs(np.mean(rangeDopplerTXRX,axis=(2,3)))
        # rangeDoppler4CFAR = np.abs(rangeDopplerTXRX[:,:,0,0])
        test1_2,test2_2 = np.min(np.abs(rangeDoppler4CFAR)),np.max(np.abs(rangeDoppler4CFAR))
        all_xyz=[]
        CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
        num_train, num_guard, alpha = [specifications['CFAR_RD_training_cells'],specifications['CFAR_RD_training_cells']], [specifications['CFAR_RD_guard_cells'],specifications['CFAR_RD_guard_cells']], specifications['CFAR_RD_alpha']
        if self.FIXEXTHRCB.isChecked():
            T = alpha * np.mean(rangeDoppler4CFAR)
            cfar_threshold = T * np.ones_like(rangeDoppler4CFAR)
            detections = np.zeros_like(rangeDoppler4CFAR)
            detections[rangeDoppler4CFAR > T] = 1
        else:
            if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled:
                detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, num_train, num_guard, alpha)
            else:
                detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR,num_train, num_guard, alpha)
        # detections,cfar_threshold = ssp.radar.utils.cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        
        # FigsAxes[1,2].plot_surface(X, Y, (cfar_threshold)+0, color='yellow', alpha=1)
        detected_points = np.where(detections == 1)
        self.axes[5].scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                    (rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')
        
        # Labels and legend
        if  self.thrCH.isChecked():
            
            self.axes[5].plot_surface(X,Y,(alpha*cfar_threshold), color='red', alpha=1)
        # FigsAxes[1,2].xaxis.set_visible(False)
        # FigsAxes[1,2].yaxis.set_visible(False)
        # FigsAxes[1,2].zaxis.set_visible(False)
        # break
        # FigsAxes[1,2].legend()
        # plt.show()
        NDetection = detected_points[0].shape[0]
        # rows = unique_PosIndex[:,2].astype(int)
        # cols = unique_PosIndex[:,3].astype(int)
        if os.path.exists(specifications['ArrayInfofile'])==False:
            rows,cols=specifications['MIMO_Antenna_Azimuth_Elevation_Order']
            VA = np.zeros((np.max(np.array(rows))+1,np.max(np.array(cols))+1),dtype=rangeDopplerTXRX.dtype)
            temp = np.zeros((rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=complex)
            for i_tx in range(rangeDopplerTXRX.shape[2]):
                for j_rx in range(rangeDopplerTXRX.shape[3]):
                    temp[i_tx,j_rx]= i_tx+1j*j_rx
            VA[np.array(rows), np.array(cols)] = temp.ravel()
            specifications['vaorder']=[]
            for i_x in range(VA.shape[0]):
                for j_y in range(VA.shape[1]):
                    specifications['vaorder'].append([int(VA[i_x,j_y].real)+1,1+int(VA[i_x,j_y].imag),1+i_x,1+j_y])
            specifications['vaorder'] = np.array(specifications['vaorder'])
            rangeVA = np.zeros((int(np.max(specifications['vaorder'][:,2])),int(np.max(specifications['vaorder'][:,3]))),dtype=rangeDopplerTXRX.dtype)
            Lambda = 1.0
            AzELscale = [1,1]
            dy = .5*Lambda*AzELscale[0]
            dz = .5*Lambda*AzELscale[1]
        elif specifications['vaprocessing']=="Az FFT":
            rangeVA = np.zeros((int(np.max(specifications['vaorder'][:,2])),int(np.max(specifications['vaorder'][:,3]))),dtype=rangeDopplerTXRX.dtype)
            Lambda = 1.0
            AzELscale = specifications['AzELscale']
            dy = .5*Lambda*AzELscale[0]
            dz = .5*Lambda*AzELscale[1]
        
        elif specifications['vaprocessing']=="Az FFT,El Estimation":
            return
        elif specifications['vaprocessing']=="Az FFT,El FFT":
            return
        elif specifications['vaprocessing']=="2D FFT":
            rangeVA = np.zeros((int(np.max(specifications['vaorder'][:,2])),int(np.max(specifications['vaorder'][:,3]))),dtype=rangeDopplerTXRX.dtype)
            Lambda = 1.0
            AzELscale = specifications['AzELscale']
            dy = .5*Lambda*AzELscale[0]
            dz = .5*Lambda*AzELscale[1]
        
        # specifications['vaorder']
        # specifications['vaorder2']
        # i_list, j_list = ssp.radar.utils.mimo.mimo_antenna_order(specifications)
        
        maxAmp = 0
        for id in range(NDetection):
            rangeTarget = d_fft[detected_points[0][id]]
            dopplerTarget = f_Doppler[detected_points[1][id]]
            
            antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
             
            
            for indx in specifications['vaorder']:
                rangeVA[int(indx[2]-1), int(indx[3]-1)] = antennaSignal[int(indx[0]-1), int(indx[1]-1)]
            # rangeVA = np.zeros((np.max(i_list)+1,np.max(j_list)+1),dtype=antennaSignal.dtype)
            # rangeVA[i_list, j_list] = antennaSignal.ravel()
            
            NFFT_Angle_OverNextPow2 = specifications['AzFFT_OverNextP2']
            NFFT_Angle = int(2 ** (np.ceil(np.log2(rangeVA.shape[0]))+NFFT_Angle_OverNextPow2))
            NFFT_Angle_OverNextPow_elevation = specifications['ElFFT_OverNextP2']
            NFFT_Angle_elevation = int(2 ** (np.ceil(np.log2(rangeVA.shape[1]))+NFFT_Angle_OverNextPow_elevation))
            AngleMap0 = np.fft.fft(rangeVA, axis=0, n=NFFT_Angle)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            AngleMap0 = np.fft.fftshift(AngleMap0, axes=0)
            AngleMap = np.fft.fft(AngleMap0, axis=1, n=NFFT_Angle_elevation)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
            AngleMap = np.fft.fftshift(AngleMap, axes=1)
            AngleMap = np.abs(AngleMap)
            
            a = np.linspace(-1, 1, AngleMap.shape[0])
            b = np.linspace(-1, 1, AngleMap.shape[1]) 
            
            a = np.fft.fftshift(np.fft.fftfreq(AngleMap.shape[0]))
            b = np.fft.fftshift(np.fft.fftfreq(AngleMap.shape[1]))
              
            X, Y = np.meshgrid(b, a)
            
            num_train, num_guard, alpha = [specifications['CFAR_Angle_training_cells'],specifications['CFAR_Angle_training_cells']], [specifications['CFAR_Angle_guard_cells'],specifications['CFAR_Angle_guard_cells']], specifications['CFAR_Angle_alpha']
            if AngleMap.shape[1]==1:
                num_guard[1]=0
            if self.FIXEXTHRCB.isChecked():
                T = alpha * np.mean(AngleMap)
                cfar_threshold_angle = T * np.ones_like(AngleMap)
                detections_angle = np.zeros_like(AngleMap)
                detections_angle[AngleMap > T] = 1
            else:
                if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled:
                    detections_angle,cfar_threshold_angle = ssp.radar.utils.cfar_ca_2D_alpha_cuda(1.0*AngleMap, num_train, num_guard, alpha)
                else:
                    detections_angle,cfar_threshold_angle = ssp.radar.utils.cfar_ca_2D_alpha(1.0*AngleMap, num_train, num_guard, alpha)
                
            detected_points_angle = np.where(detections_angle == 1)
            if maxAmp < np.abs(antennaSignal[0,0]):
                maxAmp = np.abs(antennaSignal[0,0])
                self.axes[6].cla()
                if AngleMap.shape[1]>1:
                    self.axes[6].plot_surface(X,Y,(AngleMap), cmap='viridis', alpha=1)
                
                    self.axes[6].scatter(b[detected_points_angle[1]],a[detected_points_angle[0]], 
                                (AngleMap[detected_points_angle]), color='red', s=20, label='Post-CFAR Point Cloud')
                    if  self.thrCH.isChecked():
                        self.axes[6].plot_surface(X,Y,(alpha*cfar_threshold_angle), color='red', alpha=1)
                else:
                    self.axes[6].plot(Y[:,0],AngleMap[:,0])
                    self.axes[6].plot(a[detected_points_angle[0]],AngleMap[detected_points_angle],'or')
                    if  self.thrCH.isChecked():
                        self.axes[6].plot(Y[:,0],alpha*cfar_threshold_angle[:,0])
            NDetection_angle = detected_points_angle[0].shape[0]
            for id_angle in range(NDetection_angle):
                amp = AngleMap[detected_points_angle[0][id_angle],detected_points_angle[1][id_angle]]
                
                # fy = (detected_points_angle[0][id_angle]+0) / AngleMap.shape[0] / dy
                # fz = (detected_points_angle[1][id_angle]+0) / AngleMap.shape[1] / dz
                
                # if fy > 1:
                #     fy = fy - 2
                # if fz > 1:
                #     fz = fz - 2
                fy = a[detected_points_angle[0][id_angle]] / dy
                fz = b[detected_points_angle[1][id_angle]] / dz
                azhat = np.arcsin(fy / np.sqrt((1/Lambda)**2 - fz**2))
                elhat = np.arccos(fz * Lambda)
                elhat = np.pi/2 - elhat
                
                x, y, z = ssp.utils.sph2cart(rangeTarget, azhat, elhat)
                x, y, z = y,z,-x
                # radar_center = specifications['global_location_TX_RX_Center'][2]
                # x+=radar_center[0]
                # y+=radar_center[1]
                # z+=radar_center[2]
                if 1:
                    global_location, global_rotation, global_scale = specifications['matrix_world']  
                    local_point = Vector((x, y, z))
                    global_point = global_location + global_rotation @ (local_point * global_scale)
                    bpy.ops.mesh.primitive_uv_sphere_add(radius=.03, location=global_point)
                    sphere = bpy.context.object
                    sphere.parent=pointcloud_axe_parent
                    x = global_point.x
                    y = global_point.y
                    z = global_point.z
                    
                    
                all_xyz.append([x,y,z,amp,dopplerTarget])
                #[]
                # print(f"Detected angle: azimuth={np.degrees(azhat):.2f} degrees, elevation={np.degrees(elhat):.2f} degrees")
                # print(x,y,z)
                
        points = np.array(all_xyz)
        if points.shape[0]>0:
            
            ampcolor = self.combocolor.currentText()
            if ampcolor == 'Amp':
                self.axes[7].scatter(points[:, 0], points[:, 1],points[:, 2], c=points[:, 3], marker='o') 
            if ampcolor == 'Doppler':
                self.axes[7].scatter(points[:, 0], points[:, 1],points[:, 2], c=points[:, 4], marker='o') 
                
        self.axes[7].plot(0, 0, 0, 'o', markersize=5, color='red')
        self.axes[7].plot([0,5], [0,0], [0,0], color='red')
        self.axes[7].set_xlabel('X (m)')
        self.axes[7].set_ylabel('Y (m)')
        self.axes[7].set_zlabel('Z (m)')
        self.axes[7].set_title("Detected Points")
        ssp.utils.set_axes_equal(self.axes[7])
        
        self.canvas.draw()
        self.canvas.flush_events()
        if self.combosave.currentText()=="Point Cloud":
            fn = f'PointCloud_frame_{ssp.config.CurrentFrame}.mat'
            if "Simulation Settings" in bpy.data.objects:
                if "Radar Outputs Folder" in bpy.data.objects["Simulation Settings"]:
                    fp= bpy.data.objects["Simulation Settings"]["Radar Outputs Folder"]
                    if not os.path.exists(fp):
                        os.makedirs(fp)
                    fn = os.path.join(fp,fn)
                    scipy.io.savemat(fn, {
                            'points'    : points,
                            'frame': ssp.config.CurrentFrame
                        })
        if self.combosave.currentText()=="Raw DataCube":
            fn = f'Signals_frame_{ssp.config.CurrentFrame}.mat'
            if "Simulation Settings" in bpy.data.objects:
                if "Radar Outputs Folder" in bpy.data.objects["Simulation Settings"]:
                    fp= bpy.data.objects["Simulation Settings"]["Radar Outputs Folder"]
                    if not os.path.exists(fp):
                        os.makedirs(fp)
                    fn = os.path.join(fp,fn)
                    XRadar,X_windowed_fast,rangePulseTXRX,d_fft,rangeDopplerTXRX,f_Doppler,rangeDoppler4CFAR
                    scipy.io.savemat(fn, {
                            'frame'             : ssp.config.CurrentFrame,
                            'XRadar'            : XRadar,
                            'X_windowed_fast'   : X_windowed_fast,
                            'rangePulseTXRX'    : rangePulseTXRX,
                            'd_fft'             : d_fft,
                            'rangeDopplerTXRX'  : rangeDopplerTXRX,
                            'f_Doppler'         : f_Doppler,
                            'rangeDoppler4CFAR' : rangeDoppler4CFAR,
                        })
            
        ssp.utils.increaseCurrentFrame()

class HubLoadApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SensingSP Hub")
        self.setGeometry(100, 100, 800, 600)
        self.items = []  # will hold list of (category, name)
        self.initUI()
        self.init_hub()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout(main_widget)

        # Parameter inputs
        form = QFormLayout()
        # RCS0 (int)
        self.rcs0_sb = QDoubleSpinBox()
        self.rcs0_sb.setRange(0.0, 100000.0)
        self.rcs0_sb.setValue(1.0)
        form.addRow("RCS0:", self.rcs0_sb)

        # Decimate factor (float)
        self.decimate_sb = QDoubleSpinBox()
        self.decimate_sb.setRange(0.0, 100.0)
        self.decimate_sb.setSingleStep(0.1)
        self.decimate_sb.setValue(1.0)
        form.addRow("Decimate Factor:", self.decimate_sb)

        # Rotation (3 floats)
        rot_hbox = QHBoxLayout()
        self.rot_x = QDoubleSpinBox(); self.rot_x.setRange(-360,360); self.rot_x.setValue(0.0)
        self.rot_y = QDoubleSpinBox(); self.rot_y.setRange(-360,360); self.rot_y.setValue(0.0)
        self.rot_z = QDoubleSpinBox(); self.rot_z.setRange(-360,360); self.rot_z.setValue(0.0)
        for sb in (self.rot_x, self.rot_y, self.rot_z):
            sb.setSingleStep(1.0)
            rot_hbox.addWidget(sb)
        form.addRow("Rotation (°):", rot_hbox)

        # Translation (3 floats)
        trans_hbox = QHBoxLayout()
        self.tr_x = QDoubleSpinBox(); self.tr_x.setRange(-1000,1000); self.tr_x.setValue(0.0)
        self.tr_y = QDoubleSpinBox(); self.tr_y.setRange(-1000,1000); self.tr_y.setValue(0.0)
        self.tr_z = QDoubleSpinBox(); self.tr_z.setRange(-1000,1000); self.tr_z.setValue(0.0)
        for sb in (self.tr_x, self.tr_y, self.tr_z):
            sb.setSingleStep(0.1)
            trans_hbox.addWidget(sb)
        form.addRow("Translation:", trans_hbox)

        vbox.addLayout(form)

        # Auto Next checkbox
        # self.auto_next_cb = QCheckBox("Auto Next")
        # vbox.addWidget(self.auto_next_cb)

        # Combo box for selecting one item
        self.combo = QComboBox()
        vbox.addWidget(self.combo)

        # Buttons at bottom
        btn_hbox = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Metadata")
        self.refresh_btn.clicked.connect(self.init_hub)
        btn_hbox.addWidget(self.refresh_btn)

        self.load_btn = QPushButton("Load Selected")
        self.load_btn.clicked.connect(self.load_selected)
        btn_hbox.addWidget(self.load_btn)

        vbox.addLayout(btn_hbox)
        vbox.addStretch()


        # respiration_cycles = 5
        # respiration_rpm = 12 # 12 - 20
        # respiration_time = 60.0/respiration_rpm
        # respiration_period = int( respiration_time * bpy.context.scene.render.fps /2)
        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(QLabel("respiration_cycles"))
        self.respiration_cycles = QLineEdit("50")
        btn_hbox.addWidget(self.respiration_cycles)
        
        btn_hbox.addWidget(QLabel("respiration_angle_amplitude"))
        self.respiration_angle_amplitude = QLineEdit("8")
        btn_hbox.addWidget(self.respiration_angle_amplitude)
        btn_hbox.addWidget(QLabel("respiration_rpm"))
        self.respiration_rpm = QLineEdit("18")
        btn_hbox.addWidget(self.respiration_rpm)
        btn_hbox.addWidget(QLabel("subdiv"))
        self.subdiv = QLineEdit("10")
        btn_hbox.addWidget(self.subdiv)
        btn_hbox.addWidget(QLabel("fps"))
        self.fpslineedit = QLineEdit(str(bpy.context.scene.render.fps))
        btn_hbox.addWidget(self.fpslineedit)
        vbox.addLayout(btn_hbox)
        btn_hbox = QHBoxLayout()
        self.chest_btn = QPushButton("Add Chest")
        self.chest_btn.clicked.connect(self.chestOsc)
        btn_hbox.addWidget(self.chest_btn)
        self.simplescenario_btn = QPushButton("Simple Test Scenario")
        self.simplescenario_btn.clicked.connect(self.simplescenario)
        btn_hbox.addWidget(self.simplescenario_btn)
        vbox.addLayout(btn_hbox)
    
    def simplescenario(self):
        ssp.utils.initialize_environment()
        radar = ssp.radar.utils.addRadar(
            radarSensor=ssp.radar.utils.RadarSensorsCategory.TI_IWR6843,
            location_xyz=(0, -2, 0)
        )
        radar.rotation_euler.z=0
        radar['NPulse'] = 1
        radar['PRI_us'] = 50000
        radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=.1
        radar['RF_AnalogNoiseFilter_Bandwidth_MHz']=.1
        self.subdiv.setText("0")
        self.chestOsc()

    def chestOsc(self):
        bpy.context.scene.render.fps = int(self.fpslineedit.text())
        respiration_cycles = int(self.respiration_cycles.text())
        respiration_rpm = float(self.respiration_rpm.text())
        respiration_angle_amplitude = float(self.respiration_angle_amplitude.text())
        respiration_time = 60.0/respiration_rpm
        subdiv = int(self.subdiv.text())
        respiration_period = int( respiration_time * bpy.context.scene.render.fps /2)
        rcs0 = self.rcs0_sb.value()
        rot = (self.rot_x.value(), self.rot_y.value(), self.rot_z.value())
        trans = (self.tr_x.value(), self.tr_y.value(), self.tr_z.value())
        chest_skin = ssp.environment.generate_chest_vibration_target(respiration_angle_amplitude=respiration_angle_amplitude,
                                    bone_lengths=(0.07, 0.035, 0.035),
                                    chest_dimensions=(0.175, 0.14, 1),
                                    subdivisions=subdiv,
                                    respiration_cycles=respiration_cycles,
                                    respiration_period=respiration_period)
        chest_skin.location.x+=trans[0]
        chest_skin.location.y+=trans[1]
        chest_skin.location.z+=trans[2]
        chest_skin.rotation_euler.x+=np.deg2rad(rot[0])
        chest_skin.rotation_euler.y+=np.deg2rad(rot[1])
        chest_skin.rotation_euler.z+=np.deg2rad(rot[2])
        chest_skin["RCS0"]=rcs0
        self.close()
    def init_hub(self):
        """Load metadata and populate the combo box."""
        metadata = ssp.utils.hub.load_metadata()
        if not metadata:
            QMessageBox.warning(self, "Hub Loader", "No hub metadata available.")
            return

        self.combo.clear()
        self.items.clear()

        for category, files in metadata.items():
            # ensure temp folder exists
            category_folder = os.path.join(ssp.config.temp_folder, "hub", category)
            os.makedirs(category_folder, exist_ok=True)

            for item in files:
                name = item["name"]
                display = f"{category}: {name}"
                self.combo.addItem(display)
                self.items.append((category, name))

    def load_selected(self):
        """Fetch and load the selected .blend file, handle parameters and Auto Next."""
        idx = self.combo.currentIndex()
        if idx < 0 or idx >= len(self.items):
            QMessageBox.information(self, "Hub Loader", "Please select an item to load.")
            return

        category, name = self.items[idx]

        # fetch the blend file
        try:
            blend_path = ssp.utils.hub.fetch_file(category, name)
        except Exception as e:
            QMessageBox.critical(self, "Hub Loader", f"Failed to fetch '{name}':\n{e}")
            return

        if not os.path.exists(blend_path):
            QMessageBox.critical(self, "Hub Loader", f"File not found:\n{blend_path}")
            return

        # read UI parameters
        rcs0 = self.rcs0_sb.value()
        decim = self.decimate_sb.value()
        rot = (self.rot_x.value(), self.rot_y.value(), self.rot_z.value())
        trans = (self.tr_x.value(), self.tr_y.value(), self.tr_z.value())

        # import into Blender with all parameters
        ssp.environment.add_blenderfileobjects(
            blend_path,
            RCS0=rcs0,
            decimatefactor=decim,
            rotation=rot,
            translation=trans
        )
        self.close()
class RadarArrayApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Array Configuration")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Main container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.vbox = QVBoxLayout(main_widget)

        # JSON file path with browse button
        h_json = QHBoxLayout()
        h_json.addWidget(QLabel("Radar Array File:"))
        self.linedit = QLineEdit()
        self.arrayFolder = os.path.join(ssp.config.temp_folder, "arrayFolder")
        os.makedirs(self.arrayFolder, exist_ok=True)
        self.linedit.setText(os.path.join(self.arrayFolder, "radar_array.mat"))
        h_json.addWidget(self.linedit)
        self.loadfile_button = QPushButton("Load")
        self.loadfile_button.clicked.connect(self.loadfile)
        h_json.addWidget(self.loadfile_button)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        h_json.addWidget(self.browse_button)
        self.vbox.addLayout(h_json)

        # Scale selector
        h_scale = QHBoxLayout()
        h_scale.addWidget(QLabel("Position Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["half wavelength", "m", "mm"])
        h_scale.addWidget(self.scale_combo)
        h_scale.addWidget(QLabel("Frequency (GHz):"))
        self.f0_lineedit = QLineEdit()
        self.f0_lineedit.setText("77")
        h_scale.addWidget(self.f0_lineedit)
        self.vbox.addLayout(h_scale)

        # TX array position
        h_tx = QHBoxLayout()
        h_tx.addWidget(QLabel("TX Array Positions (format x,y|x2,y2):"))
        self.tx_lineedit = QLineEdit()
        self.tx_lineedit.setText("0,0|4,0")
        h_tx.addWidget(self.tx_lineedit)
        self.vbox.addLayout(h_tx)

        # RX array position
        h_rx = QHBoxLayout()
        h_rx.addWidget(QLabel("RX Array Positions (format x,y|x2,y2|...):"))
        self.rx_lineedit = QLineEdit()
        self.rx_lineedit.setText("0,0|1,0|2,0|3,0")
        h_rx.addWidget(self.rx_lineedit)
        self.vbox.addLayout(h_rx)
        
        h_rxb = QHBoxLayout()
        h_rxb.addWidget(QLabel("RX Array Position Bias (format x,y):"))
        self.rxb_lineedit = QLineEdit()
        self.rxb_lineedit.setText("10.5,10.5")
        h_rxb.addWidget(self.rxb_lineedit)
        self.vbox.addLayout(h_rxb)

        # Default configurations
        h_config = QHBoxLayout()
        h_config.addWidget(QLabel("Default Array Configurations:"))
        self.config_combo = QComboBox()
        self.config_combo.addItems(["2x4","2x4-2", "3x4", "3x4-2", "3x4-3", "12x16", "12x16-2"])
        self.config_combo.currentIndexChanged.connect(self.on_config_changed)
        h_config.addWidget(self.config_combo)
        self.vbox.addLayout(h_config)
        
        # Default configurations

        h_config2 = QHBoxLayout()
        h_config2.addWidget(QLabel("Virtual Array Processing"))
        self.va_combo = QComboBox()
        self.va_combo.addItems(["Az FFT","Az FFT,El Estimation","Az FFT,El FFT", "2D FFT"])
        self.va_combo.currentIndexChanged.connect(self.on_config_changed)
        h_config2.addWidget(self.va_combo)
        self.vaAuto_button = QPushButton("Auto")
        self.vaAuto_button.clicked.connect(self.apply_vaAuto)
        # h_config2.addWidget(self.vaAuto_button)
        self.vbox.addLayout(h_config2)


        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("VA order (TX,RX)->[X,Y]|"))
        self.vaorder_lineedit = QLineEdit()
        h_va.addWidget(self.vaorder_lineedit)
        self.vaorder_lineedit2 = QLineEdit()
        h_va.addWidget(self.vaorder_lineedit2)
        self.vbox.addLayout(h_va)
    
        s = ''
        k=0
        for itx in range(2):
            for irx in range(4):
                k+=1
                s+=f'({itx+1},{irx+1})->[{k},1] | '
        self.vaorder_lineedit.setText(s)



        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("MIMO Technique"))
        self.mimo_combo = QComboBox()
        self.mimo_combo.addItems(["TDM","BPM","DDM"])
        self.mimo_combo.currentIndexChanged.connect(self.on_mimo_changed)
        h_va.addWidget(self.mimo_combo)
        self.mimo_lineedit = QLineEdit()
        h_va.addWidget(self.mimo_lineedit)
        self.vbox.addLayout(h_va)
        M=2
        A = np.eye(M,M,dtype=np.complex128)
        s = ''
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                s+=f'{A[i,j]},'
            s = s[:-1]
            s+=';'
        s = s[:-1]
        self.mimo_lineedit.setText(s)
        
        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("X,Y distance scaling factor (sx,sy)"))
        self.disscale_lineedit = QLineEdit("1,1")
        h_va.addWidget(self.disscale_lineedit)
        self.vbox.addLayout(h_va)
        
        # Apply and additional buttons
        h_buttons = QHBoxLayout()
        self.apply_button = QPushButton("Visualzie")
        self.apply_button.clicked.connect(self.apply_settings)
        h_buttons.addWidget(self.apply_button)


        self.download_button = QPushButton("Download predesigned from Hub")
        self.download_button.clicked.connect(self.download_from_hub)
        # h_buttons.addWidget(self.download_button)

        self.visualize_button = QPushButton("Virtual Array Calc.")
        self.visualize_button.clicked.connect(self.visualize_virtual)
        h_buttons.addWidget(self.visualize_button)

        self.vbox.addLayout(h_buttons)

        # Matplotlib canvas for plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vbox.addWidget(self.canvas)

        # Load/Save/Run buttons
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Radar Array JSON")
        self.load_button.clicked.connect(self.load_data)
        # btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(QLabel("Radar ID"))
        self.id_lineedit = QLineEdit()
        self.id_lineedit.setText("001")
        btn_layout.addWidget(self.id_lineedit)
        self.save_button = QPushButton("Save Radar Array")
        self.save_button.clicked.connect(self.save_data)
        btn_layout.addWidget(self.save_button)

        self.run_button = QPushButton("Run Radar Array Analysis")
        self.run_button.clicked.connect(self.run_analysis)
    
        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("CFAR_RD_training_cells"))
        self.CFAR_RD_training_cells = QLineEdit("20")
        h_va.addWidget(self.CFAR_RD_training_cells)
        h_va.addWidget(QLabel("CFAR_RD_guard_cells"))
        self.CFAR_RD_guard_cells = QLineEdit("10")
        h_va.addWidget(self.CFAR_RD_guard_cells)
        h_va.addWidget(QLabel("CFAR_RD_alpha"))
        self.CFAR_RD_alpha = QLineEdit("5.0")
        h_va.addWidget(self.CFAR_RD_alpha)
    # radar['CFAR_Angle_training_cells']=10
    # radar['CFAR_Angle_guard_cells']=0
    # radar['CFAR_Angle_alpha']=2.0
        self.vbox.addLayout(h_va)
        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("CFAR_Angle_training_cells"))
        self.CFAR_Angle_training_cells = QLineEdit("10")
        h_va.addWidget(self.CFAR_Angle_training_cells)
        h_va.addWidget(QLabel("CFAR_Angle_guaAngle_cells"))
        self.CFAR_Angle_guard_cells = QLineEdit("0")
        h_va.addWidget(self.CFAR_Angle_guard_cells)
        h_va.addWidget(QLabel("CFAR_Angle_alpha"))
        self.CFAR_Angle_alpha = QLineEdit("2.0")
        h_va.addWidget(self.CFAR_Angle_alpha)
        self.vbox.addLayout(h_va)

        
    


        self.add_button = QPushButton("Add Radar")
        self.add_button.clicked.connect(self.run_add)
        btn_layout.addWidget(self.add_button)

        self.vbox.addLayout(btn_layout)
        # self.vbox.addStretch()
    def run_add(self):
        radar = ssp.radar.utils.addRadarFile(self.linedit.text(),float(self.f0_lineedit.text())*1e9)
        radar['CFAR_RD_training_cells']=int(self.CFAR_RD_training_cells.text()) 
        radar['CFAR_RD_guard_cells']=int(self.CFAR_RD_guard_cells.text())
        radar['CFAR_RD_alpha']=float(self.CFAR_RD_alpha.text())
        radar['CFAR_Angle_training_cells']=int(self.CFAR_Angle_training_cells.text())
        radar['CFAR_Angle_guard_cells']=int(self.CFAR_Angle_guard_cells.text())
        radar['CFAR_Angle_alpha']=float(self.CFAR_Angle_alpha.text())
        self.close()
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Radar Array", self.arrayFolder, "Files (*.mat)")
        if file_path:
            self.linedit.setText(file_path)
            self.load_data()
    def loadfile(self):
        data = scipy.io.loadmat(self.linedit.text())
        tx_positions   = data.get('tx_positions', None)
        rx_positions   = data.get('rx_positions', None)
        vaorder        = data.get('vaorder', None)
        vaorder2        = data.get('vaorder2', None)
        PrecodingMatrix = data.get('PrecodingMatrix', None)
        rx_bias =  data.get('rx_bias', None)
        AzELscale          = data.get('AzELscale', None)
        scale          = str(data.get('scale', None)[0])
        vaprocessing   = str(data.get('vaprocessing', None)[0])
        id = str(data.get('id', None)[0])
        self.id_lineedit.setText(id)
        self.scale_combo.setCurrentText(scale)
        self.va_combo.setCurrentText(vaprocessing)
        
        text = '|'.join(f"{x},{y}" for x, y in tx_positions)
        self.tx_lineedit.setText(text)
        text = '|'.join(f"{x},{y}" for x, y in rx_positions)
        self.rx_lineedit.setText(text)
        pieces = []
        for tx, rx, v1, v2 in vaorder:
            # ensure the same formatting you expect when decoding;
            # you can tweak float formatting here if desired
            pieces.append(f"({int(tx)},{int(rx)})->[{int(v1)},{int(v2)}] | ")
        s = ''.join(pieces)
        self.vaorder_lineedit.setText(s)
        pieces = []
        for tx, rx, v1, v2 in vaorder2:
            # ensure the same formatting you expect when decoding;
            # you can tweak float formatting here if desired
            pieces.append(f"({int(tx)},{int(rx)})->[{int(v1)},{int(v2)}] | ")
        s = ''.join(pieces)
        self.vaorder_lineedit2.setText(s)
        
        s = ''
        for i in range(PrecodingMatrix.shape[0]):
            for j in range(PrecodingMatrix.shape[1]):
                s+=f'{PrecodingMatrix[i,j]},'
            s = s[:-1]
            s+=';'
        s = s[:-1]
        self.mimo_lineedit.setText(s)
        
        self.rxb_lineedit.setText(f'{rx_bias[0][0]},{rx_bias[0][1]}')
        self.disscale_lineedit.setText(f'{AzELscale[0][0]},{AzELscale[0][1]}')

    def on_config_changed(self, index):
        selected_value = self.config_combo.currentText()
        self.scale_combo.setCurrentIndex(0)
            
        if selected_value == "2x4":
            self.tx_lineedit.setText("0,0|4,0")
            self.rx_lineedit.setText("0,0|1,0|2,0|3,0")
            self.va_combo.setCurrentIndex(0) # Az FFT
            s = ''
            k=0
            for itx in range(2):
                for irx in range(4):
                    k+=1
                    s+=f'({itx+1},{irx+1})->[{k},1] | '
            self.vaorder_lineedit.setText(s)
            s = ''
            self.vaorder_lineedit2.setText(s)
        if selected_value == "2x4-2":
            self.tx_lineedit.setText("0,0|2,0")
            self.rx_lineedit.setText("0,0|1,0|0,1|1,1")
            self.va_combo.setCurrentIndex(1) # Az FFT El Estimation
            s = ''
            s+=f'(1,1)->[1,1] | '
            s+=f'(1,2)->[2,1] | '
            s+=f'(2,1)->[3,1] | '
            s+=f'(2,2)->[4,1] |'
            self.vaorder_lineedit.setText(s)
            s = ''
            s+=f'(1,1)->[1,1] |'
            s+=f'(1,3)->[1,2] |'
            self.vaorder_lineedit2.setText(s)
        if selected_value == "3x4":
            self.tx_lineedit.setText("0,0|4,0|8,0")
            self.rx_lineedit.setText("0,0|1,0|2,0|3,0")
            
            self.va_combo.setCurrentIndex(0) # Az FFT
            s = ''
            k=0
            for itx in range(3):
                for irx in range(4):
                    k+=1
                    s+=f'({itx+1},{irx+1})->[{k},1] | '
            self.vaorder_lineedit.setText(s)
            s = ''
            self.vaorder_lineedit2.setText(s)
        if selected_value == "3x4-3":
            self.tx_lineedit.setText("0,0|4,1|8,0")
            self.rx_lineedit.setText("0,0|1,0|2,0|3,0")
            self.va_combo.setCurrentIndex(1) # Az FFT El Estimation
            s = ''
            k=0
            for itx in range(3):
                for irx in range(4):
                    k+=1
                    s+=f'({itx+1},{irx+1})->[{k},1] | '
            s = s[:-1]
            self.vaorder_lineedit.setText(s)
            s = ''
            s+=f'(1,1)->[1,1] |'
            s+=f'(2,1)->[1,2] |'
            self.vaorder_lineedit2.setText(s)
        if selected_value == "3x4-2":
            self.scale_combo.setCurrentIndex(0)
            self.tx_lineedit.setText("0,0|1,0|2,0")
            self.rx_lineedit.setText("0,0|0,1|0,2|0,3")
            self.va_combo.setCurrentIndex(3) # 2D FFT
            s = ''
            k=0
            for itx in range(3):
                for irx in range(4):
                    k+=1
                    s+=f'({itx+1},{irx+1})->[{itx+1},{irx+1}] | '
            
            self.vaorder_lineedit.setText(s)
            self.vaorder_lineedit2.setText('')
        if selected_value == "12x16":
            self.tx_lineedit.setText("0,0|4,0|8,0|9,1|10,4|11,6|12,0|16,0|20,0|24,0|28,0|32,0")
            self.rx_lineedit.setText("0,0|1,0|2,0|3,0|11,0|12,0|13,0|14,0|46,0|47,0|48,0|49,0|50,0|51,0|52,0|53,0")
            self.va_combo.setCurrentIndex(2) # Az FFT - El FFT
            s = ''
            s+=f'(1,1)->[1,1] | '
            s+=f'(1,2)->[2,1] | '
            s+=f'(1,3)->[3,1] | '
            s+=f'(1,4)->[4,1] | '
            
            s+=f'(2,1)->[5,1] | '
            s+=f'(2,2)->[6,1] | '
            s+=f'(2,3)->[7,1] | '
            s+=f'(2,4)->[8,1] | '
            
            s+=f'(3,1)->[9,1] | '
            s+=f'(3,2)->[10,1] | '
            s+=f'(3,3)->[11,1] | '
            s+=f'(3,4)->[12,1] | '
            
            s+=f'(7,1)->[13,1] | '
            s+=f'(7,2)->[14,1] | '
            s+=f'(7,3)->[15,1] | '
            s+=f'(7,4)->[16,1] | '
            
            s+=f'(8,1)->[17,1] | '
            s+=f'(8,2)->[18,1] | '
            s+=f'(8,3)->[19,1] | '
            s+=f'(8,4)->[20,1] | '
            
            s+=f'(9,1)->[21,1] | '
            s+=f'(9,2)->[22,1] | '
            s+=f'(9,3)->[23,1] | '
            s+=f'(9,4)->[24,1] | '
            
            s+=f'(10,1)->[25,1] | '
            s+=f'(10,2)->[26,1] | '
            s+=f'(10,3)->[27,1] | '
            s+=f'(10,4)->[28,1] | '
            
            s+=f'(11,1)->[29,1] | '
            s+=f'(11,2)->[30,1] | '
            s+=f'(11,3)->[31,1] | '
            s+=f'(11,4)->[32,1] | '
            
            s+=f'(12,1)->[33,1] | '
            s+=f'(12,2)->[34,1] | '
            s+=f'(12,3)->[35,1] | '
            s+=f'(12,4)->[36,1] | '
            
            s+=f'(10,6)->[37,1] | '
            s+=f'(10,7)->[38,1] | '
            s+=f'(10,8)->[39,1] | '
            
            s+=f'(11,5)->[40,1] | '
            s+=f'(11,6)->[41,1] | '
            s+=f'(11,7)->[42,1] | '
            s+=f'(11,8)->[43,1] | '
            
            s+=f'(12,5)->[44,1] | '
            s+=f'(12,6)->[45,1] | '
            s+=f'(12,7)->[46,1] | '
            
            s+=f'(1,9)->[47,1] | '
            s+=f'(1,10)->[48,1] | '
            s+=f'(1,11)->[49,1] | '
            s+=f'(1,12)->[50,1] | '
            s+=f'(1,13)->[51,1] | '
            s+=f'(1,14)->[52,1] | '
            s+=f'(1,15)->[53,1] | '
            s+=f'(1,16)->[54,1] | '
            
            s+=f'(3,9)->[55,1] | '
            s+=f'(3,10)->[56,1] | '
            s+=f'(3,11)->[57,1] | '
            s+=f'(3,12)->[58,1] | '
            s+=f'(3,13)->[59,1] | '
            s+=f'(3,14)->[60,1] | '
            s+=f'(3,15)->[61,1] | '
            s+=f'(3,16)->[62,1] | '
            
            s+=f'(8,9)->[63,1] | '
            s+=f'(8,10)->[64,1] | '
            s+=f'(8,11)->[65,1] | '
            s+=f'(8,12)->[66,1] | '
            s+=f'(8,13)->[67,1] | '
            s+=f'(8,14)->[68,1] | '
            s+=f'(8,15)->[69,1] | '
            s+=f'(8,16)->[70,1] | '
            
            s+=f'(10,9)->[71,1] | '
            s+=f'(10,10)->[72,1] | '
            s+=f'(10,11)->[73,1] | '
            s+=f'(10,12)->[74,1] | '
            s+=f'(10,13)->[75,1] | '
            s+=f'(10,14)->[76,1] | '
            s+=f'(10,15)->[77,1] | '
            s+=f'(10,16)->[78,1] | '

            s+=f'(12,9)->[79,1] | '
            s+=f'(12,10)->[80,1] | '
            s+=f'(12,11)->[81,1] | '
            s+=f'(12,12)->[82,1] | '
            s+=f'(12,13)->[83,1] | '
            s+=f'(12,14)->[84,1] | '
            s+=f'(12,15)->[85,1] | '
            s+=f'(12,16)->[86,1] | '
            self.vaorder_lineedit.setText(s)
            s = ''
            s+=f'(1,1)->[1,1] |'
            s+=f'(4,1)->[1,2] |'
            s+=f'(5,1)->[1,3] |'
            s+=f'(6,1)->[1,4] |'
            self.vaorder_lineedit2.setText(s)

        
        if selected_value == "12x16-2":
            self.scale_combo.setCurrentIndex(0)
            self.tx_lineedit.setText("0,0|4,0|8,0|12,0|0,4|4,4|8,4|12,4|0,8|4,8|8,8|12,8")
            self.rx_lineedit.setText("0,0|1,0|2,0|3,0|0,1|1,1|2,1|3,1|0,2|1,2|2,2|3,2|0,3|1,3|2,3|3,3")
            self.va_combo.setCurrentIndex(3) # 2D FFT
            s = ''
            k=0
            for itx1 in range(3):
                for itx2 in range(4):
                    itx = itx1*4+itx2
                    for irx1 in range(4):
                        for irx2 in range(4):
                            irx = irx1*4+irx2
                            k+=1
                            s+=f'({itx+1},{irx+1})->[{itx2*4+irx2+1},{itx1*4+irx1+1}] | '
            self.vaorder_lineedit.setText(s)
            self.vaorder_lineedit2.setText('')
    
    
        self.on_mimo_changed(self.mimo_combo.currentIndex())
    
    def apply_vaAuto(self):
        tx_positions = [tuple(map(float, p.split(','))) for p in self.tx_lineedit.text().split('|')]
        rx_positions = [tuple(map(float, p.split(','))) for p in self.rx_lineedit.text().split('|')]
        tx_x, tx_y = zip(*tx_positions)
        rx_x, rx_y = zip(*rx_positions)
        va = []
        for tx in tx_positions:
            for rx in rx_positions:
                va.append([tx[0]+rx[0],tx[1]+rx[1]])
        va_x, va_y = zip(*va)
        
        xs = sorted({x for x,y in va})
        ys = sorted({y for x,y in va})


        # empty["antenna2azelIndex"]=[xs,ys]

        MTX = len(tx_positions)
        NRX = len(tx_positions)

        antennaSignal = np.zeros((MTX, NRX), dtype=np.complex128)
        for m in range(MTX):
            for n in range(NRX):
                antennaSignal[m, n] = (m+1) + 1j * (n+1)
        
        
        
        xs,ys=list(xs),list(ys)
        X = np.zeros((len(xs),len(ys)), dtype=complex)
        X2 = np.zeros((len(xs),len(ys)), dtype=complex)
        i_list = []
        j_list = []
        for m in range(MTX):
            for n in range(NRX):
                xv, yv = -tx_positions[m][0]+rx_positions[n][0], tx_positions[m][1]+rx_positions[n][1]
                i, j = xs.index(xv), ys.index(yv)
                if i is not None and j is not None:
                    i_list.append(i)
                    j_list.append(j)
                X[i,j] = antennaSignal[m,n]
        X2[i_list, j_list] = antennaSignal.ravel()
        er = np.linalg.norm(X-X2)
        
    def on_mimo_changed(self, index):
        selected_value = self.mimo_combo.currentText()
        tx_positions = [tuple(map(float, p.split(','))) for p in self.tx_lineedit.text().split('|')]
        M = len(tx_positions)
        A=ssp.radar.utils.mimo.MIMO_Functions().AD_matrix(M,M,selected_value)
        s = ''
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                s+=f'{A[i,j]},'
            s = s[:-1]
            s+=';'
        s = s[:-1]
        self.mimo_lineedit.setText(s)
            

    def apply_settings(self):
        # Parse positions
        tx_positions = [tuple(map(float, p.split(','))) for p in self.tx_lineedit.text().split('|')]
        rx_positions = [tuple(map(float, p.split(','))) for p in self.rx_lineedit.text().split('|')]
        rx_bias = tuple(map(float, self.rxb_lineedit.text().split(',')))
        for i in range(len(rx_positions)):
            rx_positions[i] = (rx_positions[i][0]+rx_bias[0], rx_positions[i][1]+rx_bias[1])
        
        scale = self.scale_combo.currentText()
        config = self.config_combo.currentText()
        
        # Plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        tx_x, tx_y = zip(*tx_positions)
        rx_x, rx_y = zip(*rx_positions)
        va = []
        for tx in tx_positions:
            for rx in rx_positions:
                va.append([tx[0]+rx[0],tx[1]+rx[1]])
        va_x, va_y = zip(*va)
        ax.scatter(tx_x, tx_y, marker='^', label='TX')
        ax.scatter(rx_x, rx_y, marker='x', label='RX')
        ax.scatter(va_x, va_y, marker='+', label='VA')
        ax.set_title(f'TX/RX Positions ({config}, scale={scale})')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        self.canvas.draw()
        
    def download_from_hub(self):
        # ssp.visualization.visualize_array()
        # Placeholder: implement hub download logic
        pass

    def visualize_virtual(self):
        import matplotlib
        matplotlib.use('Qt5Agg') 
        self.save_data()
        data = scipy.io.loadmat(self.linedit.text())
        tx_positions   = data.get('tx_positions', None)
        rx_positions   = data.get('rx_positions', None)
        vaorder        = data.get('vaorder', None)
        vaorder2        = data.get('vaorder2', None)
        PrecodingMatrix = data.get('PrecodingMatrix', None)
        scale          = str(data.get('scale', None)[0])
        vaprocessing   = str(data.get('vaprocessing', None)[0])
        id=data.get('id', None)
        
        # plt.figure()
        # for i in range(len(results)):
        #     plt.plot(results[i][2], results[i][3], 'rx')
        #     text = f"{int(results[i][0])},{int(results[i][1])}"
        #     plt.text(results[i][2], results[i][3], text, fontsize=7)
        # plt.gca().set_aspect('equal')
        # plt.xlabel('Y')
        # plt.ylabel('Z')
        
        MTX = len(tx_positions)
        NRX = len(rx_positions)
        
        antennaSignal = np.zeros((MTX, NRX), dtype=np.complex128)
        for m in range(MTX):
            for n in range(NRX):
                antennaSignal[m, n] = (m+1) + 1j * (n+1)
        
        Mx = int(max(rec[2] for rec in vaorder))
        Ny = int(max(rec[3] for rec in vaorder))
        
        X = np.zeros((Mx,Ny), dtype=complex)
        for i in range(len(vaorder)):
            m=int(vaorder[i][0])-1
            n=int(vaorder[i][1])-1
            ix=int(vaorder[i][2])-1
            iy=int(vaorder[i][3])-1
            X[ix,iy] = antennaSignal[m,n]
        
        plt.figure()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = X[i, j]
                text = f"{int(val.real)},{int(val.imag)}"
                plt.text(i, j, text, ha="center", va="center", fontsize=7)
        plt.gca().set_xlim(-0.5, X.shape[0]-0.5)
        plt.gca().set_ylim(X.shape[1]-0.5, -0.5)
        plt.gca().invert_yaxis() 
        plt.gca().set_xlabel("Azimuth Index")
        plt.gca().set_ylabel("Elevation Index")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().set_aspect('equal')
        plt.tight_layout()    
        plt.show()
    # Placeholder methods
    def load_data(self):
        # Placeholder: implement JSON loading
        pass

    def save_data(self):
        tx_positions = [tuple(map(float, p.split(','))) for p in self.tx_lineedit.text().split('|')]
        rx_positions = [tuple(map(float, p.split(','))) for p in self.rx_lineedit.text().split('|')]
        s=self.vaorder_lineedit.text()
        pattern = re.compile(r'\(\s*([^\],]+)\s*,\s*([^\]]+)\s*\)->\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]')
        vaorder = []
        for m in pattern.finditer(s):
            tx, rx, v1, v2 = m.groups()
            # convert to appropriate types; float if possible, else leave as string
            try:
                v1 = float(v1)
                v2 = float(v2)
            except ValueError:
                pass
            vaorder.append((int(float(tx)), int(float(rx)), v1, v2))
        s=self.vaorder_lineedit2.text()
        vaorder2 = []
        for m in pattern.finditer(s):
            tx, rx, v1, v2 = m.groups()
            # convert to appropriate types; float if possible, else leave as string
            try:
                v1 = float(v1)
                v2 = float(v2)
            except ValueError:
                pass
            vaorder2.append((int(float(tx)), int(float(rx)), v1, v2))
        scale = self.scale_combo.currentText()
        vaprocessing = self.va_combo.currentText()
        
        txt = self.mimo_lineedit.text()
        row_strs = txt.split(';')
        data = []
        for r in row_strs:
            items = [s.strip() for s in r.split(',')]
            row = [complex(item) for item in items]
            data.append(row)

        PrecodingMatrix = np.array(data, dtype=complex)
        rx_bias = tuple(map(float, self.rxb_lineedit.text().split(',')))
        id = self.id_lineedit.text()
        AzELscale = tuple(map(float, self.disscale_lineedit.text().split(',')))
        
        scipy.io.savemat(self.linedit.text(), {'tx_positions': tx_positions,'rx_positions': rx_positions,'vaorder': vaorder,'vaorder2': vaorder2,'scale': scale,'vaprocessing': vaprocessing,'PrecodingMatrix': PrecodingMatrix,'rx_bias': rx_bias,'id':id,'AzELscale': AzELscale})


    def run_analysis(self):
        # Placeholder: implement analysis
        pass

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

        # File path entry (full width)
        self.linedit = QLineEdit()
        self.linedit.setText(os.path.join(ssp.config.temp_folder, "RIS_data.mat"))
        self.vbox.addWidget(self.linedit)

        # Freq (GHz)
        hbox_f0 = QHBoxLayout()
        self.label = QLabel("Freq (GHz):")
        hbox_f0.addWidget(self.label)
        self.f0_lineedit = QLineEdit()
        self.f0_lineedit.setText("62")
        hbox_f0.addWidget(self.f0_lineedit)
        self.vbox.addLayout(hbox_f0)

        # Nx,Ny,dx,dy
        hbox_n = QHBoxLayout()
        self.label2 = QLabel("Nx,Ny,dx,dy:")
        hbox_n.addWidget(self.label2)
        self.n_lineedit = QLineEdit()
        self.n_lineedit.setText("100,100,0.5,0.5")
        hbox_n.addWidget(self.n_lineedit)
        self.vbox.addLayout(hbox_n)

        # Radar Position
        hbox_radar = QHBoxLayout()
        self.labelradarpos = QLabel("Radar Position (x,y,z):")
        hbox_radar.addWidget(self.labelradarpos)
        self.radarpos_lineedit = QLineEdit()
        self.radarpos_lineedit.setText("-0.5,0,-.5")
        hbox_radar.addWidget(self.radarpos_lineedit)
        self.vbox.addLayout(hbox_radar)

        # Target Position
        hbox_target = QHBoxLayout()
        self.labeltargetpos = QLabel("Target Position (x,y,z):")
        hbox_target.addWidget(self.labeltargetpos)
        self.targetpos_lineedit = QLineEdit()
        self.targetpos_lineedit.setText("0.5,0,-0.5")
        hbox_target.addWidget(self.targetpos_lineedit)
        self.vbox.addLayout(hbox_target)

        # Generate buttons
        self.generate_button = QPushButton("Generate RIS Scenario 1")
        self.generate_button.clicked.connect(self.generate_ris_data)
        self.vbox.addWidget(self.generate_button)

        self.generate_button2 = QPushButton("Generate RIS Scenario 2")
        self.generate_button2.clicked.connect(self.generate_ris_data2)
        self.vbox.addWidget(self.generate_button2)

        # Phase limits
        hbox_phase = QHBoxLayout()
        phase_label = QLabel("Phase Limits Down, UP, Levels:")
        hbox_phase.addWidget(phase_label)
        self.phaselims_lineedit = QLineEdit()
        self.phaselims_lineedit.setText("-160,150,100")
        hbox_phase.addWidget(self.phaselims_lineedit)
        self.vbox.addLayout(hbox_phase)

        self.generate_button3 = QPushButton("Generate RIS Scenario 3")
        self.generate_button3.clicked.connect(self.generate_ris_data3)
        self.vbox.addWidget(self.generate_button3)

        self.vbox.addStretch()

        # N Grid
        hbox_ng = QHBoxLayout()
        ng_label = QLabel("N Grid:")
        hbox_ng.addWidget(ng_label)
        self.ng_lineedit = QLineEdit()
        self.ng_lineedit.setText("40")
        hbox_ng.addWidget(self.ng_lineedit)
        self.vbox.addLayout(hbox_ng)

        # d(Wavelength), Arc(deg) Grid
        hbox_arc = QHBoxLayout()
        arc_label = QLabel("d(Wavelength), Arc(deg) Grid:")
        hbox_arc.addWidget(arc_label)
        self.arc_lineedit = QLineEdit()
        self.arc_lineedit.setText("2,40")
        hbox_arc.addWidget(self.arc_lineedit)
        self.vbox.addLayout(hbox_arc)

        # Dynamic Range
        hbox_dynr = QHBoxLayout()
        dynr_label = QLabel("Dynamic Range (dB):")
        hbox_dynr.addWidget(dynr_label)
        self.dynr_lineedit = QLineEdit()
        self.dynr_lineedit.setText("100")
        hbox_dynr.addWidget(self.dynr_lineedit)
        self.vbox.addLayout(hbox_dynr)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_analysis)
        self.vbox.addWidget(self.run_button)
        self.run_button2 = QPushButton("Run2")
        self.run_button2.clicked.connect(self.run_analysis2)
        # self.vbox.addWidget(self.run_button2)
        self.vbox.addStretch()

        # Blender buttons
        self.blender_button = QPushButton("Blender scenario BR 18")
        self.blender_button.clicked.connect(self.run_blender_button)
        self.vbox.addWidget(self.blender_button)

        self.blender_button2 = QPushButton("Blender scenario BR 18+HR 70")
        self.blender_button2.clicked.connect(self.run_blender_button2)
        self.vbox.addWidget(self.blender_button2)

        self.blenderpath_button = QPushButton("Path sim")
        self.blenderpath_button.clicked.connect(self.run_blenderpath_button)
        self.vbox.addWidget(self.blenderpath_button)

    def generate_ris_data(self):
        import matplotlib
        matplotlib.use('Qt5Agg') 
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
        import matplotlib
        matplotlib.use('Qt5Agg') 
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
    def generate_ris_data3(self):
        import matplotlib
        matplotlib.use('Qt5Agg') 
        f0 = self.f0_lineedit.text()
        f0 = float(f0)
        f0 = f0 * 1e9
        Lambda = 3e8 / f0
        nxnyLxLy = self.n_lineedit.text().split(",")
        if len(nxnyLxLy) != 4:
            QMessageBox.warning(self, "Input Error", "Please enter Nx, Ny, dx, dy in the format: Nx,Ny,dx,dy")
            return
        phaselims = self.phaselims_lineedit.text().split(",")
        if len(phaselims) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter lims")
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
        phase = np.angle(a)
        min_phase = np.deg2rad(float(phaselims[0]))
        phase[phase < min_phase] = min_phase
        max_phase = np.deg2rad(float(phaselims[1]))
        phase[phase > max_phase] = max_phase
        phase = phase - min_phase 
        phase = phase / (max_phase -min_phase)
        phase = np.floor(phase * float(phaselims[2]))
        phase = phase / float(phaselims[2])
        phase = phase * (max_phase - min_phase)
        phase = phase + min_phase 

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
        import matplotlib
        matplotlib.use('Qt5Agg') 
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
        import matplotlib
        matplotlib.use('Qt5Agg') 
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
    def run_analysis2(self):
        import matplotlib
        matplotlib.use('Qt5Agg') 
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
    if st=="Antenna Element Pattern":
        runrAntennaElementPatternapp()
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
def runradarArrayapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = RadarArrayApp()
    window.show()
    app.exec_()  # Do not use sys.exit()
def runRadarVis():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = RadarVisApp()
    window.show()
    app.exec_()  # Do not use sys.exit()

def runrAntennaElementPatternapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = AntennaElementPattern()
    window.show()
    app.exec_()  # Do not use sys.exit()
    

def runHubLoad():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    window = HubLoadApp()
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


