import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton, QHBoxLayout, QScrollArea, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen,QTransform
from PyQt5.QtWidgets import (
    QPushButton, QTableWidget, QTableWidgetItem, QGraphicsView,QTabWidget,QStackedWidget,
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
from mathutils import Quaternion
# import numpy as np
# from PyQt5.QtWidgets import (
#     QMainWindow, QApplication, QWidget, QVBoxLayout,
#     QPushButton, QFileDialog, QLabel, QHBoxLayout
# )
in_colab = 'google.colab' in sys.modules

# 2) Detect if we’re on an HPC batch job (common env‑vars: SLURM, PBS, LSF)
hpc_vars = ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID')
in_hpc = any(var in os.environ for var in hpc_vars)

# 3) Detect local GUI‑capable session:
#    - On Windows or macOS we assume a GUI is available.
#    - On Linux, require a DISPLAY to be set.
import platform
system = platform.system()  # 'Windows', 'Darwin' (macOS) or 'Linux'
if system in ('Windows', 'Darwin'):
    local_gui = True
elif system == 'Linux':
    local_gui = bool(os.environ.get('DISPLAY'))
else:
    local_gui = False

# 4) Finally: only switch backend on a local GUI session,
#    and never on HPC or Colab.
if local_gui and not in_hpc and not in_colab:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Qt5Agg')


# Use Qt5 backend for Matplotlib
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

def process_events():
    QApplication.processEvents()
    bpy.context.view_layer.update()
    for area in bpy.context.screen.areas:
        area.tag_redraw()
def guess_memory_usage(rays):
    return sys.getsizeof(rays)

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
        # ssp.environment.scenarios.predefine_movingcube_6843()
        self.initUI()

    def initUI(self):
        # Main container widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        vbox = QVBoxLayout(main_widget)
        # Controls layout: Run button, Next button, ComboBox
        controls_layout = QHBoxLayout()
        self.run_button = QPushButton("Initialize")
        self.next_button = QPushButton("Analysis Next Frame")
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
        ssp.config.restart()
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
    def temp(self):
        X_fft_fast, d_fft = ssp.radar.utils.rangeprocessing(XRadar, specifications)
        rangeDopplerTXRX, f_Doppler = ssp.radar.utils.dopplerprocessing_mimodemodulation(X_fft_fast, specifications)
        detections, cfar_threshold, rangeDoppler4CFAR = ssp.radar.utils.rangedoppler_detection_alpha(rangeDopplerTXRX, specifications)
        detected_points = np.where(detections == 1)
        Azimuth_Angles,debug_spectrums = ssp.radar.utils.angleprocessing_capon1D(rangeDopplerTXRX, detections, specifications)
        pc = ssp.radar.utils.pointscloud(d_fft[detected_points[0]], f_Doppler[detected_points[1]], Azimuth_Angles)
    
    def on_next(self):
        ssp.utils.research.simplest.ui_frame_processing(self)
        return
        for a in self.axes:
            a.cla()
        isuite = int(self.combo2.currentText().split(',')[0])
        iradar = int(self.combo2.currentText().split(',')[1])
        
        specifications = ssp.RadarSpecifications[isuite][iradar]
        
        # suite_information = ssp.environment.BlenderSuiteFinder().find_suite_information()
        # radobj = suite_information[isuite]['Radar'][iradar]['GeneralRadarSpec_Object'] 
        
        
        
           
    def on_next_old2(self):
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
        self.statusBar().showMessage("raytracing ...");QApplication.processEvents()
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        self.statusBar().showMessage("SensorsSignalGeneration ...");QApplication.processEvents()
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
        
        self.axes[1].plot(np.real(XRadar[:, 0, 0]), label="Real Part")
        self.axes[1].plot(np.imag(XRadar[:, 0, 0]), label="Imaginary Part")
        self.axes[1].set_xlabel("ADC Samples")
        self.axes[1].set_ylabel("ADC Output Level")
        self.axes[1].legend(loc='upper right')
        self.axes[0].imshow(np.abs(XRadar[:, :, 0]),extent=[1*PRI, XRadar.shape[1]*PRI, 1, XRadar.shape[0]],
                            aspect='auto', origin='lower')
        self.axes[0].set_xlabel("Slow time")
        self.axes[0].set_ylabel("ADC Samples")
        
        self.statusBar().showMessage("Range Doppler Processing ...");QApplication.processEvents()
        
        X_fft_fast, d_fft = ssp.radar.utils.rangeprocessing(XRadar, specifications)
        
        self.axes[3].plot(d_fft, np.abs(X_fft_fast[:, 0, 0]))
        self.axes[3].set_xlabel("Range (m)")
        self.axes[3].set_ylabel("Range Profile")
        self.axes[2].imshow(np.abs(X_fft_fast[:, :, 0]),
                            extent=[1*PRI, XRadar.shape[1]*PRI, d_fft[0], d_fft[-1]],
                            aspect='auto', origin='lower')
        self.axes[2].set_xlabel("Slow time")
        self.axes[2].set_ylabel("Range")
        
        M_TX=PrecodingMatrix.shape[1]#specifications['M_TX']
        L = X_fft_fast.shape[1]
        Leff = int(L/M_TX)
        if Leff == 0:
            self.canvas.draw()
            self.canvas.flush_events()
            return
        
        rangeDopplerTXRX, f_Doppler = ssp.radar.utils.dopplerprocessing_mimodemodulation(X_fft_fast, specifications)
        
        
        im = self.axes[4].imshow(np.abs(rangeDopplerTXRX[:, :, 0, 0]),
                                extent=[f_Doppler[0], f_Doppler[-1], d_fft[0], d_fft[-1]],
                                aspect='auto', origin='lower')
        self.axes[4].set_xlabel("Doppler Frequency (Hz)")
        self.axes[4].set_ylabel("Range (m)")
        
        # rangeTXRX = np.zeros((rangeDopplerTXRX.shape[0],rangeDopplerTXRX.shape[2],rangeDopplerTXRX.shape[3]),dtype=rangeDopplerTXRX.dtype)
        
        detections, cfar_threshold, rangeDoppler4CFAR = ssp.radar.utils.rangedoppler_detection_alpha(rangeDopplerTXRX, specifications,self.doCFARCB.isChecked())
        
        # rangeDoppler4CFAR -= np.min(rangeDoppler4CFAR)
        
        distance = np.linspace(d_fft[0], d_fft[-1], rangeDoppler4CFAR.shape[0])  # Replace with real distance data
        elevation = np.linspace(f_Doppler[0],f_Doppler[-1], rangeDoppler4CFAR.shape[1])  # Replace with real angle data
        X, Y = np.meshgrid(elevation, distance)
        # FigsAxes[1,2].plot_surface(X, Y, 10*np.log10(rangeDoppler4CFAR), cmap='viridis', alpha=1)
        self.axes[5].plot_surface(X, Y, (rangeDoppler4CFAR), cmap='viridis', alpha=1)
        self.axes[5].set_xlabel('Doppler (Hz)')
        self.axes[5].set_ylabel('Distance (m)')
        self.axes[5].set_zlabel('Magnitude (normalized, dB)')
        detected_points = np.where(detections == 1)
        self.axes[5].scatter(elevation[detected_points[1]], distance[detected_points[0]], 
                    (rangeDoppler4CFAR[detected_points]), color='red', s=20, label='Post-CFAR Point Cloud')
        
        points = ssp.radar.utils.xyz_angleprocessing(rangeDopplerTXRX, detections, specifications)
        
        
        # detected_points = np.where(detections == 1)
        # Azimuth_Angles,debug_spectrums = ssp.radar.utils.angleprocessing_capon1D(rangeDopplerTXRX, detections, specifications)
        # pc = ssp.radar.utils.pointscloud(d_fft[detected_points[0]], f_Doppler[detected_points[1]], Azimuth_Angles)
            
        ssp.utils.increaseCurrentFrame()
        
        self.statusBar().showMessage("Done")
    def on_next_old(self):
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
        self.statusBar().showMessage("raytracing ...");QApplication.processEvents()
        path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
        self.statusBar().showMessage("SensorsSignalGeneration ...");QApplication.processEvents()
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
        
        self.axes[1].plot(np.real(XRadar[:, 0, 0]), label="Real Part")
        self.axes[1].plot(np.imag(XRadar[:, 0, 0]), label="Imaginary Part")
        self.axes[1].set_xlabel("ADC Samples")
        self.axes[1].set_ylabel("ADC Output Level")
        self.axes[1].legend(loc='upper right')
        self.axes[0].imshow(np.abs(XRadar[:, :, 0]),extent=[1*PRI, XRadar.shape[1]*PRI, 1, XRadar.shape[0]],
                            aspect='auto', origin='lower')
        self.axes[0].set_xlabel("Slow time")
        self.axes[0].set_ylabel("ADC Samples")
        
        self.statusBar().showMessage("Range Doppler Processing ...");QApplication.processEvents()
        
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
        self.statusBar().showMessage("Range Doppler CFAR ...");QApplication.processEvents()
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
            specifications['vaprocessing'] = "Az FFT"
        elif specifications['vaprocessing']=="Az FFT":
            rangeVA = np.zeros((int(np.max(specifications['vaorder'][:,2])),int(np.max(specifications['vaorder'][:,3]))),dtype=rangeDopplerTXRX.dtype)
            Lambda = 1.0
            AzELscale = specifications['AzELscale']
            dy = .5*Lambda*AzELscale[0]
            dz = .5*Lambda*AzELscale[1]
        
        elif specifications['vaprocessing']=="Az FFT,El Estimation":
            rangeVA = np.zeros((int(np.max(specifications['vaorder'][:,2])),int(np.max(specifications['vaorder'][:,3]))),dtype=rangeDopplerTXRX.dtype)
            rangeVA_2 = np.zeros((int(np.max(specifications['vaorder2'][:,2])),int(np.max(specifications['vaorder2'][:,3]))),dtype=rangeDopplerTXRX.dtype)
            Lambda = 1.0
            AzELscale = specifications['AzELscale']
            dy = .5*Lambda*AzELscale[0]
            dz = .5*Lambda*AzELscale[1]
        
            
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
            self.statusBar().showMessage(f"Angle Processing {id} from {NDetection} ...");QApplication.processEvents()
            rangeTarget = d_fft[detected_points[0][id]]
            dopplerTarget = f_Doppler[detected_points[1][id]]
            
            antennaSignal = rangeDopplerTXRX[detected_points[0][id],detected_points[1][id],:,:]
             
            
            for indx in specifications['vaorder']:
                rangeVA[int(indx[2]-1), int(indx[3]-1)] = antennaSignal[int(indx[0]-1), int(indx[1]-1)]
            # rangeVA = np.zeros((np.max(i_list)+1,np.max(j_list)+1),dtype=antennaSignal.dtype)
            # rangeVA[i_list, j_list] = antennaSignal.ravel()
            if specifications['vaprocessing']=="Az FFT,El Estimation":
                for indx in specifications['vaorder2']:
                    rangeVA_2[int(indx[2]-1), int(indx[3]-1)] = antennaSignal[int(indx[0]-1), int(indx[1]-1)]
                
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
        
        self.statusBar().showMessage("Done")
    
    
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
        self.setGeometry(100, 100, 1100, 600)
        # ssp.environment.scenarios.predefine_movingcube_6843()
        self.initUI()

    def initUI(self):
        # Main container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.vbox_main = QVBoxLayout(main_widget)
        wid1 = QWidget()
        self.vbox = QVBoxLayout(wid1)
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
        # self.vbox.addLayout(h_json)

        h_config = QHBoxLayout()
        h_config.addWidget(QLabel("Array initialization:"))
        self.comboarrayinititype = QComboBox()
        self.comboarrayinititype.addItems(["This UI", "Blender with this System Configurations", "Blender TDM"])
        h_config.addWidget(self.comboarrayinititype)
        # self.vbox.addLayout(h_config)
        
        # Default configurations
        h_config = QHBoxLayout()
        h_config.addWidget(QLabel("Default Array Configurations:"))
        self.config_combo = QComboBox()
        self.config_combo.addItems(ssp.utils.QtUI_arrays())
        self.config_combo.currentIndexChanged.connect(self.on_config_changed)
        h_config.addWidget(self.config_combo)
        # self.vbox.addLayout(h_config)
        # Scale selector
        h_scale = h_config#QHBoxLayout()
        h_scale.addWidget(QLabel("Position Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["half wavelength", "m", "mm"])
        h_scale.addWidget(self.scale_combo)
        # h_scale.addWidget(QLabel("Frequency (GHz):"))
        # self.f0_lineedit = QLineEdit()
        # self.f0_lineedit.setText("77")
        # h_scale.addWidget(self.f0_lineedit)
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
        # self.vbox.addLayout(h_rxb)

        h_va = h_rxb#QHBoxLayout()
        h_va.addWidget(QLabel("X,Y distance scaling factor (sx,sy)"))
        self.disscale_lineedit = QLineEdit("1,1")
        h_va.addWidget(self.disscale_lineedit)
        self.vbox.addLayout(h_va)
        
        # Default configurations

        # h_config2 = QHBoxLayout()
        # h_config2.addWidget(QLabel("Virtual Array Processing"))
        # self.va_combo = QComboBox()
        # self.va_combo.addItems(["Az FFT","Az FFT,El Estimation","Az FFT,El FFT", "2D FFT"])
        # self.va_combo.currentIndexChanged.connect(self.on_config_changed)
        # h_config2.addWidget(self.va_combo)
        # self.vaAuto_button = QPushButton("Auto")
        # self.vaAuto_button.clicked.connect(self.apply_vaAuto)
        # # h_config2.addWidget(self.vaAuto_button)
        # self.vbox.addLayout(h_config2)

        h_buttons = QHBoxLayout()
        self.apply_button = QPushButton("Visualzie")
        self.apply_button.clicked.connect(self.apply_settings)
        h_buttons.addWidget(self.apply_button)

        self.add_button = QPushButton("Add Radar")
        self.add_button.clicked.connect(self.run_add)
        h_buttons.addWidget(self.add_button)
        self.vbox.addLayout(h_buttons)

        # Matplotlib canvas for plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vbox.addWidget(self.canvas)
        self.vbox.addWidget(QLabel("System Configurations:"))

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
        
        
        # Apply and additional buttons
        h_buttons = QHBoxLayout()


        self.download_button = QPushButton("Download predesigned from Hub")
        self.download_button.clicked.connect(self.download_from_hub)
        # h_buttons.addWidget(self.download_button)

        self.visualize_button = QPushButton("Virtual Array Calc.")
        self.visualize_button.clicked.connect(self.visualize_virtual)
        h_buttons.addWidget(self.visualize_button)

        self.vbox.addLayout(h_buttons)


        # Load/Save/Run buttons
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Radar Array JSON")
        self.load_button.clicked.connect(self.load_data)
        # btn_layout.addWidget(self.load_button)
        
        self.cfarwid = QWidget()
        self.vbox_cfar = QVBoxLayout(self.cfarwid)
        
        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("CFAR_RD_training_cells"))
        self.CFAR_RD_training_cells = QLineEdit("40,40")
        h_va.addWidget(self.CFAR_RD_training_cells)
        h_va.addWidget(QLabel("CFAR_RD_guard_cells"))
        self.CFAR_RD_guard_cells = QLineEdit("10,10")
        h_va.addWidget(self.CFAR_RD_guard_cells)
        h_va.addWidget(QLabel("CFAR_RD_alpha"))
        self.CFAR_RD_alpha = QLineEdit("5.0")
        h_va.addWidget(self.CFAR_RD_alpha)
        self.vbox_cfar.addLayout(h_va)
        h_va = QHBoxLayout()
        h_va.addWidget(QLabel("CFAR_Angle_training_cells"))
        self.CFAR_Angle_training_cells = QLineEdit("40,40")
        h_va.addWidget(self.CFAR_Angle_training_cells)
        h_va.addWidget(QLabel("CFAR_Angle_guaAngle_cells"))
        self.CFAR_Angle_guard_cells = QLineEdit("10,10")
        h_va.addWidget(self.CFAR_Angle_guard_cells)
        h_va.addWidget(QLabel("CFAR_Angle_alpha"))
        self.CFAR_Angle_alpha = QLineEdit("2.0")
        h_va.addWidget(self.CFAR_Angle_alpha)
        self.vbox_cfar.addLayout(h_va)
        self.vbox_cfar.addStretch()
        
    


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
        h_va.addWidget(QLabel("Select Radar Sensor"))
        self.radarsensor_combo = QComboBox()
        self.radarsensor_combo.addItems(["New","TI_AWR1642","TI_IWR6843","TI_AWR2243","TI_AWR2944","TI_Cascade_AWR2243","SISO_mmWave76GHz","Xhetru_X4","Altos"])
        h_va.addWidget(self.radarsensor_combo)
        self.radarsensor_combo.currentIndexChanged.connect(self.on_radarsensor_changed)
        
        # ssp.environment.scenarios.predefine_movingcube_6843()
        
        self.combo_radarsel = QComboBox()
        h_va.addWidget(self.combo_radarsel)
        self.updatesuiteradarspec()
        self.readradarspec_button = QPushButton("Read Radar spec from Scenario")
        self.readradarspec_button.clicked.connect(self.readradarspec)
        h_va.addWidget(self.readradarspec_button)
        self.writeradarspec_button = QPushButton("Write to Scenario")
        self.writeradarspec_button.clicked.connect(self.writeradarspec)
        h_va.addWidget(self.writeradarspec_button)
        
        
        self.vbox_main.addLayout(h_va)
        self.tab_widget = QTabWidget()
        self._build_signal_tab()
        self._build_transmit_receive_tab()
        self.tab_widget.addTab(wid1, "Array Radar")
        self._build_noise_adc_tab()
        
        self._build_radar_config_tab()
        self._build_angleProcessing_tab()
        self._build_simulations_tab()
        self._build_run_tab()
        self._build_runsim_tab()
        self.tab_widget.setCurrentIndex(2)
        self.vbox_main.addWidget(self.tab_widget)
        # self.vbox_main.addLayout(btn_layout)
        # self.vbox.addStretch()
    def on_radarsensor_changed(self,index):
        # self.radarsensor_combo.addItems(["New","TI_AWR1642","TI_IWR6843","TI_AWR2243","TI_AWR2944","TI_Cascade_AWR2243","SISO_mmWave76GHz","Xhetru_X4","Altos"])
        
        if self.radarsensor_combo.currentText()=="New":
            ""
        elif self.radarsensor_combo.currentText()=="TI_AWR1642":
            ssp.utils.set_TI_AWR1642_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="TI_IWR6843":
            ssp.utils.set_TI_IWR6843_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="TI_AWR2243":
            ssp.utils.set_TI_AWR2243_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="TI_AWR2944":
            ssp.utils.set_TI_AWR2944_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="TI_Cascade_AWR2243":
            ssp.utils.set_TI_Cascade_AWR2243_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="SISO_mmWave76GHz":
            ssp.utils.set_SISO_mmWave76GHz_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="Xhetru_X4":
            ssp.utils.set_Xhetru_X4_toQtUI(self)
        elif self.radarsensor_combo.currentText()=="Altos":
            ssp.utils.set_Altos_toQtUI(self)
            
        
    def updatesuiteradarspec(self):
        suite_information = ssp.environment.BlenderSuiteFinder().find_suite_information()
        self.combo_radarsel.clear()
        for isuite,suiteobject in enumerate(suite_information):
            for iradar,radarobject in enumerate(suiteobject['Radar']):
                self.combo_radarsel.addItem(f'{isuite},{iradar}')
        self.combo_radarsel.setCurrentIndex(self.combo_radarsel.count()-1)
    def runnextframe(self):
        self.new_window = RadarVisApp()
        self.new_window.show()
    def initsimulation(self):
        ssp.utils.trimUserInputs() 
        ssp.config.restart()
    def vissimulation(self):
        ssp.visualization.visualize_scenario()
    def _build_runsim_tab(self):
        pass
        # w = QWidget()
        # vbox = QVBoxLayout(w)
        # init = QPushButton("Initialize Simulation")
        # init.clicked.connect(self.initsimulation)
        # next = QPushButton("Next Frame")
        # next.clicked.connect(self.runnextframe)
        # hbox = QHBoxLayout()
        # hbox.addWidget(init)
        # hbox.addWidget(next)
        # vbox.addLayout(hbox)
        # self.figure_vis = Figure(figsize=(12, 8), facecolor='#4e4e4e')  # dark background
        # self.canvas_vis = FigureCanvas(self.figure_vis)
        # vbox.addWidget(self.canvas_vis)
        # self.axes_vis = []
        # for i in range(8):
        #     if i in [5, 6, 7]:
        #         ax = self.figure_vis.add_subplot(2, 4, i + 1, projection='3d')
        #         ax.set_facecolor('#6e6e6e')
        #         ax.grid(True, linestyle='--', alpha=0.3)
        #         ax.set_xlabel('X (m)')
        #         ax.set_ylabel('Y (m)')
        #         ax.set_zlabel('Z (m)')
        #         ax.tick_params(colors='white')
        #         ax.xaxis.label.set_color('white')
        #         ax.yaxis.label.set_color('white')
        #         ax.title.set_color('white')
        #     else:
        #         ax = self.figure_vis.add_subplot(2, 4, i + 1)
        #         ax.set_facecolor('#6e6e6e')
        #         ax.grid(True, linestyle='--', alpha=0.3)
        #         ax.tick_params(colors='white')
        #         ax.xaxis.label.set_color('white')
        #         ax.yaxis.label.set_color('white')
        #         ax.title.set_color('white')

        #         # Optional: Set titles for each subplot
        #         titles = [
        #             "Raw ADC Data", "Real vs Imag", "Range-Time", "Range Profile",
        #             "Range-Doppler", "", "", "Detected Points"
        #         ]
        #         if i < len(titles):
        #             ax.set_title(titles[i], fontsize=10)

        #     self.axes_vis.append(ax)

        # self.figure_vis.tight_layout(pad=3.0)


        # self.tab_widget.addTab(w, "Run")
    def visualize_pattern(self):
            # Get parameters from UI
        pattern_type = self.tx_pattern.currentText()
        max_gain = self.tx_gain.value()
        az_bw = self.tx_az_bw.value()
        el_bw = self.tx_el_bw.value()

        # Define azimuth and elevation grid
        az = np.linspace(-180, 180, 361)     # degrees
        el = np.linspace(-90, 90, 181)     # degrees
        AZ, EL = np.meshgrid(az, el)

        # Antenna direction is identity (forward)
        antenna_dir = Quaternion((1, 0, 0, 0))  # no rotation

        # Allocate gain array
        gain_map = np.zeros_like(AZ)

        # Compute gain at each (az, el) point
        for i in range(AZ.shape[0]):
            for j in range(AZ.shape[1]):
                az_rad = np.radians(AZ[i, j])
                el_rad = np.radians(EL[i, j])

                # Spherical to Cartesian unit vector
                x = np.cos(el_rad) * np.cos(az_rad)
                y = np.cos(el_rad) * np.sin(az_rad)
                z = np.sin(el_rad)
                dir_vec = Vector((x, y, z))

                gain = ssp.raytracing.antenna_gain(
                    pattern_type,
                    antenna_dir,
                    max_gain,
                    az_bw,
                    el_bw,
                    pattern_type,
                    dir_vec
                )
                gain_map[i, j] = gain

        # Normalize and convert to Cartesian for 3D plotting
        r = gain_map / np.max(gain_map)
        AZ_rad = np.radians(AZ)
        EL_rad = np.radians(EL)
        X = r * np.cos(EL_rad) * np.cos(AZ_rad)
        Y = r * np.cos(EL_rad) * np.sin(AZ_rad)
        Z = r * np.sin(EL_rad)

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z,
                            facecolors=plt.cm.viridis(gain_map / np.max(gain_map)),
                            rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=1.0)

        ax.set_title(f"Antenna Gain Pattern: {pattern_type}", fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        ssp.utils.set_axes_equal(ax)
        plt.show()
        # pattern_type = self.tx_pattern.currentText()
        # max_gain = self.tx_gain.value()
        # az_bw = self.tx_az_bw.value()
        # el_bw = self.tx_el_bw.value()

        # # Create a grid of azimuth and elevation angles
        # az = np.linspace(-90, 90, 181)
        # el = np.linspace(-90, 90, 181)
        # AZ, EL = np.meshgrid(az, el)

        # # Assume fixed antenna orientation (looking forward) and convert angles to direction vectors
        # gain_map = np.zeros_like(AZ)

        # # Create identity rotation (forward-looking antenna)
        # antenna_dir = Quaternion((1, 0, 0, 0))  # no rotation

        # for i in range(AZ.shape[0]):
        #     for j in range(AZ.shape[1]):
        #         az_rad = np.radians(AZ[i, j])
        #         el_rad = np.radians(EL[i, j])

        #         # Convert (az, el) to unit vector
        #         x = np.cos(el_rad) * np.cos(az_rad)
        #         y = np.cos(el_rad) * np.sin(az_rad)
        #         z = np.sin(el_rad)

        #         dir_vec = Vector((x, y, z))

        #         gain_map[i, j] = ssp.raytracing.antenna_gain(
        #             pattern_type,
        #             antenna_dir,
        #             max_gain,
        #             az_bw,
        #             el_bw,
        #             pattern_type,
        #             dir_vec
        #         )

        # # Convert to dB for better visualization
        # gain_dB = 10 * np.log10(np.clip(gain_map, 1e-6, None))

        # # Plot as 2D heatmap
        # plt.figure(figsize=(8, 6))
        # plt.contourf(AZ, EL, gain_dB, levels=40, cmap='viridis')
        # plt.colorbar(label='Gain (dB)')
        # plt.xlabel("Azimuth (deg)")
        # plt.ylabel("Elevation (deg)")
        # plt.title(f"Antenna Pattern: {pattern_type}")
        # plt.grid(True)
        # plt.show()

        
    def _build_transmit_receive_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.tx_pattern = QComboBox()
        self.tx_pattern.addItems(["Omni","Directional-Sinc","Rect","cos"])  # extend as needed
        form.addRow("Tx Antenna Pattern", self.tx_pattern)

        self.tx_gain = QDoubleSpinBox(); self.tx_gain.setRange(0, 50)
        self.tx_gain.setValue(3.0)
        form.addRow("Tx Element Gain (dB)", self.tx_gain)

        self.tx_az_bw = QDoubleSpinBox(); self.tx_az_bw.setRange(0, 360)
        self.tx_az_bw.setValue(120.0)
        form.addRow("Tx Azimuth BW (°)", self.tx_az_bw)

        self.tx_el_bw = QDoubleSpinBox(); self.tx_el_bw.setRange(0, 360)
        self.tx_el_bw.setValue(120.0)
        form.addRow("Tx Elevation BW (°)", self.tx_el_bw)
        
        visulaize_pattern = QPushButton("Visualize")
        visulaize_pattern.clicked.connect(self.visualize_pattern)
        # form.addRow("", visulaize_pattern)
        # Receive
        self.rx_pattern = QComboBox()
        self.rx_pattern.addItems(["Omni","Directional-Sinc","Rect","cos"])
        form.addRow("Rx Antenna Pattern", self.rx_pattern)

        self.rx_gain = QDoubleSpinBox(); self.rx_gain.setRange(0, 50)
        self.rx_gain.setValue(0.0)
        form.addRow("Rx Element Gain (dB)", self.rx_gain)

        self.rx_az_bw = QDoubleSpinBox(); self.rx_az_bw.setRange(0, 360)
        self.rx_az_bw.setValue(120.0)
        form.addRow("Rx Azimuth BW (°)", self.rx_az_bw)

        self.rx_el_bw = QDoubleSpinBox(); self.rx_el_bw.setRange(0, 360)
        self.rx_el_bw.setValue(120.0)
        form.addRow("Rx Elevation BW (°)", self.rx_el_bw)

        self.tab_widget.addTab(w, "Antenna Elements Patterns")
        
    def _on_radar_mode_change(self, text):
        if text == "FMCW":
            self.stack.setCurrentIndex(0)
        elif text == "Pulse":
            self.stack.setCurrentIndex(1)
        elif text == "UWB":
            self.stack.setCurrentIndex(2)
        elif text == "CW":
            self.stack.setCurrentIndex(3)
    def _build_signal_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.f0_lineedit = QDoubleSpinBox(); self.f0_lineedit.setRange(0, 1000)
        self.f0_lineedit.setValue( 77.0 )
        form.addRow("Start RF Frequency (GHz)", self.f0_lineedit)

        
        self.tx_power = QDoubleSpinBox();  self.tx_power.setRange(-100, +100)
        self.tx_power.setValue(12.0)
        form.addRow("Transmit Power (dBm)", self.tx_power)
        # Radar mode combobox
        self.radar_mode = QComboBox()
        self.radar_mode.addItems(["FMCW","Pulse","UWB","CW"])
        self.radar_mode.setCurrentText("FMCW")
        self.pri = QDoubleSpinBox(); self.pri.setRange(0, 1e6)
        self.pri.setValue(70.0)
        form.addRow("PRI (µs)", self.pri)
        form.addRow("Radar Mode", self.radar_mode)
        self.stack = QStackedWidget()
        self.radar_mode.currentTextChanged.connect(self._on_radar_mode_change)
        # Slobe rate
        fmcw_widget = QWidget()
        
        fmcw_layout = QFormLayout(fmcw_widget)
        
        self.slobe = QDoubleSpinBox(); self.slobe.setRange(0, 1e6)
        self.slobe.setValue(1000.0/60.0)
        fmcw_layout.addRow("FMCW Slope (MHz/µs)", self.slobe)
        
        fmcw_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        
        self.stack.addWidget(fmcw_widget)

        pulse_widget = QWidget()
        pulse_layout = QFormLayout(pulse_widget)
        pulse_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Pulse waveform file
        h = QHBoxLayout()
        self.pulse_file = QLineEdit("WaveformFile.txt")
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._pick_file(self.pulse_file))
        h.addWidget(self.pulse_file); h.addWidget(btn)
        pulse_layout.addRow("Pulse Waveform", h)
        
        self.stack.addWidget(pulse_widget)
        
        UWB_widget = QWidget()
        UWB_layout = QFormLayout(UWB_widget)
        self.stack.addWidget(UWB_widget)
        
        CW_widget = QWidget()
        CW_layout = QFormLayout(CW_widget)
        self.stack.addWidget(CW_widget)
        
        
        form.addWidget(self.stack)
        # form.addRow(QWidget())
        self.fs = QDoubleSpinBox(); self.fs.setRange(0, 1e3)
        self.fs.setValue(5.0)
        fmcw_layout.addRow("ADC Sampling Rate (MSps)", self.fs)

        self.n_adc = QSpinBox(); self.n_adc.setRange(1, 1024*8)
        self.n_adc.setValue(256)
        fmcw_layout.addRow("ADC Samples per Pulse", self.n_adc)
        self.fmcw = QCheckBox("FMCW Mode")
        self.fmcw.setChecked(True)
        fmcw_layout.addRow(self.fmcw)
        self.fmcwinfo = QLabel("...")
        self.fmcwinfo_btn = QPushButton("Info")
        self.fmcwinfo_btn.clicked.connect(self.fmcwinfo_fcn)
        fmcw_layout.addRow(self.fmcwinfo_btn, self.fmcwinfo)
        
        
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.tab_widget.addTab(w, "TX Waveform")
    
    def _build_noise_adc_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.rf_filter_bw = QDoubleSpinBox(); self.rf_filter_bw.setRange(0, 1e3)
        self.rf_filter_bw.setValue(10.0)
        form.addRow("RF Filter BW (MHz)", self.rf_filter_bw)
        
        self.rf_nf = QDoubleSpinBox(); self.rf_nf.setRange(0, 20)
        self.rf_nf.setValue(5.0)
        form.addRow("RF Noise Figure (dB)", self.rf_nf)

        self.temp_k = QDoubleSpinBox(); self.temp_k.setRange(0, 1000)
        self.temp_k.setValue(290.0)
        form.addRow("Temperature (K)", self.temp_k)

        self.adc_pk2pk = QDoubleSpinBox(); self.adc_pk2pk.setRange(0, 10)
        self.adc_pk2pk.setValue(2.0)
        form.addRow("ADC Peak‐to‐Peak", self.adc_pk2pk)

        self.adc_levels = QSpinBox(); self.adc_levels.setRange(2, 1<<16)
        self.adc_levels.setValue(256)
        form.addRow("ADC Levels", self.adc_levels)

        self.adc_imp = QDoubleSpinBox(); self.adc_imp.setRange(0, 1e3)
        self.adc_imp.setValue(300.0)
        form.addRow("ADC Impedance Factor", self.adc_imp)

        self.adc_lna = QDoubleSpinBox(); self.adc_lna.setRange(0, 100)
        self.adc_lna.setValue(50.0)
        form.addRow("ADC LNA Gain (dB)", self.adc_lna)


        self.adc_sat = QCheckBox("Enable ADC Saturation")
        form.addRow(self.adc_sat)

        self.tab_widget.addTab(w, "RX Hardware")

    def _build_angleProcessing_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.azimuth_window = QComboBox()
        self.azimuth_window.addItems(["Rectangular", "Hamming", "Hann"])
        self.azimuth_window.setCurrentText("Hamming")
        form.addRow("Azimuth Window", self.azimuth_window)
        self.azimuth_FFT_points = QSpinBox(); self.azimuth_FFT_points.setRange(-1, 4)
        form.addRow("Azimuth FFT points (-1(same length), n next pow 2)", self.azimuth_FFT_points)
        
        
        self.Elevation_window = QComboBox()
        self.Elevation_window.addItems(["Rectangular", "Hamming", "Hann"])
        self.Elevation_window.setCurrentText("Hamming")
        form.addRow("Elevation Window", self.Elevation_window)
        self.Elevation_FFT_points = QSpinBox(); self.Elevation_FFT_points.setRange(-1, 4)
        form.addRow("Elevation FFT points (-1(same length), n next pow 2)", self.Elevation_FFT_points)
        
        self.spectrum_angle_type = QComboBox()
        self.spectrum_angle_type.addItems(["FFT","Capon"])
        form.addRow("Angle Spectrum", self.spectrum_angle_type)

        self.CaponAzimuth = QLineEdit()
        self.CaponAzimuth.setText("-60:3:60:1")
        form.addRow("Capon Azimuth min:res:max:fine_res (deg)", self.CaponAzimuth)
        self.CaponElevation = QLineEdit()
        self.CaponElevation.setText("-60:3:60:1")
        form.addRow("Capon Elevation min:res:max:fine_res (deg)", self.CaponElevation)
        self.CaponDL = QLineEdit()
        self.CaponDL.setText("2")
        form.addRow("Capon Diagonal Loading Factor ( >10 -> MF)", self.CaponDL)
        

        self.cfar_angle_type = QComboBox()
        self.cfar_angle_type.addItems(["Fixed Threshold a*KSort","CA CFAR", "OS CFAR", "Fixed Threshold", "Fixed Threshold a*mean","No CFAR (max)"])
        form.addRow("CFAR Type: Angle (Azimuth-Elevation)", self.cfar_angle_type)
        form.addRow("CFAR coef.: Angle (Azimuth-Elevation)", self.CFAR_Angle_alpha)
        form.addRow("CFAR Training Cells (Az,El)%: Angle (Azimuth-Elevation)", self.CFAR_Angle_training_cells)
        form.addRow("CFAR Guard Cells (OS index)(Az,El)%: Angle (Azimuth-Elevation)", self.CFAR_Angle_guard_cells)
        self.tab_widget.addTab(w, "RX Processing-Angle")
    def _build_radar_config_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.range_window = QComboBox()
        self.range_window.addItems(["Rectangular", "Hamming", "Hann"])
        # self.range_window.setCurrentText("Hamming")
        form.addRow("Range Window", self.range_window)
        self.Range_FFT_points = QSpinBox(); self.Range_FFT_points.setRange(-1, 4)
        form.addRow("Range FFT points (-1(same length), n next pow 2)", self.Range_FFT_points)

        self.Range_Start = QDoubleSpinBox(); self.Range_Start.setRange(0, 100)
        self.Range_End = QDoubleSpinBox(); self.Range_End.setRange(0, 100)
        self.Range_End.setValue(100.0)
        self.range_limits = QHBoxLayout()
        self.range_limits.addWidget(self.Range_Start)
        self.range_limits.addWidget(self.Range_End)
        form.addRow("Range Limits (%)", self.range_limits)

        self.pulse_buffering = QCheckBox("Pulse Buffering")
        form.addRow("Vital Sign Phase Processing", self.pulse_buffering)
        
        self.clutter_removal = QCheckBox("Enable Clutter Removal")
        form.addRow("Clutter Removal", self.clutter_removal)
        self.dopplerprocessing_method = QComboBox()
        self.dopplerprocessing_method.addItems(["Simple FFT", "inv W", "Compensation"])
        form.addRow("Doppler Processing Method", self.dopplerprocessing_method)
        self.n_pulse = QSpinBox(); self.n_pulse.setRange(1, 1024*8)
        self.n_pulse.setValue(3 * 64)
        form.addRow("Number of Pulses in CPI", self.n_pulse)
        self.n_pulse_mimo = QSpinBox(); self.n_pulse_mimo.setRange(1, 1024*8)
        self.n_pulse_mimo.setValue(64)
        self.n_pulse_mimo_btn = QPushButton("Set CPI to (x * TX Number) for MIMO")
        self.n_pulse_mimo_btn.clicked.connect(self.n_pulse_mimo_btn_fcn)
        form.addRow(self.n_pulse_mimo_btn, self.n_pulse_mimo)
        
        self.doppler_window = QComboBox()
        self.doppler_window.addItems(["Rectangular", "Hamming", "Hann"])
        self.doppler_window.setCurrentText("Hamming")
        form.addRow("Doppler Window", self.doppler_window)
        self.Doppler_FFT_points = QSpinBox(); self.Doppler_FFT_points.setRange(-1, 4)
        form.addRow("Doppler FFT points (-1(same length), n next pow 2)", self.Doppler_FFT_points)
        self.RangeDopplerCFARMean = QCheckBox("Range Doppler CFAR Antenna Mean")
        form.addRow("Range Doppler CFAR Antenna Mean", self.RangeDopplerCFARMean)
        
        self.logscale = QCheckBox("Logarithmic Scale Amplitude")
        self.logscale.setChecked(True)
        form.addRow("Log Scale", self.logscale)
        
        self.cfar_rd_type = QComboBox()
        self.cfar_rd_type.addItems(["Fixed Threshold a*KSort","CA CFAR", "OS CFAR", "Fixed Threshold", "Fixed Threshold a*mean","No CFAR (max)"])
        form.addRow("CFAR Type: Range-Doppler", self.cfar_rd_type)
        form.addRow("CFAR coef.: Range-Doppler", self.CFAR_RD_alpha)
        form.addRow("CFAR Training Cells (R,D)%: Range-Doppler", self.CFAR_RD_training_cells)
        form.addRow("CFAR Guard Cells (OS index)(R,D)%: Range-Doppler", self.CFAR_RD_guard_cells)
        
        
        
        
        
        # form.addRow("CFAR",self.cfarwid)
        # # FFT zero‐pad flags
        # self.rng_fft_pad = QCheckBox("Next‐Power‐2 Range FFT")
        # form.addRow(self.rng_fft_pad)
        # self.dop_fft_pad = QCheckBox("Next‐Power‐2 Doppler FFT")
        # form.addRow(self.dop_fft_pad)
        # self.az_fft_pad = QCheckBox("Next‐Power‐2 Azimuth FFT")
        # form.addRow(self.az_fft_pad)
        # self.el_fft_pad = QCheckBox("Next‐Power‐2 Elevation FFT")
        # form.addRow(self.el_fft_pad)
        self.tab_widget.addTab(w, "RX Processing")    


    
    def _build_simulations_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Misc
        self.save_t = QCheckBox("Save Signal Generation Time")
        self.save_t.setChecked(True)
        form.addRow(self.save_t)

        self.continuous_cpi = QCheckBox("Continuous CPI per Frame")
        form.addRow(self.continuous_cpi)

        self.starttime = QLineEdit("0.000000")
        form.addRow("Start time", self.starttime)
        self.currenttime = QLabel("0.000000")
        form.addRow("Current time", self.currenttime)
        self.currentpulse = QLabel("0")
        form.addRow("Current pulse number", self.currentpulse)
        
        self.tab_widget.addTab(w, "Miscellaneous")        
    def _build_run_tab(self):
        ssp.utils.define_settings()
        w = QWidget()
        form = QFormLayout(w)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Misc
        
        self.animation_frame_rate = QLineEdit(str(bpy.context.scene.render.fps))
        form.addRow("Animation Frame Rate", self.animation_frame_rate)
        self.ray_tracing_mode = QComboBox()
        self.ray_tracing_mode.addItems(["Light", "Balanced", "Advanced"])
        form.addRow("Ray Tracing Mode", self.ray_tracing_mode)
        self.ray_tracing_bounce_number = QLineEdit(str(bpy.data.objects["Simulation Settings"]["Bounce Number"]))
        form.addRow("Ray Tracing Bounce Number", self.ray_tracing_bounce_number)
        
        self.cuda_check = QCheckBox("use CUDA")
        self.cuda_check.setChecked(ssp.config.GPU_run_available())
        if ssp.config.CUDA_is_available:
            form.addRow("GPU & CUDA is available", self.cuda_check )
            
        else:
            form.addRow("No GPU & CUDA", self.cuda_check )
            self.cuda_check.setChecked(False)
            self.cuda_check.setEnabled(False)
            
        self.spill_over_check = QCheckBox("Spill over enabled")
        form.addRow("Spill over", self.spill_over_check )
        
        self.theme = QComboBox()
        self.theme.addItems(["Theme 1", "Theme 2", "Theme 3", "Theme 4"])
        form.addRow("UI Theme", self.theme)
        
        self.animation_endframe = QLineEdit(str(bpy.context.scene.frame_end))
        form.addRow("Animation End Frame", self.animation_endframe)
        
        
        self.set_button = QPushButton("Set")
        self.set_button.clicked.connect(self.set_button_fcn)
        self.processingUI_button = QPushButton("Processing UI")
        self.processingUI_button.clicked.connect(self.processingUI_button_fcn)
        # form.addRow(self.set_button,self.processingUI_button)
        form.addRow("Set",self.set_button)
        
        # 
        hb = QHBoxLayout()
        self.range_input = QDoubleSpinBox()
        self.range_input.setRange(0, 10000)
        self.range_input.setValue(10.0)
        hb.addWidget(QLabel("Range (m)"))
        hb.addWidget(self.range_input)
        # Radial Velocity
        self.radial_velocity_input = QDoubleSpinBox()
        self.radial_velocity_input.setRange(-1000, 1000)
        self.radial_velocity_input.setValue(0.0)
        hb.addWidget(QLabel("Radial Velocity (m/s)"))
        hb.addWidget(self.radial_velocity_input)
        form.addRow(hb)
        # Azimuth
        hb = QHBoxLayout()
        self.azimuth_input = QDoubleSpinBox()
        self.azimuth_input.setRange(-180, 180)
        self.azimuth_input.setValue(0.0)
        hb.addWidget(QLabel("Azimuth (deg)"))
        hb.addWidget(self.azimuth_input)
        
        # form.addRow("Azimuth (deg)", self.azimuth_input)

        # Elevation
        self.elevation_input = QDoubleSpinBox()
        self.elevation_input.setRange(-90, 90)
        self.elevation_input.setValue(0.0)
        hb.addWidget(QLabel("Elevation (deg)"))
        hb.addWidget(self.elevation_input)
        form.addRow(hb)
        # RCS0
        hb = QHBoxLayout()
        self.rcs0_input = QDoubleSpinBox()
        self.rcs0_input.setRange(0, 10000)
        self.rcs0_input.setValue(1.0)
        hb.addWidget(QLabel("RCS0 (m)"))
        hb.addWidget(self.rcs0_input)
        # Size
        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(0.01, 100)
        self.size_input.setValue(1.0)
        hb.addWidget(QLabel("Size (m)"))
        hb.addWidget(self.size_input)
        form.addRow(hb)
        
        
        # Shape
        self.shape_input = QComboBox()
        self.shape_input.addItems(["cube", "sphere", "plane"])
        form.addRow("Shape", self.shape_input)
        
        self.addTarget_button = QPushButton("Add Target")
        self.addTarget_button.clicked.connect(self.addTarget_button_fcn)
        form.addRow(self.addTarget_button)
        delall_button = QPushButton("Delete All")
        delall_button.clicked.connect(self.delall_button_fcn)
        form.addRow(delall_button)
        
        next = QPushButton("Signal Gen. & Process")
        next.clicked.connect(self.runnextframe)
        
        init = QPushButton("Initialize Simulation")
        init.clicked.connect(self.initsimulation)
        form.addRow(init)
        vis = QPushButton("Visualize Scenario")
        vis.clicked.connect(self.vissimulation)
        form.addRow(vis,next)

        self.tab_widget.addTab(w, "Simulation")        
    def delall_button_fcn(self):
        ssp.utils.delete_all_objects()
        ssp.utils.define_settings()
        self.updatesuiteradarspec()

    def set_button_fcn(self):
        ssp.utils.useCUDA(self.cuda_check.isChecked())
        
        bpy.context.scene.render.fps = int(self.animation_frame_rate.text())
        bpy.context.scene.frame_end = int(self.animation_endframe.text())
        ssp.utils.set_raytracing_bounce(int(self.ray_tracing_bounce_number.text()))
        ["Light", "Balanced", "Advanced"]
        if self.ray_tracing_mode.currentText()=="Light":
            ssp.utils.set_RayTracing_light()
        elif self.ray_tracing_mode.currentText()=="Balanced":
            ssp.utils.set_RayTracing_balanced()
        elif self.ray_tracing_mode.currentText()=="Advanced":
            ssp.utils.set_RayTracing_advanced_intense()
        if self.theme.currentText()=="Theme 1":
            ssp.config.appSTYLESHEET = ssp.config.appSTYLESHEET1
        elif self.theme.currentText()=="Theme 2":
            ssp.config.appSTYLESHEET = ssp.config.appSTYLESHEET2
        elif self.theme.currentText()=="Theme 3":
            ssp.config.appSTYLESHEET = ssp.config.appSTYLESHEET3
        elif self.theme.currentText()=="Theme 4":
            ssp.config.appSTYLESHEET = ssp.config.appSTYLESHEET4
        self.setStyleSheet(ssp.config.appSTYLESHEET)
        if self.spill_over_check.isChecked():
            ssp.config.directReceivefromTX =  True
        else:
            ssp.config.directReceivefromTX =  False 
            ssp.config.RadarRX_only_fromscatters_itsTX = True
            ssp.config.RadarRX_only_fromitsTX = True
            ssp.config.Radar_TX_RX_isolation = True
    def processingUI_button_fcn(self):
        ssp.utils.trimUserInputs() 
        ssp.config.restart()
        while ssp.config.run():
            path_d_drate_amp = ssp.raytracing.Path_RayTracing_frame()
            Signals = ssp.integratedSensorSuite.SensorsSignalGeneration_frame(path_d_drate_amp)
            ssp.integratedSensorSuite.SensorsSignalProccessing_Chain_RangeProfile_RangeDoppler_AngleDoppler(Signals)
            ssp.utils.increaseCurrentFrame()
        
    def addTarget_button_fcn(self):
        range_m = self.range_input.value()
        azimuth_deg = self.azimuth_input.value()
        elevation_deg = self.elevation_input.value()
        rcs0 = self.rcs0_input.value()
        size_m = self.size_input.value()
        radial_velocity = self.radial_velocity_input.value()
        shape = self.shape_input.currentText()
        s = self.combo_radarsel.currentText()
        if s=="":
            return
        isuite,iradar = s.split(',')
        isuite = int(isuite)
        iradar = int(iradar)
        # ssp.utils.trimUserInputs()
        # radobj = ssp.RadarSpecifications[isuite][iradar]['BlenderObject']
        
        suite_information = ssp.environment.BlenderSuiteFinder().find_suite_information()
        radobj = suite_information[isuite]['Radar'][iradar]['GeneralRadarSpec_Object'] 
        
        ssp.radar.utils.addTarget(
            refRadar=radobj,
            range=range_m,
            azimuth=azimuth_deg,
            elevation=elevation_deg,
            RCS0=rcs0,
            size=size_m,
            radial_velocity=radial_velocity,
            shape=shape
        )
        ssp.radar.utils.apps.process_events()
    def n_pulse_mimo_btn_fcn(self):
        tx_positions = [tuple(map(float, p.split(','))) for p in self.tx_lineedit.text().split('|')]
        n_tx = len(tx_positions)
        self.n_pulse.setValue(n_tx * self.n_pulse_mimo.value())

    def read_settings(self,settings):
        ssp.utils.QtUI_to_BlenderAddonUI(self, settings)
        return
        settings["Center_Frequency_GHz"] = self.f0_lineedit.value()
        settings["Transmit_Power_dBm"] = self.tx_power.value()
        settings["PRI_us"] = self.pri.value()
        settings["RadarMode"] = self.radar_mode.currentText()
        settings["FMCW_ChirpSlobe_MHz_usec"] = self.slobe.value()
        settings["Fs_MHz"] = self.fs.value()
        settings["N_ADC"] = self.n_adc.value()
        settings["FMCW"] = self.fmcw.isChecked()
        settings["PulseWaveform"] = self.pulse_file.text()

        # --- Array & MIMO ---
        settings["ArrayInfofile"] = self.linedit.text()
        settings["Array_initialization"] = self.comboarrayinititype.currentText()
        settings["Default_Array_Config"] = self.config_combo.currentText()
        settings["Position_Scale"] = self.scale_combo.currentText()
        settings["TXPos_xy"] = self.tx_lineedit.text()
        settings["RXPos_xy"] = self.rx_lineedit.text()
        settings["RXPos_xy_bias"] = self.rxb_lineedit.text()
        
        settings["distance scaling"] = self.disscale_lineedit.text()
        settings["VA order (TX,RX)->[X,Y]|"] = self.vaorder_lineedit.text()
        settings["VA order2 (TX,RX)->[X,Y]|"] = self.vaorder_lineedit2.text()
        settings["MIMO_Tech"] = self.mimo_combo.currentText()
        settings["MIMO_W"] = self.mimo_lineedit.text()

        # --- Antenna patterns ---
        settings["Transmit_Antenna_Element_Pattern"] = self.tx_pattern.currentText()
        settings["Transmit_Antenna_Element_Gain_db"] = self.tx_gain.value()
        settings["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = self.tx_az_bw.value()
        settings["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = self.tx_el_bw.value()
        settings["Receive_Antenna_Element_Pattern"] = self.rx_pattern.currentText()
        settings["Receive_Antenna_Element_Gain_db"] = self.rx_gain.value()
        settings["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = self.rx_az_bw.value()
        settings["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = self.rx_el_bw.value()

        # --- Noise & ADC ---
        settings["RF_AnalogNoiseFilter_Bandwidth_MHz"] = self.rf_filter_bw.value()
        settings["RF_NoiseFiguredB"] = self.rf_nf.value()
        settings["Tempreture_K"] = self.temp_k.value()
        settings["ADC_peak2peak"] = self.adc_pk2pk.value()
        settings["ADC_levels"] = self.adc_levels.value()
        settings["ADC_ImpedanceFactor"] = self.adc_imp.value()
        settings["ADC_LNA_Gain_dB"] = self.adc_lna.value()
        settings["ADC_SaturationEnabled"] = self.adc_sat.isChecked()

        # --- Range-Doppler processing ---
        settings["RangeWindow"] = self.range_window.currentText()
        settings["RangeFFT_OverNextP2"] = self.Range_FFT_points.value()
        settings["Pulse_Buffering"] = self.pulse_buffering.isChecked()
        settings["ClutterRemoval_Enabled"] = self.clutter_removal.isChecked()
        settings["DopplerProcessingMIMODemod"] = self.dopplerprocessing_method.currentText()
        settings["NPulse"] = self.n_pulse.value()
        settings["DopplerWindow"] = self.doppler_window.currentText()
        settings["DopplerFFT_OverNextP2"] = self.Doppler_FFT_points.value()
        settings["RangeDopplerCFARLogScale"] = self.logscale.isChecked()
        settings["CFAR_RD_type"] = self.cfar_rd_type.currentText()
        settings["CFAR_RD_training_cells"] = self.CFAR_RD_training_cells.text()
        settings["CFAR_RD_guard_cells"] = self.CFAR_RD_guard_cells.text()
        settings["CFAR_RD_alpha"] = float(self.CFAR_RD_alpha.text())

        # --- Angle processing ---
        settings["AzimuthWindow"] = self.azimuth_window.currentText()
        settings["AzFFT_OverNextP2"] = self.azimuth_FFT_points.value()
        settings["ElevationWindow"] = self.Elevation_window.currentText()
        settings["ElFFT_OverNextP2"] = self.Elevation_FFT_points.value()
        settings["AngleSpectrum"] = self.spectrum_angle_type.currentText()
        settings["Capon Azimuth min:res:max:fine_res (deg)"] = self.CaponAzimuth.text()
        settings["Capon Elevation min:res:max:fine_res (deg)"] = self.CaponElevation.text()
        settings["CFAR_Angle_type"] = self.cfar_angle_type.currentText()
        settings["CFAR_Angle_training_cells"] = self.CFAR_Angle_training_cells.text()
        settings["CFAR_Angle_guard_cells"] = self.CFAR_Angle_guard_cells.text()
        settings["CFAR_Angle_alpha"] = float(self.CFAR_Angle_alpha.text())

        # --- Simulation settings ---
        settings["SaveSignalGenerationTime"] = self.save_t.isChecked()
        settings["continuousCPIsTrue_oneCPIpeerFrameFalse"] = self.continuous_cpi.isChecked()
        settings["t_start_radar"] = float(self.starttime.text())

        
            
    def read_settings0(self,settings):
        # --- Transmit / Receive ---
        settings["Transmit_Power_dBm"] = self.tx_power.value()
        settings["Transmit_Antenna_Element_Pattern"] = self.tx_pattern.currentText()
        settings["Transmit_Antenna_Element_Gain_db"] = self.tx_gain.value()
        settings["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = self.tx_az_bw.value()
        settings["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = self.tx_el_bw.value()

        settings["Receive_Antenna_Element_Pattern"] = self.rx_pattern.currentText()
        settings["Receive_Antenna_Element_Gain_db"] = self.rx_gain.value()
        settings["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = self.rx_az_bw.value()
        settings["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = self.rx_el_bw.value()

        # --- Signal parameters ---
        settings["Center_Frequency_GHz"] = self.center_freq.value()
        settings["PRI_us"] = self.pri.value()
        settings["Fs_MHz"] = self.fs.value()
        settings["NPulse"] = self.n_pulse.value()
        settings["N_ADC"] = self.n_adc.value()
        settings["RangeWindow"] = self.range_window.currentText()
        settings["DopplerWindow"] = self.doppler_window.currentText()

        # --- Noise & ADC ---
        settings["Tempreture_K"] = self.temp_k.value()
        settings["ADC_peak2peak"] = self.adc_pk2pk.value()
        settings["ADC_levels"] = self.adc_levels.value()
        settings["ADC_ImpedanceFactor"] = self.adc_imp.value()
        settings["ADC_LNA_Gain_dB"] = self.adc_lna.value()
        settings["RF_NoiseFiguredB"] = self.rf_nf.value()
        settings["RF_AnalogNoiseFilter_Bandwidth_MHz"] = self.rf_filter_bw.value()
        settings["ADC_SaturationEnabled"] = self.adc_sat.isChecked()

        # --- Radar config ---
        settings["FMCW"] = self.fmcw.isChecked()
        settings["FMCW_ChirpSlobe_MHz_usec"] = self.slobe.value()
        settings["RangeFFT_OverNextP2"] = int(self.rng_fft_pad.isChecked())
        settings["DopplerFFT_OverNextP2"] = int(self.dop_fft_pad.isChecked())
        settings["AzFFT_OverNextP2"] = int(self.az_fft_pad.isChecked())
        settings["ElFFT_OverNextP2"] = int(self.el_fft_pad.isChecked())
        settings["RadarMode"] = self.radar_mode.currentText()
        settings["PulseWaveform"] = self.pulse_file.text()
        settings["SaveSignalGenerationTime"] = self.save_t.isChecked()
        settings["continuousCPIsTrue_oneCPIpeerFrameFalse"] = self.continuous_cpi.isChecked()

        # --- CFAR tab ---
        settings["CFAR_RD_training_cells"] = int(self.CFAR_RD_training_cells.text())
        settings["CFAR_RD_guard_cells"] = int(self.CFAR_RD_guard_cells.text())
        settings["CFAR_RD_alpha"] = float(self.CFAR_RD_alpha.text())
        settings["CFAR_Angle_training_cells"] = int(self.CFAR_Angle_training_cells.text())
        settings["CFAR_Angle_guard_cells"] = int(self.CFAR_Angle_guard_cells.text())
        settings["CFAR_Angle_alpha"] = float(self.CFAR_Angle_alpha.text())
        self.close()
    
    def write_settings(self, settings):
        ssp.utils.BlenderAddonUI_to_QtUI(settings,self )
        specifications={}
        ssp.utils.BlenderAddonUI_to_RadarSpecifications(settings,specifications)
        return
        self.f0_lineedit.setValue(settings.get("Center_Frequency_GHz", self.f0_lineedit.value()))
        self.tx_power.setValue(settings.get("Transmit_Power_dBm", self.tx_power.value()))
        self.pri.setValue(settings.get("PRI_us", self.pri.value()))
        self.radar_mode.setCurrentText(settings.get("RadarMode", self.radar_mode.currentText()))
        self.slobe.setValue(settings.get("FMCW_ChirpSlobe_MHz_usec", self.slobe.value()))
        self.fs.setValue(settings.get("Fs_MHz", self.fs.value()))
        self.n_adc.setValue(settings.get("N_ADC", self.n_adc.value()))
        self.fmcw.setChecked(settings.get("FMCW", self.fmcw.isChecked()))
        self.pulse_file.setText(settings.get("PulseWaveform", self.pulse_file.text()))

        # --- Array & MIMO ---
        self.linedit.setText(settings.get("ArrayInfofile", self.linedit.text()))
        self.comboarrayinititype.setCurrentText(settings.get("Array_initialization", self.comboarrayinititype.currentText()))
        self.config_combo.setCurrentText(settings.get("Default_Array_Config", self.config_combo.currentText()))
        self.scale_combo.setCurrentText(settings.get("Position_Scale", self.scale_combo.currentText()))
        self.tx_lineedit.setText(settings.get("TXPos_xy", self.tx_lineedit.text()))
        self.rx_lineedit.setText(settings.get("RXPos_xy", self.rx_lineedit.text()))
        # Optional bias if present

        self.rxb_lineedit.setText(settings.get('RXPos_xy_bias', self.rxb_lineedit.text()))
        
        self.disscale_lineedit.setText(settings.get("distance scaling", self.disscale_lineedit.text()))
        self.vaorder_lineedit.setText(settings.get("VA order (TX,RX)->[X,Y]|", self.vaorder_lineedit.text()))
        self.vaorder_lineedit2.setText(settings.get("VA order2 (TX,RX)->[X,Y]|", self.vaorder_lineedit2.text()))
        self.mimo_combo.setCurrentText(settings.get("MIMO_Tech", self.mimo_combo.currentText()))
        self.mimo_lineedit.setText(settings.get("MIMO_W", self.mimo_lineedit.text()))

        # --- Antenna patterns ---
        self.tx_pattern.setCurrentText(settings.get("Transmit_Antenna_Element_Pattern", self.tx_pattern.currentText()))
        self.tx_gain.setValue(settings.get("Transmit_Antenna_Element_Gain_db", self.tx_gain.value()))
        self.tx_az_bw.setValue(settings.get("Transmit_Antenna_Element_Azimuth_BeamWidth_deg", self.tx_az_bw.value()))
        self.tx_el_bw.setValue(settings.get("Transmit_Antenna_Element_Elevation_BeamWidth_deg", self.tx_el_bw.value()))
        self.rx_pattern.setCurrentText(settings.get("Receive_Antenna_Element_Pattern", self.rx_pattern.currentText()))
        self.rx_gain.setValue(settings.get("Receive_Antenna_Element_Gain_db", self.rx_gain.value()))
        self.rx_az_bw.setValue(settings.get("Receive_Antenna_Element_Azimuth_BeamWidth_deg", self.rx_az_bw.value()))
        self.rx_el_bw.setValue(settings.get("Receive_Antenna_Element_Elevation_BeamWidth_deg", self.rx_el_bw.value()))

        # --- Noise & ADC ---
        self.rf_filter_bw.setValue(settings.get("RF_AnalogNoiseFilter_Bandwidth_MHz", self.rf_filter_bw.value()))
        self.rf_nf.setValue(settings.get("RF_NoiseFiguredB", self.rf_nf.value()))
        self.temp_k.setValue(settings.get("Tempreture_K", self.temp_k.value()))
        self.adc_pk2pk.setValue(settings.get("ADC_peak2peak", self.adc_pk2pk.value()))
        self.adc_levels.setValue(settings.get("ADC_levels", self.adc_levels.value()))
        self.adc_imp.setValue(settings.get("ADC_ImpedanceFactor", self.adc_imp.value()))
        self.adc_lna.setValue(settings.get("ADC_LNA_Gain_dB", self.adc_lna.value()))
        self.adc_sat.setChecked(settings.get("ADC_SaturationEnabled", self.adc_sat.isChecked()))

        # --- Range-Doppler processing ---
        self.range_window.setCurrentText(settings.get("RangeWindow", self.range_window.currentText()))
        self.Range_FFT_points.setValue(settings.get("RangeFFT_OverNextP2", self.Range_FFT_points.value()))
        self.pulse_buffering.setChecked(settings.get("Pulse_Buffering", self.pulse_buffering.isChecked()))
        self.clutter_removal.setChecked(settings.get("ClutterRemoval_Enabled", self.clutter_removal.isChecked()))
        self.dopplerprocessing_method.setCurrentText(settings.get("DopplerProcessingMIMODemod", self.dopplerprocessing_method.currentText()))
        self.n_pulse.setValue(settings.get("NPulse", self.n_pulse.value()))
        self.doppler_window.setCurrentText(settings.get("DopplerWindow", self.doppler_window.currentText()))
        self.Doppler_FFT_points.setValue(settings.get("DopplerFFT_OverNextP2", self.Doppler_FFT_points.value()))
        self.logscale.setChecked(settings.get("RangeDopplerCFARLogScale", self.logscale.isChecked()))
        self.cfar_rd_type.setCurrentText(settings.get("CFAR_RD_type", self.cfar_rd_type.currentText()))
        self.CFAR_RD_training_cells.setText(settings.get("CFAR_RD_training_cells", self.CFAR_RD_training_cells.text()))
        self.CFAR_RD_guard_cells.setText(settings.get("CFAR_RD_guard_cells", self.CFAR_RD_guard_cells.text()))
        self.CFAR_RD_alpha.setText(str(settings.get("CFAR_RD_alpha", self.CFAR_RD_alpha.text())))

        # --- Angle processing ---
        self.azimuth_window.setCurrentText(settings.get("AzimuthWindow", self.azimuth_window.currentText()))
        self.azimuth_FFT_points.setValue(settings.get("AzFFT_OverNextP2", self.azimuth_FFT_points.value()))
        self.Elevation_window.setCurrentText(settings.get("ElevationWindow", self.Elevation_window.currentText()))
        self.Elevation_FFT_points.setValue(settings.get("ElFFT_OverNextP2", self.Elevation_FFT_points.value()))
        self.spectrum_angle_type.setCurrentText(settings.get("AngleSpectrum", self.spectrum_angle_type.currentText()))
        self.CaponAzimuth.setText(settings.get("Capon Azimuth min:res:max:fine_res (deg)", self.CaponAzimuth.text()))
        self.CaponElevation.setText(settings.get("Capon Elevation min:res:max:fine_res (deg)", self.CaponElevation.text()))
        self.cfar_angle_type.setCurrentText(settings.get("CFAR_Angle_type", self.cfar_angle_type.currentText()))
        self.CFAR_Angle_training_cells.setText(settings.get("CFAR_Angle_training_cells", self.CFAR_Angle_training_cells.text()))
        self.CFAR_Angle_guard_cells.setText(settings.get("CFAR_Angle_guard_cells", self.CFAR_Angle_guard_cells.text()))
        self.CFAR_Angle_alpha.setText(str(settings.get("CFAR_Angle_alpha", self.CFAR_Angle_alpha.text())))

        # --- Simulation settings ---
        self.save_t.setChecked(settings.get("SaveSignalGenerationTime", self.save_t.isChecked()))
        self.continuous_cpi.setChecked(settings.get("continuousCPIsTrue_oneCPIpeerFrameFalse", self.continuous_cpi.isChecked()))
        self.starttime.setText(str(settings.get("t_start_radar", self.starttime.text())))

    def write_settings0(self, settings):
        # --- Transmit / Receive ---
        if "Transmit_Power_dBm" in settings:
            self.tx_power.setValue(settings["Transmit_Power_dBm"])
        if "Transmit_Antenna_Element_Pattern" in settings:
            self.tx_pattern.setCurrentText(settings["Transmit_Antenna_Element_Pattern"])
        if "Transmit_Antenna_Element_Gain_db" in settings:
            self.tx_gain.setValue(settings["Transmit_Antenna_Element_Gain_db"])
        if "Transmit_Antenna_Element_Azimuth_BeamWidth_deg" in settings:
            self.tx_az_bw.setValue(settings["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"])
        if "Transmit_Antenna_Element_Elevation_BeamWidth_deg" in settings:
            self.tx_el_bw.setValue(settings["Transmit_Antenna_Element_Elevation_BeamWidth_deg"])

        if "Receive_Antenna_Element_Pattern" in settings:
            self.rx_pattern.setCurrentText(settings["Receive_Antenna_Element_Pattern"])
        if "Receive_Antenna_Element_Gain_db" in settings:
            self.rx_gain.setValue(settings["Receive_Antenna_Element_Gain_db"])
        if "Receive_Antenna_Element_Azimuth_BeamWidth_deg" in settings:
            self.rx_az_bw.setValue(settings["Receive_Antenna_Element_Azimuth_BeamWidth_deg"])
        if "Receive_Antenna_Element_Elevation_BeamWidth_deg" in settings:
            self.rx_el_bw.setValue(settings["Receive_Antenna_Element_Elevation_BeamWidth_deg"])

        # --- Signal parameters ---
        if "Center_Frequency_GHz" in settings:
            self.f0_lineedit.setValue(settings["Center_Frequency_GHz"])
        if "PRI_us" in settings:
            self.pri.setValue(settings["PRI_us"])
        if "Fs_MHz" in settings:
            self.fs.setValue(settings["Fs_MHz"])
        if "NPulse" in settings:
            self.n_pulse.setValue(settings["NPulse"])
        if "N_ADC" in settings:
            self.n_adc.setValue(settings["N_ADC"])
        if "RangeWindow" in settings:
            self.range_window.setCurrentText(settings["RangeWindow"])
        if "DopplerWindow" in settings:
            self.doppler_window.setCurrentText(settings["DopplerWindow"])

        # --- Noise & ADC ---
        if "Tempreture_K" in settings:
            self.temp_k.setValue(settings["Tempreture_K"])
        if "ADC_peak2peak" in settings:
            self.adc_pk2pk.setValue(settings["ADC_peak2peak"])
        if "ADC_levels" in settings:
            self.adc_levels.setValue(settings["ADC_levels"])
        if "ADC_ImpedanceFactor" in settings:
            self.adc_imp.setValue(settings["ADC_ImpedanceFactor"])
        if "ADC_LNA_Gain_dB" in settings:
            self.adc_lna.setValue(settings["ADC_LNA_Gain_dB"])
        if "RF_NoiseFiguredB" in settings:
            self.rf_nf.setValue(settings["RF_NoiseFiguredB"])
        if "RF_AnalogNoiseFilter_Bandwidth_MHz" in settings:
            self.rf_filter_bw.setValue(settings["RF_AnalogNoiseFilter_Bandwidth_MHz"])
        if "ADC_SaturationEnabled" in settings:
            self.adc_sat.setChecked(bool(settings["ADC_SaturationEnabled"]))

        # --- Radar config ---
        if "FMCW" in settings:
            self.fmcw.setChecked(bool(settings["FMCW"]))
        if "FMCW_ChirpSlobe_MHz_usec" in settings:
            self.slobe.setValue(settings["FMCW_ChirpSlobe_MHz_usec"])
        if "RangeFFT_OverNextP2" in settings:
            self.Range_FFT_points.setValue(int(settings["RangeFFT_OverNextP2"]))
        if "DopplerFFT_OverNextP2" in settings:
            self.Doppler_FFT_points.setValue(int(settings["DopplerFFT_OverNextP2"]))
        if "AzFFT_OverNextP2" in settings:
            self.azimuth_FFT_points.setValue(int(settings["AzFFT_OverNextP2"]))
        if "ElFFT_OverNextP2" in settings:
            self.Elevation_FFT_points.setValue(int(settings["ElFFT_OverNextP2"]))
        if "RadarMode" in settings:
            self.radar_mode.setCurrentText(settings["RadarMode"])
        
        if "AngleSpectrum" in settings:
            self.spectrum_angle_type.setCurrentText(settings["AngleSpectrum"])
        if "PulseWaveform" in settings:
            self.pulse_file.setText(settings["PulseWaveform"])
        if "SaveSignalGenerationTime" in settings:
            self.save_t.setChecked(bool(settings["SaveSignalGenerationTime"]))
        if "continuousCPIsTrue_oneCPIpeerFrameFalse" in settings:
            self.continuous_cpi.setChecked(bool(settings["continuousCPIsTrue_oneCPIpeerFrameFalse"]))

        # --- CFAR tab ---
        if "CFAR_RD_training_cells" in settings:
            self.CFAR_RD_training_cells.setText(str(settings["CFAR_RD_training_cells"]))
        if "CFAR_RD_guard_cells" in settings:
            self.CFAR_RD_guard_cells.setText(str(settings["CFAR_RD_guard_cells"]))
        if "CFAR_RD_alpha" in settings:
            self.CFAR_RD_alpha.setText(str(settings["CFAR_RD_alpha"]))
        if "CFAR_Angle_training_cells" in settings:
            self.CFAR_Angle_training_cells.setText(str(settings["CFAR_Angle_training_cells"]))
        if "CFAR_Angle_guard_cells" in settings:
            self.CFAR_Angle_guard_cells.setText(str(settings["CFAR_Angle_guard_cells"]))
        if "CFAR_Angle_alpha" in settings:
            self.CFAR_Angle_alpha.setText(str(settings["CFAR_Angle_alpha"]))
    
    def fmcwinfo_fcn(self):
        n_adc = self.n_adc.value()
        fs = self.fs.value()
        slobe = self.slobe.value()
        pri = self.pri.value()
        
        CPI = self.n_pulse.value() * pri
        s= f"T = {n_adc/fs:.1f} us, B = {n_adc/fs*slobe:.1f} MHz, CPI = {CPI/1000:.1f} ms, , Frame Time = {1000./bpy.context.scene.render.fps:.1f} ms, "
        self.fmcwinfo.setText(s)
    def readradarspec(self):
        s = self.combo_radarsel.currentText()
        if s=="":
            return
        isuite,iradar = s.split(',')
        isuite = int(isuite)
        iradar = int(iradar)
        # ssp.utils.trimUserInputs()
        # radobj = ssp.RadarSpecifications[isuite][iradar]['BlenderObject']
        
        suite_information = ssp.environment.BlenderSuiteFinder().find_suite_information()
        radobj = suite_information[isuite]['Radar'][iradar]['GeneralRadarSpec_Object'] 
        self.write_settings(radobj)

    def writeradarspec(self):
        s = self.combo_radarsel.currentText()
        if s=="":
            return
        isuite,iradar = s.split(',')
        isuite = int(isuite)
        iradar = int(iradar)
        # ssp.utils.trimUserInputs()
        # radobj = ssp.RadarSpecifications[isuite][iradar]['BlenderObject']
        suite_information = ssp.environment.BlenderSuiteFinder().find_suite_information()
        radobj = suite_information[isuite]['Radar'][iradar]['GeneralRadarSpec_Object'] 
        
        
        ssp.utils.QtUI_to_BlenderAddonUI(self, radobj)
        # ssp.utils.BlenderAddonUI_to_RadarSpecifications(radobj,specifications,online_change=True)
        # self.read_settings(radobj)

    def run_add(self):
        BlenderAddon = ssp.utils.createRadarObject_from_QtUI(self)
        ssp.utils.QtUI_to_BlenderAddonUI(self, BlenderAddon)
        self.updatesuiteradarspec()
        ssp.radar.utils.apps.process_events()
        return
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
        self.scale_combo.setCurrentIndex(0)
        ssp.utils.setQtUI_arrays(self,index)
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
        # vaprocessing   = str(data.get('vaprocessing', None)[0])
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
        vaprocessing = ""
        
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

        # Freq (GHz)
        hbox_fi = QHBoxLayout()
        hbox_fi.addWidget(QLabel("Add Resp. Freq (GHz):"))
        self.fi_lineedit = QLineEdit()
        self.fi_lineedit.setText("")
        hbox_fi.addWidget(self.fi_lineedit)
        self.vbox.addLayout(hbox_fi)

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
        fis = self.fi_lineedit.text().split(",")
        fi = []
        Lambdai=[]
        for f in fis:
            if f == "":
                continue
            f = float(f)
            f = f * 1e9
            fi.append(f)
            Lambdai.append(3e8 / f)
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
                for Lambda_i in Lambdai:
                    phase = -2 * np.pi / Lambda_i * d
                    a0 =  np.exp(1j * phase)
                    pc = a * a0
                    o += np.abs(np.sum(pc))
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
                for Lambda_i in Lambdai:
                    phase = -2 * np.pi / Lambda_i * d
                    a0 =  np.exp(1j * phase)
                    pc = a * a0
                    o += np.abs(np.sum(pc))
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
                for Lambda_i in Lambdai:
                    phase = -2 * np.pi / Lambda_i * d
                    a0 =  np.exp(1j * phase)
                    pc = a * a0
                    o += np.abs(np.sum(pc))
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
                for Lambda_i in Lambdai:
                    phase = -2 * np.pi / Lambda_i * d
                    a0 =  np.exp(1j * phase)
                    pc = a * a0
                    o += np.abs(np.sum(pc))
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


