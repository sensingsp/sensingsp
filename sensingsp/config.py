import bpy
from matplotlib import pyplot as plt
from numba import cuda
import os
import platform
import subprocess
import tempfile
from . import visualization
class Config():
    def __init__(self):
        self.CUDA_is_available = cuda.is_available()
        self.system_type = self.get_system_type()
        self.latex = self.check_latex_installed()
        if not self.latex and self.system_type == 'Google Colab':
            print("use ssp.config.install_latex_in_colab() to install latex")
            self.latex = self.check_latex_installed()
        if self.latex:
            plt.rcParams['text.usetex'] = True
        else:
            plt.rcParams['text.usetex'] = False
        self.ui_available = self.check_ui_available()
        self.temp_folder = self.create_temp_folder('SensingSP')
        self.Paths = []
        self.CurrentFrame = 1
        self.CurrentTime  = 0
        self.StartTime  = 0
        
        
        self.RadarSpecifications = []
        self.suite_information = []
        
        self.directReceivefromTX = True
        self.RadarRX_only_fromscatters_itsTX = True
        self.RadarRX_only_fromitsTX = False
        self.Radar_TX_RX_isolation = True
        

        self.Detection_Parameters_xyz_start=[-.5, -2, -.2]
        self.Detection_Parameters_xyz_N=[10, 4, 5]
        self.Detection_Parameters_gridlen=.1
        self.DopplerProcessingMethod_FFT_Winv = 1
        self.EpsilonDistanceforHitTest = 10e-6 
        self.RayTracing_ReflectionPointEpsilon=1e-4
        self.ax = None
        # self.suite_information_finder = self.environment.BlenderSuiteFinder()
        # self.geocalculator = self.raytracing.BlenderGeometry.BlenderGeometry()
        # self.rayTracingFunctions = self.raytracing.ra.RayTracingFunctions()
        self.chains =[
            ('Starndard rx', "Starndard 1", "Starndard rx"),
                ('Hand Gestures', "Hand Gestures", "MM-Wave Radar-Based Recognition of Multiple Hand Gestures Using Long Short-Term Memory (LSTM) Neural Network"),
                ('RX chain 2', "RX chain 2", "RX chain 2"),
            ]
        self.apps =[
                ('FMCW Chirp Parameters Calculator', "FMCW Chirp Calculator", "FMCW Chirp Parameters Calculator"),
                ('Radar Parameters', "Radar Parameters", "Radar Parameters"),
                ('Hand Gesture MisoCNN', "Hand Gesture MisoCNN", "Hand Gesture MisoCNN"),
            ]
        self.radars =[
                ('TI AWR 1642 (2TX-4RX)', "TI AWR 1642 (2TX-4RX)", "TI AWR 1642 (2TX-4RX)"),
                ('TI IWR 6843 (3TX-4RX)', "TI IWR 6843 (3TX-4RX)", "TI IWR 6843 (3TX-4RX)"),
                ('TI Cascade AWR 2243 (12TX-16RX)', "TI Cascade AWR 2243 (12TX-16RX)", "TI Cascade AWR 2243 (12TX-16RX)"),
                ('SISO', "SISO", "SISO"),
                ('JSON', "JSON", "JSON"),
            ]
        self.sensortypes =[
                ('Lidar', "Lidar", "Lidar"),
                ('Camera', "Camera", "Camera"),
                ('RIS', "RIS", "RIS"),
                ('Comm.', "Comm.", "Comm."),
            ]
        self.scenariotypes =[
                ('2 Cubes + 6843', "2 Cubes + 6843", "2 Cubes moving"),
                ('Hand Gesture + 1642', "Hand Gesture + 1642", "Hand Gesture + 1642"),
                ('Hand Gesture + 3 Xethru Nature paper', "Hand Gesture + 3 Xethru Nature paper", "Hand Gesture + 3 Xethru Nature paper"),
                ('Pattern SISO', "Pattern SISO", "Pattern SISO"),
                ('Ray Tracing 1', "Ray Tracing 1", "Ray Tracing 1"),
                ('Ray Tracing 2', "Ray Tracing 2", "Ray Tracing 2"),
                ('Ray Tracing 3', "Ray Tracing 3", "Ray Tracing 3"),
            ]
        self.extratypes =[
                ('Environment Meshes', "Environment Meshes", "Environment Meshes"),
                ('Ray Tracing Simulation', "Ray Tracing Simulation", "Ray Tracing Simulation"),
                ('CSI Simulation', "CSI Simulation", "CSI Simulation"),
                ('RIS Simulation', "RIS Simulation", "RIS Simulation"),
                ('Open Temp Folder', "Open Temp Folder", "Open Temp Folder"),
                ('Light RayTracing', "Light RayTracing", "Light RayTracing"),
                ('Balanced RayTracing', "Balanced RayTracing", "Balanced RayTracing"),
                ('Advanced Intense RayTracing', "Advanced Intense RayTracing", "Advanced Intense RayTracing"),
            ]
        self.appSTYLESHEET = """
            /* General */
            QWidget {
                background-color: #f5f5f5;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12pt;
                color: #333333;
            }

            /* Buttons */
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #003f73;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }

            /* Line Edit */
            QLineEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d7;
            }

            /* Labels */
            QLabel {
                font-size: 12pt;
                color: #333333;
            }

            /* ComboBox */
            QComboBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #cccccc;
                width: 20px;
                background-color: #f5f5f5;
            }
            QComboBox::down-arrow {
                image: url(down-arrow-icon.png); /* Replace with your icon path */
            }

            /* Checkboxes */
            QCheckBox {
                spacing: 6px;
                font-size: 12pt;
                color: #333333;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #cccccc;
                border-radius: 2px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d7;
                border: 1px solid #0078d7;
                image: url(check-icon.png); /* Replace with your icon path */
            }

            /* Scroll Bars */
            QScrollBar:vertical {
                background-color: #f5f5f5;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #cccccc;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #0078d7;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                background: none;
            }

            /* Progress Bar */
            QProgressBar {
                background-color: #e6e6e6;
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 4px;
            }

            /* Tabs */
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #f5f5f5;
                padding: 8px;
                border: 1px solid #cccccc;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d7;
                color: white;
                border: 1px solid #0078d7;
            }
            QTabBar::tab:hover {
                background-color: #005a9e;
            }

            /* Table View */
            QTableView {
                border: 1px solid #cccccc;
                gridline-color: #eeeeee;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 8px;
                border: 1px solid #cccccc;
            }
            """
# ssp.config.directReceivefromTX =  False 
# ssp.config.RadarRX_only_fromscatters_itsTX = True
# ssp.config.RadarRX_only_fromitsTX = True
# ssp.config.Radar_TX_RX_isolation = True

# # ssp.utils.useCUDA(True)



    def release_videofigures(self):
        for video in self.Video_videos:
            video.release()
    def define_videofigures(self, N = 4, width_in_pixels = 1920 , height_in_pixels = 1080, dpi = 300, Video_fps = 30 ,video_directory = '.'):
        self.Video_Figs = []
        self.Video_Axes = []
        width_in_inches = width_in_pixels / dpi
        height_in_inches = height_in_pixels / dpi
        for _ in range(N):
            fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
            self.Video_Figs.append(fig)
            self.Video_Axes.append(ax)
            plt.close(fig) 
        self.Video_images=[]
        for fig in self.Video_Figs:
            self.Video_images.append(visualization.captureFig(fig=fig))
        self.Video_video_directory = video_directory
        self.Video_fps = Video_fps
        self.Video_videos,self.Video_videos_WH=visualization.firsttime_init_GridVideoWriters(self.Video_images,self.Video_video_directory,self.Video_fps)

    def define_axes(self,option = 1):
        if option == 1:
            fig, self.ax = plt.subplots(3,4)
            self.ax[2, 2] = fig.add_subplot(3, 4, 9 + 2 , projection='3d')
            self.ax[2, 3] = fig.add_subplot(3, 4, 9 + 3 , projection='3d')
        elif option == 2:
            fig, self.ax = plt.subplots(3,3)
            self.ax[2, 1] = fig.add_subplot(3, 3, 6 + 2 , projection='3d')
            self.ax[2, 2] = fig.add_subplot(3, 3, 6 + 3 , projection='3d')
            self.ax[1,2] = fig.add_subplot(3, 3, 3 + 3 , polar=True)
            self.ax[2,0] = fig.add_subplot(3, 3, 6 + 1 , polar=True)
        elif option == 3:
            self.Video_Figs = []
            self.Video_Axes = []
            width_in_pixels = 1920
            height_in_pixels = 1080
            dpi = 300
            width_in_inches = width_in_pixels / dpi
            height_in_inches = height_in_pixels / dpi
            for _ in range(4):
                fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
                self.Video_Figs.append(fig)
                self.Video_Axes.append(ax)
            self.Video_images=[]
            for fig in self.Video_Figs:
                self.Video_images.append(visualization.captureFig(fig=fig))
            self.Video_video_directory = '.'
            self.Video_fps = 30
            self.Video_videos,self.Video_videos_WH=visualization.firsttime_init_GridVideoWriters(self.Video_images,self.Video_video_directory,self.Video_fps)

        
        # if os.name=='nt':
        #     figManager = plt.get_current_fig_manager()
        #     figManager.window.showMaximized()
    def setDopplerProcessingMethod_FFT_Winv(self,method):
        self.DopplerProcessingMethod_FFT_Winv = method
    def restart(self):
        self.CurrentFrame = bpy.context.scene.frame_start
        self.CurrentFrame = 1
        self.StartTime  = 0
        # self.CurrentTime  += float(bpy.context.scene.render.fps)
        self.CurrentTime  = self.StartTime + (self.CurrentFrame-1) / float(bpy.context.scene.render.fps)
        self.NextTime  = self.StartTime + (self.CurrentFrame+0) / float(bpy.context.scene.render.fps)
        self.PreTime  = self.StartTime + (self.CurrentFrame-2) / float(bpy.context.scene.render.fps)
    def getCurrentFrame(self):
        return self.CurrentFrame
    def run(self):
        self.CurrentTime  = self.StartTime + (self.CurrentFrame-1) / float(bpy.context.scene.render.fps)
        self.NextTime  = self.StartTime + (self.CurrentFrame+0) / float(bpy.context.scene.render.fps)
        self.PreTime  = self.StartTime + (self.CurrentFrame-2) / float(bpy.context.scene.render.fps)
        if self.CurrentFrame + 1 > bpy.context.scene.frame_end:
            return False
        return True
    def get_system_type(self):
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU' in os.environ:
            return 'Google Colab'
        system = platform.system()
        if system == 'Windows':
            return 'Windows'
        elif system == 'Darwin':
            return 'macOS'
        elif system == 'Linux':
            if os.path.isfile('/etc/lsb-release'):
                with open('/etc/lsb-release') as f:
                    if 'Ubuntu' in f.read():
                        return 'Ubuntu'
            return 'Linux'
        else:
            return 'Unknown'
    def check_latex_installed(self):
        try:
            # Run the command 'pdflatex --version' to check if LaTeX is installed
            subprocess.run(['pdflatex', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    def install_latex_in_colab(self):
        print("Installing LaTeX in Google Colab...")
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'texlive', 'texlive-latex-extra', 'texlive-xetex', 'latexmk'], check=True)
    def check_ui_available(self):
        try:
            import matplotlib.pyplot as plt
            if 'DISPLAY' not in os.environ and platform.system() != 'Windows':
                # No display variable and not on Windows
                plt.switch_backend('Agg')  # Use a non-interactive backend
                return False
            else:
                # Try to use a backend that requires a display
                # plt.figure()
                return True
        except Exception:
            return False
    def create_temp_folder(self,folder_name):
        temp_dir = tempfile.gettempdir()
        folder_path = os.path.join(temp_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path