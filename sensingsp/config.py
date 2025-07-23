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
        self.AddRadar_ULA_N = 4
        self.AddRadar_ULA_d = .5
        self.AddRadar_URA_NxNy = [4,4]
        self.AddRadar_URA_dxdy = [.5,.5]
        self.Paths = []
        self.CurrentFrame = 1
        self.CurrentTime  = 0
        self.StartTime  = 0
        self.updateTime = 'July 23, 2025'
        
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
        self.useEpsilonDistanceforHitTest = False 
        self.RayTracing_ReflectionPointEpsilon=1e-4
        self.ax = None
        # self.suite_information_finder = self.environment.BlenderSuiteFinder()
        # self.geocalculator = self.raytracing.BlenderGeometry.BlenderGeometry()
        # self.rayTracingFunctions = self.raytracing.ra.RayTracingFunctions()
        self.chains =[
            ('Configure Receiver Chain', "Configure Receiver Chain", "Configure Receiver Chain"),
            ('Starndard rx', "Starndard 1", "Starndard rx"),
                ('Hand Gestures', "Hand Gestures", "MM-Wave Radar-Based Recognition of Multiple Hand Gestures Using Long Short-Term Memory (LSTM) Neural Network"),
                ('RX chain 2', "RX chain 2", "RX chain 2"),
            ]
        self.apps = [
                ('FMCW Chirp Parameters Calculator', "FMCW Chirp Calculator", "FMCW Chirp Parameters Calculator"),
                ('RIS Analysis', "RIS Analysis", "RIS Analysis"),
                ('Radar Parameters', "Radar Parameters", "Radar Parameters"),
                ('Hand Gesture MisoCNN', "Hand Gesture MisoCNN", "Hand Gesture MisoCNN"),
                ('Human Health Monitoring', "Human Health Monitoring", "Human Health Monitoring"),
                ('GAN Radar Waveforms', "GAN Radar Waveforms", "GAN Radar Waveforms"),
                ('SHARP Wifi Sensing', "SHARP Wifi Sensing", "SHARP Wifi Sensing"),
                ('Patch(microstrip) Antenna Pattern', "Patch(microstrip) Antenna Pattern", "Patch(microstrip) Antenna Pattern"),
                ('Antenna Element Pattern', "Antenna Element Pattern", "Antenna Element Pattern"),
            ]
        self.radars =[
                ('App&File', "App&File", "App&File"),
                ('TI AWR 1642 (2TX-4RX)', "TI AWR 1642 (2TX-4RX)", "TI AWR 1642 (2TX-4RX)"),
                ('TI IWR 6843 (3TX-4RX)', "TI IWR 6843 (3TX-4RX)", "TI IWR 6843 (3TX-4RX)"),
                ('TI Cascade AWR 2243 (12TX-16RX)', "TI Cascade AWR 2243 (12TX-16RX)", "TI Cascade AWR 2243 (12TX-16RX)"),
                ('SISO', "SISO", "SISO"),
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
                ('Altos Radar', "Altos Radar", "Altos Radar"),
                ('Pattern SISO', "Pattern SISO", "Pattern SISO"),
                ('Ray Tracing 1', "Ray Tracing 1", "Ray Tracing 1"),
                ('Ray Tracing 2', "Ray Tracing 2", "Ray Tracing 2"),
                ('Ray Tracing 3', "Ray Tracing 3", "Ray Tracing 3"),
                ('2 Slot Example', "2 Slot Example", "2 Slot Example"),
                ('2 Slot as RIS', "2 Slot as RIS", "2 Slot as RIS"),
                ('Target RCS Simulation', "Target RCS Simulation", "Target RCS Simulation"),
                ('Target RCS Simulation Plane', "Target RCS Simulation Plane", "Target RCS Simulation Plane"),
                ('Wifi Sensing', "Wifi Sensing", "Wifi Sensing"),
                ('Ray Tracing Refraction', "Ray Tracing Refraction", "Ray Tracing Refraction"),
            ]
        self.extratypes =[
                ('Surface Materials', "Surface Materials", "Surface Materials"),
                ('Environment Meshes', "Environment Meshes", "Environment Meshes"),
                ('Ray Tracing Simulation', "Ray Tracing Simulation", "Ray Tracing Simulation"),
                ('CSI Simulation', "CSI Simulation", "CSI Simulation"),
                ('RIS Simulation', "RIS Simulation", "RIS Simulation"),
                ('Antenna Pattern', "Antenna Pattern", "Antenna Pattern"),
                ('Open Temp Folder', "Open Temp Folder", "Open Temp Folder"),
                ('Light RayTracing', "Light RayTracing", "Light RayTracing"),
                ('Balanced RayTracing', "Balanced RayTracing", "Balanced RayTracing"),
                ('Advanced Intense RayTracing', "Advanced Intense RayTracing", "Advanced Intense RayTracing"),
                ('Wifi Sensing Settings', "Wifi Sensing Settings", "Wifi Sensing Settings"),
                ('Load Hub Environment', "Load Hub Environment", "Load Hub Environment"),
                ('Environment information', "Environment information", "Environment information"),
                ('Array Visualization', "Array Visualization", "Array Visualization"),
                ('SensingSP Version', "SensingSP Version", "SensingSP Version"),
            ]
        self.hub_REPO = "https://raw.githubusercontent.com/sensingsp/sensingsp-hub/main"
        self.myglobal_outsidedefined_RIS = None
        self.appSTYLESHEET1 = """
            /* ========== Base ========== */
            QWidget {
                background-color: #2B2B2B;
                color: #F0F0F0;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 11pt;
            }

            /* ========== Frames & GroupBoxes (panels) ========== */
            QFrame, QGroupBox {
                background-color: #3A3A3A;
                border: 1px solid #4E4E4E;
                border-radius: 2px;
                margin-top: 6px;
                padding: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color: #F0F0F0;
                font-weight: bold;
            }

            /* ========== PushButtons & ToolButtons ========== */
            QPushButton, QToolButton {
                background-color: #444444;
                border: 1px solid #5A5A5A;
                border-radius: 2px;
                padding: 4px 8px;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed, QToolButton:pressed {
                background-color: #666666;
            }
            QPushButton:checked, QToolButton:checked {
                background-color: #FF8C00; /* orange accent */
                border: 1px solid #CC7000;
                color: #2B2B2B;
            }
            QPushButton:disabled, QToolButton:disabled {
                background-color: #3A3A3A;
                color: #6E6E6E;
                border: 1px solid #4E4E4E;
            }

            /* ========== MenuBar & Menus ========== */
            QMenuBar {
                background-color: #2B2B2B;
                border-bottom: 1px solid #4E4E4E;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
                margin: 0 2px;
            }
            QMenuBar::item:selected {
                background-color: #444444;
            }
            QMenu {
                background-color: #3A3A3A;
                border: 1px solid #4E4E4E;
            }
            QMenu::item {
                padding: 4px 24px 4px 24px;
            }
            QMenu::item:selected {
                background-color: #555555;
            }

            /* ========== LineEdits & TextEdits ========== */
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #252525;
                border: 1px solid #4E4E4E;
                border-radius: 2px;
                padding: 4px;
                selection-background-color: #FF8C00;
                selection-color: #2B2B2B;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #FF8C00;
            }

            /* ========== ComboBoxes ========== */
            QComboBox {
                background-color: #252525;
                border: 1px solid #4E4E4E;
                border-radius: 2px;
                padding: 4px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #4E4E4E;
            }
            QComboBox::down-arrow {
                image: url(your-arrow-icon.png);
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #3A3A3A;
                border: 1px solid #4E4E4E;
                selection-background-color: #555555;
            }

            /* ========== Tabs ========== */
            QTabWidget::pane {
                border: 1px solid #4E4E4E;
                top: -1px; /* overlap the tabs */
            }
            QTabBar::tab {
                background-color: #3A3A3A;
                border: 1px solid #4E4E4E;
                border-bottom: none;
                border-top-left-radius: 2px;
                border-top-right-radius: 2px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2B2B2B;
                color: #FF8C00;
                border-color: #FF8C00;
            }
            QTabBar::tab:hover {
                background-color: #555555;
            }

            /* ========== Sliders ========== */
            QSlider::groove:horizontal {
                background: #4E4E4E;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FF8C00;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::groove:vertical {
                background: #4E4E4E;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #FF8C00;
                height: 12px;
                margin: 0 -4px;
                border-radius: 6px;
            }

            /* ========== ScrollBars ========== */
            QScrollBar:vertical {
                background-color: #2B2B2B;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #5A5A5A;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                height: 0;
            }
            QScrollBar:horizontal {
                background-color: #2B2B2B;
                height: 10px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background-color: #5A5A5A;
                min-width: 20px;
                border-radius: 4px;
            }

            /* ========== Progress Bars ========== */
            QProgressBar {
                background-color: #3A3A3A;
                border: 1px solid #4E4E4E;
                border-radius: 2px;
                text-align: center;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: #FF8C00;
                border-radius: 2px;
            }

            /* ========== Table Views ========== */
            QTableView {
                background-color: #252525;
                gridline-color: #4E4E4E;
                alternate-background-color: #2B2B2B;
            }
            QHeaderView::section {
                background-color: #3A3A3A;
                padding: 4px;
                border: 1px solid #4E4E4E;
            }
            """
        self.appSTYLESHEET2 = """
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
        self.appSTYLESHEET3 = """
            /* ========== Cyberpunk Dark Theme ========== */
            QWidget {
                background-color: #141414;
                color: #E0E0E0;
                font-family: "Roboto", sans-serif;
                font-size: 10pt;
            }

            /* Panels */
            QFrame, QGroupBox {
                background-color: rgba(20,20,20,0.9);
                border: 1px solid #1F1F1F;
                border-radius: 5px;
                margin-top: 6px;
                padding: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #00E0FF;
                font-weight: bold;
            }

            /* Buttons */
            QPushButton, QToolButton {
                background-color: transparent;
                border: 2px solid #00E0FF;
                border-radius: 4px;
                padding: 6px 14px;
                color: #00E0FF;
                text-transform: uppercase;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: rgba(0,224,255,0.1);
            }
            QPushButton:pressed, QToolButton:pressed {
                background-color: rgba(0,224,255,0.2);
            }
            QPushButton:checked, QToolButton:checked {
                background-color: #00E0FF;
                color: #141414;
            }
            QPushButton:disabled, QToolButton:disabled {
                border-color: #333333;
                color: #555555;
            }

            /* Menu Bar */
            QMenuBar {
                background: #141414;
                border-bottom: 1px solid #1F1F1F;
            }
            QMenuBar::item {
                padding: 4px 12px;
            }
            QMenuBar::item:selected {
                background: rgba(0,224,255,0.2);
            }
            QMenu {
                background: #1E1E1E;
                border: 1px solid #1F1F1F;
            }
            QMenu::item:selected {
                background: rgba(0,224,255,0.3);
            }

            /* Text Inputs */
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 6px;
                selection-background-color: #00E0FF;
                selection-color: #141414;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #00E0FF;
            }

            /* Combo Boxes */
            QComboBox {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #333333;
            }
            QComboBox::down-arrow {
                image: url(cyber-down-arrow.png);
                width: 12px; height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                selection-background-color: rgba(0,224,255,0.3);
            }

            /* Tabs */
            QTabWidget::pane {
                border: 1px solid #1F1F1F;
                top: -1px;
            }
            QTabBar::tab {
                background: #1E1E1E;
                border: 1px solid #1F1F1F;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #141414;
                color: #00E0FF;
                border-color: #00E0FF;
            }
            QTabBar::tab:hover {
                background: rgba(0,224,255,0.1);
            }

            /* Sliders */
            QSlider::groove:horizontal {
                background: #333333;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00E0FF;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::groove:vertical {
                background: #333333;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #00E0FF;
                height: 14px;
                margin: 0 -4px;
                border-radius: 7px;
            }

            /* ScrollBars */
            QScrollBar:vertical {
                background: #141414;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #333333;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00E0FF;
            }
            QScrollBar:horizontal {
                background: #141414;
                height: 10px;
            }
            QScrollBar::handle:horizontal {
                background: #333333;
                min-width: 20px;
                border-radius: 5px;
            }

            /* ProgressBar */
            QProgressBar {
                background: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 4px;
                text-align: center;
                height: 14px;
            }
            QProgressBar::chunk {
                background: #00E0FF;
                border-radius: 4px;
            }

            /* TableView */
            QTableView {
                background: #1E1E1E;
                gridline-color: #333333;
                alternate-background-color: #141414;
            }
            QHeaderView::section {
                background: #1F1F1F;
                padding: 6px;
                border: 1px solid #333333;
            }
            """
        self.appSTYLESHEET4 = """
            /* ========== Material Light Theme ========== */
            QWidget {
                background-color: #FAFAFA;
                color: #212121;
                font-family: "Roboto", sans-serif;
                font-size: 11pt;
            }

            /* Panels */
            QFrame, QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 8px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #424242;
                font-weight: 600;
            }

            /* Buttons */
            QPushButton, QToolButton {
                background-color: #6200EE;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #FFFFFF;
                font-weight: 500;
                min-height: 36px;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: #3700B3;
            }
            QPushButton:pressed, QToolButton:pressed {
                background-color: #30009C;
            }
            QPushButton:disabled, QToolButton:disabled {
                background-color: #E0E0E0;
                color: #9E9E9E;
            }

            /* Menu Bar */
            QMenuBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #E0E0E0;
            }
            QMenuBar::item {
                padding: 6px 12px;
                color: #424242;
            }
            QMenuBar::item:selected {
                background-color: #F1F1F1;
            }
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
            }
            QMenu::item:selected {
                background-color: #F1F1F1;
            }

            /* Text Inputs */
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 6px 8px;
                selection-background-color: #6200EE;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border: 2px solid #6200EE;
            }

            /* Combo Boxes */
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 6px 8px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #E0E0E0;
            }
            QComboBox::down-arrow {
                image: url(material-down-arrow.png);
                width: 12px; height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                selection-background-color: #EDE7F6;
            }

            /* Checkboxes & Radios */
            QCheckBox, QRadioButton {
                spacing: 6px;
                color: #424242;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #E0E0E0;
                border-radius: 2px;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #6200EE;
                border: none;
            }

            /* Tabs */
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #F5F5F5;
                padding: 8px 16px;
                border: 1px solid #E0E0E0;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #6200EE;
                border-bottom-color: #FFFFFF;
            }
            QTabBar::tab:hover {
                background-color: #EEEEEE;
            }

            /* Sliders */
            QSlider::groove:horizontal {
                background: #E0E0E0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6200EE;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::groove:vertical {
                background: #E0E0E0;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #6200EE;
                height: 16px;
                margin: 0 -5px;
                border-radius: 8px;
            }

            /* ScrollBars */
            QScrollBar:vertical {
                background: #F5F5F5;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #E0E0E0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6200EE;
            }
            QScrollBar:horizontal {
                background: #F5F5F5;
                height: 10px;
            }
            QScrollBar::handle:horizontal {
                background: #E0E0E0;
                min-width: 20px;
                border-radius: 5px;
            }

            /* ProgressBar */
            QProgressBar {
                background: #E0E0E0;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                text-align: center;
                height: 14px;
            }
            QProgressBar::chunk {
                background: #6200EE;
                border-radius: 4px;
            }

            /* TableView */
            QTableView {
                background: #FFFFFF;
                gridline-color: #EEEEEE;
                alternate-background-color: #F9F9F9;
            }
            QHeaderView::section {
                background: #F5F5F5;
                padding: 8px;
                border: 1px solid #E0E0E0;
            }
            """
        self.appSTYLESHEET = self.appSTYLESHEET1

# ssp.config.directReceivefromTX =  False 
# ssp.config.RadarRX_only_fromscatters_itsTX = True
# ssp.config.RadarRX_only_fromitsTX = True
# ssp.config.Radar_TX_RX_isolation = True

# # ssp.utils.useCUDA(True)
    def GPU_run_available(self):
        CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
        if CUDA_signalGeneration_Enabled:
            if self.CUDA_is_available:
                return True
        return False
    def get_frame_end(self):
        return bpy.context.scene.frame_end

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