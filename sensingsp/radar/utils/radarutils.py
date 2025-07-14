# import os, sys, subprocess
# import easygui
import bpy
import numpy as np
from mathutils import Vector, Matrix
import matplotlib.pyplot as plt
LightSpeed = 299792458.0  # Speed of light in m/s
from collections import defaultdict
import scipy
from numba import cuda
import json

import sensingsp as ssp

from enum import Enum

class RadarSensorsCategory(Enum):
    TI_AWR1642 = 1
    TI_IWR6843 = 2
    # TI_AWR2243 = 3
    # TI_AWR2944 = 4
    TI_Cascade_AWR2243 = 5
    SISO_mmWave76GHz = 6
    Xhetru_X4 = 7
    ULA_AllRX = 8
    ULA_SameTXRX = 9
    URA_AllRX = 10
    URA_SameTXRX = 11
    URA_AllTX = 12
    URA_LinearHtxVrx = 13
    ULA_AllRX_N = 14

class RadarSignalGenerationConfigurations(Enum):
    Spillover_Enabled = 1
    Spillover_Disabled = 2
    RayTracing_Light = 3
    RayTracing_Balanced = 4
    RayTracing_Advanced = 5
    CUDA_Enabled = 6
    CUDA_Disabled = 7


class MIMORadarTechnique(Enum):
    TDM = 1
    PhasedArray = 2
    BPM = 3
    DDM = 4
    


class RadarSpecJSON:
    def __init__(self, model=None, frequency_range=None, bandwidth=None, transmit_channels=None, receive_channels=None,
                 transmit_power=None, receive_noise_figure=None, processing_units=None, peripherals=None, 
                 memory=None, power_management=None, clock_source=None, package=None, operating_conditions=None):
        self.model = model
        self.frequency_range = frequency_range
        self.bandwidth = bandwidth
        self.transmit_channels = transmit_channels
        self.receive_channels = receive_channels
        self.transmit_power = transmit_power
        self.receive_noise_figure = receive_noise_figure
        self.processing_units = processing_units
        self.peripherals = peripherals
        self.memory = memory
        self.power_management = power_management
        self.clock_source = clock_source
        self.package = package
        self.operating_conditions = operating_conditions

    def save_to_json(self, file_path):
        """Save the radar specifications to a JSON file."""
        try:
            with open(file_path, 'w') as file:
                json.dump(self.__dict__, file, indent=4)
            print(f"Specifications saved to {file_path}.")
        except Exception as e:
            print(f"Error saving to JSON: {e}")

    def load_from_json(self, file_path):
        """Load the radar specifications from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.__dict__.update(data)
            print(f"Specifications loaded from {file_path}.")
        except Exception as e:
            print(f"Error loading from JSON: {e}")

# # Example Usage
# if __name__ == "__main__":
#     radar = RadarSpecJSON(
#         model="IWR1642",
#         frequency_range="76-81 GHz",
#         bandwidth="4 GHz",
#         transmit_channels=2,
#         receive_channels=4,
#         transmit_power="12.5 dBm",
#         receive_noise_figure="14 dB",
#         processing_units={"DSP": "TI C674x", "MCU": "ARM Cortex-R4F"},
#         peripherals={"SPI": 2, "UART": 2, "CAN": True, "I2C": True, "GPIO": True},
#         memory="1.5 MB",
#         power_management="Built-in LDO",
#         clock_source="40 MHz external oscillator",
#         package="161-pin, 10.4 mm × 10.4 mm BGA",
#         operating_conditions="–40°C to 105°C"
#     )

#     # Save to JSON
#     radar.save_to_json("radar_specs.json")

#     # Load from JSON
#     loaded_radar = RadarSpecJSON()
#     loaded_radar.load_from_json("radar_specs.json")
#     print(loaded_radar.__dict__)




@cuda.jit
def CUDA_cfar_ca_2D_alpha(range_doppler_map, detections, THR, num_train0, num_train1, num_guard0, num_guard1, alpha):
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    range_size, doppler_size = range_doppler_map.shape
    wr = num_train0
    wd = num_train1
    if i < range_size and j < doppler_size:
        THR[i, j] = 0
        n = 0
        threshold_sum = 0.0
        for k in range(wr):
            i1 = i - k - num_guard0
            if i1 >= 0:
                for h in range(wd):
                    j1 = j - h - num_guard1
                    if j1 >= 0:
                        threshold_sum += range_doppler_map[i1, j1]
                        n += 1
                    j1 = j + h + num_guard1
                    if j1 < doppler_size:
                        threshold_sum += range_doppler_map[i1, j1]
                        n += 1
            i1 = i + k + num_guard0
            if i1 < range_size:
                for h in range(wd):
                    j1 = j - h - num_guard1
                    if j1 >= 0:
                        threshold_sum += range_doppler_map[i1, j1]
                        n += 1
                    j1 = j + h + num_guard1
                    if j1 < doppler_size:
                        threshold_sum += range_doppler_map[i1, j1]
                        n += 1

        if n > 0:
            THR[i, j] = threshold_sum / n

        if range_doppler_map[i, j] > alpha * THR[i, j]:
            detections[i, j] = 1
        else:
            detections[i, j] = 0
def cfar_ca_2D_alpha_cuda(range_doppler_map, num_train, num_guard, alpha):
    # Allocate and transfer data to the device
    d_range_doppler_map = cuda.to_device(range_doppler_map)
    d_detections = cuda.device_array(range_doppler_map.shape, dtype=np.int32)
    d_THR = cuda.device_array(range_doppler_map.shape, dtype=np.float32)

    # Define grid and block dimensions
    blocksize = (16, 16)
    blockspergrid_x = (range_doppler_map.shape[0] + (blocksize[0] - 1)) // blocksize[0]
    blockspergrid_y = (range_doppler_map.shape[1] + (blocksize[1] - 1)) // blocksize[1]
    gridsize = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    CUDA_cfar_ca_2D_alpha[gridsize, blocksize](
        d_range_doppler_map, d_detections, d_THR, num_train[0], num_train[1], num_guard[0], num_guard[1], alpha
    )

    # Copy result back to host
    detections = d_detections.copy_to_host()
    THR = d_THR.copy_to_host()
    return detections,THR
def cfar_ca_2D_alpha(range_doppler_map, num_train, num_guard, alpha):
    wr = num_train[0]#int(np.ceil(num_train*range_doppler_map.shape[0]))
    wd = num_train[1]#int(np.ceil(num_train*range_doppler_map.shape[1]))
    detections = np.zeros_like(range_doppler_map)
    THR = np.zeros_like(range_doppler_map)
    for i in range(THR.shape[0]):
        # print(i,THR.shape[0])
        for j in range(THR.shape[1]):
            n  = 0
            for k in range(wr):
                i1=i-k-num_guard[0]
                if i1>=0:
                    for h in range(wd):
                        j1=j-h-num_guard[1]
                        if j1>=0:
                            THR[i,j]+=range_doppler_map[i1,j1]
                            n+=1
                        j1=j+h+num_guard[1]
                        if j1<range_doppler_map.shape[1]:
                            THR[i,j]+=range_doppler_map[i1,j1]
                            n+=1
                i1=i+k+num_guard[0]
                if i1<range_doppler_map.shape[0]:
                    for h in range(wd):
                        j1=j-h-num_guard[1]
                        if j1>=0:
                            THR[i,j]+=range_doppler_map[i1,j1]
                            n+=1
                        j1=j+h+num_guard[1]
                        if j1<range_doppler_map.shape[1]:
                            THR[i,j]+=range_doppler_map[i1,j1]
                            n+=1
            if n>0:
                THR[i,j]/=n
            if range_doppler_map[i,j]>alpha*THR[i,j]:
                detections[i,j]=1
    
    return detections,THR
        

def cfar_ca_2D(range_doppler_map, num_train, num_guard, prob_fa):
    """
    2D CA-CFAR implementation for target detection in a range-Doppler map.

    Parameters:
    - range_doppler_map: 2D numpy array of the range-Doppler map.
    - num_train: Number of training cells around the CUT (Cell Under Test) in each direction.
    - num_guard: Number of guard cells around the CUT in each direction.
    - prob_fa: Desired probability of false alarm.

    Returns:
    - detections: 2D binary array indicating detected targets in the range-Doppler map.
    """

    # Calculate CFAR threshold multiplier
    num_train_cells = (2 * num_train + 2 * num_guard + 1) ** 2 - (2 * num_guard + 1) ** 2 - 1
    alpha = num_train_cells * (prob_fa ** (-1 / num_train_cells) - 1)

    # Initialize output array for detections
    detections = np.zeros_like(range_doppler_map)
    THR = np.max(range_doppler_map)/alpha*np.ones_like(range_doppler_map)

    # Slide the window across the range-Doppler map
    for i in range(num_train + num_guard, range_doppler_map.shape[0] - (num_train + num_guard)):
        for j in range(num_train + num_guard, range_doppler_map.shape[1] - (num_train + num_guard)):

            # Define the training and guard cells
            training_cells = range_doppler_map[
                i - num_train - num_guard : i + num_train + num_guard + 1,
                j - num_train - num_guard : j + num_train + num_guard + 1,
            ]

            # Exclude the guard cells and CUT from the training cells
            training_cells[
                num_train:num_train + 2 * num_guard + 1, num_train:num_train + 2 * num_guard + 1
            ] = 0

            # Calculate noise level by averaging training cells
            noise_level = np.sum(training_cells) / num_train_cells

            # Set threshold
            threshold = alpha * noise_level
            THR[i, j] = threshold

            # Compare CUT with threshold
            if range_doppler_map[i, j] > threshold:
                detections[i, j] = 1

    return detections,THR
def cfar_simple_2D(range_doppler_map,alpha):
    THR = np.mean(range_doppler_map)*alpha*np.ones_like(range_doppler_map)
    detections = np.where(range_doppler_map > THR, 1, 0)
    return detections,THR
def cfar_simple_max(range_doppler_map):
    THR = np.max(range_doppler_map)*np.ones_like(range_doppler_map)
    detections = np.where(range_doppler_map == THR, 1, 0)
    return detections,THR

def calculate_n_from_beamwidth(beamwidth):
    """
    Calculate the exponent n based on the beamwidth.
    A smaller beamwidth corresponds to a larger n.
    
    Parameters:
    beamwidth (float): Beamwidth in degrees.
    
    Returns:
    float: The calculated exponent n.
    """
    # n is inversely proportional to the beamwidth
    # This is an empirical relationship, adjust as needed for specific antennas
    n = 2 / (beamwidth / 60) ** 2
    return n

def antenna_gain_from_beamwidth(azimuth_angle, elevation_angle, max_gain, azimuth_beamwidth, elevation_beamwidth,antennaType):
    """
    Calculate the antenna gain based on input azimuth and elevation angles, using beamwidths.
    
    Parameters:
    azimuth_angle (float or np.array): The azimuth angle in degrees from the boresight.
    elevation_angle (float or np.array): The elevation angle in degrees from the boresight.
    max_gain (float): The maximum gain at 0 degrees, 0 degrees.
    azimuth_beamwidth (float): The azimuth beamwidth in degrees.
    elevation_beamwidth (float): The elevation beamwidth in degrees.
    
    Returns:
    float or np.array: The antenna gain at the given azimuth and elevation angles in dBi.
    """
    
    # Convert angles to radians
    azimuth_angle_rad = np.radians(azimuth_angle)
    # print(azimuth_angle)
    elevation_angle_rad = np.radians(elevation_angle)
    
    if antennaType=='Directional-Sinc':
        norm_azimuth = azimuth_angle_rad / np.radians(azimuth_beamwidth / 2)
        norm_elevation = elevation_angle_rad / np.radians(elevation_beamwidth / 2)
        
        gain_factor_azimuth = np.sinc(norm_azimuth / np.pi)
        gain_factor_elevation = np.sinc(norm_elevation / np.pi)
    elif antennaType=='Rect':
        gain_factor_azimuth = 1 if np.abs(azimuth_angle)<azimuth_beamwidth / 2 else 1e-6
        gain_factor_elevation = 1 if np.abs(elevation_angle)<elevation_beamwidth / 2 else 1e-6
    else:
        # Calculate n for both azimuth and elevation based on beamwidths
        n_azimuth = calculate_n_from_beamwidth(azimuth_beamwidth)
        n_elevation = calculate_n_from_beamwidth(elevation_beamwidth)
        # Calculate the gain factors based on the angles and n values
        gain_factor_azimuth = np.abs(np.cos(azimuth_angle_rad)) ** n_azimuth
        gain_factor_elevation = np.abs(np.cos(elevation_angle_rad)) ** n_elevation
    
    # Total gain is the product of the gain factors and the maximum gain
    gain = np.abs(max_gain * gain_factor_azimuth * gain_factor_elevation)
    
    return gain

# # Example usage
# azimuth_angle = 10  # Example azimuth angle in degrees
# elevation_angle = 5  # Example elevation angle in degrees
# max_gain = 15  # Example maximum gain in dBi
# azimuth_beamwidth = 60  # Example azimuth beamwidth in degrees
# elevation_beamwidth = 30  # Example elevation beamwidth in degrees

# gain = antenna_gain_from_beamwidth(azimuth_angle, elevation_angle, max_gain, azimuth_beamwidth, elevation_beamwidth)
# print(f"Antenna Gain at ({azimuth_angle}°, {elevation_angle}°): {gain:.2f} dBi")


def predefined_array_configs_TI_Cascade_AWR2243(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)

    s = 0.05
    Type = 'SPOT'

    tx_positions = [
        (0, 0),
        (-4, 0),
        (-8, 0),
        (-9, 1),
        (-10, 4),
        (-11, 6),
        (-12, 0),
        (-16, 0),
        (-20, 0),
        (-24, 0),
        (-28, 0),
        (-32, 0)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty

    bx0 = -17
    bx = bx0
    by = 34
    s = 1

    rx_positions = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (11, 0),
        (11+1, 0),
        (11+2, 0),
        (11+3, 0),
        (46, 0),
        (46+1, 0),
        (46+2, 0),
        (46+3, 0),
        (53-3, 0),
        (53-2, 0),
        (53-1, 0),
        (53, 0)
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    

    empty["TXRXPos"]=[tx_positions,rx_positions]
    
    empty["antenna2azelIndex"]=[]
    return empty

def make_Radar_DAR(radar,N=[3,1],d=[.3,.3]):
    parts = radar.name.split('_')
    isuite = int(parts[1]) 
    iradar = int(parts[2])
    isubdevive = int(parts[3])
    
    
     
def predefined_array_configs_M_MIMO(isuite, iradar, location, rotation, f0=70e9,N=10):
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    m = (N-1)*.5
    for i in range(N):
        for j in range(N):
            bpy.ops.object.light_add(type=Type, radius=1, location=((i-m)*Lambda/2*N,(j-m)*Lambda/2*N, 0))
            tx = bpy.context.object
            tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
            tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1
    for i in range(N):
        for j in range(N):
            bpy.ops.object.camera_add(location=((i-m)*Lambda/2,(j-m)*Lambda/2, 0), rotation=(0, 0, 0))
            rx = bpy.context.object
            rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
            rx.parent = empty
            rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty

def predefined_array_configs_imagingURARX(isuite, iradar, location, rotation, f0=70e9,NRX=10,DHalfwave=6):
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    N=1
    m = (N-1)*.5
    for i in range(N):
        for j in range(N):
            bpy.ops.object.light_add(type=Type, radius=1, location=((i-m)*Lambda/2*N,(j-m)*Lambda/2*N, 0))
            tx = bpy.context.object
            tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
            tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1
    N=NRX
    for i in range(N):
        for j in range(N):
            bpy.ops.object.camera_add(location=((i-m)*Lambda/2*DHalfwave,(j-m)*Lambda/2*DHalfwave, 0), rotation=(0, 0, 0))
            rx = bpy.context.object
            rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
            rx.parent = empty
            rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty

def predefined_array_configs_LinearArray(isuite, iradar, location, rotation, f0=2.447e9,
                                         LinearArray_TXPos =[0],
                                         LinearArray_RXPos =[.56,.84,.98]):
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = []
    for i in LinearArray_TXPos:
        tx_positions.append((int(round(i/(Lambda/2))),0))
    
    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1

    
    rx_positions = []
    for i in LinearArray_RXPos:
        rx_positions.append((int(round(i/(Lambda/2))),0))
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( (bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    va1=[]
    va2=[]
    for i in range(len(tx_positions)):
      for j in range(len(rx_positions)):
        va1.append(tx_positions[i][0]+rx_positions[j][0])
        va2.append(tx_positions[i][1]+rx_positions[j][1])
    va1 = np.array(va1)
    va2 = np.array(va2)
    va1 -= np.min(va1)
    va2 -= np.min(va2)
    rows = np.array(va1).astype(np.int32)
    cols = np.array(va2).astype(np.int32)
    empty["antenna2azelIndex"]=[rows,cols]
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    
def predefined_array_configs_DARWu_AllRX(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0)
    ]
    
    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (1, 0),
        (2, 0),
        (3, 0),
        (31, 0),
        (32, 0),
        (33, 0)
    ]
    rx_positions = []
    
    tx=[0,4,8,26,30,34]
    rx=[1,2,3,31,32,33]
    va = np.array(sorted({i + j for i in tx for j in rx}))
    for i in va:
        rx_positions.append((int(i),0))
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( (bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    va1=[]
    va2=[]
    for i in range(len(tx_positions)):
      for j in range(len(rx_positions)):
        va1.append(tx_positions[i][0]+rx_positions[j][0])
        va2.append(tx_positions[i][1]+rx_positions[j][1])
    va1 = np.array(va1)
    va2 = np.array(va2)
    va1 -= np.min(va1)
    va2 -= np.min(va2)
    rows = np.array(va1).astype(np.int32)
    cols = np.array(va2).astype(np.int32)
    empty["antenna2azelIndex"]=[rows,cols]
    empty["TXRXPos"]=[tx_positions,rx_positions]
    
    return empty    

def predefined_array_configs_DARWu_AllTX(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0)
    ]
    tx_positions = []
    for i in range(int(192/4)):
        tx_positions.append((i+1,0))
    
    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0)
    ]
    # rx_positions = []
    # for i in range(50):
    #     rx_positions.append((i+1,0))
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( (bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    va1=[]
    va2=[]
    for i in range(len(tx_positions)):
      for j in range(len(rx_positions)):
        va1.append(tx_positions[i][0]+rx_positions[j][0])
        va2.append(tx_positions[i][1]+rx_positions[j][1])
    va1 = np.array(va1)
    va2 = np.array(va2)
    va1 -= np.min(va1)
    va2 -= np.min(va2)
    rows = np.array(va1).astype(np.int32)
    cols = np.array(va2).astype(np.int32)
    empty["antenna2azelIndex"]=[rows,cols]
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    

def predefined_array_configs_DARWu(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0),
        (4, 0),
        (8, 0),
        (26, 0),
        (30, 0),
        (34, 0),
    ]
    
    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (1, 0),
        (2, 0),
        (3, 0),
        (31, 0),
        (32, 0),
        (33, 0)
    ]


    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( (bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    va1=[]
    va2=[]
    for i in range(len(tx_positions)):
      for j in range(len(rx_positions)):
        va1.append(tx_positions[i][0]+rx_positions[j][0])
        va2.append(tx_positions[i][1]+rx_positions[j][1])
    va1 = np.array(va1)
    va2 = np.array(va2)
    va1 -= np.min(va1)
    va2 -= np.min(va2)
    rows = np.array(va1).astype(np.int32)
    cols = np.array(va2).astype(np.int32)
    empty["antenna2azelIndex"]=[rows,cols]
    # vainfo = virtualArray_info(tx_positions,rx_positions)
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    
def virtualArray_info(tx_positions,rx_positions):
    ULA = defaultdict(lambda: defaultdict(list))
    TXRX = {}
    xv,yv=[],[]
    for itx, tx_pos in enumerate(tx_positions):
        for irx, rx_pos in enumerate(rx_positions):
            xv.append(int(round((tx_pos[0] + rx_pos[0]))))
            yv.append(int(round((tx_pos[1] + rx_pos[1]))))
    mxv = np.min(np.array(xv))
    myv = np.min(np.array(yv))
    Mxv = np.max(np.array(xv))
    Myv = np.max(np.array(yv))
    Lx=Mxv-mxv+1
    Ly=Myv-myv+1
    for itx, tx_pos in enumerate(tx_positions):
        for irx, rx_pos in enumerate(rx_positions):
            x = int(round((tx_pos[0] + rx_pos[0])))-mxv
            y = int(round((tx_pos[1] + rx_pos[1])))-myv

            # Append to ULA and set TXRX
            ULA[x][y].append((itx, irx))
            TXRX[(itx, irx)] = (x, y)
    NonZ =[]
    for j in range(Ly):
        nz =[]
        for i in range(Lx):
            x=len(ULA[i][j])
            if x>0:
                nz.append(i)
        NonZ.append(np.array(nz))
    return [ULA,TXRX,Lx,Ly,NonZ]

    
def predefined_array_configs_3TX1RX(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0),
        (-4, 0),
        (-8, 0)
    ]
    # empty["TX_az"]=[0, -4, -8]
    # empty["TX_el"]=[0,  0, 0]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -6
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0)
    ]

    # antenna_signal_example = np.zeros((len(tx_positions),len(rx_positions)),dtype=np.complex128)
    # for i in range(len(tx_positions)):
    #     for j in range(len(rx_positions)):
    #         antenna_signal_example[i,j]=i+1j*j
    # azel_signal_example = np.zeros((12,2),dtype=np.complex128)
    
    rows = [0,1,2]
    # cols = [0,0,0,0,1,1,1,1,0,0,0 ,0]
    cols = [0,0,0]
    
    # azel_signal_example[rows, cols] = antenna_signal_example.ravel()

    # empty["RX_az"]=[0, 1, 2, 3]
    # empty["RX_el"]=[0,  0, 0]
    empty["antenna2azelIndex"]=[rows,cols]
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty   

def predefined_array_configs_TI_IWR6843(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0),
        (4, 1),
        (8, 0)
    ]
    # empty["TX_az"]=[0, -4, -8]
    # empty["TX_el"]=[0,  0, 0]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -6
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0)
    ]

    # antenna_signal_example = np.zeros((len(tx_positions),len(rx_positions)),dtype=np.complex128)
    # for i in range(len(tx_positions)):
    #     for j in range(len(rx_positions)):
    #         antenna_signal_example[i,j]=i+1j*j
    # azel_signal_example = np.zeros((12,2),dtype=np.complex128)
    
    rows = [0,1,2,3,4,5,6,7,8,9,10,11]
    rows = rows[::-1]
    # cols = [0,0,0,0,1,1,1,1,0,0,0 ,0]
    cols = [0,0,0,0,0,0,0,0,0,0,0 ,0]
    
    # azel_signal_example[rows, cols] = antenna_signal_example.ravel()

    # empty["RX_az"]=[0, 1, 2, 3]
    # empty["RX_el"]=[0,  0, 0]
    empty["antenna2azelIndex"]=[rows,cols]
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( (bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    

def predefined_array_configs_TI_AWR1642(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0),
        (-4, 0)
    ]
    # empty["TX_az"]=[0, -4, -8]
    # empty["TX_el"]=[0,  0, 0]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -6
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0),
        (-1, 0),
        (-2, 0),
        (-3, 0)
    ]

    # antenna_signal_example = np.zeros((len(tx_positions),len(rx_positions)),dtype=np.complex128)
    # for i in range(len(tx_positions)):
    #     for j in range(len(rx_positions)):
    #         antenna_signal_example[i,j]=i+1j*j
    # azel_signal_example = np.zeros((12,2),dtype=np.complex128)
    
    rows = [0,1,2,3,4,5,6,7]
    # cols = [0,0,0,0,1,1,1,1,0,0,0 ,0]
    cols = [0,0,0,0,0,0,0,0,0,0,0 ,0]
    
    # azel_signal_example[rows, cols] = antenna_signal_example.ravel()

    # empty["RX_az"]=[0, 1, 2, 3]
    # empty["RX_el"]=[0,  0, 0]
    empty["antenna2azelIndex"]=[rows,cols]
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( (bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    

def predefined_array_configs_URA(isuite, iradar, location, rotation,NTX=[3,3],NRX=[2,2],dazimuth=.5,delevation=.5, f0=70e9): 
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = []
    k=0
    for i in range(NTX[0]):
        for j in range(NTX[1]):
            bpy.ops.object.light_add(type=Type, radius=1, location=(NRX[0]*(i-np.mean(np.arange(NTX[0])))*Lambda*dazimuth, NRX[1]*(j-np.mean(np.arange(NTX[1])))*Lambda*delevation, 0))
            tx = bpy.context.object
            tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{k+1:05}'
            k+=1
            tx.parent = empty
            tx_positions.append((NRX[0]*(i-np.mean(np.arange(NTX[0])))*2.0*dazimuth, NRX[1]*(j-np.mean(np.arange(NTX[1])))*2.0*delevation))

    s = 1
    k=0
    rx_positions = []
    for i in range(NRX[0]):
        for j in range(NRX[1]):
            bpy.ops.object.camera_add(location=((i-np.mean(np.arange(NRX[0])))*Lambda*dazimuth, (j-np.mean(np.arange(NRX[1])))*Lambda*delevation, 0), rotation=(0, 0, 0))
            rx = bpy.context.object
            rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
            rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{k+1:05}'
            k+=1
            rx.parent = empty
            rx.data.lens = 10
            rx_positions.append(((i-np.mean(np.arange(NRX[0])))*2.0*dazimuth, (j-np.mean(np.arange(NRX[1])))*2.0*delevation))
    empty["TXRXPos"]=[tx_positions,rx_positions]
    empty["antenna2azelIndex"]=[[],[]]
    
    p2y,p2z=[],[]
    Mz = NTX[1]
    My = NTX[0]
    Nz = NRX[1]
    Ny = NRX[0]
    for imy in range(My):
        for imz in range(Mz):
            for iny in range(Ny):
                for inz in range(Nz):
                    p2z.append(inz+imz*Nz)
                    p2y.append(iny+imy*Ny)
    empty["antenna2azelIndex"]=[p2y,p2z]
    return empty    



def predefined_array_configs_3by1(isuite, iradar, location, rotation, f0=70e9):  
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0),
        (-4, 0),
        (-8, 0)
    ]
    # empty["TX_az"]=[0, -4, -8]
    # empty["TX_el"]=[0,  0, 0]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -6
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0)
    ]

    # antenna_signal_example = np.zeros((len(tx_positions),len(rx_positions)),dtype=np.complex128)
    # for i in range(len(tx_positions)):
    #     for j in range(len(rx_positions)):
    #         antenna_signal_example[i,j]=i+1j*j
    # azel_signal_example = np.zeros((12,2),dtype=np.complex128)
    
    rows = [0,1,2,3,4,5,6,7,8,9,10,11]
    # cols = [0,0,0,0,1,1,1,1,0,0,0 ,0]
    cols = [0,0,0,0,0,0,0,0,0,0,0 ,0]
    
    # azel_signal_example[rows, cols] = antenna_signal_example.ravel()

    # empty["RX_az"]=[0, 1, 2, 3]
    # empty["RX_el"]=[0,  0, 0]
    empty["antenna2azelIndex"]=[rows,cols]
    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    
def predefined_array_configs_TI_IWR6843_az(isuite, iradar, location, rotation, f0=70e9):  # 3 x 4
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0),
        (-4, 0),
        (-8, 0)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -6
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0)
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    empty["antenna2azelIndex"]=[[],[]]
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty   

def predefined_array_configs_JSON(isuite, iradar, location, rotation, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            Suitename = f'SuitePlane_{isuite}'
            Suite_obj = bpy.data.objects[Suitename]

            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
            empty = bpy.context.object
            empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
            empty.parent = Suite_obj
            empty = setDefaults(empty,f0)
            s = 0.05
            Type = 'SPOT'
            tx_positions = [
                (0, 0),
                (-4, 0),
                (-8, 0)
            ]

            for i, pos in enumerate(tx_positions):
                bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
                tx = bpy.context.object
                tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
                tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
                tx.parent = empty


            bx0 = -6
            bx = bx0
            by = 0
            s = 1

            rx_positions = [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0)
            ]

            for i, pos in enumerate(rx_positions):
                bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
                rx = bpy.context.object
                rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
                rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
                rx.parent = empty
                rx.data.lens = 10
            empty["antenna2azelIndex"]=[[],[]]
            empty["TXRXPos"]=[tx_positions,rx_positions]
            return empty
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None
def predefined_array_configs_SISO(isuite, iradar, location, rotation, f0=70e9,Pulse1FMCW0=0):
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    if Pulse1FMCW0 == 1 :
        empty['RadarMode']='Pulse'
        empty['PulseWaveform']='WaveformFile.txt'
        
        empty['Fs_MHz']=1500
        # empty['Ts_ns']=1000/empty['Fs_MHz']
        empty['Range_End']=100
        
    
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -1
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    
    rows = [0]
    cols = [0]
    empty["antenna2azelIndex"]=[rows,cols]
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty


def predefined_array_configs_txrx(tx_positions,rx_positions,isuite, iradar, location, rotation, f0=70e9,Pulse1FMCW0=0):
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    if Pulse1FMCW0 == 1 :
        empty['RadarMode']='Pulse'
        empty['PulseWaveform']='WaveformFile.txt'
        
        empty['Fs_MHz']=1500
        # empty['Ts_ns']=1000/empty['Fs_MHz']
        empty['Range_End']=100
        
    
    s = 0.05
    Type = 'SPOT'
    
    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(-pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = 0
    bx = bx0
    by = 0
    s = 1

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    
    rows = [0]
    cols = [0]
    empty["antenna2azelIndex"]=[rows,cols]
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty


def predefined_array_configs_ALTOS_V1(isuite, iradar, location, rotation, f0=70e9):
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)

    s = 0.05
    Type = 'SPOT'
    YSpace = 1
    tx_positions = [
        (0, 0),
        (-4, 0),
        (-8, 0),
        (-12, 0),
        (0, 4*YSpace),
        (-4, 4*YSpace),
        (-8, 4*YSpace),
        (-12, 4*YSpace),
        (0, 8*YSpace),
        (-4, 8*YSpace),
        (-8, 8*YSpace),
        (-12, 8*YSpace)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty

    bx0 = -5
    bx = bx0
    by = -5
    s = 1

    rx_positions = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1*YSpace),
        (1, 1*YSpace),
        (2, 1*YSpace),
        (3, 1*YSpace),
        (0, 2*YSpace),
        (1, 2*YSpace),
        (2, 2*YSpace),
        (3, 2*YSpace),
        (0, 3*YSpace),
        (1, 3*YSpace),
        (2, 3*YSpace),
        (3, 3*YSpace),
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    

    empty["TXRXPos"]=[tx_positions,rx_positions]
    # p2y,p2z=[],[]
    # Mz = 3
    # My = 4
    # Nz = 4
    # Ny = 4
    # for imy in range(My):
    #     for imz in range(Mz):
    #         for iny in range(Ny):
    #             for inz in range(Nz):
    #                 p2z.append(inz+imz*Nz)
    #                 p2y.append(iny+imy*Ny)
    M = len(tx_positions)
    N = len(rx_positions)
    virt = [ (-tx_positions[m][0]+rx_positions[n][0],
            tx_positions[m][1]+rx_positions[n][1])
            for m in range(M) for n in range(N) ]
    xs = sorted({x for x,y in virt})
    ys = sorted({y for x,y in virt})
    empty["antenna2azelIndex"]=[xs,ys]
    return empty
def predefined_array_configs_infineon_BGT60UTR11AIP(isuite, iradar, location, rotation, f0=60e9):  
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    empty['Fs_MHz']=4
    # empty['Ts_ns']=1000/empty['Fs_MHz']
    empty['FMCW_ChirpTime_us'] = 5600/400
    empty['FMCW_Bandwidth_GHz'] = 5.6
    empty['FMCW_ChirpSlobe_MHz_usec'] = 400 #1000*empty['FMCW_Bandwidth_GHz']/empty['FMCW_ChirpTime_us']
    empty['PRI_us']=empty['FMCW_ChirpTime_us']
    
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -6
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    return empty    


def predefined_array_configs_infineon_BGT60TR13C(isuite, iradar, location, rotation, f0=60e9):  
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    empty['Fs_MHz']=4
    # empty['Ts_ns']=1000/empty['Fs_MHz']
    empty['FMCW_ChirpTime_us'] = 5500/400
    empty['FMCW_Bandwidth_GHz'] = 5.5
    empty['FMCW_ChirpSlobe_MHz_usec'] = 400 #1000*empty['FMCW_Bandwidth_GHz']/empty['FMCW_ChirpTime_us']
    empty['PRI_us']=empty['FMCW_ChirpTime_us']
    
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty


    bx0 = -3
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0),
        (1, 1),
        (2, 0)
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    return empty    

def predefined_array_configs_infineon_BGT24LTR11_CW(isuite, iradar, location, rotation, f0=24e9):  
    Lambda = LightSpeed / f0
    Suitename = f'SuitePlane_{isuite}'
    Suite_obj = bpy.data.objects[Suitename]

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=location, rotation=rotation, scale=(1, 1, 1))
    empty = bpy.context.object
    empty.name = f'RadarPlane_{isuite}_{iradar}_{0}'
    empty.parent = Suite_obj
    empty = setDefaults(empty,f0)
    empty['RadarMode']='CW'
    empty['N_ADC']=1
    
    s = 0.05
    Type = 'SPOT'
    tx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(tx_positions):
        bpy.ops.object.light_add(type=Type, radius=1, location=(pos[0]*Lambda/2, pos[1]*Lambda/2, 0))
        tx = bpy.context.object
        tx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        tx.name = f'TX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        tx.parent = empty

    bx0 = -2
    bx = bx0
    by = 0
    s = 1

    rx_positions = [
        (0, 0)
    ]

    for i, pos in enumerate(rx_positions):
        bpy.ops.object.camera_add(location=( -(bx+pos[0])*Lambda/2, (by+pos[1])*Lambda/2, 0), rotation=(0, 0, 0))
        rx = bpy.context.object
        rx.scale = (s*Lambda/2, s*Lambda/2, s*Lambda/2)
        rx.name = f'RX_{isuite}_{iradar}_{1}_{0}_{i+1:05}'
        rx.parent = empty
        rx.data.lens = 10
    empty["TXRXPos"]=[tx_positions,rx_positions]
    return empty    
    
def setDefaults(empty,f0):
    empty["Transmit_Power_dBm"] = 12.0
    empty["Transmit_Antenna_Element_Pattern"] = "Omni"
    empty["Transmit_Antenna_Element_Gain_db"] = 3.0
    empty["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = 120.0
    empty["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = 120.0
    empty["Receive_Antenna_Element_Gain_db"] = 0.0
    empty["Receive_Antenna_Element_Pattern"] = "Omni"
    empty["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = 120.0
    empty["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = 120.0
    empty["Center_Frequency_GHz"] = f0/1e9
    empty['PRI_us']=70.0
    empty['Fs_MHz']=5.0
    # empty['Ts_ns']=1000/empty['Fs_MHz']
    empty['NPulse'] = 3 * 64
    empty['N_ADC']  = 256
    empty['RangeWindow']  = 'Hamming'
    empty['DopplerWindow']  = 'Hamming'
    # empty['N_FFT_ADC']  = 128
    # empty['N_FFT_Doppler']  = 128
    # empty['Lambda_mm']=1000*LightSpeed/empty["Center_Frequency_GHz"]/1e9
    # empty['FMCW_ChirpTime_us'] = 60
    # empty['FMCW_Bandwidth_GHz'] = 1
    empty['Tempreture_K'] = 290.0
    empty['FMCW_ChirpSlobe_MHz_usec'] = 1000.0/60.0    #1000*empty['FMCW_Bandwidth_GHz']/empty['FMCW_ChirpTime_us']
    empty['RangeFFT_OverNextP2'] = 0
    empty['Range_Start']=0
    empty['Range_End']=100
    empty['DopplerProcessingMIMODemod']='Simple'
    empty['CFAR_RD_guard_cells']=2
    empty['CFAR_RD_training_cells']=10
    empty['CFAR_RD_false_alarm_rate']=1e-3
    empty['STC_Enabled']=False #
    empty['MTI_Enabled']=False #
    empty['DopplerFFT_OverNextP2']=0
    empty['AzFFT_OverNextP2']=0
    empty['ElFFT_OverNextP2']=0
    empty['CFAR_Angle_guard_cells']=1
    empty['CFAR_Angle_training_cells']=3
    empty['CFAR_Angle_false_alarm_rate']=.1
    empty['CFAR_RD_alpha']=30.0
    empty['CFAR_Angle_alpha']=5.0
    empty["FMCW"] = True
    empty['ADC_peak2peak']=2.0
    empty['ADC_levels']=256
    empty['ADC_ImpedanceFactor']=300.0
    empty['ADC_LNA_Gain_dB']=50.0
    empty['RF_NoiseFiguredB']=5.0
    empty['RF_AnalogNoiseFilter_Bandwidth_MHz']=10.0
    empty['ADC_SaturationEnabled']=False
    empty['RadarMode']='FMCW'# 'Pulse' 'CW'
    empty['PulseWaveform']='WaveformFile.txt'
    
    empty['t_start_radar']=0
    empty['MaxRangeScatter']=1e12
    empty['SaveSignalGenerationTime']=True
    empty['continuousCPIsTrue_oneCPIpeerFrameFalse']=False
    # empty['t_start_radar']=0
    empty['MIMO_Tech']='TDM'
    empty['ArrayInfofile']=''
    
    
    
    
    # empty['Timing'] = RadarTiming(t_start_radar=0.0, t_start_manual_restart_tx=1.0, t_last_pulse=10.0,
    #                 t_current_pulse=5.0, pri_sequence=[0.1, 0.2, 0.15], n_pulse=7, n_last_cpi=1024)

    
    return empty
    # , levels,,,



def set_radar_displayrangeLim(radar,Rmin,Rmax):
    rangeResolutionmaxUnambigiousRange = rangeResolution_and_maxUnambigiousRange(radar)
    if rangeResolutionmaxUnambigiousRange is not None:
        res, m = rangeResolutionmaxUnambigiousRange
        radar['Range_Start'] = 100*Rmin/m
        radar['Range_End'] = 100*Rmax/m
    
def rangeResolution_and_maxUnambigiousRange(radar):
    if radar['RadarMode']=='FMCW':
        BW = radar['FMCW_ChirpSlobe_MHz_usec']*1e12*radar['N_ADC']*radar['Ts_ns']*1e-9
        Res=3e8/(2*BW)
        MaxR =  radar['N_ADC']*Res
        return Res,MaxR
    return None

def FMCW_Chirp_Parameters(rangeResolution,N_ADC,ChirpTimeMax,radialVelocityResolution,CentralFreq):
    print("Suggestions:")
    B = 299792458.0 / (2*rangeResolution)
    print(f"FMCW Bandwidth (GHz) = {B/1e9:.3f}")
    print(f"with {N_ADC} samples, and {ChirpTimeMax*1e6} us chirp ADC time, ADC Sampleing rate is {N_ADC*1.0/ChirpTimeMax/1e6:.2f} MSps")
    print(f"Chirp Slope = {B*1e-6/(ChirpTimeMax*1e6):.2f} MHz/us")
    print(f"Max Range = {N_ADC*rangeResolution:.2f} m")
    WaveLength = 299792458.0 / CentralFreq
    CPI_Time =  WaveLength / (2*radialVelocityResolution)
    print(f"with radial Velocity Resolution {radialVelocityResolution} m/s, CPI Time = {CPI_Time*1e3:.2f} msec")
    PRI = ChirpTimeMax
    print(f"Pulse Number in CPI < {int(CPI_Time/PRI)} ")

def set_FMCW_Chirp_Parameters(radar,slope,fsps,N_ADC,NPulse,PRI_us):
    radar["FMCW_ChirpSlobe_MHz_usec"]=slope
    radar['Fs_MHz']= fsps
    # radar['Ts_ns']=1000/radar['Fs_MHz']
    radar['N_ADC'] = N_ADC
    radar['NPulse'] = NPulse
    radar['PRI_us']=PRI_us
def compute_spectrogram(x, fs, window_size, hop_size, nfft):
    n_windows = (len(x) - window_size) // hop_size + 1
    spectrogram_matrix = np.zeros((nfft, n_windows), dtype=complex)
    
    for i in range(n_windows):
        windowed_signal = x[i * hop_size : i * hop_size + window_size] * np.hanning(window_size)
        spectrogram_matrix[:, i] = np.fft.fft(windowed_signal, nfft)
    
    time_vector = np.arange(n_windows) * hop_size / fs
    # freq_vector = np.fft.fftfreq(nfft, 1 / fs)
    freq_vector = np.arange(nfft)/nfft*fs
    
    return freq_vector, time_vector, np.abs(spectrogram_matrix)

def plot_spectrogram(freq_vector, time_vector, spectrogram_matrix,doFFTShift=False):
    if doFFTShift:
        plt.pcolormesh(time_vector*1e6, freq_vector/1e9, 10 *np.log10(np.fft.fftshift(spectrogram_matrix,axes=0)), shading='gouraud')
        plt.ylabel(f'Frequency (GHz) + {freq_vector[-1]/1e9/2}')
    else:
        plt.pcolormesh(time_vector*1e6, freq_vector/1e9, 10 *np.log10(spectrogram_matrix), shading='gouraud')
        plt.ylabel('Frequency (GHz)')
    plt.xlabel('Time (us)')
    plt.title('Spectrogram')
    plt.colorbar(label='dB')
    plt.show()

def plot_signal(t, x, N=10000,M=100,ND = 100):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    for i in range(int(len(x)/N) + 1):
        ax.clear()  # Clear the axes
        
        # Plot the real and imaginary components of the signal
        ax.plot(t[i*N:i*N+M]*1e6, np.real(x[i*N:i*N+M]), label='Real Part')
        ax.plot(t[i*N:i*N+M]*1e6, np.imag(x[i*N:i*N+M]), label='Imaginary Part')
        
        # Label the axes and add a legend
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        
        plt.draw()  # Redraw the figure
        plt.pause(0.1)  # Pause briefly to allow GUI events to be processed
        
        # Check if the loop reaches the end of the signal
        if (i+1)*N >= len(x):
            break
    ax.clear()  # Clear the axes
        
    # Plot the real and imaginary components of the signal
    ax.plot(t[::ND]*1e6, np.real(x[::ND]), label='Real Part')
    ax.plot(t[::ND]*1e6, np.imag(x[::ND]), label='Imaginary Part')
    
    # Label the axes and add a legend
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    
    plt.draw()  # Redraw the figure
    plt.pause(0.1)  # Pause briefly to allow GUI events to be processed
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot
    
def compute_spectrum(x, fs):
    nfft = 8 * len(x)  # Zero-padding to improve frequency resolution
    spectrum = np.fft.fft(x, nfft)
    freq_vector = np.arange(len(spectrum))/len(spectrum)*fs
    
    return freq_vector, np.abs(spectrum)

def plot_spectrum(freq_vector, spectrum,dBScale=True):
    if dBScale:
        plt.plot(freq_vector / 1e9, 10 *np.log10(spectrum))  # Convert frequency to GHz for plotting
        plt.ylabel('Spectrum (dB)')
    else:
        plt.plot(freq_vector / 1e9, spectrum)  # Convert frequency to GHz for plotting
        plt.ylabel('Spectrum')
    plt.xlabel('Frequency (GHz)')
    plt.show()

def rangeprocessing(x,specifications):
    fast_time_window = scipy.signal.windows.hamming(x.shape[0])
    fast_time_window = scipy.signal.windows.chebwin(x.shape[0],at=100)
#     hann_window = windows.hann(N)
# blackman_window = windows.blackman(N)
# kaiser_window = windows.kaiser(N, beta=14)  # β = 14 gives ~-60 dB side lobes
# chebyshev_window = windows.chebwin(N, at=100)
    X_windowed_fast = x * fast_time_window[:, np.newaxis, np.newaxis]
    if specifications['RadarMode'] == 'Pulse':
        waveform_MF = specifications['PulseWaveform_Loaded']
        matched_filter = np.conj(waveform_MF[::-1])
        Xr = np.apply_along_axis(lambda x: np.convolve(x, matched_filter, mode='full'), axis=0, arr=X_windowed_fast)
        Xr=Xr[matched_filter.shape[0]-1:,:,:]
        d_fft = np.arange(Xr.shape[0]) * LightSpeed * specifications['Ts'] /2
    else:
        NFFT_Range_OverNextPow2 =  specifications['RangeFFT_OverNextP2']
        NFFT_Range = int(2 ** (np.ceil(np.log2(x.shape[0]))+NFFT_Range_OverNextPow2))
        Xr = np.fft.fft(X_windowed_fast, axis=0, n=NFFT_Range)  # beat freq = Slobe * 2 * d / c =   ind / nfft * Fs ->
        d_fft = np.arange(NFFT_Range) * LightSpeed / 2 / specifications['FMCW_ChirpSlobe'] / NFFT_Range / specifications['Ts']
    Range_Start = specifications['Range_Start']
    Range_End = specifications['Range_End']
    d1i = int(Xr.shape[0]*Range_Start/100.0)
    d2i = int(Xr.shape[0]*Range_End/100.0)
    d_fft = d_fft[d1i:d2i]
    Xr = Xr[d1i:d2i,:,:] #(67 Range, 96 Pulse, 4 RX)
    
    return Xr,d_fft

def dopplerprocessing_mimodemodulation(x,specifications):
    M_TX=specifications['M_TX']
    L = x.shape[1]
    Leff = int(L/M_TX)
    NFFT_Doppler_OverNextPow2 = specifications['DopplerFFT_OverNextP2']
    MTI_Apply = specifications['MTI_Enabled']
    if specifications['DopplerProcessingMIMODemod']!='Simple':
        N_Doppler = Leff
        N_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
        f_Doppler = np.arange(0,N_Doppler)/N_Doppler/specifications['PRI']/M_TX - 1/specifications['PRI']/M_TX / 2
        PRF_TDM = 1.0/specifications['PRI'] / M_TX
        f_Doppler = np.arange(0, N_Doppler) / N_Doppler * PRF_TDM - PRF_TDM/2

        PrecodingMatrixInv = np.linalg.pinv(specifications['PrecodingMatrix'])
        rangeDopplerTXRX = np.zeros((x.shape[0], f_Doppler.shape[0], M_TX, x.shape[2]),dtype=complex)
        # slow_time_window = scipy.signal.windows.boxcar(x.shape[1])
        slow_time_window = scipy.signal.windows.hamming(x.shape[1])
        x_windowed = x * slow_time_window[ np.newaxis , : , np.newaxis]
        
        for idop , f_Doppler_i in enumerate(f_Doppler):
            dopplerSteeringVector = np.exp(1j*2*np.pi*f_Doppler_i*np.arange(L)*specifications['PRI'])
            X_doppler_comp = x_windowed * np.conj(dopplerSteeringVector[np.newaxis,:,np.newaxis])
            rangeTXRX = np.einsum('ijk,lj->ilk', X_doppler_comp, PrecodingMatrixInv)
            rangeDopplerTXRX[:, idop, :, :] = rangeTXRX
            # if MIMO_phase_compensation_Enable:
            #     TDM_MIMO_phase_compensation = np.ones((S_IF_krange_ldoppler_pva.shape[1], S_IF_krange_ldoppler_pva.shape[2]),dtype=S_IF_krange_ldoppler_pva.dtype)
            #     T = 1/PRF_TDM / MTX
            #     for ifd in range(len(fd)):
            #         for m in range(MTX):
            #             n = np.arange(NRX)
            #             fdi = -fd[ifd]
            #             TDM_MIMO_phase_compensation[ ifd , m * NRX + n] = np.exp(1j * 2 * np.pi * fdi * m * T)
            #     S_IF_krange_ldoppler_pva *= TDM_MIMO_phase_compensation[np.newaxis, : , : ]

    else:
        rangePulseTXRX = np.zeros((x.shape[0], Leff, M_TX, x.shape[2]),dtype=complex)
        for ipulse in range(Leff):
            ind = ipulse*M_TX
            rangePulseTXRX[:,ipulse,:,:]=x[:,ind:ind+M_TX,:]
        if MTI_Apply:
            rangePulseTXRX = np.diff(rangePulseTXRX,axis=1)
        slow_time_window = scipy.signal.windows.hamming(rangePulseTXRX.shape[1])
        rangePulseTXRX_windowed = rangePulseTXRX * slow_time_window[ np.newaxis , : , np.newaxis, np.newaxis]
        NFFT_Doppler = int(2 ** (np.ceil(np.log2(Leff))+NFFT_Doppler_OverNextPow2))
        rangeDopplerTXRX = np.fft.fft(rangePulseTXRX_windowed, axis=1, n=NFFT_Doppler)
        rangeDopplerTXRX = np.fft.fftshift(rangeDopplerTXRX,axes=1)
        f_Doppler = np.arange(0,NFFT_Doppler)/NFFT_Doppler/specifications['PRI']/M_TX - 1/specifications['PRI']/M_TX / 2
        
    return rangeDopplerTXRX,f_Doppler

def rangedoppler_detection(x):
    rangeDoppler4CFAR = np.mean(np.abs(x),axis=(2,3))
    num_train = 50  # Number of training cells
    num_guard = 4   # Number of guard cells
    prob_fa = 1e-3  # Desired probability of false alarm
    # alpha = specifications['CFAR_Angle_alpha']
    # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D(1.0*rangeDoppler4CFAR, num_train, num_guard, prob_fa)
    CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
    if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled and 0:
        detections,cfar_threshold = cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
        # detections,cfar_threshold = cfar_simple_max(1.0*rangeDoppler4CFAR)
    else:
        # detections,cfar_threshold = cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
        detections,cfar_threshold = cfar_simple_2D(1.0*rangeDoppler4CFAR, 30)
    
    return detections,cfar_threshold,rangeDoppler4CFAR
def rangedoppler_detection_alpha(x,specifications):
    rangeDoppler4CFAR = np.mean(np.abs(x),axis=(2,3))
    num_train = 50  # Number of training cells
    num_guard = 4   # Number of guard cells
    prob_fa = 1e-3  # Desired probability of false alarm
    alpha = specifications['CFAR_RD_alpha']
    # detections,cfar_threshold = ssp.radar.utils.cfar_ca_2D(1.0*rangeDoppler4CFAR, num_train, num_guard, prob_fa)
    CUDA_signalGeneration_Enabled = bpy.data.objects["Simulation Settings"]["CUDA SignalGeneration Enabled"]
    if ssp.config.CUDA_is_available and CUDA_signalGeneration_Enabled and 0:
        detections,cfar_threshold = cfar_ca_2D_alpha_cuda(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
        # detections,cfar_threshold = cfar_simple_max(1.0*rangeDoppler4CFAR)
    else:
        # detections,cfar_threshold = cfar_ca_2D_alpha(1.0*rangeDoppler4CFAR, [25,15], [10,10], 5)
        detections,cfar_threshold = cfar_simple_2D(1.0*rangeDoppler4CFAR, alpha)
    
    return detections,cfar_threshold,rangeDoppler4CFAR

def angleprocessing_CoArray(x,detections,d_fft,specifications,FigsAxes):
    ULA_TXRX_Lx_Ly_NonZ_AllRX=specifications['ULA_TXRX_Lx_Ly_NonZ']
    xnew = np.zeros((x.shape[0],x.shape[1],ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=x.dtype)    
    for new_ix in range(xnew.shape[0]):
        for new_iy in range(xnew.shape[1]):
            itxirxs = ULA_TXRX_Lx_Ly_NonZ_AllRX[0][new_ix][new_iy]
            for itxirx in itxirxs:
                xnew[:,:,new_ix,new_iy] += x[:,:,itxirx[0],itxirx[1]]
    virtualarray = xnew[:,:,ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0],0]
    # test=virtualarray[0,0,:3]-xnew[0,0,:3,0]
    coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D_RangeDopplerVA(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0], virtualarray)
    
    angle_window = scipy.signal.windows.hamming(coArraySignal.shape[2])
    coArraySignal_windowed = coArraySignal #* angle_window[ np.newaxis  , np.newaxis, :]
        
    
    AzFFT_OverNextP2 = specifications['AzFFT_OverNextP2']
    NAng = int(2 ** (np.ceil(np.log2(coArraySignal.shape[2]))+AzFFT_OverNextP2))
    o = np.abs(np.fft.fftshift(np.fft.fft(coArraySignal_windowed,axis=2,n=NAng),axes=2))
    o = np.mean(np.abs(o),axis=(1))
    FigsAxes[0,0].cla()
    FigsAxes[0,0].imshow(10 * np.log10(o), extent=[-1,1, d_fft[0], d_fft[-1]],aspect='auto', cmap='viridis', origin='lower')
    FigsAxes[1,0].cla()
    FigsAxes[1,0].imshow(o, extent=[-1,1, d_fft[0], d_fft[-1]],aspect='auto', cmap='viridis', origin='lower')
    
    FigsAxes[1,1].cla()
    max_index = np.unravel_index(np.argmax(o), o.shape)
    FigsAxes[1,1].plot(np.linspace(-1,1,o.shape[1]),10 * np.log10(o[max_index[0],:]))
    plt.show()
    
    
def angleprocessing(x,detections,specifications,FigsAxes=None):
    if FigsAxes is not None:
        FigsAxes[0].cla()
    detected_points = np.where(detections == 1)
    NDetection = detected_points[0].shape[0]
    ULA_TXRX_Lx_Ly_NonZ_AllRX=specifications['ULA_TXRX_Lx_Ly_NonZ']
    
    AzFFT_OverNextP2 = specifications['AzFFT_OverNextP2']
    
    Naz = ULA_TXRX_Lx_Ly_NonZ_AllRX[2]
    NAng = int(2 ** (np.ceil(np.log2(Naz))+AzFFT_OverNextP2))
    # fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
    for id in range(NDetection):
        antennaSignal = x[detected_points[0][id],detected_points[1][id],:,:]
        if 1:
            guard = 2
            sdi_range = np.arange(x.shape[1])
            sdi_filtered = sdi_range[np.abs(sdi_range - detected_points[1][id]) > guard]
            secondary_data = x[detected_points[0][id], sdi_filtered, :, :]
            LinearSpace_virtualarray = np.zeros((ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
            LinearSpace_virtualarray_SD = np.zeros((secondary_data.shape[0],ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
            for new_ix in range(LinearSpace_virtualarray.shape[0]):
                for new_iy in range(LinearSpace_virtualarray.shape[1]):
                    itxirxs = ULA_TXRX_Lx_Ly_NonZ_AllRX[0][new_ix][new_iy]
                    for itxirx in itxirxs:
                        LinearSpace_virtualarray[new_ix,new_iy] += antennaSignal[itxirx[0],itxirx[1]]
                        LinearSpace_virtualarray_SD[:,new_ix,new_iy] += secondary_data[:,itxirx[0],itxirx[1]]
                    if len(itxirxs)>0:
                        LinearSpace_virtualarray[new_ix,new_iy] /= len(itxirxs)  
                        LinearSpace_virtualarray_SD[:,new_ix,new_iy] /= len(itxirxs)
            virtualarray = LinearSpace_virtualarray[ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
            virtualarray_SD = LinearSpace_virtualarray_SD[:,ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
            # coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0], virtualarray)
            XSD = virtualarray_SD.squeeze(axis=2)
            XSD = XSD.T
            Sig = XSD @ np.conj(XSD.T)
            Siginv = np.linalg.pinv(Sig+.01*np.linalg.norm(Sig)*np.eye(Sig.shape[0]))
            az = np.linspace(-90,90,NAng)
            o=np.zeros_like(az)
            for iaz in range(az.shape[0]):
                s = steeringvector(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0],az[iaz])
                w = Siginv@s / (np.conj(s.T)@Siginv@s)[0,0]
                o[iaz]=np.abs(np.conj(w.T)@virtualarray)
            # ax1.plot(az,o)
            if FigsAxes is not None:
                FigsAxes[0].plot(az,o)
        if 1:
            LinearSpace_virtualarray = np.zeros((ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
            for new_ix in range(LinearSpace_virtualarray.shape[0]):
                for new_iy in range(LinearSpace_virtualarray.shape[1]):
                    itxirxs = ULA_TXRX_Lx_Ly_NonZ_AllRX[0][new_ix][new_iy]
                    for itxirx in itxirxs:
                        LinearSpace_virtualarray[new_ix,new_iy] += antennaSignal[itxirx[0],itxirx[1]]
                    if len(itxirxs)>0:
                        LinearSpace_virtualarray[new_ix,new_iy] /= len(itxirxs)  
            virtualarray = LinearSpace_virtualarray[ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
            # coArraySignal,coArrayStructure = ssp.radar.utils.VAsignal2coArraySignal_1D(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0], virtualarray)
            if 1:
                o = np.abs(np.fft.fftshift(np.fft.fft(virtualarray,axis=0,n=NAng)))
                # ax2.plot(o)
                # FigsAxes[0].plot(az,o)
            if 1:
                o = np.abs(np.fft.fftshift(np.fft.fft(LinearSpace_virtualarray,axis=0,n=NAng)))
                # ax3.plot(o)
                
                # FigsAxes[0].plot(az,o)
            
    # plt.show()
def pointscloud(ranges,dopplers,angles_v):
    o = []
    for i,r in enumerate(ranges):
        d = dopplers[i]
        for az in angles_v[i]:
            o.append([r*np.cos(np.deg2rad(az)),r*np.sin(np.deg2rad(az)),d])
    return np.array(o)
    
def angleprocessing_capon1D(x,detections,specifications,FigsAxes=None):
    detected_points = np.where(detections == 1)
    NDetection = detected_points[0].shape[0]
    ULA_TXRX_Lx_Ly_NonZ_AllRX=specifications['ULA_TXRX_Lx_Ly_NonZ']
    
    AzFFT_OverNextP2 = specifications['AzFFT_OverNextP2']
    
    alpha = specifications['CFAR_Angle_alpha']
    
    Naz = ULA_TXRX_Lx_Ly_NonZ_AllRX[2]
    NAng = int(2 ** (np.ceil(np.log2(Naz))+AzFFT_OverNextP2))
    # fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
    # plt.figure(figsize=(10, 7))
    Azimuth_Angles=[]
    debug_spectrums = []
    for id in range(NDetection):
        antennaSignal = x[detected_points[0][id],detected_points[1][id],:,:]
        guard = 2
        sdi_range = np.arange(x.shape[1])
        sdi_filtered = sdi_range[np.abs(sdi_range - detected_points[1][id]) > guard]
        secondary_data = x[detected_points[0][id], sdi_filtered, :, :]
        LinearSpace_virtualarray = np.zeros((ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
        LinearSpace_virtualarray_SD = np.zeros((secondary_data.shape[0],ULA_TXRX_Lx_Ly_NonZ_AllRX[2],ULA_TXRX_Lx_Ly_NonZ_AllRX[3]),dtype=antennaSignal.dtype)
        for new_ix in range(LinearSpace_virtualarray.shape[0]):
            for new_iy in range(LinearSpace_virtualarray.shape[1]):
                itxirxs = ULA_TXRX_Lx_Ly_NonZ_AllRX[0][new_ix][new_iy]
                for itxirx in itxirxs:
                    LinearSpace_virtualarray[new_ix,new_iy] += antennaSignal[itxirx[0],itxirx[1]]
                    LinearSpace_virtualarray_SD[:,new_ix,new_iy] += secondary_data[:,itxirx[0],itxirx[1]]
                if len(itxirxs)>0:
                    LinearSpace_virtualarray[new_ix,new_iy] /= len(itxirxs)  
                    LinearSpace_virtualarray_SD[:,new_ix,new_iy] /= len(itxirxs)
        virtualarray = LinearSpace_virtualarray[ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
        virtualarray_SD = LinearSpace_virtualarray_SD[:,ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0]]
        XSD = virtualarray_SD.squeeze(axis=2)
        XSD = XSD.T
        Sig = XSD @ np.conj(XSD.T)
        Siginv = np.linalg.pinv(Sig+.01*np.linalg.norm(Sig)*np.eye(Sig.shape[0]))
        az = np.linspace(-90,90,NAng)
        o=np.zeros_like(az)
        for iaz in range(az.shape[0]):
            s = steeringvector(ULA_TXRX_Lx_Ly_NonZ_AllRX[4][0],az[iaz])
            w = Siginv@s / (np.conj(s.T)@Siginv@s)[0,0]
            o[iaz]=np.abs(np.conj(w.T)@virtualarray)
        # ax1.plot(az,o)
        debug_spectrums.append(o)
        THR_angle = np.mean(o)*alpha*np.ones_like(o)
        detections_angle = np.where(o > THR_angle, 1, 0)
        detected_points_angle = np.where(detections_angle == 1)
        # plt.plot(az,o)
        # plt.plot(az[detected_points_angle],o[detected_points_angle],'or')
        Azimuth_Angles.append(az[detected_points_angle])
    return Azimuth_Angles,debug_spectrums
def steeringvector(va,az):
    s=np.exp(1j*np.pi*va*np.sin(np.deg2rad(az))).reshape(va.shape[0],1)
    return s
     

def addTarget(refRadar=None, range=10, azimuth=0, elevation=0, RCS0=1, size=1.0, radial_velocity=0,shape='cube'):
    """
    Adds a target cube in a Blender scene with specified parameters.
    The cube's location and orientation are adjusted based on azimuth, elevation, and reference radar.

    Parameters:
        refRadar: The reference radar object for alignment (optional).
        range: Distance of the target from the radar (default 10).
        azimuth: Azimuth angle in degrees (default 0).
        elevation: Elevation angle in degrees (default 0).
        RCS0: Radar cross-section value (default 1).
        size: Size of the cube (default 1.0).

    Returns:
        cube: The cube object representing the target.
        empty: The empty parent object used for positioning and orientation.
    """
    # Create the cube representing the target
    # size /=2.0
    if shape=='cube':
        bpy.ops.mesh.primitive_cube_add(
            size=size,
            align='WORLD',
            location=(range + 0.5 * size, 0, 0),  # Temporary location
            scale=(1, 1, 1)
        )
    elif shape=='sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=size/2,
            align='WORLD',
            location=(range + 0.5 * size, 0, 0),  # Temporary location
            scale=(1, 1, 1)
        )
    elif shape=='plane':
        bpy.ops.mesh.primitive_plane_add(
            size=size,
            align='WORLD',
            location=(range, 0, 0),  
            scale=(1, 1, 1),
            rotation=(0,np.pi/2,0)
        )
    else:
        raise ValueError('shape should be cube or sphere')
    cube = bpy.context.object
    cube["RCS0"] = RCS0  # Store RCS value as a custom property
    cube["Backscatter N"] = 1  # Store RCS value as a custom property
    cube["Backscatter Dev (deg)"] = 0.0  # Store RCS value as a custom property
    cube["SpecularDiffusionFactor"] = 1.0  # Store RCS value as a custom property
    if np.abs(radial_velocity)> 0:
        T = 1.0/bpy.context.scene.render.fps
        start_frame = 1
        end_frame = 2
        cube.keyframe_insert(data_path="location", frame=start_frame)
        dx = radial_velocity * T
        cube.location.x = range + dx + 0.5 * size
        cube.keyframe_insert(data_path="location", frame=end_frame)

    
    # Create an empty object to act as the parent of the cube
    bpy.ops.object.empty_add(
        type='PLAIN_AXES',
        align='WORLD',
        location=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=(0.01, 0.01, 0.01)
    )
    empty = bpy.context.object
    cube.parent = empty  # Parent the cube to the empty

    # Set up azimuth and elevation angles in radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    if refRadar is not None:
        # Decompose the reference radar's world matrix
        global_location, global_rotation, _ = refRadar.matrix_world.decompose()

        # Set the empty object's location to match the radar's location
        empty.location = global_location

        # Compute the direction vector from azimuth and elevation
        dir = ssp.raytracing.dir_from_azel_matrix(azimuth_rad, elevation_rad, global_rotation)

        # Convert the direction vector to azimuth and elevation in spherical coordinates
        _, az_new, el_new = ssp.utils.cart2sph(dir.x, dir.y, dir.z)

        empty.rotation_euler = (0, -el_new, az_new)

    return cube, empty
def printAntennaInfp(radar):
    print(radar['Transmit_Power_dBm'])#=30
    print(radar["Transmit_Antenna_Element_Pattern"])# = "NotOmni"
    print(radar["Transmit_Antenna_Element_Gain_db"])# = 32
    print(radar["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"])# = 2.5
    print(radar["Transmit_Antenna_Element_Elevation_BeamWidth_deg"])# = 2.6
    print(radar["Receive_Antenna_Element_Gain_db"])# = 32
    print(radar["Receive_Antenna_Element_Pattern"])# = "NotOmni"
    print(radar["Receive_Antenna_Element_Azimuth_BeamWidth_deg"])# = 2.5
    print(radar["Receive_Antenna_Element_Elevation_BeamWidth_deg"])# = 2.6
def addRadar(radarSensor=RadarSensorsCategory.TI_AWR1642,location_xyz=[0,0,0]):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    suiteIndex=len(suite_planes)-1
    if len(suite_planes)==0:
        suiteIndex=0
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = suite_planes[-1]
    radar_planes = ssp.environment.BlenderSuiteFinder().find_radar_planes(obj)
    radarIndex = max([int(plane.name.split('_')[2]) for plane in radar_planes if plane.parent == obj] or [-1]) + 1
    if radarSensor==RadarSensorsCategory.TI_Cascade_AWR2243:
        radar = ssp.radar.utils.predefined_array_configs_TI_Cascade_AWR2243(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9) 
        return radar
    if radarSensor==RadarSensorsCategory.Xhetru_X4:
        radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=8e9)
        radarRange = 1.2
        fast_time_bins = 189
        rangebins = radarRange / fast_time_bins
        gesture_motion_slowTimeFrames = 90
        recordedTimePerSample = 4.5   
        radar['RadarMode']='Pulse'
        radar['PulseWaveform']='UWB'
        radar['N_ADC']  = fast_time_bins
        radar['Fs_MHz'] =  ssp.LightSpeed / (2*rangebins) / 1e6
        radar['RF_AnalogNoiseFilter_Bandwidth_MHz'] =  1500
        radar['PRI_us']= 1e6*recordedTimePerSample/gesture_motion_slowTimeFrames
        radar['NPulse'] = gesture_motion_slowTimeFrames
        radar['Range_End']=100 
        return radar
    
    if radarSensor==RadarSensorsCategory.TI_IWR6843:
        radar = ssp.radar.utils.predefined_array_configs_TI_IWR6843(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9) 
        return radar
    # if radarSensor==RadarSensorsCategory.:
    #     radar = ssp.radar.utils.predefined_array_configs_infineon_BGT24LTR11_CW(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[1])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9) 
    #     return radar
    if radarSensor==RadarSensorsCategory.TI_AWR1642:
        radar = ssp.radar.utils.predefined_array_configs_TI_AWR1642(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9)
        rangeResolution,radialVelocityResolution,N_ADC,ChirpTimeMax,CentralFreq=0.039,0.13,256,60e-6,76e9
        ssp.radar.utils.FMCW_Chirp_Parameters(rangeResolution,N_ADC,ChirpTimeMax,radialVelocityResolution,CentralFreq)

        ssp.radar.utils.set_FMCW_Chirp_Parameters(radar,slope=64.06,fsps=4.2,N_ADC=256,NPulse=256,PRI_us=60)
        radar['RangeFFT_OverNextP2'] = 0
        radar['Range_End']=100*.6/9.98 
        radar['DopplerFFT_OverNextP2']=0
        radar['CFAR_RD_alpha']=15
        radar['CFAR_Angle_alpha']=2
        radar['Transmit_Antenna_Element_Pattern']='Directional-Sinc' # add rect pattern
        radar["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = 60
        radar["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = 60
        return radar
    if radarSensor==RadarSensorsCategory.ULA_AllRX:
        N = ssp.config.AddRadar_ULA_N
        d = ssp.config.AddRadar_ULA_d
        radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9,LinearArray_TXPos=[0],LinearArray_RXPos=[-i*3e8/76e9*d for i in range(N*N)])
        return radar
    if radarSensor==RadarSensorsCategory.ULA_AllRX_N:
        N = ssp.config.AddRadar_ULA_N
        d = ssp.config.AddRadar_ULA_d
        radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9,LinearArray_TXPos=[0],LinearArray_RXPos=[-i*3e8/76e9*d for i in range(N)])
        return radar
    if radarSensor==RadarSensorsCategory.ULA_SameTXRX:
        N = ssp.config.AddRadar_ULA_N
        d = ssp.config.AddRadar_ULA_d
        radar = ssp.radar.utils.predefined_array_configs_LinearArray(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9,LinearArray_TXPos=[-N*(i-0)*3e8/76e9*d for i in range(N)],LinearArray_RXPos=[-i*3e8/76e9*d for i in range(N)])
        return radar
    if radarSensor==RadarSensorsCategory.SISO_mmWave76GHz:
        radar = ssp.radar.utils.predefined_array_configs_SISO(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=76e9) 
        return radar
    if radarSensor==RadarSensorsCategory.URA_AllRX:
        N,M = ssp.config.AddRadar_URA_NxNy
        f0=76e9
        radar = ssp.radar.utils.predefined_array_configs_URA(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)),NTX=[1,1],NRX=[N,M],dazimuth=.5,delevation=.5,f0=f0)
        return radar
    if radarSensor==RadarSensorsCategory.URA_SameTXRX:
        N,M = ssp.config.AddRadar_URA_NxNy
        f0=76e9
        radar = ssp.radar.utils.predefined_array_configs_URA(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)),NTX=[N,M],NRX=[N,M],dazimuth=.5,delevation=.5,f0=f0)
        return radar
    if radarSensor==RadarSensorsCategory.URA_LinearHtxVrx:
        N,M = ssp.config.AddRadar_URA_NxNy
        f0=76e9
        radar = ssp.radar.utils.predefined_array_configs_URA(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)),NTX=[N,1],NRX=[1,M],dazimuth=.5,delevation=.5,f0=f0)
        return radar
    if radarSensor==RadarSensorsCategory.URA_AllTX:
        N,M = ssp.config.AddRadar_URA_NxNy
        f0=76e9
        radar = ssp.radar.utils.predefined_array_configs_URA(isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)),NTX=[N,M],NRX=[1,1],dazimuth=.5,delevation=.5,f0=f0)
        return radar
    return None

def addRadarFile(filepath,f0=77e9,location_xyz=[0,0,0]):
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    suiteIndex=len(suite_planes)-1
    if len(suite_planes)==0:
        suiteIndex=0
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = suite_planes[-1]
    radar_planes = ssp.environment.BlenderSuiteFinder().find_radar_planes(obj)
    radarIndex = max([int(plane.name.split('_')[2]) for plane in radar_planes if plane.parent == obj] or [-1]) + 1

    data = scipy.io.loadmat(filepath)
    tx_positions   = data.get('tx_positions', None)
    rx_positions   = data.get('rx_positions', None)
    rx_bias =  data.get('rx_bias', None)[0]
    AzELscale =  data.get('AzELscale', None)[0]
    
    for i in range(len(rx_positions)):
        rx_positions[i] = (AzELscale[0]*rx_positions[i][0]+rx_bias[0], AzELscale[1]*rx_positions[i][1]+rx_bias[1])
    for i in range(len(tx_positions)):
        tx_positions[i] = (AzELscale[0]*tx_positions[i][0], AzELscale[1]*tx_positions[i][1])
    
    vaorder        = data.get('vaorder', None)
    vaorder2        = data.get('vaorder2', None)
    PrecodingMatrix = data.get('PrecodingMatrix', None)
    scale          = str(data.get('scale', None)[0])
    vaprocessing   = str(data.get('vaprocessing', None)[0])
    id = str(data.get('id', None)[0])
    radar = ssp.radar.utils.predefined_array_configs_txrx(tx_positions=tx_positions,rx_positions=rx_positions,isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=f0) 
    radar['ArrayInfofile']=filepath
    return radar
