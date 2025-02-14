import threading
import serial
import time
import struct
import numpy as np
# import matplotlib.pyplot as plt
# with open("rf_data_binary_data.bin", 'rb') as f:
#     binary_data = f.read()
# Nbinary_data = len(binary_data)
# START_SEQUENCE = 0x7C7C7C7C
# RESERVED_LENGTH = 4
# offset = 0
# start_sequence = int.from_bytes(binary_data[offset:offset+4], byteorder='little')
# check1 = start_sequence != START_SEQUENCE
# packet_length = struct.unpack('<I', binary_data[4:8])[0]
# n_samples = struct.unpack('<I', binary_data[19:23])[0] // 2  # Divide by 2 for I and Q pairs

# n_samples=700

# st = 1 + 22 + 4*2
# iq_data = np.frombuffer(binary_data[st:st + n_samples * 4 ], dtype='<f4')
    
# plt.plot(iq_data)
# plt.show()
# xxxxxxxxxxxx


def parse_rf_data(binary_data):
    START_SEQUENCE = 0x7C7C7C7C
    RESERVED_LENGTH = 4
    offset = 0
    start_sequence = int.from_bytes(binary_data[offset:offset+4], byteorder='little')
    if start_sequence != START_SEQUENCE:
        return []
    
    # packet_length = struct.unpack('<I', binary_data[4:8])[0]
    # n_samples = struct.unpack('<I', binary_data[19:23])[0] // 2  # Divide by 2 for I and Q pairs

    n_samples=700

    st = 1 + 22 + 4*2

    if len(binary_data) < n_samples * 4 + st:
        return []
    rf = np.frombuffer(binary_data[st:st + n_samples * 4 ], dtype='<f4')
    return rf
    # with open("rf_data_binary_data.bin", 'wb') as f:
    #     f.write(binary_data)
    offset += 4

    # Read Packet Length
    packet_length = int.from_bytes(binary_data[offset:offset+4], byteorder='little')
    offset += 4

    # Skip Reserved field
    offset += RESERVED_LENGTH

    # Read Data Type Identifiers
    data_type = binary_data[offset]
    offset += 1

    sub_type = binary_data[offset]
    offset += 1

    # Ensure the data type matches RF Mode
    if data_type != 0xA0 or sub_type != 0x12:
        return []

    # Read ContentId
    content_id = int.from_bytes(binary_data[offset:offset+4], byteorder='little')
    offset += 4

    # Read Info (e.g., frame counter)
    info = int.from_bytes(binary_data[offset:offset+4], byteorder='little')
    offset += 4

    # Read Length (Number of float values)
    length = int.from_bytes(binary_data[offset:offset+4], byteorder='little')
    offset += 4

    # Read DataItems (Array of float values)
    data_items = np.frombuffer(binary_data[offset:offset+(length*4)], dtype=np.float32)

    return {
        'ContentId': content_id,
        'Info': info,
        'Length': length,
        'DataItems': data_items
    }


def parse_baseband_iq(data):
    # Check the start sequence
    if data[0:4] != bytes.fromhex("7c7c7c7c"):
        return []
    
    # Extract packet length (4 bytes after the start sequence)
    packet_length = struct.unpack('<I', data[4:8])[0]
    
    # Number of samples (extracted from metadata at offset 19:23)
    n_samples = struct.unpack('<I', data[19:23])[0] // 2  # Divide by 2 for I and Q pairs
    
    # Check if the data length matches the expected size
    if len(data) < n_samples * 4 * 2 + 23:  # 4 bytes per float, 2 for I/Q, +23 for header
        return []
    
    # Parse IQ data from the appropriate offset
    iq_data = np.frombuffer(data[23:23 + n_samples * 4 * 2], dtype='<f4')
    I = iq_data[:n_samples]  # First n_samples are I components
    Q = iq_data[n_samples:]  # Last n_samples are Q components
    IQ = I + 1j * Q  # Combine I and Q to form complex numbers
    
    return IQ

def set_detection_zone_command(start_range, end_range):
    """
    Generates a command to set the start and end range for the XeThru radar module.

    :param start_range: Start of the detection range (in meters, float).
    :param end_range: End of the detection range (in meters, float).
    :return: Byte array representing the command to set the detection zone.
    """
    # Protocol constants
    START_BYTE = 0x7D
    COMMAND_TYPE = 0x50  # XTS_SPC_X4Driver
    SET_SUBCOMMAND = 0x10  # XTS_SPCA_SET
    PARAM_ID_FRAME_AREA = 0x14  # XTS_SPCXI_FRAMEAREA
    END_BYTE = 0x7E

    # Convert the start and end ranges to IEEE 754 floating-point format (4 bytes each)
    start_range_bytes = struct.pack('<f', start_range)  # Little-endian float
    end_range_bytes = struct.pack('<f', end_range)      # Little-endian float

    # Construct the command body
    command_body = (
        bytes([COMMAND_TYPE, SET_SUBCOMMAND, PARAM_ID_FRAME_AREA]) +
        start_range_bytes +
        end_range_bytes
    )

    # Calculate the checksum (XOR of all bytes in the command body)
    checksum = START_BYTE  # Start with the START_BYTE in checksum calculation
    for byte in command_body:
        checksum ^= byte
    checksum &= 0xFF  # Ensure checksum is a single byte

    # Construct the final command
    command = (
        bytes([START_BYTE]) +
        command_body +
        bytes([checksum]) +
        bytes([END_BYTE])
    )
    
    return bytes.fromhex(command.hex())


class XeThruDevice:
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.MaxBufferSize = 1200
        self.serial = None
        # Threading
        self.data_thread = None
        self.stop_event = threading.Event()

        # Optionally store incoming data or parse it on the fly
        self.data_buffer = bytearray()
        self.decoded = []
        self.connected = False
        
        self.cmd_ping = bytes.fromhex("7d01eeaaeaae7c7e")
        self.cmd_stop_profile_exec = bytes.fromhex("7d20134e7e")
        self.cmd_set_manual_mode = bytes.fromhex("7d20124f7e")
        self.cmd_set_downconversion_0 = bytes.fromhex("7d501013000000002e7e")
        self.cmd_set_downconversion_1 = bytes.fromhex("7d501013000000012f7e")
        self.cmd_set_fps_to_0 = bytes.fromhex("7d50101000000000000000002d7e")
        self.cmd_set_fps_to_10 = bytes.fromhex("7d501010000000000020414c7e")
        self.cmd_set_fps_to_50 = bytes.fromhex("7d50101000000000004842277e")
        self.cmd_set_fps_to_90 = bytes.fromhex("7d5010100000000000b442db7e")
        # self.cmd_set_fps_to_120 = bytes.fromhex("7d5010100000000042f000007e")
        # self.cmd_set_fps_to_150 = bytes.fromhex("7d50101000000000431600007e")
        # self.cmd_set_fps_to_180 = bytes.fromhex("7d50101000000000433400007e")
        # self.cmd_set_fps_to_200 = bytes.fromhex("7d50101000000000434800007e")
        # self.cmd_set_fps_to_300 = bytes.fromhex("7d50101000000000439600007e")
        # self.cmd_set_fps_to_500 = bytes.fromhex("7d5010100000000043fa00007e")
        self.cmd_set_range_0to9 = bytes.fromhex("7d5010140000000000000000001041787e")
        self.cmd_set_range_2to11 = bytes.fromhex("7d5010140000000000004000003041187e")
        self.cmd_set_range_5to14 = bytes.fromhex("7d5010140000000000a04000006041e87e")
        self.cmd_set_range_8to17 = bytes.fromhex("7d5010140000000000004100008841a17e")
        self.cmd_set_range_21to30 = bytes.fromhex("7d5010140000000000a8410000f041717e")
        self.cmd_set_range_0to1_2 = bytes.fromhex("7d501014000000009a99993f8c7e")
          # New command

        self.flag_seq_start = bytes.fromhex("7c7c7c7c")
        self.sensor_start = bytes.fromhex("7d20015c7e")  # Start radar
        self.sensor_stop = bytes.fromhex("7d20134e7e")  # Stop radar


        

        # Command cycle and associated x-axis ranges
        self.range_commands = [
            (self.cmd_set_range_0to9, (0, 9)),
            (self.cmd_set_range_2to11, (2, 11)),
            (self.cmd_set_range_5to14, (5, 14)),
            (self.cmd_set_range_8to17, (8, 17)),
            (self.cmd_set_range_21to30, (21, 30)),
        ]
        
    def connect(self):
        """
        Opens serial connections to the mmWave device and starts the data reading thread.
        """
        self.connected = True
        try:
            # Open config port
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )

            print(f"Connected to port: {self.port}")
            
            self.init()
            # Start background thread to read data
            self.stop_event.clear()
            self.data_thread = threading.Thread(target=self._read_data_loop, daemon=True)
            self.data_thread.start()
            
        except serial.SerialException as e:
            print(f"Error opening serial ports: {e}")
            self.disconnect()
            self.connected = False

    def _read_data_loop(self):
        while not self.stop_event.is_set():
            if self.serial and self.serial.is_open:
                # Read available data from the data port
                try:
                    if self.serial.in_waiting == 0:
                        continue
                    data = self.serial.read_all()
                    if data:
                        self.data_buffer.extend(data)
                        self.process_data(data)
                except serial.SerialException as e:
                    print(f"Data reading error: {e}")
                    break
                except UnicodeDecodeError as e:
                    # Handle bytes that can't be decoded
                    print(f"Decode error: {e}")
            else:
                # If port is not open, break the loop
                break
            # Adjust sleep time if needed to reduce CPU usage
            time.sleep(0.001)
        print("Data reading thread stopped.")

    def disconnect(self):
        """
        Stops data thread and closes the serial ports.
        """
        print("Disconnecting device...")
        # Signal the data loop to exit
        self.stop_event.set()

        # Close data thread gracefully
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1)

        # Close ports
        if self.serial and self.serial.is_open:
            self.serial.close()
        print("Disconnected device.")
    def init(self):        
        self.serial.write(self.cmd_ping)
        self.serial.flush()
            
        self.serial.write(self.cmd_set_manual_mode)
        self.serial.flush()
        self.set_RF_mode(0)
        self.serial.write(self.cmd_set_fps_to_10)
        self.serial.flush()
        start_range = 1  # Start range in meters
        end_range = 3    # End range in meters
        command = set_detection_zone_command(start_range, end_range)
        self.serial.write(command)
        # self.serial.write(self.cmd_set_range_0to1_2)
        self.serial.flush()
    
    def set_RF_mode(self,mode):
        self.RF_mode = mode
        if self.RF_mode == 0:
            self.serial.write(self.cmd_set_downconversion_1) # 1463 Bytes 1439
        else:
            self.serial.write(self.cmd_set_downconversion_0) # 4096 Bytes
        self.serial.flush()
    def process_data(self, data):

        # print(len(self.data_buffer),len(data))
        if len(self.data_buffer) > 10000:
            self.data_buffer = bytearray()
        if len(data) > 100:
            if self.RF_mode == 0:
                IQ=parse_baseband_iq(data)
                if len(IQ) > 0:
                    self.decoded.append([IQ,time.perf_counter()])
                    while len(self.decoded)>self.MaxBufferSize:
                        self.decoded.pop(0)
            else:    
                rf = parse_rf_data(data)
                if len(rf) > 0:
                    self.decoded.append([rf,time.perf_counter()])
                    while len(self.decoded)>self.MaxBufferSize:
                        self.decoded.pop(0)
        return
        if len(self.data_buffer) < 23:
            return
        if self.data_buffer[0:4] != self.flag_seq_start:
            return
        packet_length = struct.unpack('<I', self.data_buffer[4:8])[0]
        n_samples = struct.unpack('<I', self.data_buffer[19:23])[0] // 2
        if len(self.data_buffer) < n_samples * 4 * 2 + 23:
            return

        iq_data = np.frombuffer(self.data_buffer[23:23 + n_samples * 4 * 2], dtype='<f4')
        I = iq_data[:n_samples]  # First n_samples are I components
        Q = iq_data[n_samples:]  # Last n_samples are Q components
        parsed_data = I + 1j * Q  # Combine I and Q to form complex numbers
        self.decoded.append([parsed_data,time.perf_counter()])
        if len(self.decoded)>self.MaxBufferSize:
            self.decoded.pop(0)
        # print("I Q",len(parsed_data))
        # test = parse_baseband_iq(self.data_buffer)
        # d = test ["SigI"]-I
        # d2 = test ["SigQ"]-Q
        self.data_buffer = self.data_buffer[n_samples * 4 * 2 + 23:]
