import threading
import serial
import time
import struct
import numpy as np

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
        self.cmd_set_fps_to_10 = bytes.fromhex("7d501010000000000020414c7e")
        self.cmd_set_fps_to_50 = bytes.fromhex("7d50101000000000004842277e")
        self.cmd_set_fps_to_90 = bytes.fromhex("7d5010100000000000b442db7e")
        self.cmd_set_range_0to9 = bytes.fromhex("7d5010140000000000000000001041787e")
        self.cmd_set_range_2to11 = bytes.fromhex("7d5010140000000000004000003041187e")
        self.cmd_set_range_5to14 = bytes.fromhex("7d5010140000000000a04000006041e87e")
        self.cmd_set_range_8to17 = bytes.fromhex("7d5010140000000000004100008841a17e")
        self.cmd_set_range_21to30 = bytes.fromhex("7d5010140000000000a8410000f041717e")
        self.cmd_set_range_0to1_2 = bytes.fromhex("7d501014000000009a99993f8c7e")
        self.flag_seq_start = bytes.fromhex("7c7c7c7c")
        

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

    def send_command(self, command: str):
        if self.serial and self.serial.is_open:
            1
            # if not command.endswith('\n'):
            #     command += '\n'
            # self.serial.write(command.encode('utf-8'))
            # self.serial.flush()
            # print(f"Sent command: {command.strip()}")
            # # Read response until timeout
            # timeout = .5
            # end_time = time.time() + timeout
            # response = b""
            # while time.time() < end_time:
            #     bytes_waiting = self.serial_config.in_waiting
            #     if bytes_waiting > 0:
            #         # Read everything waiting in the buffer
            #         chunk = self.serial_config.read(bytes_waiting)
            #         response += chunk

            #         if b"mmwDemo:/>" in response:
            #             break
            #         # Extend the timeout slightly if new data continues to arrive
            #         # (optional: helps catch slower responses)
            #         end_time = time.time() + timeout
            #     else:
            #         # If nothing new arrived, take a short break to let more data come in
            #         time.sleep(0.05)

            # # Convert bytes to string
            # response_str = response.decode('utf-8', errors='ignore').strip()
            # print(f"Response command: {response_str}")
            # print(f"________________")
            
        else:
            print("Configuration port is not open. Cannot send command.")

    def _read_data_loop(self):
        while not self.stop_event.is_set():
            if self.serial and self.serial.is_open:
                # Read available data from the data port
                try:
                    if self.serial.in_waiting == 0:
                        continue
                    data = self.serial.read_all()
                    if data:
                        # Store or parse the incoming data
                        # self.data_buffer.extend(data)
                        self.data_buffer=data
                        
                        # print(len(self.data_buffer),len(data))
                        self.process_data()

                        # print(f"Data read: {len(data)}, {len(self.data_buffer)}")
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
        self.serial.write(self.cmd_set_downconversion_1)
        self.serial.flush()
        self.serial.write(self.cmd_set_fps_to_90)
        self.serial.flush()
        start_range = 1  # Start range in meters
        end_range = 3    # End range in meters
        command = set_detection_zone_command(start_range, end_range)
        self.serial.write(command)
        # self.serial.write(self.cmd_set_range_0to1_2)
        self.serial.flush()
        
    def process_data(self):
        if len(self.decoded)>1:
            self.decoded = [self.decoded[-1]]
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
        self.decoded.append([parsed_data])
        # print("I Q",len(parsed_data))
        
        self.data_buffer = self.data_buffer[n_samples * 4 * 2 + 23:]
# downloaded_file = ssp.utils.hub.fetch_file("animation", "WalkingMan")
# downloaded_file = ssp.utils.hub.fetch_random_file()
# ssp.utils.hub.available_files()
# ssp.hardware.radar.DeviceUI.runapp()
# ssp.ai.extras.ConwaysGameofLife.runapp()
# ssp.ai.radarML.HandGestureMisoCNN.runradarmisoCNNapp()
# ssp.ai.radarML.HumanHealthMonitoringConvAE_BiLSTM.runradarConvAEBiLSTMapp()
# ssp.ai.radarML.GANWaveforms.runradarWaveformapp()
# ssp.environment.deform_scenario_1(angLim=10,Lengths = [.2,.1,.1],elps=[.25,.14,.5],sbd=4,cycles=8,cycleHLen=15)
# ssp.environment.handGesture_simple(G=1)