import threading
import serial
import time
import struct

def read_config_file(file_path):
    with open(file_path, 'r') as file:
        commands = file.readlines()
    return [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('%')]


def find_magic_word(data):
    MAGIC_WORD = [2, 1, 4, 3, 6, 5, 8, 7]
    magic_word = bytearray(MAGIC_WORD)
    for i in range(len(data) - len(magic_word) + 1):
        if data[i:i + len(magic_word)] == magic_word:
            return i
    return -1

def parse_header(data, offset):
    header_format = '<QIIIIIIII'
    header_size = struct.calcsize(header_format)
    header = struct.unpack_from(header_format, data, offset)
    return {
        'magic_word': header[0],
        'version': header[1],
        'total_packet_length': header[2],
        'platform': header[3],
        'frame_number': header[4],
        'time_cpu_cycles': header[5],
        'num_detected_obj': header[6],
        'num_tlvs': header[7],
        'subframe_number': header[8],
        'header_size': header_size
    }

def parse_tlv(data, offset):
    tlv_header_format = '<II'
    tlv_header_size = struct.calcsize(tlv_header_format)
    tlv_type, tlv_length = struct.unpack_from(tlv_header_format, data, offset)
    tlv_data = data[offset + tlv_header_size:offset + tlv_length]
    return {
        'type': tlv_type,
        'length': tlv_length,
        'data': tlv_data,
        'tlv_header_size': tlv_header_size
    }

def parse_detected_objects(tlv_data):
    detected_objects = []
    object_format = '<ffff'
    object_size = struct.calcsize(object_format)
    for i in range(0, len(tlv_data), object_size):
        x, y, z, velocity = struct.unpack_from(object_format, tlv_data, i)
        detected_objects.append({'x': x, 'y': y, 'z': z, 'velocity': velocity})
    return detected_objects

def parse_range_profile(tlv_data):
    range_profile = []
    point_format = '<H'
    point_size = struct.calcsize(point_format)
    for i in range(0, len(tlv_data), point_size):
        point = struct.unpack_from(point_format, tlv_data, i)[0]
        range_profile.append(point / 512.0)  # Convert from Q9 format
    return range_profile

def parse_noise_profile(tlv_data):
    noise_profile = []
    point_format = '<H'
    point_size = struct.calcsize(point_format)
    for i in range(0, len(tlv_data), point_size):
        point = struct.unpack_from(point_format, tlv_data, i)[0]
        noise_profile.append(point / 512.0)  # Convert from Q9 format
    return noise_profile

class MMWaveDevice:
    def __init__(self, config_port: str, data_port: str,
                 config_baudrate: int = 115200, data_baudrate: int = 921600):
        self.config_port = config_port
        self.data_port = data_port
        self.config_baudrate = config_baudrate
        self.data_baudrate = data_baudrate

        self.serial_config = None
        self.serial_data = None

        # Threading
        self.data_thread = None
        self.stop_event = threading.Event()

        # Optionally store incoming data or parse it on the fly
        self.data_buffer = bytearray()
        self.decoded = []
        self.connected = False

    def connect(self):
        """
        Opens serial connections to the mmWave device and starts the data reading thread.
        """
        self.connected = True
        try:
            # Open config port
            self.serial_config = serial.Serial(
                port=self.config_port,
                baudrate=self.config_baudrate,
                timeout=1
            )

            # Open data port
            self.serial_data = serial.Serial(
                port=self.data_port,
                baudrate=self.data_baudrate,
                timeout=1
            )
            print(f"Connected to config port: {self.config_port}, data port: {self.data_port}")

            # Start background thread to read data
            self.stop_event.clear()
            self.data_thread = threading.Thread(target=self._read_data_loop, daemon=True)
            self.data_thread.start()
        except serial.SerialException as e:
            print(f"Error opening serial ports: {e}")
            self.disconnect()
            self.connected = False

    def send_command(self, command: str):
        """
        Send a command to the mmWave device via the config port.

        :param command: A string command to send (e.g. 'sensorStart\n').
        """
        if self.serial_config and self.serial_config.is_open:
            # Ensure it ends with a newline (depends on your device’s command format)
            if not command.endswith('\n'):
                command += '\n'
            self.serial_config.write(command.encode('utf-8'))
            self.serial_config.flush()
            print(f"Sent command: {command.strip()}")
            # Read response until timeout
            timeout = .5
            end_time = time.time() + timeout
            response = b""
            while time.time() < end_time:
                bytes_waiting = self.serial_config.in_waiting
                if bytes_waiting > 0:
                    # Read everything waiting in the buffer
                    chunk = self.serial_config.read(bytes_waiting)
                    response += chunk

                    if b"mmwDemo:/>" in response:
                        break
                    # Extend the timeout slightly if new data continues to arrive
                    # (optional: helps catch slower responses)
                    end_time = time.time() + timeout
                else:
                    # If nothing new arrived, take a short break to let more data come in
                    time.sleep(0.05)

            # Convert bytes to string
            response_str = response.decode('utf-8', errors='ignore').strip()
            print(f"Response command: {response_str}")
            print(f"________________")
            
        else:
            print("Configuration port is not open. Cannot send command.")

    def _read_data_loop(self):
        """
        Internal method run by the data thread, continuously reading data
        from the mmWave device.
        """
        print("Data reading thread started.")
        while not self.stop_event.is_set():
            if self.serial_data and self.serial_data.is_open:
                # Read available data from the data port
                try:
                    data = self.serial_data.read(512)
                    if data:
                        # Store or parse the incoming data
                        self.data_buffer.extend(data)
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
        if self.serial_config and self.serial_config.is_open:
            self.serial_config.close()
        if self.serial_data and self.serial_data.is_open:
            self.serial_data.close()

        print("Disconnected device.")

    def send_config_file(self,CONFIG_FILE):
        config_commands = read_config_file(CONFIG_FILE)
        for command in config_commands:
            self.send_command(command)
        time.sleep(0.05)
    def process_data(self):
        # self.data_buffer
        if len(self.decoded)>1:
            self.decoded = [self.decoded[-1]]
        MinPackLen = 200
        while True:
            parsed_data = {}
            magic_word_index = find_magic_word(self.data_buffer)
            if magic_word_index == -1:
                break
            if len(self.data_buffer)<MinPackLen:
                break
            header = parse_header(self.data_buffer, magic_word_index)
            # print(len(data),header['total_packet_length'])
            if len(self.data_buffer) < header['total_packet_length']:
                break
            offset = magic_word_index + header['header_size']
            for _ in range(header['num_tlvs']):
                tlv = parse_tlv(self.data_buffer, offset)
                offset += tlv['length']+2*4
                if tlv['type'] == 1:
                    1
                    # print("objs")
                    # 1#parsed_data['detected_objects'] = parse_detected_objects(tlv['data'])
                elif tlv['type'] == 2:
                    parsed_data['range_profile'] = parse_range_profile(tlv['data'])
                    print(header['time_cpu_cycles']," range",len(parsed_data['range_profile']))
                    # plot 
                elif tlv['type'] == 3:
                    parsed_data['noise_profile'] = parse_noise_profile(tlv['data'])
                    # print("noise",len(parsed_data['noise_profile']))
                    
                elif tlv['type'] <20:
                    1
                else:
                    print("!!!!!!!!",tlv['type'])
                    # parsed_data['noise_profile'] = parse_noise_profile(tlv['data'])
            self.decoded.append([parsed_data])
            self.data_buffer = self.data_buffer[magic_word_index+header['total_packet_length']:]
        
