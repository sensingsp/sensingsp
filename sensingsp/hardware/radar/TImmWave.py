import serial
import serial.tools.list_ports
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget

class TISensor(QWidget):

    CommandPortSignal = pyqtSignal(str)
    rangeProfileDetectedTargets = pyqtSignal(list, list, list)
    tiTLVTypes = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.Commandport = None
        self.Dataport = None
        self.alldata = bytearray()
        self.cloudPoints = []

    def print_ports(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            print(f"{port.device} - {port.description}")

    def set_ports(self, command_port_name, data_port_name):
        self.Commandport = serial.Serial(
            port=command_port_name, baudrate=115200, parity=serial.PARITY_NONE
        )
        self.Dataport = serial.Serial(
            port=data_port_name, baudrate=921600, timeout=None
        )

        if self.Commandport.is_open:
            print("Command Port is Open")
        else:
            print("Command Port is Not Open!!!!")

        if self.Dataport.is_open:
            print("Data Port is Open")
        else:
            print("Data Port is Not Open!!!!")

    def command_ready_read(self):
        if self.Commandport and self.Commandport.is_open:
            data = self.Commandport.read_all()
            data_str = data.decode('utf-8', errors='ignore')
            self.CommandPortSignal.emit(data_str)

    def data_ready_read(self):
        if self.Dataport and self.Dataport.is_open:
            data = self.Dataport.read_all()
            self.alldata.extend(data)

            hdr = [2, 1, 4, 3, 6, 5, 8, 7]
            hdr_index = -1

            for i in range(len(data)):
                if data[i:i+len(hdr)] == bytearray(hdr):
                    hdr_index = i
                    break

            if hdr_index > -1:
                # Process header and data
                self.parse_data_packet(data[hdr_index:])

    def parse_data_packet(self, data):
        types = []

        hdr_length = 8
        version_length = 4
        total_packet_length_length = 4
        platform_length = 4
        frame_number_length = 4
        time_cpu_cycles_length = 4
        num_detected_obj_length = 4
        num_tlvs_length = 4
        subframe_number_length = 4

        magic_word_offset = hdr_length
        version_offset = magic_word_offset
        total_packet_length_offset = version_offset + version_length
        platform_offset = total_packet_length_offset + total_packet_length_length
        frame_number_offset = platform_offset + platform_length
        time_cpu_cycles_offset = frame_number_offset + frame_number_length
        num_detected_obj_offset = time_cpu_cycles_offset + time_cpu_cycles_length
        num_tlvs_offset = num_detected_obj_offset + num_detected_obj_length
        subframe_number_offset = num_tlvs_offset + num_tlvs_length

        if len(data) > subframe_number_offset:
            version = int.from_bytes(data[version_offset:version_offset + version_length], 'little')
            total_packet_length = int.from_bytes(data[total_packet_length_offset:total_packet_length_offset + total_packet_length_length], 'little')
            platform = int.from_bytes(data[platform_offset:platform_offset + platform_length], 'little')
            frame_number = int.from_bytes(data[frame_number_offset:frame_number_offset + frame_number_length], 'little')
            time_cpu_cycles = int.from_bytes(data[time_cpu_cycles_offset:time_cpu_cycles_offset + time_cpu_cycles_length], 'little')
            num_detected_obj = int.from_bytes(data[num_detected_obj_offset:num_detected_obj_offset + num_detected_obj_length], 'little')
            num_tlvs = int.from_bytes(data[num_tlvs_offset:num_tlvs_offset + num_tlvs_length], 'little')
            subframe_number = int.from_bytes(data[subframe_number_offset:subframe_number_offset + subframe_number_length], 'little')

            print(f"Version: {version}, Total Packet Length: {total_packet_length}, Platform: {platform}, Frame Number: {frame_number}, Time in CPU Cycles: {time_cpu_cycles}, Num Detected Obj: {num_detected_obj}, Num TLVs: {num_tlvs}, Subframe Number: {subframe_number}")

            self.tiTLVTypes.emit([version, total_packet_length, platform, frame_number, time_cpu_cycles, num_detected_obj, num_tlvs, subframe_number])

    def sensor_stop(self):
        self.write_command("sensorStop")

    def sensor_start(self):
        self.write_command("sensorStart 0")

    def write_command(self, command):
        if self.Commandport and self.Commandport.is_open:
            command_bytes = command.encode('utf-8') + b'\n'
            self.Commandport.write(command_bytes)
            self.Commandport.flush()
            return len(command_bytes)
        return 0

# Example usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    sensor = TISensor()
    sensor.print_ports()
    sensor.set_ports("COM1", "COM2")
    sensor.command_ready_read()
    sensor.data_ready_read()
    sensor.sensor_start()
    sensor.sensor_stop()

    app.exec_()
