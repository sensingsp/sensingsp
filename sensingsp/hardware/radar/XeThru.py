from PyQt5.QtCore import QCoreApplication, QTimer, QObject, pyqtSlot
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo

class Xethru(QObject):
    def __init__(self):
        super().__init__()
        self.sensors = X4RangePulseAntennaData()
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
        self.flag_seq_start = bytes.fromhex("7c7c7c7c")
        self.serial = QSerialPort()

        # Command cycle and associated x-axis ranges
        self.range_commands = [
            (self.cmd_set_range_0to9, (0, 9)),
            (self.cmd_set_range_2to11, (2, 11)),
            (self.cmd_set_range_5to14, (5, 14)),
            (self.cmd_set_range_8to17, (8, 17)),
            (self.cmd_set_range_21to30, (21, 30)),
        ]
        self.current_command_index = 0

        # Create plot figure and axis once
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_title('Radar Data (Magnitude)')
        self.ax.set_xlabel('Range (m)')
        self.ax.set_ylabel('Amplitude')
        self.fig.show()
        self.rangeLim = (0,1)
        self.rangeLim_RecSignal = []
        
        # Timer to change commands every 5 seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.change_range_command)
        self.timer.start(5000)  # 5 seconds interval

    def initX4(self, s):
        if not s.isOpen():
            o = s.open(QSerialPort.ReadWrite)
            if o==False:
                CanNotOpen
            self.waitsec(0.1)
        s.write(self.cmd_ping)
        self.waitsec(0.1)
        s.write(self.cmd_set_manual_mode)
        self.waitsec(0.1)
        s.write(self.cmd_set_downconversion_1)
        if self.sensors.setFPS == 50:
            self.waitsec(0.1)
            s.write(self.cmd_set_fps_to_50)
        
        self.waitsec(0.1)
        s.write(self.cmd_set_range_0to9)
        self.rangeLim=(0, 9)
        

    def allports(self, manualPort=''):
        ports = QSerialPortInfo.availablePorts()
        for v in ports:
            check_manual = False
            if manualPort != '':
                if manualPort in v.description():
                    check_manual = True
            if "Bossa" in v.description() or "XeThru" in v.description() or check_manual:
                already_exists = False
                for radar in self.sensors.radar:
                    if radar.serial.portName() == v.portName():
                        already_exists = True
                        if not radar.serial.isOpen():
                            self.initX4(radar.serial)

                if not already_exists:
                    new_x4 = X4RangePulseData(QSerialPort(v))
                    self.sensors.radar.append(new_x4)
                    new_x4.serial.setBaudRate(QSerialPort.Baud115200)
                    new_x4.serial.readyRead.connect(self.serialreadyRead)
                    self.initX4(new_x4.serial)

    def waitsec(self, sec):
        QTimer.singleShot(int(sec * 1000), lambda: None)

    def change_range_command(self):
        # Cycle through the commands and send them to the serial port
        cmd, self.rangeLim = self.range_commands[self.current_command_index]

        if self.sensors.radar:
            radar_serial = self.sensors.radar[-1].serial
            if radar_serial.isOpen():
                print(f"Sending command: {cmd.hex()}, setting range {self.rangeLim[0]} to {self.rangeLim[1]}")
                radar_serial.write(cmd)

        # Move to the next command
        self.current_command_index = (self.current_command_index + 1) % len(self.range_commands)

    @pyqtSlot()
    def serialreadyRead(self):
        p = self.sender()
        b = p.readAll()
        if len(b) > 50:
            while len(b):
                if len(b) < 23:
                    return
                if b[0:4] != self.flag_seq_start:
                    return
                packet_length = struct.unpack('<I', b[4:8])[0]
                n_samples = struct.unpack('<I', b[19:23])[0] // 2
                if len(b) < n_samples * 4 * 2 + 23:
                    return

                d = X4RangeData()
                # Use numpy to handle I/Q data
                iq_data = np.frombuffer(b[23:23 + n_samples * 4 * 2], dtype='<f4')
                I = iq_data[:n_samples]  # First n_samples are I components
                Q = iq_data[n_samples:]  # Last n_samples are Q components
                d.r = I + 1j * Q  # Combine I and Q to form complex numbers

                # Find the radar and append the new data
                for radar in self.sensors.radar:
                    if radar.serial.portName() == p.portName():
                        radar.append(d)

                # Plot the absolute value of the most recent data
                if np.random.rand() > 0.8:  # Only plot with some probability for demonstration
                    self.update_plot(10 * np.log10(np.abs(self.sensors.radar[0].r[-1].r)))
                    self.sensors.radar[0].r = []

                # Move the buffer forward
                b = b[n_samples * 4 * 2 + 23:]

    def update_plot(self, data):
        ok=False
        for lim,sig in self.rangeLim_RecSignal:
            if lim == self.rangeLim:
                sig.append(data)
                ok=True
        if ok==False:
            self.rangeLim_RecSignal.append([self.rangeLim,[data]])
        print(len(self.rangeLim_RecSignal))
        r = np.linspace(self.rangeLim[0],self.rangeLim[1],len(data))
        self.line.set_data(r, data)

        self.ax.cla()
        for lim,sig in self.rangeLim_RecSignal:
            d=sig[-1]
            r = np.linspace(lim[0],lim[1],len(d))
            self.ax.plot(r, d)

        self.ax.set_title('Radar Data (Magnitude)')
        self.ax.set_xlabel('Range (m)')
        self.ax.set_ylabel('Amplitude')
        
        self.ax.relim()
        self.ax.set_ylim(-60, -5)
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
