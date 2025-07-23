        
import sensingsp as ssp
import re
import numpy as np
from mathutils import Vector
def QtUI_to_BlenderAddonUI(QtUI,BlenderAddon=None):
    if BlenderAddon==None:
        BlenderAddon = createRadarObject_from_QtUI(QtUI)
    
    BlenderAddon["Center_Frequency_GHz"] = QtUI.f0_lineedit.value()
    BlenderAddon["Transmit_Power_dBm"] = QtUI.tx_power.value()
    BlenderAddon["PRI_us"] = QtUI.pri.value()
    BlenderAddon["RadarMode"] = QtUI.radar_mode.currentText()
    BlenderAddon["FMCW_ChirpSlobe_MHz_usec"] = QtUI.slobe.value()
    BlenderAddon["Fs_MHz"] = QtUI.fs.value()
    BlenderAddon["N_ADC"] = QtUI.n_adc.value()
    BlenderAddon["FMCW"] = QtUI.fmcw.isChecked()
    BlenderAddon["PulseWaveform"] = QtUI.pulse_file.text()

    # --- Array & MIMO ---
    BlenderAddon["ArrayInfofile"] = QtUI.linedit.text()
    BlenderAddon["Array_initialization"] = QtUI.comboarrayinititype.currentText()
    BlenderAddon["Default_Array_Config"] = QtUI.config_combo.currentText()
    BlenderAddon["Position_Scale"] = QtUI.scale_combo.currentText()
    BlenderAddon["TXPos_xy"] = QtUI.tx_lineedit.text()
    BlenderAddon["RXPos_xy"] = QtUI.rx_lineedit.text()
    BlenderAddon["RXPos_xy_bias"] = QtUI.rxb_lineedit.text()
    
    BlenderAddon["distance scaling"] = QtUI.disscale_lineedit.text()
    BlenderAddon["VA order (TX,RX)->[X,Y]|"] = QtUI.vaorder_lineedit.text()
    BlenderAddon["VA order2 (TX,RX)->[X,Y]|"] = QtUI.vaorder_lineedit2.text()
    BlenderAddon["MIMO_Tech"] = QtUI.mimo_combo.currentText()
    BlenderAddon["MIMO_W"] = QtUI.mimo_lineedit.text()

    # --- Antenna patterns ---
    BlenderAddon["Transmit_Antenna_Element_Pattern"] = QtUI.tx_pattern.currentText()
    BlenderAddon["Transmit_Antenna_Element_Gain_db"] = QtUI.tx_gain.value()
    BlenderAddon["Transmit_Antenna_Element_Azimuth_BeamWidth_deg"] = QtUI.tx_az_bw.value()
    BlenderAddon["Transmit_Antenna_Element_Elevation_BeamWidth_deg"] = QtUI.tx_el_bw.value()
    BlenderAddon["Receive_Antenna_Element_Pattern"] = QtUI.rx_pattern.currentText()
    BlenderAddon["Receive_Antenna_Element_Gain_db"] = QtUI.rx_gain.value()
    BlenderAddon["Receive_Antenna_Element_Azimuth_BeamWidth_deg"] = QtUI.rx_az_bw.value()
    BlenderAddon["Receive_Antenna_Element_Elevation_BeamWidth_deg"] = QtUI.rx_el_bw.value()

    # --- Noise & ADC ---
    BlenderAddon["RF_AnalogNoiseFilter_Bandwidth_MHz"] = QtUI.rf_filter_bw.value()
    BlenderAddon["RF_NoiseFiguredB"] = QtUI.rf_nf.value()
    BlenderAddon["Tempreture_K"] = QtUI.temp_k.value()
    BlenderAddon["ADC_peak2peak"] = QtUI.adc_pk2pk.value()
    BlenderAddon["ADC_levels"] = QtUI.adc_levels.value()
    BlenderAddon["ADC_ImpedanceFactor"] = QtUI.adc_imp.value()
    BlenderAddon["ADC_LNA_Gain_dB"] = QtUI.adc_lna.value()
    BlenderAddon["ADC_SaturationEnabled"] = QtUI.adc_sat.isChecked()

    # --- Range-Doppler processing ---
    BlenderAddon["RangeWindow"] = QtUI.range_window.currentText()
    BlenderAddon["RangeFFT_OverNextP2"] = QtUI.Range_FFT_points.value()
    BlenderAddon["Range_Start"] = QtUI.Range_Start.value()
    BlenderAddon["Range_End"] = QtUI.Range_End.value()
    BlenderAddon["Pulse_Buffering"] = QtUI.pulse_buffering.isChecked()
    BlenderAddon["ClutterRemoval_Enabled"] = QtUI.clutter_removal.isChecked()
    BlenderAddon["DopplerProcessingMIMODemod"] = QtUI.dopplerprocessing_method.currentText()
    BlenderAddon["NPulse"] = QtUI.n_pulse.value()
    BlenderAddon["DopplerWindow"] = QtUI.doppler_window.currentText()
    BlenderAddon["DopplerFFT_OverNextP2"] = QtUI.Doppler_FFT_points.value()

    BlenderAddon['RangeDoppler CFAR Mean'] = QtUI.RangeDopplerCFARMean.isChecked()
    BlenderAddon["RangeDopplerCFARLogScale"] = QtUI.logscale.isChecked()
    BlenderAddon["CFAR_RD_type"] = QtUI.cfar_rd_type.currentText()
    BlenderAddon["CFAR_RD_type"] = QtUI.cfar_rd_type.currentText()
    BlenderAddon["CFAR_RD_training_cells"] = QtUI.CFAR_RD_training_cells.text()
    BlenderAddon["CFAR_RD_guard_cells"] = QtUI.CFAR_RD_guard_cells.text()
    BlenderAddon["CFAR_RD_alpha"] = float(QtUI.CFAR_RD_alpha.text())

    # --- Angle processing ---
    BlenderAddon["AzimuthWindow"] = QtUI.azimuth_window.currentText()
    BlenderAddon["AzFFT_OverNextP2"] = QtUI.azimuth_FFT_points.value()
    BlenderAddon["ElevationWindow"] = QtUI.Elevation_window.currentText()
    BlenderAddon["ElFFT_OverNextP2"] = QtUI.Elevation_FFT_points.value()
    BlenderAddon["AngleSpectrum"] = QtUI.spectrum_angle_type.currentText()
    BlenderAddon["Capon Azimuth min:res:max:fine_res (deg)"] = QtUI.CaponAzimuth.text()
    BlenderAddon["Capon Elevation min:res:max:fine_res (deg)"] = QtUI.CaponElevation.text()
    
    BlenderAddon["Capon DL"] = QtUI.CaponDL.text()
    BlenderAddon["CFAR_Angle_type"] = QtUI.cfar_angle_type.currentText()
    BlenderAddon["CFAR_Angle_training_cells"] = QtUI.CFAR_Angle_training_cells.text()
    BlenderAddon["CFAR_Angle_guard_cells"] = QtUI.CFAR_Angle_guard_cells.text()
    BlenderAddon["CFAR_Angle_alpha"] = float(QtUI.CFAR_Angle_alpha.text())

    # --- Simulation BlenderAddon ---
    BlenderAddon["SaveSignalGenerationTime"] = QtUI.save_t.isChecked()
    BlenderAddon["continuousCPIsTrue_oneCPIpeerFrameFalse"] = QtUI.continuous_cpi.isChecked()
    BlenderAddon["t_start_radar"] = float(QtUI.starttime.text())

def BlenderAddonUI_to_QtUI(BlenderAddon,QtUI):
    QtUI.f0_lineedit.setValue(BlenderAddon.get("Center_Frequency_GHz", QtUI.f0_lineedit.value()))
    QtUI.tx_power.setValue(BlenderAddon.get("Transmit_Power_dBm", QtUI.tx_power.value()))
    QtUI.pri.setValue(BlenderAddon.get("PRI_us", QtUI.pri.value()))
    QtUI.radar_mode.setCurrentText(BlenderAddon.get("RadarMode", QtUI.radar_mode.currentText()))
    QtUI.slobe.setValue(BlenderAddon.get("FMCW_ChirpSlobe_MHz_usec", QtUI.slobe.value()))
    QtUI.fs.setValue(BlenderAddon.get("Fs_MHz", QtUI.fs.value()))
    QtUI.n_adc.setValue(BlenderAddon.get("N_ADC", QtUI.n_adc.value()))
    QtUI.fmcw.setChecked(BlenderAddon.get("FMCW", QtUI.fmcw.isChecked()))
    QtUI.pulse_file.setText(BlenderAddon.get("PulseWaveform", QtUI.pulse_file.text()))

    # --- Array & MIMO ---
    QtUI.linedit.setText(BlenderAddon.get("ArrayInfofile", QtUI.linedit.text()))
    QtUI.comboarrayinititype.setCurrentText(BlenderAddon.get("Array_initialization", QtUI.comboarrayinititype.currentText()))
    QtUI.config_combo.setCurrentText(BlenderAddon.get("Default_Array_Config", QtUI.config_combo.currentText()))
    QtUI.scale_combo.setCurrentText(BlenderAddon.get("Position_Scale", QtUI.scale_combo.currentText()))
    QtUI.tx_lineedit.setText(BlenderAddon.get("TXPos_xy", QtUI.tx_lineedit.text()))
    QtUI.rx_lineedit.setText(BlenderAddon.get("RXPos_xy", QtUI.rx_lineedit.text()))
    QtUI.rxb_lineedit.setText(BlenderAddon.get('RXPos_xy_bias', QtUI.rxb_lineedit.text()))
    QtUI.disscale_lineedit.setText(BlenderAddon.get("distance scaling", QtUI.disscale_lineedit.text()))
    QtUI.vaorder_lineedit.setText(BlenderAddon.get("VA order (TX,RX)->[X,Y]|", QtUI.vaorder_lineedit.text()))
    QtUI.vaorder_lineedit2.setText(BlenderAddon.get("VA order2 (TX,RX)->[X,Y]|", QtUI.vaorder_lineedit2.text()))
    QtUI.mimo_combo.setCurrentText(BlenderAddon.get("MIMO_Tech", QtUI.mimo_combo.currentText()))
    QtUI.mimo_lineedit.setText(BlenderAddon.get("MIMO_W", QtUI.mimo_lineedit.text()))

    # --- Antenna patterns ---
    QtUI.tx_pattern.setCurrentText(BlenderAddon.get("Transmit_Antenna_Element_Pattern", QtUI.tx_pattern.currentText()))
    QtUI.tx_gain.setValue(BlenderAddon.get("Transmit_Antenna_Element_Gain_db", QtUI.tx_gain.value()))
    QtUI.tx_az_bw.setValue(BlenderAddon.get("Transmit_Antenna_Element_Azimuth_BeamWidth_deg", QtUI.tx_az_bw.value()))
    QtUI.tx_el_bw.setValue(BlenderAddon.get("Transmit_Antenna_Element_Elevation_BeamWidth_deg", QtUI.tx_el_bw.value()))
    QtUI.rx_pattern.setCurrentText(BlenderAddon.get("Receive_Antenna_Element_Pattern", QtUI.rx_pattern.currentText()))
    QtUI.rx_gain.setValue(BlenderAddon.get("Receive_Antenna_Element_Gain_db", QtUI.rx_gain.value()))
    QtUI.rx_az_bw.setValue(BlenderAddon.get("Receive_Antenna_Element_Azimuth_BeamWidth_deg", QtUI.rx_az_bw.value()))
    QtUI.rx_el_bw.setValue(BlenderAddon.get("Receive_Antenna_Element_Elevation_BeamWidth_deg", QtUI.rx_el_bw.value()))

    # --- Noise & ADC ---
    QtUI.rf_filter_bw.setValue(BlenderAddon.get("RF_AnalogNoiseFilter_Bandwidth_MHz", QtUI.rf_filter_bw.value()))
    QtUI.rf_nf.setValue(BlenderAddon.get("RF_NoiseFiguredB", QtUI.rf_nf.value()))
    QtUI.temp_k.setValue(BlenderAddon.get("Tempreture_K", QtUI.temp_k.value()))
    QtUI.adc_pk2pk.setValue(BlenderAddon.get("ADC_peak2peak", QtUI.adc_pk2pk.value()))
    QtUI.adc_levels.setValue(BlenderAddon.get("ADC_levels", QtUI.adc_levels.value()))
    QtUI.adc_imp.setValue(BlenderAddon.get("ADC_ImpedanceFactor", QtUI.adc_imp.value()))
    QtUI.adc_lna.setValue(BlenderAddon.get("ADC_LNA_Gain_dB", QtUI.adc_lna.value()))
    QtUI.adc_sat.setChecked(BlenderAddon.get("ADC_SaturationEnabled", QtUI.adc_sat.isChecked()))

    # --- Range-Doppler processing ---
    QtUI.range_window.setCurrentText(BlenderAddon.get("RangeWindow", QtUI.range_window.currentText()))
    QtUI.Range_FFT_points.setValue(BlenderAddon.get("RangeFFT_OverNextP2", QtUI.Range_FFT_points.value()))
    QtUI.Range_Start.setValue(BlenderAddon.get("Range_Start", QtUI.Range_Start.value()))
    QtUI.Range_End.setValue(BlenderAddon.get("Range_End", QtUI.Range_End.value()))
    QtUI.pulse_buffering.setChecked(BlenderAddon.get("Pulse_Buffering", QtUI.pulse_buffering.isChecked()))
    QtUI.clutter_removal.setChecked(BlenderAddon.get("ClutterRemoval_Enabled", QtUI.clutter_removal.isChecked()))
    QtUI.dopplerprocessing_method.setCurrentText(BlenderAddon.get("DopplerProcessingMIMODemod", QtUI.dopplerprocessing_method.currentText()))
    QtUI.n_pulse.setValue(BlenderAddon.get("NPulse", QtUI.n_pulse.value()))
    QtUI.doppler_window.setCurrentText(BlenderAddon.get("DopplerWindow", QtUI.doppler_window.currentText()))
    QtUI.Doppler_FFT_points.setValue(BlenderAddon.get("DopplerFFT_OverNextP2", QtUI.Doppler_FFT_points.value()))
    QtUI.logscale.setChecked(BlenderAddon.get("RangeDopplerCFARLogScale", QtUI.logscale.isChecked()))
    

    QtUI.RangeDopplerCFARMean.setChecked(BlenderAddon.get("RangeDoppler CFAR Mean", QtUI.RangeDopplerCFARMean.isChecked()))
    QtUI.cfar_rd_type.setCurrentText(BlenderAddon.get("CFAR_RD_type", QtUI.cfar_rd_type.currentText()))
    QtUI.CFAR_RD_training_cells.setText(BlenderAddon.get("CFAR_RD_training_cells", QtUI.CFAR_RD_training_cells.text()))
    QtUI.CFAR_RD_guard_cells.setText(BlenderAddon.get("CFAR_RD_guard_cells", QtUI.CFAR_RD_guard_cells.text()))
    QtUI.CFAR_RD_alpha.setText(str(BlenderAddon.get("CFAR_RD_alpha", QtUI.CFAR_RD_alpha.text())))

    # --- Angle processing ---
    QtUI.azimuth_window.setCurrentText(BlenderAddon.get("AzimuthWindow", QtUI.azimuth_window.currentText()))
    QtUI.azimuth_FFT_points.setValue(BlenderAddon.get("AzFFT_OverNextP2", QtUI.azimuth_FFT_points.value()))
    QtUI.Elevation_window.setCurrentText(BlenderAddon.get("ElevationWindow", QtUI.Elevation_window.currentText()))
    QtUI.Elevation_FFT_points.setValue(BlenderAddon.get("ElFFT_OverNextP2", QtUI.Elevation_FFT_points.value()))
    QtUI.spectrum_angle_type.setCurrentText(BlenderAddon.get("AngleSpectrum", QtUI.spectrum_angle_type.currentText()))
    QtUI.CaponAzimuth.setText(BlenderAddon.get("Capon Azimuth min:res:max:fine_res (deg)", QtUI.CaponAzimuth.text()))
    QtUI.CaponElevation.setText(BlenderAddon.get("Capon Elevation min:res:max:fine_res (deg)", QtUI.CaponElevation.text()))
    
    QtUI.CaponDL.setText(BlenderAddon.get("Capon DL", QtUI.CaponDL.text()))
    QtUI.cfar_angle_type.setCurrentText(BlenderAddon.get("CFAR_Angle_type", QtUI.cfar_angle_type.currentText()))
    QtUI.CFAR_Angle_training_cells.setText(BlenderAddon.get("CFAR_Angle_training_cells", QtUI.CFAR_Angle_training_cells.text()))
    QtUI.CFAR_Angle_guard_cells.setText(BlenderAddon.get("CFAR_Angle_guard_cells", QtUI.CFAR_Angle_guard_cells.text()))
    QtUI.CFAR_Angle_alpha.setText(str(BlenderAddon.get("CFAR_Angle_alpha", QtUI.CFAR_Angle_alpha.text())))

    # --- Simulation BlenderAddon ---
    QtUI.save_t.setChecked(BlenderAddon.get("SaveSignalGenerationTime", QtUI.save_t.isChecked()))
    QtUI.continuous_cpi.setChecked(BlenderAddon.get("continuousCPIsTrue_oneCPIpeerFrameFalse", QtUI.continuous_cpi.isChecked()))
    QtUI.starttime.setText(str(BlenderAddon.get("t_start_radar", QtUI.starttime.text())))

def BlenderAddonUI_to_RadarSpecifications(BlenderAddon,specifications,online_change=False):
    specifications['BlenderObject']=BlenderAddon
    
    specifications['Lambda']=ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9
    # Transmit_Power_dBm  used in Tx-Power of suites 
    specifications['PRI']=BlenderAddon['PRI_us']*1e-6
    specifications['RadarMode']=BlenderAddon['RadarMode']
    specifications['FMCW_ChirpSlobe'] = BlenderAddon['FMCW_ChirpSlobe_MHz_usec']*1e12
    specifications['Ts']=1e-6/BlenderAddon['Fs_MHz']
    specifications['N_ADC']  = BlenderAddon['N_ADC']
    specifications['FMCW']  = BlenderAddon['FMCW']
    specifications['PulseWaveform']  = BlenderAddon['PulseWaveform']
    # specifications['MIMO_Antenna_Azimuth_Elevation_Order']=BlenderAddon['antenna2azelIndex']
    # # --- Antenna patterns ---
    # QtUI.tx_pattern.setCurrentText(BlenderAddon.get("Transmit_Antenna_Element_Pattern", QtUI.tx_pattern.currentText()))
    # QtUI.tx_gain.setValue(BlenderAddon.get("Transmit_Antenna_Element_Gain_db", QtUI.tx_gain.value()))
    # QtUI.tx_az_bw.setValue(BlenderAddon.get("Transmit_Antenna_Element_Azimuth_BeamWidth_deg", QtUI.tx_az_bw.value()))
    # QtUI.tx_el_bw.setValue(BlenderAddon.get("Transmit_Antenna_Element_Elevation_BeamWidth_deg", QtUI.tx_el_bw.value()))
    # QtUI.rx_pattern.setCurrentText(BlenderAddon.get("Receive_Antenna_Element_Pattern", QtUI.rx_pattern.currentText()))
    # QtUI.rx_gain.setValue(BlenderAddon.get("Receive_Antenna_Element_Gain_db", QtUI.rx_gain.value()))
    # QtUI.rx_az_bw.setValue(BlenderAddon.get("Receive_Antenna_Element_Azimuth_BeamWidth_deg", QtUI.rx_az_bw.value()))
    # QtUI.rx_el_bw.setValue(BlenderAddon.get("Receive_Antenna_Element_Elevation_BeamWidth_deg", QtUI.rx_el_bw.value()))

    # --- Noise & ADC ---
    specifications['RF_AnalogNoiseFilter_Bandwidth'] = BlenderAddon['RF_AnalogNoiseFilter_Bandwidth_MHz']*1e6
    specifications['RF_NoiseFiguredB'] = BlenderAddon['RF_NoiseFiguredB']
    specifications['Tempreture_K']=BlenderAddon['Tempreture_K']
    specifications['ADC_peak2peak'] = BlenderAddon['ADC_peak2peak']
    specifications['ADC_levels'] = BlenderAddon['ADC_levels']
    specifications['ADC_ImpedanceFactor'] = BlenderAddon['ADC_ImpedanceFactor']
    specifications['ADC_LNA_Gain'] = 10**(BlenderAddon['ADC_LNA_Gain_dB']/10)
    specifications['ADC_SaturationEnabled'] = BlenderAddon['ADC_SaturationEnabled']
    
    # --- Range-Doppler processing ---
    specifications['RangeWindow'] = BlenderAddon['RangeWindow']
    specifications['RangeFFT_OverNextP2'] = BlenderAddon['RangeFFT_OverNextP2']
    specifications['Range_Start'] = BlenderAddon['Range_Start']
    specifications['Range_End'] = BlenderAddon['Range_End']
    specifications['Pulse_Buffering'] = BlenderAddon['Pulse_Buffering']
    specifications['ClutterRemoval_Enabled'] = BlenderAddon['ClutterRemoval_Enabled']
    specifications['DopplerProcessingMIMODemod'] = BlenderAddon['DopplerProcessingMIMODemod']
    specifications['NPulse'] = BlenderAddon['NPulse']
    specifications['RangeDopplerCFARLogScale'] = BlenderAddon['RangeDopplerCFARLogScale']
    specifications['DopplerWindow'] = BlenderAddon['DopplerWindow']
    specifications['DopplerFFT_OverNextP2'] = BlenderAddon['DopplerFFT_OverNextP2']
    specifications['RangeDoppler CFAR Mean'] = BlenderAddon['RangeDoppler CFAR Mean']
    specifications['CFAR_RD_type'] = BlenderAddon['CFAR_RD_type']
    specifications['CFAR_RD_training_cells'] = BlenderAddon['CFAR_RD_training_cells']
    specifications['CFAR_RD_guard_cells'] = BlenderAddon['CFAR_RD_guard_cells']
    specifications['CFAR_RD_alpha'] = BlenderAddon['CFAR_RD_alpha']
    specifications['CFAR_RD_false_alarm_rate'] = "BlenderAddon['CFAR_RD_false_alarm_rate']"
    
    # --- Angle processing ---
    specifications['AzimuthWindow'] = BlenderAddon['AzimuthWindow']
    specifications['AzFFT_OverNextP2'] = BlenderAddon['AzFFT_OverNextP2']
    specifications['ElevationWindow'] = BlenderAddon['ElevationWindow']
    specifications['ElFFT_OverNextP2'] = BlenderAddon['ElFFT_OverNextP2']
    specifications['AngleSpectrum'] = BlenderAddon['AngleSpectrum']
    a, b, c, d = map(float, BlenderAddon['Capon Azimuth min:res:max:fine_res (deg)'].split(':'))
    specifications['Capon Azimuth min:res:max:fine_res'] =[a,b,c,d] 
    a, b, c, d = map(float, BlenderAddon['Capon Elevation min:res:max:fine_res (deg)'].split(':'))
    specifications['Capon Elevation min:res:max:fine_res'] =[a,b,c,d] 
    specifications['Capon DL'] = float(BlenderAddon['Capon DL'])
    specifications['CFAR_Angle_type'] = BlenderAddon['CFAR_Angle_type']
    specifications['CFAR_Angle_guard_cells'] = BlenderAddon['CFAR_Angle_guard_cells']
    specifications['CFAR_Angle_training_cells'] = BlenderAddon['CFAR_Angle_training_cells']
    specifications['CFAR_Angle_false_alarm_rate'] = "BlenderAddon['CFAR_Angle_false_alarm_rate']"
    specifications['CFAR_Angle_alpha'] = BlenderAddon['CFAR_Angle_alpha']

    specifications['distance scaling'] = BlenderAddon['distance scaling']

    if "TXRXPos" in BlenderAddon:
        tx = np.array(BlenderAddon["TXRXPos"][0])
        rx = np.array(BlenderAddon["TXRXPos"][1])
        array_geometry = []
        for txi in tx:
            for rxi in rx:
                array_geometry.append([0,txi[0]+rxi[0], txi[1]+rxi[1]])
        array_geometry = np.array(array_geometry)
        specifications['array_geometry'] = array_geometry
    else:
        specifications['array_geometry'] = None
    # --- Simulation BlenderAddon ---

    specifications['SaveSignalGenerationTime']=BlenderAddon['SaveSignalGenerationTime']
    specifications['MaxRangeScatter']="BlenderAddon['MaxRangeScatter']"
    specifications['continuousCPIsTrue_oneCPIpeerFrameFalse']=BlenderAddon['continuousCPIsTrue_oneCPIpeerFrameFalse']
    
    if online_change == False:
        specifications['RadarTiming'] = ssp.utils.RadarTiming(t_start_radar=BlenderAddon['t_start_radar'], 
                                                    t_start_manual_restart_tx=0,
                                                    t_last_pulse=0.0,
                                                    t_current_pulse=0, 
                                                    pri_sequence=[specifications['PRI']], 
                                                    n_pulse=0,
                                                    n_last_cpi=0)
        specifications['CPI_Buffer']=[]
        specifications['RadarBuffer']=ssp.utils.RadarBuffer(specifications['NPulse'])
        

    # # --- Array & MIMO ---
    # QtUI.linedit.setText(BlenderAddon.get("ArrayInfofile", QtUI.linedit.text()))
    # QtUI.comboarrayinititype.setCurrentText(BlenderAddon.get("Array_initialization", QtUI.comboarrayinititype.currentText()))
    # QtUI.config_combo.setCurrentText(BlenderAddon.get("Default_Array_Config", QtUI.config_combo.currentText()))

    # QtUI.mimo_combo.setCurrentText(BlenderAddon.get("MIMO_Tech", QtUI.mimo_combo.currentText()))
    
    
    if BlenderAddon['Array_initialization']=="This UI":
        1
    if BlenderAddon['Array_initialization']=="Blender with this System Configurations":
        2
    elif BlenderAddon['Array_initialization']=="Blender TDM":
        3

    specifications['ArrayInfofile']=BlenderAddon['ArrayInfofile']


    # specifications['MIMO_Antenna_Azimuth_Elevation_Order']=BlenderAddon['antenna2azelIndex']
    
    s=BlenderAddon["VA order (TX,RX)->[X,Y]|"]
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
    specifications['vaorder']=np.array(vaorder)
    
    s=BlenderAddon["VA order2 (TX,RX)->[X,Y]|"]
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
    specifications['vaorder2']=np.array(vaorder2)
    
    # scale = BlenderAddon["Position_Scale"]
    # vaprocessing = QtUI.va_combo.currentText()
    
    txt = BlenderAddon["MIMO_W"]
    row_strs = txt.split(';')
    data = []
    for r in row_strs:
        items = [s.strip() for s in r.split(',')]
        row = [complex(item) for item in items]
        data.append(row)

    PrecodingMatrix = np.array(data, dtype=complex)
    specifications['PrecodingMatrix'] = np.tile(PrecodingMatrix, (int(np.ceil(specifications['NPulse'] / PrecodingMatrix.shape[0])), 1))[:specifications['NPulse'], :]

    # x=BlenderAddon['TX']
    # specifications['FMCW_Bandwidth']=BlenderAddon['FMCW_Bandwidth_GHz']*1e9
    # specifications['PrecodingMatrix'] = np.eye(len(Suite_Position[isuite]['Radar'][iradar]['TX-Position']),dtype=np.complex128)
    specifications['M_TX'] = len(ssp.environment.BlenderSuiteFinder().find_tx(BlenderAddon))
    specifications['N_RX'] = len(ssp.environment.BlenderSuiteFinder().find_rx(BlenderAddon))
    specifications['MIMO_Tech'] = BlenderAddon['MIMO_Tech']
    specifications['STC_Enabled'] = BlenderAddon['STC_Enabled']
    specifications['MTI_Enabled'] = BlenderAddon['MTI_Enabled']
    
    # specifications['PrecodingMatrix'] = ssp.radar.utils.mimo_Functions.AD_matrix(NPulse=specifications['NPulse'],
    #                                                             M=len(radarobject['TX']),
    #                                                             tech=specifications['MIMO_Tech'])
    
    
    specifications['matrix_world']=BlenderAddon.matrix_world.decompose()
    # if "ArrayInfofile" in radarobject["GeneralRadarSpec_Object"].keys():
    #     specifications['ArrayInfofile']=BlenderAddon['ArrayInfofile']
    # else:
    #     specifications['ArrayInfofile']=None
        
    # modifyArrayinfowithFile(specifications)
    k=0
    global_location_Center = Vector((0,0,0))
    global_location_TX = []
    for itx,txobj in enumerate(ssp.environment.BlenderSuiteFinder().find_tx(BlenderAddon)):
        global_location, global_rotation, global_scale = txobj.matrix_world.decompose()
        global_location_TX.append(global_location)
        global_location_Center += global_location
        k+=1
    global_location_RX = []
    for irx,rxobj in enumerate(ssp.environment.BlenderSuiteFinder().find_rx(BlenderAddon)):
        global_location, global_rotation, global_scale = rxobj.matrix_world.decompose()
        global_location_RX.append(global_location)
        global_location_Center += global_location
        k+=1
    global_location_Center /= k
    specifications['global_location_TX_RX_Center'] = [global_location_TX,global_location_RX,global_location_Center]
    return
    azindex = []
    elindex = []
    for itx,txobj in enumerate(radarobject['TX']):
        # global_location, global_rotation, global_scale = txobj.matrix_world.decompose()
        local_location, local_rotation, local_scale = txobj.matrix_local.decompose()
        local_location_HW = local_location / (ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9)* 2
        azTx=round(local_location_HW.x)
        elTx=round(local_location_HW.y)
        # print("itx,local_location:",itx,local_location,txobj.name)
        for irx,rxobj in enumerate(radarobject['RX']):
            local_location, local_rotation, local_scale = rxobj.matrix_local.decompose()
            local_location_HW = local_location / (ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9) * 2
            azRx=round(local_location_HW.x)
            elRx=round(local_location_HW.y)
            # print(iradar,azTx+azRx,elRx+elTx)
            azindex.append(azTx+azRx)
            elindex.append(elTx+elRx)
    #       print("irx,local_location:",irx,local_location,rxobj.name)
    # print("azindex:",azindex)
    # print("elindex:",elindex)


    azindex = azindex - np.min(azindex)+1
    elindex = elindex - np.min(elindex)+1
    antennaIndex2VAx = np.zeros((len(radarobject['TX']),len(radarobject['RX'])))
    antennaIndex2VAy = np.zeros((len(radarobject['TX']),len(radarobject['RX'])))
    k=0
    for itx in range(antennaIndex2VAx.shape[0]):
        for irx in range(antennaIndex2VAx.shape[1]):
            antennaIndex2VAx[itx,irx] = azindex[k]-1
            antennaIndex2VAy[itx,irx] = elindex[k]-1
            k+=1

    specifications['MIMO_AntennaIndex2VA']=[antennaIndex2VAx,antennaIndex2VAy,np.max(elindex),np.max(azindex)]
    antenna_Pos0_Wavelength_TX=[]
    for itx,txobj in enumerate(radarobject['TX']):
        local_location, local_rotation, local_scale = txobj.matrix_local.decompose()
        local_location_HW = local_location / (ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9)
        antenna_Pos0_Wavelength_TX.append(local_location_HW)
    antenna_Pos0_Wavelength_RX=[]
    for irx,rxobj in enumerate(radarobject['RX']):
            local_location, local_rotation, local_scale = rxobj.matrix_local.decompose()
            local_location_HW = local_location / (ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9)
            antenna_Pos0_Wavelength_RX.append(local_location_HW)
    specifications['antenna_Pos0_Wavelength']=[antenna_Pos0_Wavelength_TX,antenna_Pos0_Wavelength_RX]


    PosIndex = []
    for itx,txobj in enumerate(radarobject['TX']):
        # global_location, global_rotation, global_scale = txobj.matrix_world.decompose()
        local_location, local_rotation, local_scale = txobj.matrix_local.decompose()
        local_location_HW = local_location / (ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9)* 2
        azTx=local_location_HW.x
        elTx=local_location_HW.y
        # print("itx,local_location:",itx,local_location,txobj.name)
        for irx,rxobj in enumerate(radarobject['RX']):
            local_location, local_rotation, local_scale = rxobj.matrix_local.decompose()
            local_location_HW = local_location / (ssp.constants.LightSpeed/BlenderAddon["Center_Frequency_GHz"]/1e9) * 2
            azRx=local_location_HW.x
            elRx=local_location_HW.y
            PosIndex.append([azTx+azRx,elTx+elRx,itx,irx])
    specifications['Local_location_TXplusRX_Center'] = PosIndex

    # x = np.zeros((np.max(elindex),np.max(azindex)))
    # for itx,txobj in enumerate(radarobject['TX']):
    #   for irx,rxobj in enumerate(radarobject['RX']):
    #     x[int(antennaIndex2VAy[itx,irx]),int(antennaIndex2VAx[itx,irx])]=1

    # print(iradar,azindex,elindex,np.max(azindex),np.max(elindex),x)
    # specifications['RangePulseRX']= np.zeros((specifications['N_ADC'],specifications['NPulse'],len(Suite_Position[isuite]['Radar'][iradar]['RX-Position'])),dtype=np.complex128)
    
    tx_positions,rx_positions=BlenderAddon["TXRXPos"]
    specifications['TXRXPos']=BlenderAddon["TXRXPos"]
    vainfo = ssp.radar.utils.virtualArray_info(tx_positions,rx_positions)
    specifications['ULA_TXRX_Lx_Ly_NonZ']=vainfo

def createRadarObject_from_QtUI(QtUI):
    tx_positions = [tuple(map(float, p.split(','))) for p in QtUI.tx_lineedit.text().split('|')]
    rx_positions = [tuple(map(float, p.split(','))) for p in QtUI.rx_lineedit.text().split('|')]
    rx_bias = tuple(map(float, QtUI.rxb_lineedit.text().split(',')))
    # id = QtUI.id_lineedit.text()
    AzELscale = tuple(map(float, QtUI.disscale_lineedit.text().split(',')))
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    suiteIndex=len(suite_planes)-1
    if len(suite_planes)==0:
        suiteIndex=0
        ssp.integratedSensorSuite.define_suite(suiteIndex, location=Vector((0, 0, 0)), rotation=Vector((0, 0, 0)))
    suite_planes = ssp.environment.BlenderSuiteFinder().find_suite_planes()
    obj = suite_planes[-1]
    radar_planes = ssp.environment.BlenderSuiteFinder().find_radar_planes(obj)
    radarIndex = max([int(plane.name.split('_')[2]) for plane in radar_planes if plane.parent == obj] or [-1]) + 1

    
    for i in range(len(rx_positions)):
        rx_positions[i] = (AzELscale[0]*rx_positions[i][0]+rx_bias[0], AzELscale[1]*rx_positions[i][1]+rx_bias[1])
    for i in range(len(tx_positions)):
        tx_positions[i] = (AzELscale[0]*tx_positions[i][0], AzELscale[1]*tx_positions[i][1])
    f0=1e9*QtUI.f0_lineedit.value()
    location_xyz = [0, 0, 0]
    BlenderAddon = ssp.radar.utils.predefined_array_configs_txrx(tx_positions=tx_positions,rx_positions=rx_positions,isuite=suiteIndex, iradar=radarIndex, location=Vector((location_xyz[0], location_xyz[1],location_xyz[2])), rotation=Vector((np.pi/2,0, -np.pi/2)), f0=f0) 
    return BlenderAddon
    
def QtUI_arrays():
    return ["2x4","2x4-2", "3x4", "3x4-2", "3x4-3", "12x16", "12x16-2"]

def setQtUI_arrays(QtUI, index):
    selected_value = QtUI.config_combo.currentText()        
    if selected_value == "2x4":
        QtUI.tx_lineedit.setText("0,0|4,0")
        QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0")
        s = ''
        k=0
        for itx in range(2):
            for irx in range(4):
                k+=1
                s+=f'({itx+1},{irx+1})->[{k},1] | '
        QtUI.vaorder_lineedit.setText(s)
        s = ''
        QtUI.vaorder_lineedit2.setText(s)
    if selected_value == "2x4-2":
        QtUI.tx_lineedit.setText("0,0|2,0")
        QtUI.rx_lineedit.setText("0,0|1,0|0,1|1,1")
        s = ''
        s+=f'(1,1)->[1,1] | '
        s+=f'(1,2)->[2,1] | '
        s+=f'(2,1)->[3,1] | '
        s+=f'(2,2)->[4,1] |'
        QtUI.vaorder_lineedit.setText(s)
        s = ''
        s+=f'(1,1)->[1,1] |'
        s+=f'(1,3)->[1,2] |'
        QtUI.vaorder_lineedit2.setText(s)
    if selected_value == "3x4":
        QtUI.tx_lineedit.setText("0,0|4,0|8,0")
        QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0")
        s = ''
        k=0
        for itx in range(3):
            for irx in range(4):
                k+=1
                s+=f'({itx+1},{irx+1})->[{k},1] | '
        QtUI.vaorder_lineedit.setText(s)
        s = ''
        QtUI.vaorder_lineedit2.setText(s)
    if selected_value == "3x4-3":
        QtUI.tx_lineedit.setText("0,0|4,1|8,0")
        QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0")
        # QtUI.va_combo.setCurrentIndex(1) # Az FFT El Estimation
        s = ''
        k=0
        for itx in range(3):
            for irx in range(4):
                k+=1
                s+=f'({itx+1},{irx+1})->[{k},1] | '
        s = s[:-1]
        QtUI.vaorder_lineedit.setText(s)
        s = ''
        s+=f'(1,1)->[1,1] |'
        s+=f'(2,1)->[1,2] |'
        QtUI.vaorder_lineedit2.setText(s)
    if selected_value == "3x4-2":
        QtUI.scale_combo.setCurrentIndex(0)
        QtUI.tx_lineedit.setText("0,0|1,0|2,0")
        QtUI.rx_lineedit.setText("0,0|0,1|0,2|0,3")
        # QtUI.va_combo.setCurrentIndex(3) # 2D FFT
        s = ''
        k=0
        for itx in range(3):
            for irx in range(4):
                k+=1
                s+=f'({itx+1},{irx+1})->[{itx+1},{irx+1}] | '
        
        QtUI.vaorder_lineedit.setText(s)
        QtUI.vaorder_lineedit2.setText('')
    if selected_value == "12x16":
        QtUI.tx_lineedit.setText("0,0|4,0|8,0|9,1|10,4|11,6|12,0|16,0|20,0|24,0|28,0|32,0")
        QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0|11,0|12,0|13,0|14,0|46,0|47,0|48,0|49,0|50,0|51,0|52,0|53,0")
        # QtUI.va_combo.setCurrentIndex(2) # Az FFT - El FFT
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
        QtUI.vaorder_lineedit.setText(s)
        s = ''
        s+=f'(1,1)->[1,1] |'
        s+=f'(4,1)->[1,2] |'
        s+=f'(5,1)->[1,3] |'
        s+=f'(6,1)->[1,4] |'
        QtUI.vaorder_lineedit2.setText(s)
    if selected_value == "12x16-2":
        QtUI.scale_combo.setCurrentIndex(0)
        QtUI.tx_lineedit.setText("0,0|4,0|8,0|12,0|0,4|4,4|8,4|12,4|0,8|4,8|8,8|12,8")
        QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0|0,1|1,1|2,1|3,1|0,2|1,2|2,2|3,2|0,3|1,3|2,3|3,3")
        # QtUI.va_combo.setCurrentIndex(3) # 2D FFT
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
        QtUI.vaorder_lineedit.setText(s)
        QtUI.vaorder_lineedit2.setText('')

def set_TI_AWR1642_toQtUI(QtUI):
    QtUI.f0_lineedit.setValue(76)
    QtUI.tx_power.setValue(12)
    QtUI.pri.setValue(70)
    QtUI.radar_mode.setCurrentText("FMCW")
    QtUI.slobe.setValue(17)
    QtUI.fs.setValue(12.5)
    QtUI.n_adc.setValue(256)
    QtUI.fmcw.setChecked(True)
    QtUI.pulse_file.setText("")

    # --- Array & MIMO ---
    QtUI.linedit.setText("")
    QtUI.comboarrayinititype.setCurrentText("This UI")
    QtUI.config_combo.setCurrentText("2x4")
    QtUI.scale_combo.setCurrentText("half wavelength")
    QtUI.tx_lineedit.setText("0,0|4,0")
    QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0")
    QtUI.rxb_lineedit.setText("10,0")
    QtUI.disscale_lineedit.setText("1,1")
    QtUI.vaorder_lineedit.setText("(1,1)->[1,1] | (1,2)->[2,1] | (1,3)->[3,1] | (1,4)->[4,1] | (2,1)->[5,1] | (2,2)->[6,1] | (2,3)->[7,1] | (2,4)->[8,1] | ")
    QtUI.vaorder_lineedit2.setText("")
    QtUI.mimo_combo.setCurrentText("TDM")
    QtUI.mimo_lineedit.setText("(1+0j),0j;0j,(1+0j)")

    # --- Antenna patterns ---
    QtUI.tx_pattern.setCurrentText("Directional-Sinc")
    QtUI.tx_gain.setValue(6)
    QtUI.tx_az_bw.setValue(60)
    QtUI.tx_el_bw.setValue(60)
    QtUI.rx_pattern.setCurrentText("Directional-Sinc")
    QtUI.rx_gain.setValue(6)
    QtUI.rx_az_bw.setValue(60)
    QtUI.rx_el_bw.setValue(60)
    # --- Noise & ADC ---
    QtUI.rf_filter_bw.setValue(12.5)
    QtUI.rf_nf.setValue(14)
    QtUI.temp_k.setValue(300)
    QtUI.adc_pk2pk.setValue(2)
    QtUI.adc_levels.setValue(256)
    QtUI.adc_imp.setValue(300)
    QtUI.adc_lna.setValue(50)
    QtUI.adc_sat.setChecked(False)

    # --- Range-Doppler processing ---
    QtUI.range_window.setCurrentText("Hamming")
    QtUI.Range_FFT_points.setValue(0)
    QtUI.Range_Start.setValue(0)
    QtUI.Range_End.setValue(100)
    QtUI.pulse_buffering.setChecked(False)
    QtUI.clutter_removal.setChecked(False)
    QtUI.dopplerprocessing_method.setCurrentText("Simple FFT")
    QtUI.n_pulse.setValue(128)
    QtUI.doppler_window.setCurrentText("Hamming")
    QtUI.Doppler_FFT_points.setValue(0)
    QtUI.logscale.setChecked(False)
    QtUI.RangeDopplerCFARMean.setChecked(True)
    QtUI.cfar_rd_type.setCurrentText("Fixed Threshold a*KSort")
    QtUI.CFAR_RD_training_cells.setText("40,40")
    QtUI.CFAR_RD_guard_cells.setText("10,10")
    QtUI.CFAR_RD_alpha.setText("5")

    # --- Angle processing ---
    QtUI.azimuth_window.setCurrentText("Hamming")
    QtUI.azimuth_FFT_points.setValue(0)
    QtUI.Elevation_window.setCurrentText("Hamming")
    QtUI.Elevation_FFT_points.setValue(0)
    QtUI.spectrum_angle_type.setCurrentText("FFT")
    QtUI.CaponAzimuth.setText("-60:3:60:1")
    QtUI.CaponElevation.setText("-60:3:60:1")
    
    QtUI.CaponDL.setText("2")
    QtUI.cfar_angle_type.setCurrentText("No CFAR (max)")
    QtUI.CFAR_Angle_training_cells.setText("40,40")
    QtUI.CFAR_Angle_guard_cells.setText("10,10")
    QtUI.CFAR_Angle_alpha.setText("2")

    # --- Simulation BlenderAddon ---
    QtUI.save_t.setChecked(True)
    QtUI.continuous_cpi.setChecked(False)
    QtUI.starttime.setText("0.000000")

def set_TI_IWR6843_toQtUI(QtUI):
    1
def set_TI_AWR2243_toQtUI(QtUI):
    1
def set_TI_AWR2944_toQtUI(QtUI):
    1
def set_TI_Cascade_AWR2243_toQtUI(QtUI):
    1
def set_SISO_mmWave76GHz_toQtUI(QtUI):
    1
def set_Xhetru_X4_toQtUI(QtUI):
    1
def set_Altos_toQtUI(QtUI):
    QtUI.f0_lineedit.setValue(76)
    QtUI.tx_power.setValue(12)
    QtUI.pri.setValue(50)
    QtUI.radar_mode.setCurrentText("FMCW")
    QtUI.slobe.setValue(17)
    QtUI.fs.setValue(12.5)
    QtUI.n_adc.setValue(256)
    QtUI.fmcw.setChecked(True)
    QtUI.pulse_file.setText("")

    # --- Array & MIMO ---
    QtUI.linedit.setText("")
    QtUI.comboarrayinititype.setCurrentText("This UI")
    QtUI.config_combo.setCurrentText("12x16-2")
    QtUI.scale_combo.setCurrentText("half wavelength")
    QtUI.tx_lineedit.setText("0,0|4,0|8,0|12,0|0,4|4,4|8,4|12,4|0,8|4,8|8,8|12,8")
    QtUI.rx_lineedit.setText("0,0|1,0|2,0|3,0|0,1|1,1|2,1|3,1|0,2|1,2|2,2|3,2|0,3|1,3|2,3|3,3")
    QtUI.rxb_lineedit.setText("50,0")
    QtUI.disscale_lineedit.setText("3,3")
    QtUI.vaorder_lineedit.setText("(1,1)->[1,1] | (1,2)->[2,1] | (1,3)->[3,1] | (1,4)->[4,1] | (1,5)->[1,2] | (1,6)->[2,2] | (1,7)->[3,2] | (1,8)->[4,2] | (1,9)->[1,3] | (1,10)->[2,3] | (1,11)->[3,3] | (1,12)->[4,3] | (1,13)->[1,4] | (1,14)->[2,4] | (1,15)->[3,4] | (1,16)->[4,4] | (2,1)->[5,1] | (2,2)->[6,1] | (2,3)->[7,1] | (2,4)->[8,1] | (2,5)->[5,2] | (2,6)->[6,2] | (2,7)->[7,2] | (2,8)->[8,2] | (2,9)->[5,3] | (2,10)->[6,3] | (2,11)->[7,3] | (2,12)->[8,3] | (2,13)->[5,4] | (2,14)->[6,4] | (2,15)->[7,4] | (2,16)->[8,4] | (3,1)->[9,1] | (3,2)->[10,1] | (3,3)->[11,1] | (3,4)->[12,1] | (3,5)->[9,2] | (3,6)->[10,2] | (3,7)->[11,2] | (3,8)->[12,2] | (3,9)->[9,3] | (3,10)->[10,3] | (3,11)->[11,3] | (3,12)->[12,3] | (3,13)->[9,4] | (3,14)->[10,4] | (3,15)->[11,4] | (3,16)->[12,4] | (4,1)->[13,1] | (4,2)->[14,1] | (4,3)->[15,1] | (4,4)->[16,1] | (4,5)->[13,2] | (4,6)->[14,2] | (4,7)->[15,2] | (4,8)->[16,2] | (4,9)->[13,3] | (4,10)->[14,3] | (4,11)->[15,3] | (4,12)->[16,3] | (4,13)->[13,4] | (4,14)->[14,4] | (4,15)->[15,4] | (4,16)->[16,4] | (5,1)->[1,5] | (5,2)->[2,5] | (5,3)->[3,5] | (5,4)->[4,5] | (5,5)->[1,6] | (5,6)->[2,6] | (5,7)->[3,6] | (5,8)->[4,6] | (5,9)->[1,7] | (5,10)->[2,7] | (5,11)->[3,7] | (5,12)->[4,7] | (5,13)->[1,8] | (5,14)->[2,8] | (5,15)->[3,8] | (5,16)->[4,8] | (6,1)->[5,5] | (6,2)->[6,5] | (6,3)->[7,5] | (6,4)->[8,5] | (6,5)->[5,6] | (6,6)->[6,6] | (6,7)->[7,6] | (6,8)->[8,6] | (6,9)->[5,7] | (6,10)->[6,7] | (6,11)->[7,7] | (6,12)->[8,7] | (6,13)->[5,8] | (6,14)->[6,8] | (6,15)->[7,8] | (6,16)->[8,8] | (7,1)->[9,5] | (7,2)->[10,5] | (7,3)->[11,5] | (7,4)->[12,5] | (7,5)->[9,6] | (7,6)->[10,6] | (7,7)->[11,6] | (7,8)->[12,6] | (7,9)->[9,7] | (7,10)->[10,7] | (7,11)->[11,7] | (7,12)->[12,7] | (7,13)->[9,8] | (7,14)->[10,8] | (7,15)->[11,8] | (7,16)->[12,8] | (8,1)->[13,5] | (8,2)->[14,5] | (8,3)->[15,5] | (8,4)->[16,5] | (8,5)->[13,6] | (8,6)->[14,6] | (8,7)->[15,6] | (8,8)->[16,6] | (8,9)->[13,7] | (8,10)->[14,7] | (8,11)->[15,7] | (8,12)->[16,7] | (8,13)->[13,8] | (8,14)->[14,8] | (8,15)->[15,8] | (8,16)->[16,8] | (9,1)->[1,9] | (9,2)->[2,9] | (9,3)->[3,9] | (9,4)->[4,9] | (9,5)->[1,10] | (9,6)->[2,10] | (9,7)->[3,10] | (9,8)->[4,10] | (9,9)->[1,11] | (9,10)->[2,11] | (9,11)->[3,11] | (9,12)->[4,11] | (9,13)->[1,12] | (9,14)->[2,12] | (9,15)->[3,12] | (9,16)->[4,12] | (10,1)->[5,9] | (10,2)->[6,9] | (10,3)->[7,9] | (10,4)->[8,9] | (10,5)->[5,10] | (10,6)->[6,10] | (10,7)->[7,10] | (10,8)->[8,10] | (10,9)->[5,11] | (10,10)->[6,11] | (10,11)->[7,11] | (10,12)->[8,11] | (10,13)->[5,12] | (10,14)->[6,12] | (10,15)->[7,12] | (10,16)->[8,12] | (11,1)->[9,9] | (11,2)->[10,9] | (11,3)->[11,9] | (11,4)->[12,9] | (11,5)->[9,10] | (11,6)->[10,10] | (11,7)->[11,10] | (11,8)->[12,10] | (11,9)->[9,11] | (11,10)->[10,11] | (11,11)->[11,11] | (11,12)->[12,11] | (11,13)->[9,12] | (11,14)->[10,12] | (11,15)->[11,12] | (11,16)->[12,12] | (12,1)->[13,9] | (12,2)->[14,9] | (12,3)->[15,9] | (12,4)->[16,9] | (12,5)->[13,10] | (12,6)->[14,10] | (12,7)->[15,10] | (12,8)->[16,10] | (12,9)->[13,11] | (12,10)->[14,11] | (12,11)->[15,11] | (12,12)->[16,11] | (12,13)->[13,12] | (12,14)->[14,12] | (12,15)->[15,12] | (12,16)->[16,12] |  ")
    QtUI.vaorder_lineedit2.setText("")
    QtUI.mimo_combo.setCurrentText("TDM")
    QtUI.mimo_lineedit.setText("(1+0j),0j,0j,0j,0j,0j,0j,0j,0j,0j,0j,0j;0j,(1+0j),0j,0j,0j,0j,0j,0j,0j,0j,0j,0j;0j,0j,(1+0j),0j,0j,0j,0j,0j,0j,0j,0j,0j;0j,0j,0j,(1+0j),0j,0j,0j,0j,0j,0j,0j,0j;0j,0j,0j,0j,(1+0j),0j,0j,0j,0j,0j,0j,0j;0j,0j,0j,0j,0j,(1+0j),0j,0j,0j,0j,0j,0j;0j,0j,0j,0j,0j,0j,(1+0j),0j,0j,0j,0j,0j;0j,0j,0j,0j,0j,0j,0j,(1+0j),0j,0j,0j,0j;0j,0j,0j,0j,0j,0j,0j,0j,(1+0j),0j,0j,0j;0j,0j,0j,0j,0j,0j,0j,0j,0j,(1+0j),0j,0j;0j,0j,0j,0j,0j,0j,0j,0j,0j,0j,(1+0j),0j;0j,0j,0j,0j,0j,0j,0j,0j,0j,0j,0j,(1+0j)")

    # --- Antenna patterns ---
    QtUI.tx_pattern.setCurrentText("Directional-Sinc")
    QtUI.tx_gain.setValue(6)
    QtUI.tx_az_bw.setValue(60)
    QtUI.tx_el_bw.setValue(60)
    QtUI.rx_pattern.setCurrentText("Directional-Sinc")
    QtUI.rx_gain.setValue(6)
    QtUI.rx_az_bw.setValue(60)
    QtUI.rx_el_bw.setValue(60)
    # --- Noise & ADC ---
    QtUI.rf_filter_bw.setValue(12.5)
    QtUI.rf_nf.setValue(14)
    QtUI.temp_k.setValue(300)
    QtUI.adc_pk2pk.setValue(2)
    QtUI.adc_levels.setValue(256)
    QtUI.adc_imp.setValue(300)
    QtUI.adc_lna.setValue(50)
    QtUI.adc_sat.setChecked(False)

    # --- Range-Doppler processing ---
    QtUI.range_window.setCurrentText("Hamming")
    QtUI.Range_FFT_points.setValue(0)
    QtUI.Range_Start.setValue(0)
    QtUI.Range_End.setValue(100)
    QtUI.pulse_buffering.setChecked(False)
    QtUI.clutter_removal.setChecked(False)
    QtUI.dopplerprocessing_method.setCurrentText("Simple FFT")
    QtUI.n_pulse.setValue(768)
    QtUI.doppler_window.setCurrentText("Hamming")
    QtUI.Doppler_FFT_points.setValue(0)
    QtUI.logscale.setChecked(False)
    QtUI.RangeDopplerCFARMean.setChecked(True)
    QtUI.cfar_rd_type.setCurrentText("Fixed Threshold a*KSort")
    QtUI.CFAR_RD_training_cells.setText("40,40")
    QtUI.CFAR_RD_guard_cells.setText("10,10")
    QtUI.CFAR_RD_alpha.setText("5")

    # --- Angle processing ---
    QtUI.azimuth_window.setCurrentText("Hamming")
    QtUI.azimuth_FFT_points.setValue(0)
    QtUI.Elevation_window.setCurrentText("Hamming")
    QtUI.Elevation_FFT_points.setValue(0)
    QtUI.spectrum_angle_type.setCurrentText("FFT")
    QtUI.CaponAzimuth.setText("-60:3:60:1")
    QtUI.CaponElevation.setText("-60:3:60:1")
    
    QtUI.CaponDL.setText("2")
    QtUI.cfar_angle_type.setCurrentText("Fixed Threshold a*KSort")
    QtUI.CFAR_Angle_training_cells.setText("40,40")
    QtUI.CFAR_Angle_guard_cells.setText("10,10")
    QtUI.CFAR_Angle_alpha.setText("2")

    # --- Simulation BlenderAddon ---
    QtUI.save_t.setChecked(True)
    QtUI.continuous_cpi.setChecked(False)
    QtUI.starttime.setText("0.000000")

