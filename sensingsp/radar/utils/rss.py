import numpy as np

def CFAR_Window_Selection_F( Len, indices_fd, Guard_Len, Wing_Len):
  selected_indices = []

  min1 = min(indices_fd) - Guard_Len
  min2 = min1 - Wing_Len
  max1 = max(indices_fd) + Guard_Len
  max2 = max1 + Wing_Len

  for i in range(Len):
      if i >= min2 and i < min1:
          selected_indices.append(i)
      elif i > max1 and i <= max2:
          selected_indices.append(i)

  return np.array(selected_indices)
def find_indices_within_distance(d_fft, R, Res):
  if R<d_fft[0]-Res:
    return np.array([])
  if R>d_fft[-1]+Res:
    return np.array([])

  # Compute the absolute differences
  differences = np.abs(d_fft - R)

  # Find indices where the difference is less than Res
  indices_within_distance = np.where(differences < Res)[0]

  # If there are no indices within the specified distance, find the index of minimum distance
  if indices_within_distance.size == 0:
      min_distance_index = np.argmin(differences)
      return np.array([min_distance_index])
  else:
      return indices_within_distance



def apply_adc(signal, peak2peak, levels,ImpedanceFactor,LNA_Gain,SaturationEnabled):
  signal *= ImpedanceFactor*LNA_Gain
  return apply_adc_real(np.real(signal),peak2peak, levels,SaturationEnabled)+1j*apply_adc_real(np.imag(signal),peak2peak, levels,SaturationEnabled)
def apply_adc_real(signal, peak2peak, levels,SaturationEnabled):        
  signal = (signal + (peak2peak / 2)) / peak2peak
  if SaturationEnabled:
    signal = np.clip(signal, 0, 1)
  signal = np.round(signal * (levels - 1)-(levels - 1)/2)
  signal /= 2*levels
  return signal

