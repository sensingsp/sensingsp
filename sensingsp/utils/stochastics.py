import numpy as np
def Complex_Noise_Buffer(N_ADC, NPulse, NRX, T=290, B=1e6):
    """
    Generate a radar data cube buffer with complex noise based on KTB power.

    Parameters:
    N_ADC (int): Number of ADC samples (fast time).
    NPulse (int): Number of pulses (slow time).
    NRX (int): Number of receivers.
    T (float): Temperature in Kelvin. Default is 290 K.
    B (float): Bandwidth in Hz. Default is 1 MHz.

    Returns:
    np.ndarray: Complex noise buffer with dimensions (N_ADC, NPulse, NRX).
    """
    K = 1.38e-23  # Boltzmann constant in J/K
    P_noise = K * T * B  # Noise power
    sigma = np.sqrt(P_noise / 2)  # Standard deviation of noise for real and imaginary parts

    # Generate noise buffer
    real_noise = sigma * np.random.randn(N_ADC, NPulse, NRX)
    imag_noise = sigma * np.random.randn(N_ADC, NPulse, NRX)

    complex_noise_buffer = real_noise + 1j * imag_noise

    return complex_noise_buffer.astype(np.complex128)