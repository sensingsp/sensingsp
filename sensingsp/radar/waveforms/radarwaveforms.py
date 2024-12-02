import numpy as np

def barker_code(n):
    """Generate a Barker code of length n."""
    barker_codes = {
        2: np.array([1, -1]),
        3: np.array([1, 1, -1]),
        4: np.array([1, 1, -1, 1]),
        5: np.array([1, 1, 1, -1, 1]),
        7: np.array([1, 1, 1, -1, -1, 1, -1]),
        11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
        13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    }
    return barker_codes.get(n, None)

def frank_code(order):
    """Generate a Frank code of given order."""
    phase_shifts = np.linspace(0, 2 * np.pi, order, endpoint=False)
    code_matrix = np.exp(1j * np.outer(np.arange(order), phase_shifts))
    return code_matrix.flatten()


def golomb_sequence(p):
    """Generate a Golomb sequence using a primitive polynomial of order p."""
    if p not in (3, 5, 7):
        raise ValueError("Golomb sequences are typically generated for p = 3, 5, 7.")

    if p == 3:
        primitive_polynomial = [1, 0, 1]  # x^3 + x + 1
    elif p == 5:
        primitive_polynomial = [1, 0, 0, 1, 1]  # x^5 + x^2 + 1
    elif p == 7:
        primitive_polynomial = [1, 0, 0, 0, 0, 1, 1]  # x^7 + x^3 + 1

    # Initialize the register
    register = np.ones(p, dtype=int)
    sequence = []

    # Generate the sequence
    for _ in range(2**p - 1):
        feedback = register[-1]
        sequence.append(feedback)
        register[1:] = register[:-1]
        register[0] = 0
        for i in range(p):
            register[0] ^= primitive_polynomial[i] & feedback

    return np.array(sequence)

def generate_pulse_train(t, x, z=2, N=5):
    # Create a zero array of size `z * len(x)`
    zx = np.zeros(int(z * len(x)), dtype=x.dtype)
    
    # Concatenate the original signal with the zero array to form xc
    xc = np.concatenate((x, zx))
    
    # Initialize the output pulse train with the original signal
    pulse_train = xc.copy()
    
    # Append `N-1` additional repetitions of xc to form the pulse train
    for i in range(N - 1):
        pulse_train = np.concatenate((pulse_train, xc))
    
    # Generate a new time array corresponding to the pulse train length
    t_new = np.arange(len(pulse_train)) * (t[1] - t[0])
    
    return t_new, pulse_train

def generate_fmcw_signal(f0 = 77e9, B=4e9, T=40e-6, RF1_BaseBand0=True,Nyq = 1):    
    if RF1_BaseBand0:
        fs = 2*(f0+B) * Nyq 
        t = np.arange(0, T, 1 / fs)
        phase_t = 2 * np.pi * f0 * t 
        phase_t += np.pi * B / T * t**2
        x = np.cos( phase_t )
    else:
        fs = Nyq * B
        t = np.arange(0, T, 1 / fs)
        phase_t =  np.pi * B / T * t**2
        x = np.exp(1j * phase_t)
    return t , x , fs