
import sensingsp as ssp
import numpy as np
from mathutils import Vector
import torch
import numpy as np
from scipy.signal import fftconvolve

# --------------------- Environment & Ray Tracing ---------------------
class Plane:
    def __init__(self, p0, normal, eps_r=5.0, sigma=0.01):
        self.p0 = np.array(p0, dtype=float)
        self.n = np.array(normal, dtype=float) / np.linalg.norm(normal)
        self.eps_r = eps_r
        self.sigma = sigma


def reflect_point(src, p0, n):
    """Reflect point src across plane defined by point p0 and normal n."""
    v = src - p0
    return src - 2 * np.dot(v, n) * n


def path_length_and_validity(src, rx, planes_sequence):
    """
    Check that the ray from src to rx reflects off each plane in planes_sequence in order.
    Return (distance, valid).
    Simplified: only checks line-of-sight for each segment, without occluders.
    """
    pts = [src]
    for pl in planes_sequence:
        # find intersection point of ray with plane
        ray_dir = pts[-1] - rx
        denom = np.dot(ray_dir, pl.n)
        if np.isclose(denom, 0):
            return None, False
        t = np.dot(pl.p0 - rx, pl.n) / denom
        P = rx + t * ray_dir
        # check if P lies in plane bounds: omitted, assume infinite plane
        pts.append(P)
    pts.append(rx)
    # compute total distance
    d = sum(np.linalg.norm(pts[i] - pts[i+1]) for i in range(len(pts)-1))
    return d, True


def image_method(tx, rx, planes, fc=5.8e9, max_order=2):
    """
    Compute paths using the image method up to max_order reflections.
    Returns list of {'delay', 'gain'} dicts.
    """
    c = 3e8
    paths = []
    def recurse(src, order, seq):
        if order > max_order:
            return
        for pl in planes:
            img = reflect_point(src, pl.p0, pl.n)
            d, valid = path_length_and_validity(img, rx, seq + [pl])
            if valid:
                # free-space loss
                L_fs = (4*np.pi*fc*d/c)**2
                # simple reflection loss: constant factor
                L_ref = np.prod([0.8 for _ in seq + [pl]])
                gain = 1/np.sqrt(L_fs) * L_ref
                paths.append({'delay': d/c, 'gain': gain})
            recurse(img, order+1, seq + [pl])
    recurse(tx, 0, [])
    return paths

# --------------------- CIR Construction ---------------------

def build_cir(paths, Fs):
    """Build discrete-time CIR from paths list."""
    max_delay = max(p['delay'] for p in paths)
    L = int(np.ceil(max_delay * Fs)) + 1
    h = np.zeros(L, dtype=complex)
    for p in paths:
        idx = int(round(p['delay'] * Fs))
        h[idx] += p['gain']
    return h

# --------------------- OFDM Modulation (802.11n/ac) ---------------------

def ofdm_params(bw_mhz, use_sgi=False):
    """Return nfft, cp_len, data_idx, pilot_idx for given bandwidth in MHz."""
    bw = bw_mhz * 1e6
    subcar_spacing = 312.5e3
    nfft = int(bw / subcar_spacing)
    # CP length: 0.8us -> samples at 20MHz equiv = 16; SGI 0.4us -> 8
    cp = 8 if use_sgi else 16
    # define pilot & data indices (centered around nfft/2+1)
    if nfft == 64:
        pilot_locs = np.array([-21, -7, 7, 21])
    elif nfft == 128:
        pilot_locs = np.array([-53, -25, -11, 11, 25, 53])
    elif nfft == 256:
        pilot_locs = np.array([-105, -71, -7, 7, 71, 105, -49, 49])
    else:
        raise ValueError("Unsupported FFT size")
    pilot_idx = (pilot_locs + nfft//2 + 1).astype(int) - 1
    all_bins = np.arange(nfft)
    data_idx = np.setdiff1d(all_bins, np.concatenate(([nfft//2], pilot_idx)))
    return nfft, cp, data_idx, pilot_idx


def ofdm_modulate(bits, bw_mhz=20, M=4, use_sgi=False):
    """Generate OFDM baseband I/Q from bits for one symbol."""
    nfft, cp, data_idx, pilot_idx = ofdm_params(bw_mhz, use_sgi)
    # QAM mapping
    k = int(np.log2(M))
    symbols = qam_mod(bits, M)
    # prepare frequency domain grid
    X = np.zeros(nfft, dtype=complex)
    # pilot symbols: all +1
    X[pilot_idx] = 1+0j
    X[data_idx[:min(len(symbols),data_idx.shape[0])]] = symbols[:min(len(symbols),data_idx.shape[0])]
    # IFFT
    x_time = np.fft.ifft(X)
    # add cyclic prefix
    ofdm_sym = np.hstack([x_time[-cp:], x_time])
    return ofdm_sym


def qam_mod(bits, M):
    """Simple Gray-coded square QAM mapper."""
    k = int(np.log2(M))
    if len(bits) % k != 0:
        raise ValueError("Number of bits not multiple of log2(M)")
    # reshape
    bit_groups = bits.reshape((-1, k))
    # decimal mapping
    ints = bit_groups.dot(1 << np.arange(k)[::-1])
    # constellation points
    m_sqrt = int(np.sqrt(M))
    real = 2*(ints % m_sqrt) - m_sqrt + 1
    imag = 2*(ints // m_sqrt) - m_sqrt + 1
    return (real + 1j*imag) / np.sqrt(2*(m_sqrt**2 - 1)/3)

# --------------------- CSI Estimation ---------------------

def estimate_csi(rx_sym, bw_mhz=20, use_sgi=False):
    """Estimate CSI for one OFDM symbol using pilot-based interpolation."""
    nfft, cp, data_idx, pilot_idx = ofdm_params(bw_mhz, use_sgi)
    # remove CP
    y = rx_sym[cp:cp+nfft]
    Y = np.fft.fft(y)
    # pilots all 1, so CSI on pilots = Y[pilot_idx]
    H_p = Y[pilot_idx]
    # linear interpolation across full spectrum
    H_full = np.interp(np.arange(nfft), pilot_idx, H_p)
    # return CSI on data subcarriers
    return H_full[data_idx]
