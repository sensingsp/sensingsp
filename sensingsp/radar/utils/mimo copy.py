
import numpy as np
from scipy.linalg import hadamard
import scipy.special as sp
import scipy.stats as stats
import scipy.signal.windows as scipy_windows
import math
import numba

class MIMO_Functions:
  
    @numba.jit(nopython=True)
    def sv(self, az, fd, W):
        NPulse = W.shape[0]
        M = W.shape[1]
        a = np.exp(1j * np.pi * np.arange(M) * np.sin(np.deg2rad(az)))
        d = np.exp(1j * 2 * np.pi * np.arange(NPulse) * fd)
        s = (W @ a) * d
        return s

    @numba.jit(nopython=True)
    def AD_matrix(self, NPulse, M, tech='TDM'):
        W = np.zeros((NPulse, M), dtype=np.complex128)
        if tech == "DDM":
            for p in range(NPulse):
                for m in range(M):
                    W[p, m] = np.exp(1j * 2 * np.pi * m * p / M)
        elif tech == "DDM2":
            for p in range(NPulse):
                for m in range(M):
                    W[p, m] = np.exp(1j * 2 * np.pi * m * p / NPulse)
        elif tech == "DDM_RandPhase":
            for p in range(NPulse):
                for m in range(M):
                    W[p, m] = np.exp(1j * 2 * np.pi * np.random.rand())
        elif tech == "TDM":
            for p in range(NPulse):
                for m in range(M):
                    W[p, m] = 0
                    if p % M == m:
                        W[p, m] = 1
        elif tech == "BPM":
            H = hadamard(M, dtype=complex)
            for p in range(NPulse):
                for m in range(M):
                    W[p, m] = H[p % M, m]
        else:
            raise ValueError("Not Valid!")
        return W

@numba.jit(nopython=True)
def generate_complex_gaussian_samples(cholesky_covariance_matrix, K):
    N = cholesky_covariance_matrix.shape[0]
    real_part = np.random.randn(N, K)
    imag_part = np.random.randn(N, K)
    Z = (real_part + 1j * imag_part) / np.sqrt(2)
    samples = cholesky_covariance_matrix @ Z
    return samples

@numba.jit(nopython=True)
def array_detector_ABC(A, B, C):
    return A @ B @ C

@numba.jit(nopython=True)
def array_detector_ABC_inv(A, B, C):
    return np.linalg.pinv(array_detector_ABC(A, B, C))

@numba.jit(nopython=True)
def array_detector_Hermitian(A):
    return np.conj(A.T)

@numba.jit(nopython=True)
def array_detector_MF(x, Rinv, S):
    v = array_detector_ABC(array_detector_Hermitian(S), Rinv, x)
    A = array_detector_ABC(array_detector_Hermitian(S), Rinv, S)
    A = np.linalg.pinv(A)
    t = array_detector_ABC(array_detector_Hermitian(v), A, v)[0, 0]
    return np.abs(t)

@numba.jit(nopython=True)
def MF_threshold(Q, pfa):
    # Q is the degrees of freedom (related to the number of samples), pfa is the desired false alarm probability
    return stats.chi2.ppf(1 - pfa, 2 * Q) / 2

@numba.jit(nopython=True)
def MF_pfa(Q, threshold):
    return 1 - stats.chi2.cdf(threshold * 2, 2 * Q)

@numba.jit(nopython=True)
def MF_pd(Q, threshold, SCINR_Post):
    return 1 - stats.ncx2.cdf(x=2 * threshold, df=2 * Q, nc=2 * SCINR_Post)

@numba.jit(nopython=True)
def array_detector_ED(x, Rinv):
    t = array_detector_ABC(array_detector_Hermitian(x), Rinv, x)[0, 0]
    return np.abs(t)

@numba.jit(nopython=True)
def array_detector_ACE(x, Rinv, S):
    return array_detector_MF(x, Rinv, S) / array_detector_ED(x, Rinv)

@numba.jit(nopython=True)
def array_detector_Kelly(x, Rinv, S):
    return array_detector_MF(x, Rinv, S) / (1 + array_detector_ED(x, Rinv))

@numba.jit(nopython=True)
def Kelly_pfa(Q, NSD, NDim, threshold):
    return 1 - sp.betainc(Q, NSD - NDim + 1, threshold)

@numba.jit(nopython=True)
def Kelly_threshold(Q, NSD, NDim, pfa):
    return sp.betaincinv(Q, NSD - NDim + 1, 1 - pfa)

@numba.jit(nopython=True)
def array_detector_Rao(x, R, S):
    Rinv = np.linalg.pinv(R + array_detector_Hermitian(x) @ x)
    v = array_detector_ABC(array_detector_Hermitian(S), Rinv, x)
    A = array_detector_ABC(array_detector_Hermitian(S), Rinv, S)
    A = np.linalg.pinv(A)
    t = array_detector_ABC(array_detector_Hermitian(v), A, v)[0, 0]
    return np.abs(t)

@numba.jit(nopython=True)
def array_detector_MMED(R):
    eigenvalues = np.linalg.eigvals(R)
    return np.abs(np.max(eigenvalues) / np.min(eigenvalues))

@numba.jit(nopython=True)
def array_detector_MED(x, R):
    eigenvalues = np.linalg.eigvals(R)
    e = array_detector_Hermitian(x) @ x
    return np.abs(e[0, 0] / np.min(eigenvalues))

@numba.jit(nopython=True)
def array_detector_RankOne_SingleShot(X):
    R = X @ array_detector_Hermitian(X)
    eigenvalues = np.linalg.eigvals(R)
    return np.abs(np.max(eigenvalues) / np.trace(R))

@numba.jit(nopython=True)
def array_detector_SCM_N(X):
    return X @ array_detector_Hermitian(X)

@numba.jit(nopython=True)
def array_detector_cos2(S, Sp, Rinv):
    A = array_detector_ABC(array_detector_Hermitian(S), Rinv, S)
    B = array_detector_ABC(array_detector_Hermitian(S), Rinv, Sp)
    BhAinvB = array_detector_ABC(array_detector_Hermitian(B), np.linalg.pinv(A), B)
    C = array_detector_ABC(array_detector_Hermitian(Sp), Rinv, Sp)
    ones = np.ones((S.shape[1], 1))
    num = array_detector_Hermitian(ones) @ BhAinvB @ ones
    denum = array_detector_Hermitian(ones) @ C @ ones
    return np.abs(num[0, 0] / denum[0, 0])

@numba.jit(nopython=True)
def array_detector_SINR(S, alpha, Rinv):
    A = array_detector_ABC(array_detector_Hermitian(S), Rinv, S)
    SINR = array_detector_ABC(array_detector_Hermitian(alpha), A, alpha)
    return np.abs(SINR[0, 0])

@numba.jit(nopython=True)
def array_detector_ROC_MonteCarlo_sortxH0(xH0, xH1, N=200):
    THR_v = xH0   
    THR_v = THR_v[THR_v.argsort()]
    pfa_v = np.logspace(0, -np.log10(THR_v.shape[0]), N)
    indices = []
    Pd_o  = []
    Pfa_o = []
    for pfai in pfa_v:
        ind = round((1 - pfai) * (THR_v.shape[0] - 1))
        if ind not in indices:
            indices.append(ind)
            THR = THR_v[ind]
            pfa = np.sum(xH0 > THR) / THR_v.shape[0]
            pd = np.sum(xH1 > THR) / THR_v.shape[0]
            Pd_o.append(pd)
            Pfa_o.append(pfa)
    return np.array(Pfa_o), np.array(Pd_o)

@numba.jit(nopython=True)
def array_detector_ROC_AMF_ACE_MonteCarlo(Min_Pfa, S, SigmaSQRT, NSD, alpha0_SINR, NPfa=100, progbar=True):
    t_mc = []
    alpha = alpha0_SINR * np.ones((S.shape[1], 1))
    signal_H1_0 = S @ alpha
    N_MC_1 = int(NPfa / Min_Pfa)
    mc_100 = 0
    for mc in range(N_MC_1):
        if progbar:
            if mc_100 < mc * 100 / N_MC_1:
                mc_100 += 1
                print(mc_100)
        noiseSD = generate_complex_gaussian_samples(SigmaSQRT, NSD)
        NSigmaHat = noiseSD @ np.conj(noiseSD.T)
        NDim = NSigmaHat.shape[0]
        NSigmaHat_DL = NSigmaHat + 0 * np.eye(NDim)
        NSigmaHat_DL_inv = np.linalg.pinv(NSigmaHat_DL)
        N_Trick_MC = 1
        for _ in range(N_Trick_MC):
            noiseH0 = generate_complex_gaussian_samples(SigmaSQRT, 1)
            signal_H0 = noiseH0
            t0_ACE = array_detector_ACE(signal_H0, NSigmaHat_DL_inv, S)
            t0_AMF = array_detector_MF(signal_H0, NSigmaHat_DL_inv, S)
            noiseH1 = generate_complex_gaussian_samples(SigmaSQRT, 1)
            signal_H1 = noiseH1 + signal_H1_0 
            t1_ACE = array_detector_ACE(signal_H1, NSigmaHat_DL_inv, S)
            t1_AMF = array_detector_MF(signal_H1, NSigmaHat_DL_inv, S)
            t_mc.append([t0_AMF, t0_ACE, t1_AMF, t1_ACE])
    t_mc = np.array(t_mc)
    Pfa_AMF, Pd_AMF = array_detector_ROC_MonteCarlo_sortxH0(t_mc[:, 0], t_mc[:, 2])
    Pfa_ACE, Pd_ACE = array_detector_ROC_MonteCarlo_sortxH0(t_mc[:, 1], t_mc[:, 3])
    return [Pfa_AMF, Pd_AMF], [Pfa_ACE, Pd_ACE]
