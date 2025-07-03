import numpy as np
from scipy.linalg import hadamard
import scipy.special as sp
import scipy.stats as stats
import scipy.signal.windows as scipy_windows
import math
import numba
import numpy as np
# paper: caltech 2D + tx H MR. rx V MR. 
def matlab_like_hadamard(n, classname='double'):
    """
    Generate a Hadamard matrix of order n.

    Parameters:
    n (int): Order of the Hadamard matrix.
    classname (str): Type of the output matrix ('double' or 'single'). Default is 'double'.

    Returns:
    numpy.ndarray: Hadamard matrix of order n.

    Raises:
    ValueError: If n is not valid for generating a Hadamard matrix.
    """
    if classname not in ['single', 'double']:
        raise ValueError("classname must be 'single' or 'double'")

    # Check if n, n/12, or n/20 is a power of 2
    def is_power_of_two(x):
        return x > 0 and (x & (x - 1)) == 0

    candidates = [n, n // 12, n // 20]
    valid_indices = [i for i, val in enumerate(candidates) if is_power_of_two(val)]

    if len(valid_indices) == 0 or n <= 0:
        raise ValueError("Invalid input: n must be a positive integer where n, n/12, or n/20 is a power of 2.")

    k = valid_indices[0]
    e = int(np.log2(candidates[k]))

    if k == 0:  # N = 1 * 2^e
        H = np.ones((1, 1), dtype=np.float32 if classname == 'single' else np.float64)

    elif k == 1:  # N = 12 * 2^e
        base = np.array([
            [-1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1],
            [-1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1]
        ])
        toeplitz_part = np.array([np.roll(base[0], i) for i in range(len(base[0]))])
        H = np.vstack([np.ones((1, 12)), np.hstack([np.ones((11, 1)), toeplitz_part])])

    elif k == 2:  # N = 20 * 2^e
        base = np.array([
            [-1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1],
            [ 1, -1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1]
        ])
        hankel_part = np.array([np.roll(base[0], i) for i in range(len(base[0]))])
        H = np.vstack([np.ones((1, 20)), np.hstack([np.ones((19, 1)), hankel_part])])

    # Kronecker product construction
    for _ in range(e):
        H = np.block([[H, H], [H, -H]])

    return H.astype(np.float32 if classname == 'single' else np.float64)



class MIMO_Functions:
  
  def sv(self,az,fd,W):
    NPulse=W.shape[0]
    M=W.shape[1]
    a=np.exp(1j*np.pi*np.arange(M)*np.sin(np.deg2rad(az)))
    d=np.exp(1j*2*np.pi*np.arange(NPulse)*fd)
    s = (W @ a)*d
    return s

  def plot_Angle_Doppler(self,W,st,ax):
    x = self.sv(20,0,W)
    azv = np.linspace(-90,90,100)
    fdv = np.linspace(-.5,.5,500)
    Im = np.zeros((azv.shape[0],fdv.shape[0]),dtype=np.float64)
    for i in range(azv.shape[0]):
      for j in range(fdv.shape[0]):
        Im[i,j] = (np.abs(np.conj(self.sv(azv[i],fdv[j],W)).T@x))
    im = ax.imshow(Im.T, extent=(-90, 90, -0.5, 0.5), aspect='auto', cmap='viridis')  # Added cmap for better color representation
    ax.set_title(st)
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label('Log-Power (dB)')

  def AD_matrix(self,NPulse,M,tech='TDM'):
    W = np.zeros((NPulse,M),dtype=np.complex128)
    match tech:
      case "DDM":
        for p in range(NPulse):
          for m in range(M):
            W[p,m]=np.exp(1j*2*np.pi*m*p/M)
      case "DDM2":
        for p in range(NPulse):
          for m in range(M):
            W[p,m]=np.exp(1j*2*np.pi*m*p/NPulse)
      case "DDM_RandPhase":
        for p in range(NPulse):
          for m in range(M):
            W[p,m]=np.exp(1j*2*np.pi*np.random.rand())
      case "TDM":
        for p in range(NPulse):
          for m in range(M):
            W[p,m]=0
            if p%M == m:
              W[p,m]=1
      case "BPM":
        # H = hadamard(M, dtype=complex)
        H = matlab_like_hadamard(M)
        for p in range(NPulse):
          for m in range(M):
            W[p,m]=H[p%M,m]
      case _:
          print("Not Valid!")
    return W
  

def steeringVector_position(specifications,p0,Coherent_Distributed=False):
  global_location_TX,global_location_RX,global_location_Center = specifications['global_location_TX_RX_Center']
  sv = np.zeros((len(global_location_TX)*len(global_location_RX),1),dtype=complex)
  k=0
  for itx , txPos in enumerate(global_location_TX):
    dtx = np.linalg.norm(p0-txPos)
    for irx , rxPos in enumerate(global_location_RX):
      drx = np.linalg.norm(p0-rxPos)
      sv[k,0]=np.exp(1j*2*np.pi/specifications['Lambda']*(dtx+drx))
      k+=1
  if Coherent_Distributed == False:
    sv/=sv[0]
  return sv

def update_block_diag_matrix(Z, vec):
    if Z is None:
        return vec
    rowsZ, colsZ = Z.shape
    newZ = np.zeros((rowsZ + vec.shape[0], colsZ + vec.shape[1]),dtype=complex)
    newZ[:rowsZ, :colsZ] = Z
    newZ[rowsZ:, colsZ:] = vec
    return newZ

def Distributed_MIMO_steeringMatrix_position(RadarSpecifications,p0,allSuites=1):
  if allSuites==1:
    steeringMatrix = None
    for radarSpecifications in RadarSpecifications:
      for specifications in radarSpecifications:
        steeringMatrix = update_block_diag_matrix(steeringMatrix,steeringVector_position(specifications,p0))
    return steeringMatrix
  elif allSuites==2:
    steeringMatrix=[]
    for radarSpecifications in RadarSpecifications:
      steeringMatrix0 = None
      for specifications in radarSpecifications:
        steeringMatrix0 = update_block_diag_matrix(steeringMatrix0,steeringVector_position(specifications,p0))
      steeringMatrix.append(steeringMatrix0)
    return steeringMatrix
  elif allSuites==3:
    steeringMatrix=[]
    for radarSpecifications in RadarSpecifications:
      for specifications in radarSpecifications:
        steeringMatrix.append(steeringVector_position(specifications,p0))
    return steeringMatrix
  
def generate_covariance_matrix(N, rho):
    covariance_matrix = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            # covariance_matrix[i, j] = np.exp(-spread * abs(i - j))
            covariance_matrix[i, j] = (rho**abs(i - j))*np.exp(1j* np.pi/2 *(i - j))
    return covariance_matrix
  
def generate_covariance_matrix_Distributed(Nv, rhov,c=0):
  covariance_matrix = None
  for i in range(len(Nv)):
    cm = generate_covariance_matrix(Nv[i], rhov[i])
    covariance_matrix = update_block_diag_matrix(covariance_matrix,cm)
  for i in range(covariance_matrix.shape[0]):
    for j in range(i):
      if covariance_matrix[i,j]==0:
        a= c * np.exp(1j*np.random.rand()*2*np.pi)
        covariance_matrix[i,j]=a
        covariance_matrix[j,i]=np.conj(a)
  return covariance_matrix

#@numba.jit(nopython=True)
def generate_complex_gaussian_samples(cholesky_covariance_matrix, K):
    N = cholesky_covariance_matrix.shape[0]
    real_part = np.random.randn(N, K)
    imag_part = np.random.randn(N, K)
    Z = (real_part + 1j * imag_part) / np.sqrt(2)
    samples = cholesky_covariance_matrix @ Z
    return samples

def pfaTHR_tH0(tH0,pfa_v):
  N_MC = tH0.shape[0]
  detector_THR = tH0[tH0.argsort()]
  indices = []
  Pfa_THR  = []
  for pfai in pfa_v:
      ind = round((1 - pfai) * (N_MC - 1))
      if ind not in indices:
          indices.append(ind)
          THR = detector_THR[ind]
          pfa = np.sum(tH0>THR)/N_MC
          Pfa_THR.append([pfa,THR])
  return np.array(Pfa_THR)

def ROC_tH1H0(tH0H1,pfa_v):
    N_MC = tH0H1.shape[0]
    detector_THR = tH0H1[:, 0]   
    detector_THR = detector_THR[detector_THR.argsort()]
    indices = []
    Pd_Pfa_THR  = []
    for pfai in pfa_v:
        ind = round((1 - pfai) * (N_MC - 1))
        if ind not in indices:
            indices.append(ind)
            THR = detector_THR[ind]
            pfa = np.sum(tH0H1[:, 0]>THR)/N_MC
            pd  = np.sum(tH0H1[:, 1]>THR)/N_MC
            Pd_Pfa_THR.append([pd,pfa,THR])
    return np.array(Pd_Pfa_THR)
  
def ROC_tH1(tH1,pfa_v,THR_v):
    Pd_Pfa  = []
    for i in range(pfa_v.shape[0]):
        pd  = np.sum(tH1>THR_v[i])/tH1.shape[0]
        Pd_Pfa.append([pd,pfa_v[i]])
    return np.array(Pd_Pfa)
#@numba.jit(nopython=True)
def array_detector_ABC(A,B,C):
  return A@B@C

def array_detector_ABC_inv(A,B,C):
  return np.linalg.pinv(array_detector_ABC(A,B,C))

def array_detector_Hermitian(A):
  return np.conj(A.T)

## MF
#@numba.jit(nopython=True)
def array_detector_MF(x,Rinv,S):
  v = array_detector_ABC(array_detector_Hermitian(S),Rinv,x)
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  A = np.linalg.pinv(A)
  t = array_detector_ABC(array_detector_Hermitian(v),A,v)[0,0]
  return np.abs(t)

def MF_pfa(Q , threshold):
    P_fa = 1-stats.chi2.cdf(threshold*2, 2*Q)
    # P_fa = np.exp(-threshold) * sum((threshold**q) / math.factorial(q) for q in range(Q))
    return P_fa

def MF_threshold(Q,pfa):
  return stats.chi2.ppf(1 - pfa, 2 * Q) / 2

def MF_pd(Q , threshold,SCINR_Post):
  # term1 = np.sqrt(2 * SCINR_Post)
  # term2 = np.sqrt(2 * threshold)
  # non_centrality = term1**2
  # Pd = 1 - stats.ncx2.cdf(term2, 2*Q, non_centrality)
  # return Pd
  return 1-stats.ncx2.cdf(x=2 * threshold,df=2*Q,nc=2 * SCINR_Post)
    # P_d_value = sp.marcumq(Q, np.sqrt(2 * SCINR_Post), np.sqrt(2 * threshold))
    # return P_d_value

def array_detector_EstimatedAmplitude(x,Rinv,S):
  v = array_detector_ABC(array_detector_Hermitian(S),Rinv,x)
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  return np.linalg.pinv(A)@v
#@numba.jit(nopython=True)
def array_detector_ED(x,Rinv):
  t = array_detector_ABC(array_detector_Hermitian(x),Rinv,x)[0,0]
  return np.abs(t)

def ED_pfa(Q , threshold):
    P_fa = 1-stats.chi2.cdf(threshold*2, 2*Q)
    # P_fa = np.exp(-threshold) * sum((threshold**q) / math.factorial(q) for q in range(Q))
    return P_fa
#@numba.jit(nopython=True)
def array_detector_ACE(x,Rinv,S):
  return array_detector_MF(x,Rinv,S)/array_detector_ED(x,Rinv)

## Kelly
def array_detector_Kelly(x,Rinv,S):
  return array_detector_MF(x,Rinv,S)/(1+array_detector_ED(x,Rinv))

def Kelly_pfa(Q, NSD, NDim, threshold):
    P_fa = 1 - sp.betainc(Q, NSD - NDim + 1, threshold)
    return P_fa

def Kelly_threshold(Q, NSD, NDim,pfa):
  return sp.betaincinv(Q, NSD - NDim + 1, 1 - pfa)

def Kelly_DistributedIndependentSameRadars_pfa(Q, NSDi, NDimi, threshold):
    P_fa = 0
    for q in range(Q):
      P_fa += (NSDi-NDimi+1)**q / math.factorial(q) * (np.log(threshold)**q)
    P_fa *= threshold ** (-(NSDi-NDimi+1))
    return P_fa

def array_detector_Rao(x,R,S):
  Rinv = np.linalg.pinv(R+array_detector_Hermitian(x)@x)
  v = array_detector_ABC(array_detector_Hermitian(S),Rinv,x)
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  A = np.linalg.pinv(A)
  t = array_detector_ABC(array_detector_Hermitian(v),A,v)[0,0]
  return np.abs(t)

def array_detector_MMED(R):
  eigenvalues = np.linalg.eigvals(R)
  return np.abs(np.max(eigenvalues)/np.min(eigenvalues))

def array_detector_MED(x,R):
  eigenvalues = np.linalg.eigvals(R)
  e = array_detector_Hermitian(x)@x
  return np.abs( e[0,0] / np.min(eigenvalues))

def array_detector_RankOne_SingleShot(X):
  R = X @ array_detector_Hermitian(X)
  eigenvalues = np.linalg.eigvals(R)
  return np.abs( np.max(eigenvalues) / np.trace(R))

def array_detector_SCM_N(X):
  return X @ array_detector_Hermitian(X)

def array_detector_cos2(S,Sp,Rinv):
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  B = array_detector_ABC(array_detector_Hermitian(S),Rinv,Sp)
  BhAinvB = array_detector_ABC(array_detector_Hermitian(B),np.linalg.pinv(A),B)
  C = array_detector_ABC(array_detector_Hermitian(Sp),Rinv,Sp)
  ones = np.ones((S.shape[1],1))
  num = array_detector_Hermitian(ones) @ BhAinvB @ ones
  denum = array_detector_Hermitian(ones) @ C @ ones 
  return np.abs(num[0,0]/denum[0,0])

def array_detector_SINR(S,alpha,Rinv):
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  SINR = array_detector_ABC(array_detector_Hermitian(alpha),A,alpha)
  return np.abs(SINR[0,0])

def array_detector_AMFDeemphasis(x,R,S,eps_AMFD):
  Rinv = np.linalg.pinv(R+eps_AMFD*array_detector_Hermitian(x)@x)
  v = array_detector_ABC(array_detector_Hermitian(S),Rinv,x)
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  A = np.linalg.pinv(A)
  t = array_detector_ABC(array_detector_Hermitian(v),A,v)[0,0]
  return np.abs(t)

def array_detector_Kalson(x,Rinv,S,eps_Kalson):
  return array_detector_MF(x,Rinv,S)/(1+eps_Kalson*array_detector_ED(x,Rinv))

def array_detector_SD(x,Rinv,H): # [50, 51]
  return array_detector_Kelly(x,Rinv,H)

def array_detector_ABORT(x,Rinv,S):
  return (1+array_detector_MF(x,Rinv,S))/(2+array_detector_ED(x,Rinv))

def array_detector_WABORT(x,Rinv,S):
  t = (1-array_detector_Kelly(x,Rinv,S))**2
  A = array_detector_ABC(array_detector_Hermitian(S),Rinv,S)
  ones = np.ones((S.shape[1],1))
  a = np.abs(denum = array_detector_Hermitian(ones) @ A @ ones)[0,0] # Not sure
  return 1 / ((1+a)*t)

def array_detector_CAD(x,Rinv,S,eps_CAD):
  uv = array_detector_ED(x,Rinv) - array_detector_ED(x,Rinv)*(1+eps_CAD**2) # check [41, 58]
  u = 1 if uv >= 0 else 0
  t = array_detector_ED(x,Rinv) - 1/(1-eps_CAD**2)*(np.sqrt(array_detector_ED(x,Rinv)-array_detector_MF(x,Rinv,S))-eps_CAD*np.sqrt(array_detector_MF(x,Rinv,S)))**2 * u  
  return t


def array_detector_CARD(x,Rinv,S,eps_CARD):
  uv = eps_CARD * np.sqrt(array_detector_MF(x,Rinv,S))-np.sqrt(array_detector_ED(x,Rinv)-array_detector_MF(x,Rinv,S))
  u = 1 if uv >= 0 else -1
  return u * uv**2

def array_detector_2SROB(x,Rinv,S,N,K):
  p = array_detector_ED(x,Rinv)-array_detector_MF(x,Rinv,S)
  THR = N/K  # K > 2N
  if p>THR:
    return (THR/p)**THR * np.exp(-THR+array_detector_ED(x,Rinv))
  else:
    return np.exp(array_detector_MF(x,Rinv,S))

def array_detector_1SROB(x,Rinv,S,zeta):
  e = array_detector_ED(x,Rinv)
  p = e - array_detector_MF(x,Rinv,S)
  g = 1 + p
  if p >= 1/(zeta-1):
    Delta = (zeta-1)**(1/zeta) / (1-1/zeta)
    g = (1+Delta) * ( p ** (1/zeta) )
  return (1+e)/g

def array_detector_ROB(x,Rinv,S,zetaeps):
  p = array_detector_ED(x,Rinv)-array_detector_MF(x,Rinv,S)
  xh2 = array_detector_ED(x,Rinv)
  THR = 1/(zetaeps-1)
  if p>THR:
    return ( (1+xh2) * (1-1/zetaeps) ) / ( ((zetaeps-1)*p)**(1/zetaeps) )
  else:
    return (1+xh2)/(1+p)

def hamming(N):
  return scipy_windows.hamming(N)

#@numba.jit(nopython=True)
def array_detector_ROC_MonteCarlo_sortxH0(xH0,xH1,N=200):
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
            pfa = np.sum(xH0>THR)/THR_v.shape[0]
            pd = np.sum(xH1>THR)/THR_v.shape[0]
            Pd_o.append(pd)
            Pfa_o.append(pfa)
    return np.array(Pfa_o),np.array(Pd_o)

#@numba.jit(nopython=True)
def array_detector_ROC_AMF_ACE_MonteCarlo(Min_Pfa,S,SigmaSQRT,NSD,alpha0_SINR,NPfa = 100,progbar = True):
    t_mc = []
    alpha = alpha0_SINR * np.ones((S.shape[1],1))
    signal_H1_0 = S @ alpha
    N_MC_1 = int(NPfa/Min_Pfa)
    mc_100=0
    for mc in range(N_MC_1):
        if progbar:
          if mc_100 < mc * 100 / N_MC_1:
              mc_100+=1
              print(mc_100)
        noiseSD = generate_complex_gaussian_samples(SigmaSQRT,NSD)
        NSigmaHat = noiseSD @ np.conj(noiseSD.T)
        NDim=NSigmaHat.shape[0]
        NSigmaHat_DL = NSigmaHat + 0 * np.eye(NDim)
        NSigmaHat_DL_inv = np.linalg.pinv(NSigmaHat_DL)
        N_Trick_MC = 1
        for _ in range(N_Trick_MC):
            noiseH0 = generate_complex_gaussian_samples(SigmaSQRT,1)
            signal_H0 = noiseH0
            t0_ACE = array_detector_ACE(signal_H0,NSigmaHat_DL_inv,S) 
            t0_AMF = array_detector_MF(signal_H0,NSigmaHat_DL_inv,S) 
            noiseH1 = generate_complex_gaussian_samples(SigmaSQRT,1)
            signal_H1 = noiseH1 + signal_H1_0 
            t1_ACE = array_detector_ACE(signal_H1,NSigmaHat_DL_inv,S) 
            t1_AMF = array_detector_MF(signal_H1,NSigmaHat_DL_inv,S) 
            t_mc.append([t0_AMF,t0_ACE,t1_AMF,t1_ACE]) 
    t_mc = np.array(t_mc)
    Pfa_AMF,Pd_AMF=array_detector_ROC_MonteCarlo_sortxH0(t_mc[:, 0],t_mc[:, 2])
    Pfa_ACE,Pd_ACE=array_detector_ROC_MonteCarlo_sortxH0(t_mc[:, 1],t_mc[:, 3])
    return [Pfa_AMF,Pd_AMF],[Pfa_ACE,Pd_ACE]


def array_detector_ROC_Kelly_MonteCarlo(Pfa_v,S,SigmaSQRT,NSD,alpha0_SINR,NMC = [100,100],progbar = True):
  NC1 = NMC[0]
  NC2 = NMC[1]
  iC = Pfa_v * 0
  NC = Pfa_v * 0
  NDim = SigmaSQRT.shape[0]
  for _ in range(NC1):
      noiseSD = generate_complex_gaussian_samples(SigmaSQRT,NSD)
      NSigmaHat = noiseSD @ np.conj(noiseSD.T)
      NSigmaHat_DL = NSigmaHat + 0 * np.eye(NDim) 
      NSigmaHat_DL_inv = np.linalg.pinv(NSigmaHat_DL)
      for ics,s in enumerate(Pfa_v):
          thr_Kelly = Kelly_threshold(S.shape[1], NSD, NDim,s)
          alpha = alpha0_SINR * np.ones((S.shape[1],1))
          signal_H1_0 = S @ alpha
          for __ in range(NC2):
              noiseH1 = generate_complex_gaussian_samples(SigmaSQRT,1)
              signal_H1 = noiseH1 + signal_H1_0 
              t1_Kelly = array_detector_Kelly(signal_H1,NSigmaHat_DL_inv,S)
              if t1_Kelly>thr_Kelly:
                  iC[ics]+=1
              NC[ics]+=1
  pd_v = iC / NC
  return pd_v

def array_detector_findTHR_AMF_ACE(Pfa,S,SigmaSQRT,NSD,NPfa = 100,progbar = True):
    t_mc = []
    N_MC_1 = int(NPfa/Pfa)
    mc_100=0
    for mc in range(N_MC_1):
        if progbar:
          if mc_100 < mc * 100 / N_MC_1:
              mc_100+=1
              print(mc_100)
        noiseSD = generate_complex_gaussian_samples(SigmaSQRT,NSD)
        NSigmaHat = noiseSD @ np.conj(noiseSD.T)
        NDim=NSigmaHat.shape[0]
        NSigmaHat_DL = NSigmaHat + 0 * np.eye(NDim)
        NSigmaHat_DL_inv = np.linalg.pinv(NSigmaHat_DL)
        N_Trick_MC = 1
        for _ in range(N_Trick_MC):
            noiseH0 = generate_complex_gaussian_samples(SigmaSQRT,1)
            signal_H0 = noiseH0
            t0_ACE = array_detector_ACE(signal_H0,NSigmaHat_DL_inv,S) 
            t0_AMF = array_detector_MF(signal_H0,NSigmaHat_DL_inv,S) 
            t_mc.append([t0_AMF ,t0_ACE]) 

    t_mc = np.array(t_mc)
    N_MC_2 = t_mc.shape[0]
    pfa_v = np.logspace(-np.log10(N_MC_2), 0, 200)
    Pfa_THR_AMF = pfaTHR_tH0(t_mc[:, 0],pfa_v)
    Pfa_THR_ACE = pfaTHR_tH0(t_mc[:, 1],pfa_v)
    thr_AMF = Pfa_THR_AMF[np.argmin(np.abs(Pfa_THR_AMF[:,0]-Pfa)),1]
    thr_ACE = Pfa_THR_ACE[np.argmin(np.abs(Pfa_THR_ACE[:,0]-Pfa)),1]
    return thr_AMF,thr_ACE


def array_detector_PdSNR_AMF_ACE_Kelly_MonteCarlo(Fixed_Pfa,S,SigmaSQRT,NSD,SCINR_Post_v,SCINR_PostProcessingM,NMC = [100,100],progbar = True):
  NC1 = NMC[0]
  NC2 = NMC[1]
  NDim = S.shape[0]
  Q = S.shape[1]
  
  thr_AMF,thr_ACE = array_detector_findTHR_AMF_ACE(Fixed_Pfa,S,SigmaSQRT,NSD) 
  thr_Kelly = Kelly_threshold(Q, NSD, NDim,Fixed_Pfa)
  
  iC = SCINR_Post_v * 0
  iCAMF = SCINR_Post_v * 0
  iCACE = SCINR_Post_v * 0
  NC = SCINR_Post_v * 0
  for isd in range(NC1):
      noiseSD = generate_complex_gaussian_samples(SigmaSQRT,NSD)
      NSigmaHat = noiseSD @ np.conj(noiseSD.T)
      NSigmaHat_DL = NSigmaHat + 0 * np.eye(NDim) 
      NSigmaHat_DL_inv = np.linalg.pinv(NSigmaHat_DL)
      for ics,s in enumerate(SCINR_Post_v):
          alpha0 = 10**((s - SCINR_PostProcessingM)/20)
          alpha = alpha0 * np.ones((Q,1))
          signal_H1_0 = S @ alpha
          for isd2 in range(NC2):
              noiseH1 = generate_complex_gaussian_samples(SigmaSQRT,1)
              signal_H1 = noiseH1 + signal_H1_0 
              t1_Kelly = array_detector_Kelly(signal_H1,NSigmaHat_DL_inv,S)
              if t1_Kelly>thr_Kelly:
                  iC[ics]+=1
              t1_Kelly = array_detector_MF(signal_H1,NSigmaHat_DL_inv,S)
              if t1_Kelly>thr_AMF:
                  iCAMF[ics]+=1
              t1_Kelly = array_detector_ACE(signal_H1,NSigmaHat_DL_inv,S)
              if t1_Kelly>thr_ACE:
                  iCACE[ics]+=1
              NC[ics]+=1
  return iCAMF / NC,iCACE / NC,iC / NC
  
def array_detector_PdNSecondary_AMF_ACE_Kelly_MonteCarlo(Fixed_Pfa,S,SigmaSQRT,NSDTimes_v,SCINR_Post_Fixed,SCINR_PostProcessingM,NMC = [100,100],progbar = True):
  NC1 = NMC[0]
  NC2 = NMC[1]
  NDim = S.shape[0]
  Q = S.shape[1]
  thr_AMF=[]
  thr_ACE=[]
  thr_Kelly=[]
  for NSDTimes in NSDTimes_v:
    NSD = int(NSDTimes*NDim)
    thr_AMF0,thr_ACE0 = array_detector_findTHR_AMF_ACE(Fixed_Pfa,S,SigmaSQRT,NSD,NC1) 
    thr_Kelly0 = Kelly_threshold(Q, NSD, NDim,Fixed_Pfa)
    thr_AMF.append(thr_AMF0)
    thr_ACE.append(thr_ACE0)
    thr_Kelly.append(thr_Kelly0)
    
  iC = NSDTimes_v * 0
  iCAMF = NSDTimes_v * 0
  iCACE = NSDTimes_v * 0
  NC = NSDTimes_v * 0
  for ics,s in enumerate(NSDTimes_v):
      NSD =int(s*NDim)
      for isd in range(NC1):
        noiseSD = generate_complex_gaussian_samples(SigmaSQRT,NSD)
        NSigmaHat = noiseSD @ np.conj(noiseSD.T)
        NSigmaHat_DL = NSigmaHat + 0 * np.eye(NDim) 
        NSigmaHat_DL_inv = np.linalg.pinv(NSigmaHat_DL)
        alpha0 = 10**((SCINR_Post_Fixed - SCINR_PostProcessingM)/20)
        alpha = alpha0 * np.ones((Q,1))
        signal_H1_0 = S @ alpha
        for isd2 in range(NC2):
            noiseH1 = generate_complex_gaussian_samples(SigmaSQRT,1)
            signal_H1 = noiseH1 + signal_H1_0 
            t1_Kelly = array_detector_Kelly(signal_H1,NSigmaHat_DL_inv,S)
            if t1_Kelly>thr_Kelly[ics]:
                iC[ics]+=1
            t1_Kelly = array_detector_MF(signal_H1,NSigmaHat_DL_inv,S)
            if t1_Kelly>thr_AMF[ics]:
                iCAMF[ics]+=1
            t1_Kelly = array_detector_ACE(signal_H1,NSigmaHat_DL_inv,S)
            if t1_Kelly>thr_ACE[ics]:
                iCACE[ics]+=1
            NC[ics]+=1
  return iCAMF / NC,iCACE / NC,iC / NC
  
def VAsignal2coArraySignal_1D_RangeDopplerVA(virtual_array_Position, RDvirtual_array_Signal):
    coprime_differences = []  # To store unique element differences
    coprime_struct = {}  # Dictionary to store co-prime pairs
    
    # Calculate all pairwise differences for the co-prime array
    for i in range(len(virtual_array_Position)):
        for j in range(len(virtual_array_Position)):
            difference = virtual_array_Position[i] - virtual_array_Position[j]
            if difference not in coprime_differences:
                coprime_differences.append(difference)
                coprime_struct[difference] = [(i, j)]
            else:
                coprime_struct[difference].append((i, j))
    
    coprime_differences = np.array(sorted(coprime_differences))
    sorted_indices = np.argsort(coprime_differences)
    
    coArraySignal = np.zeros((RDvirtual_array_Signal.shape[0],RDvirtual_array_Signal.shape[1],len(coprime_differences)), dtype=complex)
    # co_prime_steering_vectors = np.zeros((len(coprime_differences), len(sin_angles)), dtype=complex)
    
    for i, diff in enumerate(coprime_differences):
        pairs = coprime_struct[diff]
        
        # Average over all pairs for each co-prime difference
        for idx1, idx2 in pairs:
            coArraySignal[:,:,i] += RDvirtual_array_Signal[:,:,idx1] * np.conj(RDvirtual_array_Signal[:,:,idx2])
        coArraySignal[:,:,i] /= len(pairs)  # Average
        
        # Calculate the steering vector for each co-prime difference
        # co_prime_steering_vectors[i, :] = np.exp(1j * np.pi * diff * sin_angles)
    
    coArrayStructure = [coprime_struct,coprime_differences]
    return coArraySignal,coArrayStructure
def VAsignal2coArraySignal_1D(virtual_array_Position, virtual_array_Signal):
    coprime_differences = []  # To store unique element differences
    coprime_struct = {}  # Dictionary to store co-prime pairs
    
    # Calculate all pairwise differences for the co-prime array
    for i in range(len(virtual_array_Position)):
        for j in range(len(virtual_array_Position)):
            difference = virtual_array_Position[i] - virtual_array_Position[j]
            if difference not in coprime_differences:
                coprime_differences.append(difference)
                coprime_struct[difference] = [(i, j)]
            else:
                coprime_struct[difference].append((i, j))
    
    coprime_differences = np.array(sorted(coprime_differences))
    sorted_indices = np.argsort(coprime_differences)
    
    coArraySignal = np.zeros(len(coprime_differences), dtype=complex)
    # co_prime_steering_vectors = np.zeros((len(coprime_differences), len(sin_angles)), dtype=complex)
    
    for i, diff in enumerate(coprime_differences):
        pairs = coprime_struct[diff]
        
        # Average over all pairs for each co-prime difference
        for idx1, idx2 in pairs:
            coArraySignal[i] += virtual_array_Signal[idx1] * np.conj(virtual_array_Signal[idx2])
        coArraySignal[i] /= len(pairs)  # Average
        
        # Calculate the steering vector for each co-prime difference
        # co_prime_steering_vectors[i, :] = np.exp(1j * np.pi * diff * sin_angles)
    
    coArrayStructure = [coprime_struct,coprime_differences]
    return coArraySignal,coArrayStructure
def VAsignal2coArraySignal_1D_sum(virtual_array_Position, virtual_array_Signal):
    coprime_differences = []  # To store unique element differences
    coprime_struct = {}  # Dictionary to store co-prime pairs
    
    # Calculate all pairwise differences for the co-prime array
    for i in range(len(virtual_array_Position)):
        for j in range(len(virtual_array_Position)):
            difference = virtual_array_Position[i] - virtual_array_Position[j]
            if difference not in coprime_differences:
                coprime_differences.append(difference)
                coprime_struct[difference] = [(i, j,-1)]
            else:
                coprime_struct[difference].append((i, j,-1))
            difference = virtual_array_Position[i] + virtual_array_Position[j]
            if difference not in coprime_differences:
                coprime_differences.append(difference)
                coprime_struct[difference] = [(i, j,1)]
            else:
                coprime_struct[difference].append((i, j,1))
    
    coprime_differences = np.array(sorted(coprime_differences))
    sorted_indices = np.argsort(coprime_differences)
    
    coArraySignal = np.zeros(len(coprime_differences), dtype=complex)
    # co_prime_steering_vectors = np.zeros((len(coprime_differences), len(sin_angles)), dtype=complex)
    
    for i, diff in enumerate(coprime_differences):
        pairs = coprime_struct[diff]
        
        # Average over all pairs for each co-prime difference
        for idx1, idx2,sd in pairs:
            if sd == -1:
              coArraySignal[i] += virtual_array_Signal[idx1] * np.conj(virtual_array_Signal[idx2])
            else:
              coArraySignal[i] += virtual_array_Signal[idx1] * (virtual_array_Signal[idx2])
        coArraySignal[i] /= len(pairs)  # Average
        
        # Calculate the steering vector for each co-prime difference
        # co_prime_steering_vectors[i, :] = np.exp(1j * np.pi * diff * sin_angles)
    
    coArrayStructure = [coprime_struct,coprime_differences]
    return coArraySignal,coArrayStructure

def VAsignal2coArraySignal_2D(virtual_array_Position, virtual_array_Signal=None, Option="Fwd_x"):
    coArray = []
    posPair = {}
    num_positions = len(virtual_array_Position)
    for i in range(num_positions):
        for j in range(num_positions):
            add = False
            if Option == "Fwd_x":
              if virtual_array_Position[i][0] - virtual_array_Position[j][0]>=0:
                add=True
            if Option == "Bcw_x":
              if virtual_array_Position[i][0] - virtual_array_Position[j][0]<=0:
                add=True
            if Option == "FwdBcw_x":
                add=True
            if Option == "Fwd_xy":
              if virtual_array_Position[i][0] - virtual_array_Position[j][0]>=0:
                if virtual_array_Position[i][1] - virtual_array_Position[j][1]>=0:
                  add=True
              
            if add:
                difference = virtual_array_Position[i] - virtual_array_Position[j]
                diff_tuple = tuple(difference.tolist())
                
                if diff_tuple not in coArray:
                    coArray.append(diff_tuple)
                    posPair[diff_tuple] = [(i, j,0)]
                else:
                    posPair[diff_tuple].append((i, j,0))
                    
    return None, coArray, posPair


def MIMO_minimum_redundancy_array():
  Tx = [0,1,3]
  Rx = [0,6,13,40,60]

def minimum_redundancy_array(N, array_type="restricted"):
    restricted_arrays = {
        1: [],
        5: [1, 3, 3, 2],
        6: [1, 5, 3, 2, 2],
        7: [1, 3, 6, 2, 3, 2],
        8: [1, 3, 6, 6, 2, 3, 2],
        9: [1, 3, 6, 6, 6, 2, 3, 2],
        10: [1, 2, 3, 7, 7, 7, 4, 4, 1],
        11: [1, 2, 3, 7, 7, 7, 7, 4, 4, 1]
    }

    general_arrays = {
        1: [],
        2: [1],
        3: [1, 2],
        4: [1,3,2],
        5: [4, 1, 2, 6],
        6: [6, 1, 2, 2, 8],
        7: [14, 1, 3, 6, 2, 5],
        8: [8, 10, 1, 3, 2, 7, 8],
        10: [16, 1,11, 8, 6, 4, 3,2, 22],
        11: [18, 1,3, 9, 11, 6, 8, 2, 5, 28]
    }

    # Select the correct array dictionary
    if array_type == "restricted":
        selected_arrays = restricted_arrays
    elif array_type == "general":
        selected_arrays = general_arrays
    else:
        raise ValueError("Invalid array type. Choose 'restricted' or 'general'.")

    # Return the array configuration for the given N
    if N in selected_arrays:
        return selected_arrays[N]
    else:
        raise ValueError(f"No array configuration found for N = {N}.")

def mimo_antenna_order(specifications):
  tx_positions,rx_positions = specifications["TXRXPos"]
  xs, ys = specifications['MIMO_Antenna_Azimuth_Elevation_Order']
  xs,ys=list(xs),list(ys)
  i_list = []
  j_list = []
  for m in range(len(tx_positions)):
    for n in range(len(rx_positions)):
      xv, yv = -tx_positions[m][0]+rx_positions[n][0], tx_positions[m][1]+rx_positions[n][1]
      i, j = xs.index(xv), ys.index(yv)
      if i is not None and j is not None:
        i_list.append(i)
        j_list.append(j)
  return np.array(i_list), np.array(j_list)