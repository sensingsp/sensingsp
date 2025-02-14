from scipy.io import loadmat
from matplotlib import pyplot as plt
from matplotlib.image import imread
import os
import numpy as np
def run():
    path = "P:/datasets/Carry Object/2021_05_11_ph_oc001/"
    path2 = os.path.join(path, "images_0")
    path = os.path.join(path, "radar_raw_frame")
    file = os.path.join(path, "000004.mat")
    data = loadmat(file)
    data = data['adcData']
    print(data.shape) # samples (256), chirps (61), receivers (16), transmitters (12)

    tx_positions = [
        (1,11, 6),
        (2,10, 4),
        (3,9 , 1),
        (4,32, 0),
        (5,28, 0),
        (6,24, 0),
        (7,20, 0),
        (8,16, 0),
        (9,12, 0),
        (10,8, 0),
        (11,4, 0),
        (12,0, 0)
    ]
    bx = -17
    by = 34
    rx_positions = [
        (1,0, 0),
        (2,1, 0),
        (3,2, 0),
        (4,3, 0),
        (5,53-3, 0),
        (6,53-2, 0),
        (7,53-1, 0),
        (8,53, 0),
        (9,11, 0),
        (10,11+1, 0),
        (11,11+2, 0),
        (12,11+3, 0),
        (13,46, 0),
        (14,46+1, 0),
        (15,46+2, 0),
        (16,46+3, 0),
    ]
    x,y=[],[]
    for tx in tx_positions:
        for rx in rx_positions:
            x.append(tx[1]+rx[1])
            y.append(tx[2]+rx[2])
    dall=[]
    vaitxirx = []
    for itx,tx in enumerate(tx_positions):
        for irx,rx in enumerate(rx_positions):
            if tx[2]+rx[2]==0:
                d = tx[1]+rx[1]
                if d in dall:
                    continue
                dall.append(d)
                vaitxirx.append([d,itx,irx])
                # print(f"tx{itx+1} -> rx{irx+1} = {d}")
    vaitxirx = np.array(vaitxirx)
    isort = np.argsort(vaitxirx[:,0])
    vaitxirx = vaitxirx[isort]
                

    fig, axs = plt.subplots(1, 3, figsize=(18, 12))
        
    for ifile in range(148):
        file = os.path.join(path, f"{ifile+2:06}.mat")
        file2 = os.path.join(path2, f"{ifile+2:010}.jpg")
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        data = loadmat(file)
        data = data['adcData']

        data = np.fft.fft(data,axis=0,n=4*data.shape[0])
        ranges = np.arange(data.shape[0])*.06/4
        data = np.fft.fftshift(np.fft.fft(data,axis=1,n=data.shape[1]),axes=1)
        rangedopplermap = np.abs(np.sum(data,axis=(2,3)))
        rangedopplermap = rangedopplermap/np.max(rangedopplermap)
        rangedopplermap = 20*np.log10(rangedopplermap)
        axs[0].imshow(rangedopplermap)
        # plt.show()
        if 0:
            if 0:
                data = data.transpose(0,1,3,2).reshape(data.shape[0],data.shape[1],data.shape[2]*data.shape[3]) # 4 times
            else:
                data = data.reshape(data.shape[0],data.shape[1],data.shape[2]*data.shape[3]) # 8 times
        else:
            if 1:
                data = data[:, :, vaitxirx[:,2], vaitxirx[:,1]]
            else:
                data = data[:, :, vaitxirx[:,2], vaitxirx[:,1]]

        data = np.fft.fftshift(np.fft.fft(data,axis=2,n=4*data.shape[2]),axes=2)
        
        d_Wavelength = .5
        normalized_freq = np.arange(-data.shape[2]/2, data.shape[2]/2) / data.shape[2] 
        sintheta = normalized_freq / d_Wavelength
        azimuths = np.arcsin(sintheta)
        r, theta = np.meshgrid(ranges, azimuths)
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        RangeAngleMap = 20*np.log10(np.sum(np.abs(data),axis=1))
        RangeAngleMap = RangeAngleMap.T
        # x = np.concatenate((x, x[::-1,:]), axis=0)
        # y = np.concatenate((y, -y[::-1,:]), axis=0)
        # RangeAngleMap = np.concatenate((RangeAngleMap, RangeAngleMap[::-1,:]), axis=0)

        RangeAngleMap -= np.max(RangeAngleMap)
        # With phase compensation (sharp target)

        mesh = axs[1].pcolormesh(x, y, RangeAngleMap , vmin=-20, vmax=0, shading='auto', cmap='hot')
        # axs[1].set_title("Range-azimuth map with phase compensation")
        axs[1].axis('equal')
        axs[1].set_xlabel("y (m)")
        axs[1].set_ylabel("x (m)")
        # fig.colorbar(mesh, ax=axs[1], label="Amplitude")
        # axes.plot(rangeTarget*np.sin(np.deg2rad(azimuth)),rangeTarget*np.cos(np.deg2rad(azimuth)),'r*')

        # plt.tight_layout()
        axs[1].set_xlim(-2,2)
        # axes.set_ylim(0,2*8)
        
        axs[1].set_title(f"frame {ifile+2}")
        if os.path.exists(file2):
            img = imread(file2)
            axs[2].imshow(img)
            axs[2].axis('off')  # Hide axes
        # rangeanglemap = np.abs(np.sum(data,axis=(1)))
        # rangeanglemap = rangeanglemap/np.max(rangeanglemap)
        # rangeanglemap = 20*np.log10(rangeanglemap)
        # axs[1].imshow(rangeanglemap)
        plt.draw()  
        plt.gcf().canvas.flush_events() 
        plt.pause(.001)
    plt.show()

