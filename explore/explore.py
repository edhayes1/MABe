import torch
import copy
from scipy import fftpack
import numpy as np


submission_clips = np.load('raw_data/submission_data.npy',allow_pickle=True).item()
user_train = np.load('raw_data/user_train.npy',allow_pickle=True).item()

sequence_names = list(user_train['sequences'].keys())
sequence_key = sequence_names[1]
single_sequence = user_train["sequences"][sequence_key]

def fft_filter(data):
    data = np.reshape(data, (data.shape[0], 2, 1))
    # Taken from scipy-lectures
    fft = fftpack.fft2(data)
    keep_fraction = 0.1
    fft2 = fft.copy()

    _, r, c = fft2.shape

    fft2[:,int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    fft2[:,:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

    new_data = fftpack.ifft2(fft2).real
    new_data = np.squeeze(new_data)
    return new_data


def clean_data(data):
    clean_data = copy.deepcopy(data)
    for m in range(3):
        holes = np.where(clean_data[0,m,:,0]==0)
        if not holes:
            continue
        for h in holes[0]:
            sub = np.where(clean_data[:,m,h,0]!=0)
            if(sub and sub[0].size > 0):
                clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
            else:
                return np.empty((0))
    
    for fr in range(1,np.shape(clean_data)[0]):
        for m in range(3):
            holes = np.where(clean_data[fr,m,:,0]==0)
            if not holes:
                continue
            for h in holes[0]:
                clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]

    for m in range(3):
        for kp in range(12):
            clean_data[:, m, kp, :] = fft_filter(clean_data[:, m, kp, :])

    return clean_data

keypoint_sequence = single_sequence['keypoints']
filled_sequence = clean_data(keypoint_sequence)


