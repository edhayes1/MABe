# "TREBA" by Sun, Jennifer J and Kennedy, Ann and Zhan, Eric and Anderson, David J and Yue, Yisong and Perona, Pietro is licensed under CC BY-NC-SA 4.0 license.
# https://github.com/neuroethology/TREBA/blob/c522e169738f5225298cd4577e5df9085130ce8a/util/datasets/mouse_v1/augmentations/augmentation_functions.py

# Baseline code (not used)

import numpy as np
from tomcat.consts import DEFAULT_GRID_SIZE, NUM_MICE, BODY_PART_2_INDEX
from sklearn.decomposition import TruncatedSVD
import copy
from scipy.interpolate import interp1d

def fill_holes(data):
    '''Interpolate missing data. Stolen from notebook'''
    clean_data = copy.deepcopy(data)

    for m in range(NUM_MICE):
        mouse = clean_data[:, m]
        shape = mouse.shape
        mouse = mouse.reshape(shape[0], -1)

        indexes = np.arange(shape[0])
        good_indexes, = np.where((mouse != 0).all(axis=1))

        if good_indexes.shape[0] < 20:
            clean_data[:, m] = np.zeros(shape)
            continue

        f = interp1d(
            good_indexes, 
            mouse[good_indexes], 
            bounds_error=False, 
            copy=False, 
            fill_value=(mouse[good_indexes[0]], mouse[good_indexes[-1]]), 
            kind='linear',
            axis=0
        )

        clean_data[:, m] = f(indexes).reshape(shape)

    return clean_data


def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[1] // 2
    shift = [int(DEFAULT_GRID_SIZE / 2), int(DEFAULT_GRID_SIZE / 2)] * state_dim
    scale = [int(DEFAULT_GRID_SIZE / 2), int(DEFAULT_GRID_SIZE / 2)] * state_dim
    return np.divide(data - shift, scale)

def transform_to_centered_data(data, center_index):
    # data shape is seq_len, 3, 12, 2 -> seq_len*3, 12, 2
    data = data.reshape(-1, *data.shape[2:])

    # Center the data using given center_index
    mouse_center = data[:, center_index, :]
    centered_data = data - mouse_center[:, np.newaxis, :]

    # Rotate such that keypoints Tail base and neck are parallel with the y axis
    tail_base = BODY_PART_2_INDEX['tail_base']
    neck = BODY_PART_2_INDEX['neck']
    mouse_rotation = np.arctan2(
        data[:, tail_base, 0] - data[:, neck, 0], data[:, tail_base, 1] - data[:, neck, 1])

    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))

    # Encode mouse rotation as sine and cosine
    mouse_rotation = np.concatenate([np.sin(mouse_rotation)[:, np.newaxis], np.cos(
        mouse_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))

    centered_data = centered_data.reshape((-1, 24))

    # mean = np.mean(centered_data, axis=0)
    # centered_data = centered_data - mean
    return mouse_center, mouse_rotation, centered_data

def transform_to_svd_components(data,
                                center_index=7,
                                svd_computer=None
                                ):
    seq_len, num_mice = data.shape[:2]
    
    mouse_center, mouse_rotation, centered_data = transform_to_centered_data(data, center_index)
    # Compute SVD components
    if svd_computer:
        keypoint_data = svd_computer.transform(centered_data)
    else:
        keypoint_data = centered_data

    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([mouse_center, mouse_rotation, keypoint_data], axis=1)
    data = data.reshape(seq_len, num_mice, -1)

    return data, svd_computer

def get_svd_from_dataset(raw_keypoints,
                        center_index=7,
                        n_components=5,
                        n_iter=5,
                        svd_computer=None
                        ):
    print('*** Computing SVD ***')
    data = raw_keypoints.reshape(-1, NUM_MICE, NUM_KEYPOINTS, 2)
    data = normalize(data)
    
    _, _, centered_data = transform_to_centered_data(data, center_index)

    # Compute SVD components
    svd_computer = TruncatedSVD(n_components=n_components, n_iter=n_iter)
    svd_computer = svd_computer.fit(centered_data) 

    return svd_computer


class Features:
    def __init__(self) -> None:
        self.features = [self.center_distance, self.speed]
    
    def center_distance(self, keypoints):
        center = [DEFAULT_GRID_SIZE//2, DEFAULT_GRID_SIZE//2]
        keypoints = keypoints.mean(2)
        dist = np.linalg.norm(keypoints-center, axis=2)
        return dist/DEFAULT_GRID_SIZE

    def speed(self, keypoints):
        keypoints = keypoints.mean(2)
        displacements = np.diff(keypoints, prepend=keypoints[None, 0], axis=0)
        displacements = np.linalg.norm(displacements, axis=2)
        return displacements
    
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        all_feats = []
        for feature in self.features:
            feat = feature(keypoints)
            all_feats.append(feat)