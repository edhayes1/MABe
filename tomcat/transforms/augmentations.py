# Adapted from TREBA
# "TREBA" by Sun, Jennifer J and Kennedy, Ann and Zhan, Eric and Anderson, David J and Yue, Yisong and Perona, Pietro is licensed under CC BY-NC-SA 4.0 license.
# https://github.com/neuroethology/TREBA/blob/c522e169738f5225298cd4577e5df9085130ce8a/util/datasets/mouse_v1/augmentations/augmentation_functions.py

import numpy as np
from tomcat.consts import DEFAULT_GRID_SIZE
from typing import Dict
import torch
from torchvision import transforms


class Tensor:
    def __call__(self, keypoints: np.ndarray) -> torch.tensor:
        return torch.tensor(keypoints)

class Flatten:
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        return keypoints.reshape(keypoints.shape[0], -1)

class Scale:
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        return keypoints / DEFAULT_GRID_SIZE

class GaussianNoise:
    def __init__(self, p=0.5, mu=0, sigma=2) -> None:
        self.p = p
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            noise = np.random.normal(self.mu, self.sigma, keypoints.shape).astype(np.float32)
            keypoints = keypoints + noise

        return keypoints
        
class Rotation:
    def __init__(self, p=0.5, rotation_range=np.pi) -> None:
        self.p = p
        self.rotation_range = rotation_range

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return keypoints

        original = keypoints.copy()
        keypoints = keypoints.transpose(1,0,2,3)

        image_center = [DEFAULT_GRID_SIZE/2, DEFAULT_GRID_SIZE/2]    

        mouse_rotation = np.repeat(np.random.uniform(low = -1*self.rotation_range, high = self.rotation_range), keypoints.shape[1])
        R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                    [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))
        
        keypoints[0] = np.matmul(R, (keypoints[0] - image_center).transpose(0, 2, 1)).transpose(0,2,1) + image_center
        keypoints[1] = np.matmul(R, (keypoints[1] - image_center).transpose(0, 2, 1)).transpose(0,2,1) + image_center

        # Check if possible for trajectory to fit within borders
        bounded = ((np.amax(keypoints[:, :, :, 0]) - np.amin(keypoints[:, :, :, 0])) < DEFAULT_GRID_SIZE) and ((np.amax(keypoints[:, :, :, 1]) - np.amin(keypoints[:, :, :, 1])) < DEFAULT_GRID_SIZE)

        if bounded:
            # Shift all points to within borders first
            horizontal_shift = np.amax(keypoints[:, :, :, 0] - DEFAULT_GRID_SIZE)
            horizontal_shift_2 = np.amin(keypoints[:, :, :, 0])
            if horizontal_shift > 0:
                keypoints[:, :, :, 0] = keypoints[:, :, :, 0] - horizontal_shift
            if horizontal_shift_2 < 0:
                keypoints[:, :, :, 0] = keypoints[:, :, :, 0] - horizontal_shift_2
        
            vertical_shift = np.amax(keypoints[:, :, :, 1] - DEFAULT_GRID_SIZE)
            vertical_shift_2 = np.amin(keypoints[:, :, :, 1])
            if vertical_shift > 0:
                keypoints[:, :, :, 1] = keypoints[:, :, :, 1] - vertical_shift
            if vertical_shift_2 < 0:
                keypoints[:, :, :, 1] = keypoints[:, :, :, 1] - vertical_shift_2
        

            max_horizontal_shift = np.amin(DEFAULT_GRID_SIZE - keypoints[:, :, :, 0])
            min_horizontal_shift = np.amin(keypoints[:, :, :, 0])
            max_vertical_shift = np.amin(DEFAULT_GRID_SIZE - keypoints[:, :, :, 1])
            min_vertical_shift = np.amin(keypoints[:, :, :, 1])
            horizontal_shift = np.random.uniform(low = -1*min_horizontal_shift, high = max_horizontal_shift)
            vertical_shift = np.random.uniform(low = -1*min_vertical_shift, high = max_vertical_shift)

            keypoints[:, :, :, 0] = keypoints[:, :, :, 0] + horizontal_shift
            keypoints[:, :, :, 1] = keypoints[:, :, :, 1] + vertical_shift

            keypoints = keypoints.transpose(1,0,2,3)

            return keypoints
        else:
            return original

class Reflect:
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def reflect_points(self, points, A, B, C):
        # A * x + B * y + C = 0
        new_points = np.zeros(points.shape)

        M = np.sqrt(A*A + B*B)
        A = A/M
        B = B/M
        C = C/M

        D = A * points[:, :, :, 0] + B * points[:, :, :, 1] + C

        new_points[:, :, :, 0] = points[:, :, :, 0] - 2 * A * D
        new_points[:, :, :, 1] = points[:, :, :, 1] - 2 * B * D

        return new_points

    def __call__(self, keypoints):
        if np.random.random() > self.p:
            return keypoints
        
        if np.random.random() > 0.5:
            new_keypoints = self.reflect_points(keypoints, 0, 1, -DEFAULT_GRID_SIZE//2)  
        else:
            new_keypoints = self.reflect_points(keypoints, 1, 0, -DEFAULT_GRID_SIZE//2) 
        return new_keypoints

training_augmentations = transforms.Compose([
                                Rotation(p=0.5),
                                GaussianNoise(p=0.5),
                                Reflect(p=0.5)
                            ])
                            