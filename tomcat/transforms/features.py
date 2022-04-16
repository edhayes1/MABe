# Features inspired by The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice
# Cristina Segalin, Jalani Williams, Tomomi Karigo, May Hui, Moriel Zelikowsky, Jennifer J. Sun, Pietro Perona, David J. Anderson, Ann Kennedy
# https://www.biorxiv.org/content/10.1101/2020.07.26.222299v1

# "TREBA" by Sun, Jennifer J and Kennedy, Ann and Zhan, Eric and Anderson, David J and Yue, Yisong and Perona, Pietro is licensed under CC BY-NC-SA 4.0 license.
# https://github.com/neuroethology/TREBA/blob/c522e169738f5225298cd4577e5df9085130ce8a/util/datasets/mouse_v1/augmentations/augmentation_functions.py


import numpy as np
from tomcat.consts import DEFAULT_GRID_SIZE, BODY_PART_2_INDEX as B2I, NUM_MICE
import torch


class AccelerationFeature:
    def __init__(self, keypoints, center=False) -> None:
        self.center = center
        self.max = self.get_stats(keypoints, center)

    def get_stats(self, keypoints, center):
        if center:
            mouse_length = _distance(
                np.delete(keypoints, B2I['center'], 3),
                keypoints[:, :, :, B2I['center'], None, :]
            )

        else:
            mouse_length = _distance(
                keypoints[:, :, :, B2I['nose']], 
                keypoints[:, :, :, B2I['tail_base']]
            )

        return 2 * mouse_length.mean(axis=(0, 1, 2))

    def __call__(self, sequence):
        diff = _get_diff(
            sequence, 
            center=self.center,
            seq_dim=0,
            body_part_dim=2,
            padding=((1, 0), (0, 0), (0, 0)),
            clip=self.max
        )

        max = self.max / 2

        diff_diff = _get_diff(
            diff, 
            center=False,
            seq_dim=0,
            body_part_dim=2,
            padding=((1, 0), (0, 0), (0, 0)),
            clip=max,
            norm=False
        )

        return diff_diff / max
        

class VelocityFeature:
    '''
        Velocity feature for each body part.
        Optional centering takes velocity relative to center body part.
        Calculates mean and standard over the dataset at init.
        Clips at mouse length - this is chosen pretty much randomly
        Adds edge padding
    '''
    def __init__(self, keypoints, center=False) -> None:
        self.max = self.get_stats(keypoints, center)
        self.center = center
    
    def get_stats(self, keypoints, center):
        '''
            Normalise velocity by mouse length, 
            Assume mouse can't move more than one mouse length in a frame.
        '''
        if center:
            mouse_length = _distance(
                np.delete(keypoints, B2I['center'], 3),
                keypoints[:, :, :, B2I['center'], None, :]
            )

        else:
            mouse_length = _distance(
                keypoints[:, :, :, B2I['nose']], 
                keypoints[:, :, :, B2I['tail_base']]
            )

        return 2 * mouse_length.mean(axis=(0, 1, 2))

    def __call__(self, sequence) -> np.array:
        diff = _get_diff(
            sequence, 
            center=self.center,
            seq_dim=0,
            body_part_dim=2,
            padding=((1, 0), (0, 0), (0, 0))
        )
        return diff / self.max

class BodyLengthFeature:
    '''
        Body length feature.
        Find the distance between the nose and tail base.
        Could be useful when mouse is standing up.
        Calculates mean and standard over the dataset at init.
        Clip at 150.
    '''

    def __init__(self, keypoints) -> None:
        self.mean, self.std = self.get_stats(keypoints)

    def get_stats(self, keypoints):
        lengths = _distance(
            keypoints[:, :, :, B2I['nose']], 
            keypoints[:, :, :, B2I['tail_base']]
        )
        return lengths.mean(axis=(0, 1, 2)), lengths.std(axis=(0, 1, 2))
    
    def __call__(self, sequence):
        lengths = _distance(
            sequence[:, :, B2I['nose']], 
            sequence[:, :, B2I['tail_base']]
        )
        return (lengths - self.mean)/self.std


class PawToEarDistFeature:
    '''
        Calculates the average distance of the paws to the ears
        Might be useful if the mouse is cleaning itself.
    '''
    def __init__(self, keypoints) -> None:
        self.mean, self.std = self.get_stats(keypoints)

    def get_stats(self, keypoints):
        left_dist = _distance(
            keypoints[:, :, :, B2I['ear_left']], 
            keypoints[:, :, :, B2I['forepaw_left']]
        )
        right_dist = _distance(
            keypoints[:, :, :, B2I['ear_right']], 
            keypoints[:, :, :, B2I['forepaw_right']]
        )

        dist = (left_dist + right_dist)/2
        return dist.mean(axis=(0, 1, 2)), dist.std(axis=(0, 1, 2))
    
    def __call__(self, sequence):
        left_dist = _distance(
            sequence[:, :, B2I['ear_left']], 
            sequence[:, :, B2I['forepaw_left']]
        )

        right_dist = _distance(
            sequence[:, :, B2I['ear_right']], 
            sequence[:, :, B2I['forepaw_right']]
        )
        
        dist = (left_dist + right_dist)/2
        return (dist - self.mean)/self.std


class DirectionChangeFeature:
    def __call__(self, sequence) -> np.array:
        heading = _angle(sequence[:, :, B2I['nose']], sequence[:, :, B2I['center']])
        heading_change = _get_diff(heading, norm=False, padding=((1, 0), (0, 0)))
        encoded_heading_change = _encode_angle(heading_change)

        return encoded_heading_change


class TailBoxFeature:
    def __init__(self, keypoints) -> None:
        self.max_size, self.max_area = self.get_stats(keypoints)
    
    def get_stats(self, keypoints):
        bottom, top = _bounding_box(keypoints[:, :, :, -3:, :])
        lengths = np.sqrt((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2)
        areas = (top[0] - bottom[0]) * (top[1] - bottom[1])
        return lengths.max((0,1,2)), areas.max((0,1,2))


    def get_box_ratio(self, points1, points2) -> np.array:
        '''
            Calculates ratio of height and width given two sets of points
            Normalises between 0 and 1 by doing 1 - (1 / 1 + ratio)
            Replaces 0 divide with 1.
        '''
        d_x = points1[:, :, 0] - points2[:, :, 0]
        d_y = points1[:, :, 1] - points2[:, :, 1]

        return 1 - (1/(1 + np.divide(np.abs(d_x), np.abs(d_y), out=np.zeros_like(d_x), where=d_y!=0)))

    def box_ratio(self, sequence) -> np.array:
        """Computes tail box ratio. the smaller the value,
        the more of a line it is!"""
        tail_base = sequence[:, :, B2I['tail_base'], :]
        tail_tip = sequence[:, :, B2I['tail_tip'], :]
        return self.get_box_ratio(tail_base, tail_tip)

    def box_size(self, sequence) -> np.float32:
        """Computes tail bounding box length, Normalise by Grid size"""
        bottom, top = _bounding_box(sequence[:, :, -3:, :])
        lengths = np.sqrt((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2)
        areas = (top[0] - bottom[0]) * (top[1] - bottom[1])
        return lengths / self.max_size, areas / self.max_area

class DistanceEdgeFeature:
    def __call__(self, sequence):
        x_dist = np.minimum(sequence[:, :, B2I['center'], 0], DEFAULT_GRID_SIZE - sequence[:, :, B2I['center'], 0])
        y_dist = np.minimum(sequence[:, :, B2I['center'], 1], DEFAULT_GRID_SIZE - sequence[:, :, B2I['center'], 1])

        return x_dist / DEFAULT_GRID_SIZE, y_dist / DEFAULT_GRID_SIZE

class InternalAngleFeature:
    def __call__(self, sequence):
        # Left
        center2ear = sequence[:, :, B2I['ear_left']] - sequence[:, :, B2I['center']]
        center2paw = sequence[:, :, B2I['center']] - sequence[:, :, B2I['hindpaw_left']]
        dot = np.einsum('bmi,bmi->bm', center2ear, center2paw)
        norm = (np.linalg.norm(center2ear) * np.linalg.norm(center2paw))
        angle_left = np.arccos(np.divide(dot, norm, out=np.zeros_like(dot), where=norm!=0))

        # Right
        center2ear = sequence[:, :, B2I['ear_right']] - sequence[:, :, B2I['center']]
        center2paw = sequence[:, :, B2I['center']] - sequence[:, :, B2I['hindpaw_right']]
        dot = np.einsum('bmi,bmi->bm', center2ear, center2paw)
        norm = (np.linalg.norm(center2ear) * np.linalg.norm(center2paw))
        angle_right = np.arccos(np.divide(dot, norm, out=np.zeros_like(dot), where=norm!=0))

        return _encode_angle(angle_left), _encode_angle(angle_right)
        

class MouseFeatures:
    '''
    Calculates features for individual mice, nothing to do with interactions
    '''
    def __init__(self, keypoints) -> None:
        # initialise features
        self.body_length = BodyLengthFeature(keypoints)
        self.paw_to_head_dist = PawToEarDistFeature(keypoints)
        self.velocity = VelocityFeature(keypoints)
        self.centered_velocity = VelocityFeature(keypoints, center=True)
        self.tail_features = TailBoxFeature(keypoints)
        self.direction_change = DirectionChangeFeature()
        self.acceleration = AccelerationFeature(keypoints)
        self.centered_acceleration = AccelerationFeature(keypoints, center=True)
        self.distance_from_edge = DistanceEdgeFeature()
        self.internal_angle = InternalAngleFeature()

    def __call__(self, sequence) -> np.float32:
        features = []
        features.append(self.body_length(sequence))
        features.append(self.paw_to_head_dist(sequence))
        features.append(self.distance_from_center(sequence))
        features.extend(self.distance_from_edge(sequence))
        features.append(self.tail_features.box_ratio(sequence))
        features.extend(self.tail_features.box_size(sequence))
        features.append(self.velocity(sequence))
        features.append(self.centered_velocity(sequence))
        features.append(self.acceleration(sequence))
        features.append(self.centered_acceleration(sequence))
        features.append(self.direction_change(sequence))
        features.extend(self.internal_angle(sequence))

        # Concat features
        features = torch.tensor(np.concatenate([v[:, :, None] if v.ndim == 2 else v for v in features], axis=-1), dtype=torch.float32)

        return features

    def distance_from_center(self, sequence) -> np.float32:
        """computes distance from centre of grid
        (normalised by centre point)"""
        grid_center = DEFAULT_GRID_SIZE//2
        mouse_center = sequence[:, :, B2I['center']]
        return _distance(mouse_center, grid_center, normalise=grid_center)   


def _get_diff(sequence, 
            d=1, 
            center=False, 
            clip=None,
            seq_dim=0,
            body_part_dim=2,
            padding=None,
            norm=True):
    '''get the difference between current and the next in sequence'''
    
    if center:
        mouse_center = sequence.take(B2I['center'], axis=body_part_dim)
        sequence = np.delete(sequence, B2I['center'], axis=body_part_dim)
        sequence = sequence - np.expand_dims(mouse_center, axis=body_part_dim)

    from_d = sequence.take(range(d, sequence.shape[seq_dim]), axis=seq_dim)
    to_d = sequence.take(range(0, sequence.shape[seq_dim]-d), axis=seq_dim)
    diff = from_d - to_d

    if norm:
        diff = np.linalg.norm(diff, axis=-1)

    if diff.shape[0] == 0:
        return np.zeros(sequence.shape[:-1]) if norm else np.zeros(sequence.shape)
    
    if padding:
        diff = np.pad(diff, padding, mode='edge')
    
    if clip is not None:
        diff = np.clip(diff, -clip, clip)
    return diff

def _distance(seq1, seq2, normalise=1, clip=None):
    '''Returns the distance between two points, normalised by parameter'''
    dist = np.linalg.norm(seq1 - seq2, axis=-1)
    
    if clip is not None:
        dist = np.clip(dist, -clip, clip)
    
    return dist / normalise

def _angle(seq1, seq2):
    '''Finds the angle between m1 and m2, returns radians'''
    return ((np.arctan2(seq1[..., 0] - seq2[..., 0], seq1[..., 1] - seq2[..., 1]) + np.pi / 2) % (np.pi * 2))

def _encode_angle(angle):
    '''Encodes an angle as (cos(x), sin(x))'''
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)

def _bounding_box(points, body_part_axis=2):
    '''
        Gets top left and bottom right of keypoints for mouse
    '''
    x_min = points[..., 0].min(axis=body_part_axis)
    y_min = points[..., 1].min(axis=body_part_axis)
    x_max = points[..., 0].max(axis=body_part_axis)
    y_max = points[..., 1].max(axis=body_part_axis)

    tl = x_min, y_min
    br = x_max, y_max
    return tl, br
