# Features inspired by The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice
# Cristina Segalin, Jalani Williams, Tomomi Karigo, May Hui, Moriel Zelikowsky, Jennifer J. Sun, Pietro Perona, David J. Anderson, Ann Kennedy
# https://www.biorxiv.org/content/10.1101/2020.07.26.222299v1

# "TREBA" by Sun, Jennifer J and Kennedy, Ann and Zhan, Eric and Anderson, David J and Yue, Yisong and Perona, Pietro is licensed under CC BY-NC-SA 4.0 license.
# https://github.com/neuroethology/TREBA/blob/c522e169738f5225298cd4577e5df9085130ce8a/util/datasets/mouse_v1/augmentations/augmentation_functions.py

import numpy as np
from tomcat.consts import DEFAULT_GRID_SIZE, BODY_PART_2_INDEX as B2I, NUM_MICE
import torch
from tomcat.transforms.features import _get_diff, _distance, _angle, _encode_angle, _bounding_box

class BoundingBoxIOUFeature:
    def area(self, box):
        tl, br = box
        x = np.clip((br[0] - tl[0]), 0, None)
        y = np.clip((br[1] - tl[1]), 0, None)
        return x * y

    def __call__(self, m1, m2):
        m1_box = _bounding_box(m1[:, :-3, :], body_part_axis=1)
        m2_box = _bounding_box(m2[:, :-3, :], body_part_axis=1)

        m1_area = self.area(m1_box)
        m2_area = self.area(m2_box)

        int_tl_x = np.maximum(m1_box[0][0], m2_box[0][0])
        int_tl_y = np.maximum(m1_box[0][1], m2_box[0][1])
        int_br_x = np.minimum(m1_box[1][0], m2_box[1][0])
        int_br_y = np.minimum(m1_box[1][1], m2_box[1][1])

        int_area = self.area(((int_tl_x, int_tl_y), (int_br_x, int_br_y)))
        union_area = (m1_area + m2_area - int_area)
        
        return np.divide(int_area, union_area, out=np.zeros_like(int_area), where=union_area!=0)


class VelocityComponentFeature:
    def __init__(self, keypoints) -> None:
        self.diff_max = self.get_stats(keypoints)

    def get_stats(self, keypoints):

        mouse_length = _distance(
                keypoints[:, :, :, B2I['nose']], 
                keypoints[:, :, :, B2I['tail_base']]
            )

        diff_max = mouse_length.mean(axis=(0, 1, 2))

        return 2 * diff_max

    def __call__(self, m1, m2):
        m1_heading = _angle(m1[:, B2I['neck']], m1[:, B2I['tail_base']])
        m2_angle = _angle(m1[:, B2I['neck']], m2[:, B2I['center']])

        direction = m1_heading - m2_angle
        diff = _get_diff(m1[:, B2I['center']], padding=((1,0)))

        velocity_component = np.cos(direction) * diff
        return velocity_component / self.diff_max


class NoseToNoseDistanceFeature:
    def __init__(self, keypoints) -> None:
        self.dist_max, self.diff_max = self.get_stats(keypoints)
    
    def get_stats(self, keypoints):
        m1_nose = keypoints[:, :, 0, B2I['nose']]
        m2_nose = keypoints[:, :, 1, B2I['nose']]

        dist = _distance(m1_nose, m2_nose)
        dist_max = dist.max((0, 1))

        mouse_length = _distance(
                keypoints[:, :, :, B2I['nose']], 
                keypoints[:, :, :, B2I['tail_base']]
            )

        diff_max = mouse_length.mean(axis=(0, 1, 2))

        return dist_max, 2 * diff_max
    
    def __call__(self, m1, m2):
        m1_nose = m1[:, B2I['nose']]
        m2_nose = m2[:, B2I['nose']]
        dist = _distance(m1_nose, m2_nose)
        diff = _get_diff(dist, padding=((1,0)), norm=False)
        return dist / self.dist_max, diff / self.diff_max


class VelocityDifferenceFeature:
    def __init__(self, keypoints) -> None:
        self.diff_max = self.get_stats(keypoints)
    
    def get_stats(self, keypoints):
        mouse_length = _distance(
                keypoints[:, :, :, B2I['nose']], 
                keypoints[:, :, :, B2I['tail_base']]
            )
        return 2 * mouse_length.mean(axis=(0, 1, 2))
    
    def __call__(self, m1, m2):
        m1_center = m1[:, B2I['center']]
        m2_center = m2[:, B2I['center']]

        diff_m1 = _get_diff(
            m1_center[:, None, :],
            seq_dim=0,
            body_part_dim=1,
            padding=((1, 0), (0, 0)),
            clip=self.diff_max
        )

        diff_m2 = _get_diff(
            m2_center[:, None, :],
            seq_dim=0,
            body_part_dim=1,
            padding=((1, 0), (0, 0)),
            clip=self.diff_max
        )

        diff = diff_m1 - diff_m2

        return diff / self.diff_max

class NoseToBaseDistanceFeature:
    def __init__(self, keypoints) -> None:
        self.dist_max, self.diff_max = self.get_stats(keypoints)
    
    def get_stats(self, keypoints):
        m1_nose = keypoints[:, :, 0, B2I['nose']]
        m2_base = keypoints[:, :, 1, B2I['tail_base']]

        dist = _distance(m1_nose, m2_base)
        dist_max = dist.max((0, 1))

        mouse_length = _distance(
                keypoints[:, :, :, B2I['nose']], 
                keypoints[:, :, :, B2I['tail_base']]
            )

        diff_max = mouse_length.mean(axis=(0, 1, 2))

        return dist_max, 2 * diff_max
    
    def __call__(self, m1, m2):
        m1_nose = m1[:, B2I['nose']]
        m2_base = m2[:, B2I['tail_base']]
        dist = _distance(m1_nose, m2_base)
        diff = _get_diff(dist, padding=((1,0)), norm=False)
        return dist / self.dist_max, diff / self.diff_max


class DirectionDifferenceFeature:
    
    def __call__(self, m1, m2):
        m1_heading = _angle(m1[:, B2I['nose']], m1[:, B2I['center']])
        m2_heading = _angle(m2[:, B2I['nose']], m2[:, B2I['center']])

        dir_diff = m1_heading - m2_heading

        dir_diff_encoded = _encode_angle(dir_diff)

        return dir_diff_encoded


class NoseToCenterAngleFeature:
    
    def __call__(self, m1, m2, relative=False):
        '''Finds angle from m1 nose to m2 center'''
        m1_nose = m1[:, B2I['nose']]
        m1_center = m1[:, B2I['center']]
        m2_center = m2[:, B2I['center']]

        angle = _angle(m1_nose, m2_center)

        if relative:
            heading = _angle(m1_nose, m1_center)
            angle = angle - heading
        
        encoded_angle = _encode_angle(angle)

        return encoded_angle 


class MouseToMouseFeatures:
    '''
    Gets features BETWEEN mice.
    For example nose-to-tail distance, nose to centroid angle
    These are handled and used differently during modelling.
    Output type is a little complicated: 

    Returns: [
            (mouse 0) 0: {[feature_a: {1: np.array, 2: np.array}, ...]},
            (mouse 1) 1: {[feature_a: {0: np.array, 2: np.array}, ...]},
            (mouse 2) 2: {[feature_a: {0: np.array, 1: np.array}, ...]},
        ]
    '''
    def __init__(self, keypoints) -> None:
        self.nose_to_center_angle = NoseToCenterAngleFeature()
        self.nose_to_center_distance = NoseToBaseDistanceFeature(keypoints)
        self.nose_to_nose_distance = NoseToNoseDistanceFeature(keypoints)
        self.direction_difference = DirectionDifferenceFeature()
        self.bounding_box_iou = BoundingBoxIOUFeature()
        self.velocity_difference = VelocityDifferenceFeature(keypoints)
        self.velocity_component = VelocityComponentFeature(keypoints)
    
    def __call__(self, sequence) -> np.float32:
        features = {}
        for m1 in range(NUM_MICE):
  
            features[m1] = self._get_inter_feats(sequence, m1)

        return features

    def _get_inter_feats(self, sequence, m1):
        m1_seq = sequence[:, m1]
        features = {}
        for m2 in range(NUM_MICE):
            if m1 == m2:
                continue

            m2_seq = sequence[:, m2]

            feats = []
            feats.append(self.nose_to_center_angle(m1_seq, m2_seq))
            feats.append(self.nose_to_center_angle(m1_seq, m2_seq, relative=True))
            feats.extend(self.nose_to_center_distance(m1_seq, m2_seq))
            feats.extend(self.nose_to_nose_distance(m1_seq, m2_seq))
            feats.append(self.direction_difference(m1_seq, m2_seq))
            feats.append(self.bounding_box_iou(m1_seq, m2_seq))
            feats.append(self.velocity_difference(m1_seq, m2_seq))
            feats.append(self.velocity_component(m1_seq, m2_seq))

            features[m2] = torch.tensor(np.concatenate([i[:, None] if i.ndim == 1 else i for i in feats], axis=-1), dtype=torch.float32)
        
        return features

