import numpy as np
from tomcat.consts import DEFAULT_GRID_SIZE, BODY_PART_2_INDEX as B2I, NUM_MICE
import torch
from tomcat.transforms.features import _get_diff, _distance, _angle, _encode_angle, _bounding_box


class AreaFeature:
    '''
        Area of triangle between mice
    '''
    def __init__(self, keypoints) -> None:
        self.max = self.get_stats(keypoints)

    def get_stats(self, keypoints):
        centers = keypoints[:, :, :, B2I['center']]
        m1 = centers[:, :, 0]
        m2 = centers[:, :, 1]
        m3 = centers[:, :, 2]

        m1_m2 = m2 - m1
        m1_m3 = m3 - m1
        areas = np.abs(0.5 * (m1_m2[:, :, 0] * m1_m3[:, :, 1] - m1_m3[:, :, 0] * m1_m2[:, :, 1]))
        return areas.max((0, 1))

    def __call__(self, sequence):
        centers = sequence[:, :, B2I['center']]
        m1 = centers[:, 0]
        m2 = centers[:, 1]
        m3 = centers[:, 2]

        m1_m2 = m2 - m1
        m1_m3 = m3 - m1
        return np.abs(0.5 * (m1_m2[:, 0] * m1_m3[:, 1] - m1_m3[:, 0] * m1_m2[:, 1])) / self.max

class GroupFeatures:
    '''
        Features for the entire group of mice
    '''
    def __init__(self, keypoints) -> None:
        self.area = AreaFeature(keypoints)

    def __call__(self, sequence):
        features = []
        features.append(self.area(sequence))

        features = torch.tensor(np.concatenate([v[:, None] if v.ndim == 1 else v for v in features], axis=-1), dtype=torch.float32)

        return features