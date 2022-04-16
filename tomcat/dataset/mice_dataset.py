import __future__
from sklearn.decomposition import TruncatedSVD
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tomcat import consts
from typing import Union, List, Tuple
from tomcat.transforms.keypoints_transform import transform_to_svd_components, normalize, fill_holes
from tomcat.transforms.features import MouseFeatures
from tomcat.transforms.inter_mouse_features import MouseToMouseFeatures
from tomcat.transforms.group_features import GroupFeatures

def get_split_indices(num:int, split: float) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(0, num, 1, dtype=int)
    np.random.shuffle(indices)
    split_index = int(num * split)

    idx_train = indices[:split_index]
    idx_valid = indices[split_index:]

    return (idx_train, idx_valid)

class BaseMouseDataset(Dataset):
    """
    Primary Mouse (+Features) dataset.
    Also includes preliminary preprocessing functions
    to facilitate data engineering.
    """

    def __init__(
        self,
        path: Path,
        scale: bool = True,
        sample_frequency: int = consts.DEFAULT_FRAME_RATE,
        indices: Union[np.ndarray, None] = None,
        mouse_features: MouseFeatures = None,
        inter_mouse_features: MouseToMouseFeatures = None,
        group_features: GroupFeatures = None
    ):
        self.path = path
        self.sample_frequency = sample_frequency  # downsample frames if needed
        self.scale = scale

        # defined if data has been loaded
        self.has_annotations = None
        self.annotation_names = []
        self.annotations = {}

        # defined when data has been preprocessed.
        self.items = None
        self.keypoints = None
        self.num_features = 84 + 60*3 # #72 #
        self.n_frames = None

        self.load_data()
        self.preprocess(indices)
        
        # Set after set_svd is called, this is optional.
        self.svd = None

        # Set Mouse features
        self.mouse_feats = mouse_features if mouse_features else MouseFeatures(self.keypoints)
        self.inter_mouse_feats = inter_mouse_features if inter_mouse_features else MouseToMouseFeatures(self.keypoints)
        self.group_feats = group_features if group_features else GroupFeatures(self.keypoints)

    def get_kwargs(self) -> dict:
        """returns positional arguments"""
        return {
            "path": self.path,
            "frame_rate": self.frame_rate,
            "sample_frequency": self.sample_frequency,
            "flatten": self.flatten,
            "scale": self.scale,
        }

    def load_data(self) -> None:
        """Loads dataset"""
        self.raw_data = np.load(self.path, allow_pickle=True).item()

    def check_annotations(self) -> None:
        """Annotation check handler"""
        self.has_annotations = "vocabulary" in self.raw_data.keys()
        if self.has_annotations:
            self.annotation_names = self.raw_data["vocabulary"]
    
    def featurise_keypoints(self, keypoints):
        keypoints = normalize(keypoints)
        keypoints, _ = transform_to_svd_components(keypoints, svd_computer=self.svd)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints

    def create_task_annotations(self, keypoints):
        mouse_task_annotations = self.mouse_feats(keypoints)
        inter_mouse_task_annotations = self.inter_mouse_feats(keypoints)
        group_task_annotations = self.group_feats(keypoints)

        return mouse_task_annotations, inter_mouse_task_annotations, group_task_annotations

    def preprocess(self, indices=Union[np.ndarray, None]) -> Dataset:
        """
        Does initial preprocessing on entire dataset.
        """
        self.check_annotations()

        sequences = self.raw_data["sequences"]

        seq_ids = list(sequences.keys())

        if type(indices) == np.ndarray:
            seq_ids = [seq_ids[i] for i in indices]

        keypoints = np.array(
            [sequences[idx]["keypoints"] for idx in seq_ids], dtype=np.float32
        )

        # Get annotations
        if self.has_annotations:
            for i, task in enumerate(self.annotation_names):
                
                annotations = np.array(
                    [sequences[idx]["annotations"][i, :] for idx in seq_ids], dtype=np.float32
                )

                # annotations = self.downsample(annotations, self.sample_frequency)
                self.annotations[task] = annotations

        self.items = seq_ids
        self.keypoints = keypoints
        self.n_frames = keypoints.shape[0]
    
    def set_svd(self, svd: TruncatedSVD):
        self.svd = svd

    @staticmethod
    def downsample(keypoints: np.ndarray, sample_frequency) -> np.ndarray:
        """Downsamples frames"""
        return keypoints[:, ::sample_frequency, ...]

    def get_num_frames(self):
        return self.keypoints.shape[1]

    def __len__(self):
        return self.keypoints.shape[0]