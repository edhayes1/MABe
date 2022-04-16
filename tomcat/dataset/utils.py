from tomcat.dataset.mice_dataset import get_split_indices
from tomcat.transforms.augmentations import training_augmentations
from tomcat.consts import DEFAULT_NUM_TRAINING_POINTS, DEFAULT_NUM_TESTING_POINTS
import numpy as np
from tomcat.transforms.features import MouseFeatures
from tomcat.transforms.inter_mouse_features import MouseToMouseFeatures
from tomcat.transforms.group_features import GroupFeatures

def get_features(datasets):
    all_keypoints = []
    for path, index, _ in datasets:
        raw_data = np.load(path, allow_pickle=True).item()
        sequences = raw_data["sequences"]

        seq_ids = list(sequences.keys())
        seq_ids = [seq_ids[i] for i in index]

        keypoints = np.array(
            [sequences[idx]["keypoints"] for idx in seq_ids], dtype=np.float32
        )
        all_keypoints.append(keypoints)

    all_keypoints = np.concatenate(all_keypoints)

    mouse_features = MouseFeatures(keypoints)
    inter_mouse_features = MouseToMouseFeatures(keypoints)
    group_features = GroupFeatures(keypoints)
    
    features = {
        'mouse_features': mouse_features, 
        'inter_mouse_features': inter_mouse_features, 
        'group_features': group_features
        }
    return features


def get_multitask_datasets(args, MaskedDataset):
    types = []

    if args.train_path:
        types.append((args.train_path, *get_split_indices(DEFAULT_NUM_TRAINING_POINTS, args.train_ratio)))
    
    if args.test_path:
        types.append((args.test_path, *get_split_indices(DEFAULT_NUM_TESTING_POINTS, 1.0)))

    features = get_features(types)

    train_datasets = []
    val_datasets = []
    for path, train_indices, val_indices in types:

        train_dataset = MaskedDataset(
            path=path,
            max_seq_length=args.max_seq_length, 
            mask_prob=args.mask_prob,
            augmentations=training_augmentations,
            indices=train_indices,
            **features
        )
        train_datasets.append(train_dataset)

        if len(val_indices):
            val_dataset = MaskedDataset(
                path=path, 
                max_seq_length=args.max_seq_length, 
                mask_prob=args.mask_prob,
                indices=val_indices,
                **features
            )
            val_datasets.append(val_dataset)

    return train_datasets, val_datasets[0], features
    
