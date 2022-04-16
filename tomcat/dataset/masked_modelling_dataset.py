from tomcat.dataset.mice_dataset import BaseMouseDataset
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from tomcat.transforms.augmentations import Flatten, Scale, GaussianNoise, Rotation, Tensor

class MaskedModelingDataset(BaseMouseDataset):
    '''
        Masked Modeling Dataset

        Notes:
            Masking is not done here, we only decide on the sequence indices where masking should take place,
            mask embeddings are then applied in the forward pass.

            Labels are identical to the input.
    '''
    
    def __init__(self, 
                max_seq_length: int = 50, 
                mask_prob: float = 0.15, 
                augmentations: transforms.Compose = None, 
                **kwargs):
        
        super().__init__(**kwargs)

        self.max_keypoints_len = max_seq_length - 2 # account for special tokens
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob

        self.augmentations = augmentations
    
    def sample_random_sequence(self):
        '''Collect a training sequence at random'''
        idx = np.random.randint(0, len(self))
        return self.keypoints[idx]

    def sample_random_keypoints(self):
        '''Collect a training sample at random'''
        seq = self.sample_random_sequence()
        seq_idx = np.random.randint(0, len(seq))
        feats = self.featurise_keypoints(seq[None, seq_idx])
        feats = feats.reshape(1, -1)
        return feats

    def mask_keypoints(self, keypoints: torch.tensor) -> Tuple[torch.tensor, ...]:
        '''
        Mask input keypoints according to mask_prob:
        Follow BERT:    
            80% masked points are denoted for masking in forward pass
            10% are replaced with a randomly sampled input
            10% are left

        Returns:
            Keypoints (torch.tensor): input keypoints
            Attention Mask (torch.tensor): attention mask
            Mask (torch.tensor): denoted input positions to be replaced with mask embedding
            Labels (torch.tensor): simply a copy of the input
        '''
        seq_len = len(keypoints)
        labels = keypoints.clone()
        label_mask = torch.zeros((seq_len, 1), dtype=torch.bool)
        mask = torch.zeros((seq_len, 1), dtype=torch.bool)
        attention_mask = torch.ones((seq_len, 1), dtype=torch.bool)

        for i in range(1, seq_len -1):
            rand = np.random.random()
            if rand < self.mask_prob:
                label_mask[i] = 1

                if rand < 0.8 * self.mask_prob:
                    # mask, don't need to change the features, these will instead be masked in the model forward pass
                    mask[i] = 1

                elif rand < 0.9 * self.mask_prob:
                    # random
                    keypoints[i] = self.sample_random_keypoints()
        
        return keypoints, attention_mask, mask, labels, label_mask


    def get_training_sample(self, sequence: np.ndarray) -> dict:
        '''
        Returns a training sample
        
        Randomly samples a section with length self.max_keypoints_len of the input sequence.
        Adds dummy inputs for special CLS and SEP tokens, these are replaced in the forward pass with learnable embeddings

        Applies masking and returns model inputs

        '''
        start = np.random.randint(0, sequence.shape[0] - self.max_keypoints_len)
        end = start + self.max_keypoints_len
        keypoints = sequence[start:end, :]

        if self.augmentations:
            keypoints = self.augmentations(keypoints)
        
        # Do scale, flatten and tensor AFTER features
        feats = self.featurise_keypoints(keypoints)

        # flatten for now
        feats = feats.reshape(self.max_keypoints_len, -1)

        # Add dummy inputs for CLS and SEP tokens
        dummy = torch.zeros((1, feats.shape[1]))
        feats = torch.cat([dummy, feats, dummy], 0)

        feats, attention_mask, mask, labels, label_mask = self.mask_keypoints(feats)

        return {
            'keypoints': feats, 
            'attention_mask': attention_mask, 
            'masked_positions': mask, 
            'labels': labels,
            'label_mask': label_mask
            }

    def __getitem__(self, idx: int) -> dict:
        sequence = self.keypoints[idx]

        inputs = self.get_training_sample(sequence)
        
        return inputs
    
    def collator(self, batch: list) -> dict:
        '''
        Data Collator
        Stacks and pads the input batch
        '''
        keypoints = torch.stack([i['keypoints'] for i in batch])
        attention_mask = torch.stack([i['attention_mask'] for i in batch])
        masked_positions = torch.stack([i['masked_positions'] for i in batch])
        labels = torch.stack([i['labels'] for i in batch])
        label_mask = torch.stack([i['label_mask'] for i in batch])

        return {
            'keypoints': keypoints, 
            'attention_mask': attention_mask, 
            'masked_positions': masked_positions, 
            'labels': labels,
            'label_mask': label_mask
            }


