from typing import Union, Mapping
import torch
import numpy as np
from torch.utils.data import DataLoader
from tomcat.dataset.mice_dataset import BaseMouseDataset
import math
from tomcat.consts import NUM_MICE
from sklearn.decomposition import TruncatedSVD
from tomcat.transforms.keypoints_transform import normalize, transform_to_svd_components

class FullSequenceSepDataset(BaseMouseDataset):
    '''
    Full Sequence Dataset
    Returns a full sequence as a batch of subsequences - used for testing

    Notes:
        Doesn't currently work with down-sampling
        Each worker returns an entire sequence, chunked up into max_seq_lengths and batched.
        DataLoader batch size MUST be one.
        
        If the sequence is not perfectly divided up by max_seq_length, 
        then the final entry in the batch will be made up into a full length sequence:
            These extras need to be removed from the embeddings before submitting
        
        Special tokens (CLS, SEP) also need to be removed after model forward pass
        See tomcat.model.mouse_bert MouseBERT.extract_embeddings
        
    '''

    def __init__(self, max_seq_length: int = 50, **kwargs):
        super().__init__(**kwargs)

        self.max_keypoints_length = max_seq_length - 2 
        self.max_seq_length = max_seq_length # includes special tokens
    
    def _get_original_sequence_length(self):
        return self.keypoints.shape[1]

    def add_special_tokens(self, tensor, default=0):
        batch_size, length, mice, dim = tensor.shape
        new_tensor = torch.full((batch_size, mice * length + 4, dim), fill_value=default, dtype=tensor.dtype)
        pos = 1
        for m in range(mice):
            start = pos
            end = pos + length
            new_tensor[:, start:end] = tensor[:, :, m]
            pos += length + 1

        return new_tensor
    
    def create_segments(self, batch_size, total_seq_len, seq_len):
        segments = torch.zeros((batch_size, total_seq_len), dtype=torch.long)
        pos = seq_len + 2
        for m in range(NUM_MICE):
            segments[:, pos:] += 1
            pos += 1 + seq_len
        
        return segments

    def split_overlap(self, sequence):
        size = self.max_keypoints_length
        step = self.max_keypoints_length // 2
        sequences = [sequence[:size]] + [sequence[i : i + size] for i in range(step//2, len(sequence)-size, step)] + [sequence[-size:]]
        return np.stack(sequences)
    
    def unsplit_overlap(self, embeddings):
        total_seq_len = self._get_original_sequence_length()
        step = self.max_keypoints_length // 2
        up_to_last = np.concatenate([embeddings[0][:step]] + [i[step//2:-step//2] for i in embeddings[1:-1]], axis=0)
        
        leftover = total_seq_len - len(up_to_last)
        return torch.tensor(np.concatenate([up_to_last, embeddings[-1][-leftover:]], axis=0))

    def reshape_nested(self, N, data):
        if isinstance(data, Mapping):
            return type(data)({k: self.reshape_nested(N, v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.reshape_nested(N, v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.view(N, self.max_keypoints_length, -1)
        return data

    def get_sample(self, sequence: np.ndarray) -> dict:
        '''
        Gets the test batch
        Splits input into self.max_keypoints_length sized chunks and stacks them as a batch
        Also return attention mask all ones.
        '''

        # Split input sequence into fixed length overlapping sub sequences.
        batch_keypoints = self.split_overlap(sequence)
        N = len(batch_keypoints)

        # (batch, max_keypoints_len, features)
        batch_keypoints = batch_keypoints.reshape(-1, *batch_keypoints.shape[2:])

        feats = self.featurise_keypoints(batch_keypoints)
        mouse_task_annotations, inter_mouse_task_annotations, group_task_annotations = self.create_task_annotations(batch_keypoints)
        feats = torch.cat([feats, mouse_task_annotations], dim=2)
        attention_mask = torch.ones(N, self.max_keypoints_length, NUM_MICE, 1, dtype=torch.bool)

        feats = feats.reshape(N, self.max_keypoints_length, NUM_MICE, -1)
        inter_mouse_task_annotations = self.reshape_nested(N, inter_mouse_task_annotations)

        feats = self.add_special_tokens(feats)
        attention_mask = self.add_special_tokens(attention_mask, default=1)

        segments = self.create_segments(N, feats.shape[1], self.max_keypoints_length)

        return {
            'keypoints': feats, 
            'attention_mask': attention_mask, 
            'segments': segments,
            'inter_mouse_task_annotations': inter_mouse_task_annotations
            }

    def __getitem__(self, idx: int) -> dict:
        seq_id = self.items[idx]
        sequence = self.keypoints[idx]

        inputs = self.get_sample(sequence)
        inputs['seq_id'] = seq_id

        annotations = {}
        if self.has_annotations:
            annotations = {k: torch.tensor(v[idx]) for (k,v) in self.annotations.items()}
            
        inputs = {**inputs, **annotations}
        return inputs

    def collate(self, batch):
        '''
        Collate function
        Expect batch to be a list with single element - Dataloader batch size MUST be one
        Returns the sequence id to keep track of which embeddings belong to which sequence.
        '''
        
        batch = batch[0]
        seq_id = batch.pop('seq_id')
        return seq_id, batch
    
    def get_dataloader(self) -> DataLoader:
        '''
        Get a dataloader to match this dataset, since the settings are very specific.
        '''

        return DataLoader(
            self, 
            batch_size=1, 
            num_workers=0, 
            collate_fn=self.collate,
            pin_memory=True
        )