import torch
import numpy as np
from torch.utils.data import DataLoader
from tomcat.dataset.mice_dataset import BaseMouseDataset
import math
from tomcat import consts
from sklearn.decomposition import TruncatedSVD
from tomcat.transforms.keypoints_transform import normalize, transform_to_svd_components

class FullSequenceDataset(BaseMouseDataset):
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

    def split_overlap(self, sequence):
        size = self.max_keypoints_length
        step = self.max_keypoints_length // 4
        sequences = [sequence[:size]] + [sequence[i : i + size] for i in range(step//2, len(sequence)-size, step)] + [sequence[-size:]]
        return np.stack(sequences)
    
    def unsplit_overlap(self, embeddings):
        step = self.max_keypoints_length // 4
        embeddings = [embeddings[0][:step]] + [i[step//2:-step//2] for i in embeddings[1:-1]] + [embeddings[-1][-step:]]
        return torch.cat(embeddings, dim=1)

    def get_sample(self, sequence: np.ndarray) -> dict:
        '''
        Gets the test batch
        Splits input into self.max_keypoints_length sized chunks and stacks them as a batch
        Also return attention mask all ones.
        '''

        # Split input sequence into fixed length sub sequences. Note last array could be different length
        N = math.ceil(sequence.shape[0] / self.max_keypoints_length)
        split_indices = list(range(self.max_keypoints_length, sequence.shape[0], self.max_keypoints_length))
        sequences = np.array_split(sequence, split_indices)

        # Fix last array, so includes full length sequence
        len_last = sequences[-1].shape[0]
        if len_last != self.max_keypoints_length:
            diff = self.max_keypoints_length - len_last
            sequences[-1] = np.concatenate([sequences[-2][-diff :], sequences[-1]], 0)

        # (batch, max_keypoints_len, features)
        batch_keypoints = np.stack(sequences)
        batch_keypoints = batch_keypoints.reshape(-1, *batch_keypoints.shape[2:])
        feats = self.featurise_keypoints(batch_keypoints)
        feats = feats.reshape(N, self.max_keypoints_length, -1)

        # Add dummy inputs for CLS and SEP tokens
        dummy = torch.zeros((N, 1, self.num_features), dtype=torch.float32)
        keypoints = torch.cat([dummy, feats, dummy], 1)

        attention_mask = torch.ones((N, self.max_seq_length, 1), dtype=torch.bool)

        return {'keypoints': keypoints, 'attention_mask': attention_mask}

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