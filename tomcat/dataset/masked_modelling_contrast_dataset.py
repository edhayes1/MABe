from tomcat.dataset.masked_modelling_sep_span_dataset import MaskedModelingSepSpanDataset
from tomcat.consts import NUM_MICE
import numpy as np
import torch

class MaskedModellingContrastDataset(MaskedModelingSepSpanDataset):
    '''
        Samples two distinct sections from the same video for contrastive learning
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_random_start_end(self, length):
        start = np.random.randint(0, length - self.max_keypoints_len)
        end = start + self.max_keypoints_len
        return start, end

    def get_non_overlapping(self, length):
        start, end = self.get_random_start_end(length)
        start2, end2 = self.get_random_start_end(length)
        while start < start2 < end or start < end2 < end:
            start2, end2 = self.get_random_start_end(length)
        
        return (start, end), (start2, end2)

    def get_sub_sequence(self, idx: int, sequence, start, end):
        keypoints = sequence[start:end, :]

        annotations = {}
        if self.has_annotations:
            annotations = {k: torch.tensor(v[idx][start:end]) for (k,v) in self.annotations.items()}
            annotations = {k: v[:, None, None].repeat(1, NUM_MICE, 1) for (k,v) in annotations.items()}
        
        return keypoints, annotations
    
    def sample_random_keypoints(self, length=1):
        '''Collect a training sample at random'''
        seq = self.sample_random_sequence()
        seq_idx = np.random.randint(0, len(seq)-length)
        mouse_idx = np.random.randint(0, NUM_MICE)
        keypoints = seq[seq_idx:seq_idx + length]
        feats = self.featurise_keypoints(keypoints)[:, mouse_idx]
        mouse_task_annotations = self.mouse_feats(keypoints)[:, mouse_idx]
        return torch.cat([feats, mouse_task_annotations], dim=1)

    def get_training_sample(self, idx: int) -> dict:
        '''
        Returns a training sample
        
        Randomly samples a section with length self.max_keypoints_len of the input sequence.
        Adds dummy inputs for special CLS and SEP tokens, these are replaced in the forward pass with learnable embeddings

        Applies masking and returns model inputs

        '''

        sequence = self.keypoints[idx]

        all_inputs = []

        non_overlapping = self.get_non_overlapping(sequence.shape[0])

        for start, end in non_overlapping:

            keypoints, annotations = self.get_sub_sequence(idx, sequence, start, end)

            keypoints = self.shuffle_mice(keypoints)

            if self.augmentations:
                keypoints = self.augmentations(keypoints)
            
            feats = self.featurise_keypoints(keypoints)
            mouse_task_annotations, inter_mouse_task_annotations, group_task_annotations = self.create_task_annotations(keypoints)

            # Experiment with adding task annotations to input
            feats = torch.cat([feats, mouse_task_annotations], dim=2)

            feats, attention_mask, masked_positions, labels, label_mask = self.mask_keypoints(feats)

            feats = self.add_special_tokens(feats)
            attention_mask = self.add_special_tokens(attention_mask, 1)
            masked_positions = self.add_special_tokens(masked_positions)
            labels = self.add_special_tokens(labels)
            label_mask = self.add_special_tokens(label_mask)
            task_annotations = self.add_special_tokens(mouse_task_annotations)

            if annotations:
                annotations = {k: self.add_special_tokens(v) for (k,v) in annotations.items()}

            # Do segment ids
            segments = self.create_segments(len(feats), self.max_keypoints_len)

            inputs = {
                'keypoints': feats, 
                'attention_mask': attention_mask, 
                'masked_positions': masked_positions, 
                'labels': labels,
                'label_mask': label_mask,
                'segments': segments,
                # 'task_annotations': task_annotations,
                'inter_mouse_task_annotations': inter_mouse_task_annotations,
                'group_task_annotations': group_task_annotations
            }

            all_inputs.append({**inputs, **annotations})

        return all_inputs


    def collator(self, batch: tuple) -> dict:
        '''
        Data Collator
        Stacks the input batch
        
        We need special treatment for inter_mouse_task_annotations, since it is a nested dict
        '''

        inputs = {}
        keys = batch[0][0].keys()
        for k in keys:
            if batch[0][0][k] is None:
                continue

            if k == 'inter_mouse_task_annotations':
                inter_mouse_task_annotations = {}
                for m1 in range(NUM_MICE):
                    inter_mouse_task_annotations[m1] = {m2: torch.stack([i[0][k][m1][m2] for i in batch] + [i[1][k][m1][m2] for i in batch]) for m2 in batch[0][0][k][m1].keys()}
                
                inputs[k] = inter_mouse_task_annotations
            
            else:
                inputs[k] = torch.stack([i[0][k] for i in batch] + [i[1][k] for i in batch])

        return inputs