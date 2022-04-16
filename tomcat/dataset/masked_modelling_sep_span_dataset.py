# Adapted from:
# "SpanBERT Improving Pre-training by Representing and Predicting Spans" by Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy is licensed under CC-BY-NC 4.0 license.
# https://github.com/facebookresearch/SpanBERT

from cProfile import label
from tomcat.dataset.masked_modelling_sep_dataset import MaskedModelingSepDataset
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict
from torchvision import transforms
from tomcat.consts import NUM_MICE
import math

class MaskedModelingSepSpanDataset(MaskedModelingSepDataset):
    '''
        Masked Modeling Dataset

        Notes:
            Masking is not done here, we only decide on the sequence indices where masking should take place,
            mask embeddings are then applied in the forward pass.

            Labels are identical to the input.
    '''
    def __init__(self, min_span: int = 3, max_span: int = 20, geometric_p: float = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.lens = list(range(min_span, max_span + 1))
        
        self.len_distrib = [geometric_p * (1-geometric_p)**(i - min_span) for i in range(min_span, max_span + 1)] if geometric_p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib] # Normalise
        # print(self.len_distrib, self.lens)

    def mask_keypoints(self, keypoints: torch.tensor) -> Dict[str, torch.tensor]:
        '''
        Mask input keypoints according to mask_prob:
        Follow BERT:    
            80% masked points are denoted for masking in forward pass
            10% are replaced with a randomly sampled input
            10% are left

        Returns:
            Feats (torch.tensor): input feats
            Attention Mask (torch.tensor): attention mask
            Mask (torch.tensor): denoted input positions to be replaced with mask embedding
            Labels (torch.tensor): simply a copy of the input
        '''
        seq_len = len(keypoints)
        labels = keypoints.clone()
        label_mask = torch.zeros((seq_len, NUM_MICE, 1), dtype=torch.bool)
        masked_positions = torch.zeros((seq_len, NUM_MICE, 1), dtype=torch.bool)
        attention_mask = torch.ones((seq_len, NUM_MICE, 1), dtype=torch.bool)

        for m in range(NUM_MICE):
            mouse_keypoints, mouse_masked_positions, mouse_label_mask = self.mask_mouse(keypoints[:, m]) 
            keypoints[:, m] = mouse_keypoints
            label_mask[:, m] = mouse_label_mask
            masked_positions[:, m] = mouse_masked_positions

        return keypoints, attention_mask, masked_positions, labels, label_mask

    def mask_mouse(self, sequence):
        """
        mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        seq_length = len(sequence)
        mask_num = math.ceil(seq_length * self.mask_prob)
        mask = set()
        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            
            start = np.random.choice(seq_length)
            if start in mask:
                continue
            spans.append([start, start])
            end = start
            while end - start < span_len and end < seq_length and len(mask) < mask_num:
                mask.add(end)
                spans[-1][-1] = end
                end += 1

        spans = self.merge_intervals(spans)

        keypoints, masked_positions, label_mask = self.span_masking(sequence, spans, mask)
        
        return keypoints, masked_positions, label_mask

    
    def span_masking(self, keypoints, spans, mask):
        seq_len = len(keypoints)
        label_mask = torch.zeros((seq_len, 1), dtype=torch.bool)
        masked_positions = torch.zeros((seq_len, 1), dtype=torch.bool)

        assert len(mask) == sum([e - s + 1 for s,e in spans])

        for start, end in spans:
            if end - start == 0:
                end += 1
                
            label_mask[start:end] = 1
            rand = np.random.random()

            if rand < 0.8:
                masked_positions[start:end] = 1

            elif rand < 0.9:
                keypoints[start:end] = self.sample_random_keypoints(end-start)
            
        return keypoints, masked_positions, label_mask

    def merge_intervals(self, intervals):
        intervals = sorted(intervals, key=lambda x : x[0])
        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] + 1 < interval[0]:
                merged.append(interval)
            else:
                # otherwise, there is overlap, so we merge the current and previous
                # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
