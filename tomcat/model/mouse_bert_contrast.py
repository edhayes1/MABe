# Adapted from Pytorch Lightning. Licensed under Apache 2.0
# https://github.com/PyTorchLightning/pytorch-lightning

# from tomcat.model.mouse_bert_sep import MouseBERTSep
from tomcat.model.mouse_bert_inter_input import MouseBERTSep
import torch
from typing import Dict
from tomcat.consts import NUM_MICE
from torch import nn
from torch.nn import functional as F
import math

class ContrastiveProjection(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(ContrastiveProjection, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, z):
        z = self.mlp(z)
        return F.normalize(z, dim=-1)


class MouseBERTContrast(MouseBERTSep):
    def __init__(self, **kwargs) -> None:
        super(MouseBERTContrast, self).__init__(**kwargs)

        self.contrastive_projection = ContrastiveProjection(NUM_MICE * self.output_size, self.output_size)
        self.temperature = 0.1
    
    def forward(self,
        keypoints: torch.tensor = None, 
        attention_mask: torch.tensor = None,
        label_mask: torch.tensor = None,
        labels: torch.tensor = None,
        masked_positions: torch.tensor = None,
        segments: torch.tensor = None,
        chases: torch.tensor = None,
        lights: torch.tensor = None,
        sequence_replace_label: torch.tensor = None,
        task_annotations: torch.tensor = None,
        inter_mouse_task_annotations: Dict = None,
        group_task_annotations: torch.tensor = None,
        eps: float = 1e-6
        ):
        '''Adapted from Pytorch lightning version of SimCLR'''

        outputs = super(MouseBERTContrast, self).forward(
            keypoints,
            attention_mask,
            label_mask,
            labels,
            masked_positions,
            segments,
            chases,
            lights,
            sequence_replace_label,
            task_annotations,
            inter_mouse_task_annotations,
            group_task_annotations
        )

        z = self.contrastive_projection(outputs['pooled_embeds'])

        N = int(z.shape[0] / 2)

        cov = torch.mm(z, z.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.full(neg.shape, math.e ** (1 / self.temperature), device=neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(z[:N] * z[N:], dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        contrastive_loss = 0.1 * -torch.log(pos / (neg + eps)).mean()

        outputs['loss'] += contrastive_loss
        outputs['contrastive_loss'] = contrastive_loss

        return outputs
        