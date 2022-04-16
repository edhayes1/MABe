from transformers import BertConfig, BertModel
from torch import nn
import torch
from tomcat.dataset.mice_dataset import get_split_indices
from tomcat.consts import NUM_MICE
from typing import Dict
from torch.nn import functional as F


class Pooler(nn.Module):
    def __init__(self, input_size):
        super(Pooler, self).__init__()

        self.attn = nn.Linear(input_size, 1)
    
    def forward(self, embeddings):
        weights = F.softmax(self.attn(embeddings), dim=1)
        return torch.sum(weights * embeddings, dim=1)

class LinearClassifier(nn.Module):
    '''
        Simple linear classifier on top of embeddings (single layer perceptron)
        Optional dropout
    '''
    def __init__(self, hidden_dim, output_dim, dropout=False):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.2)
        self.do_dropout = dropout

    def forward(self, embedding, labels):
        if self.do_dropout:
            embedding = self.dropout(embedding)
        
        preds = self.classifier(embedding)
        loss = self.loss(preds, labels)
        return loss, preds


class RegressionHead(nn.Module):
    '''
        General regression head for continuous features
        Can be used with dropout, or with a mask
    '''

    def __init__(self, hidden_dim, output_dim, dropout=0.0, return_pred=False, two_layer=False) -> None:
        super(RegressionHead, self).__init__()
        if two_layer:
            self.regression_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.regression_layer = nn.Linear(hidden_dim, output_dim)

        self.loss = nn.MSELoss(reduction='none')
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.return_pred = return_pred

    def forward(self, x, labels, mask=None):
        if self.dropout:
            x = self.dropout(x)

        preds = self.regression_layer(x)

        if mask is not None:
            loss = ((mask * self.loss(preds, labels)).sum(dim=1)/mask.sum(dim=1)).mean()
        else:
            loss = self.loss(preds, labels).mean()
        
        if self.return_pred:
            return loss, preds
        else:
            return loss


class MouseBERTSep(nn.Module):
    '''
    MouseBERT
        Just your average BERT encoder, performing masked modeling over the input
        Input could be keypoints, or any other (continuous) features.

        All special tokens are added in the forward pass

        Single segment (for now)
    '''


    def __init__(self, bert_config: BertConfig = None, num_features: int = None) -> None:
        '''
        Initialise transformer encoder, embeddings and additional heads
        '''
        
        super(MouseBERTSep, self).__init__()

        self.output_size = int(bert_config.output_size/NUM_MICE)
        self.num_features = int(num_features/NUM_MICE)

        self.transformer = BertModel(bert_config)
        self.output_projection = nn.Linear(bert_config.hidden_size, self.output_size)
        self.mm_head = RegressionHead(self.output_size, self.num_features)
        self.pooler = Pooler(NUM_MICE * self.output_size)

        self.seq_replace_head = LinearClassifier(self.output_size, 1, dropout=0.2)

        self.chases_head = LinearClassifier(self.output_size, 1, dropout=0.2)
        self.lights_head = LinearClassifier(self.output_size, 1, dropout=0.2)

        # Todo don't hard code...
        self.task_head = RegressionHead(self.output_size, 59, dropout=0.1)
        self.inter_mouse_task_head = RegressionHead(self.output_size, 9, dropout=0.1)
        self.group_task_head = RegressionHead(NUM_MICE * self.output_size, 1, dropout=0.2)

        self.dropout = nn.Dropout(0.2)

        self.points_embedding = nn.Linear(self.num_features, bert_config.hidden_size)

        # Special_tokens: CLS, SEP and MASK
        self.cls_embed = nn.Embedding(1, bert_config.hidden_size)
        self.sep_embed = nn.Embedding(1, bert_config.hidden_size)
        self.mask_embed = nn.Embedding(1, bert_config.hidden_size)

    def extract_embeddings(self, embeddings: torch.tensor) -> torch.tensor:
        '''
        Extracts the embeddings for each frame from raw model outputs
        Removes special tokens and extra embeddings added to input 
        '''
        
        # Get rid of CLS and SEP tokens
        seq_len = int((embeddings.shape[1] - 4)/NUM_MICE)
        mouse_embeds = []
        pos = 1
        for m in range(NUM_MICE):
            mouse_embeds.append(embeddings[:, pos:pos+seq_len, :])
            pos += 1 + seq_len

        return mouse_embeds

    def create_inter_mouse_embeddings(self, embeddings) -> torch.tensor:
        '''
            This function creates embeddings for inter-mouse interactions
            We create an embedding for every possible pairing
            (m0, m1), (m0, m2), (m1, m0), (m1, m2)... etc
            We use these embeddings to predict inter (between) mouse tasks.
        '''
        pairs = {}

        # Now create the pairings:
        for m1 in range(NUM_MICE):
            pairs[m1] = {}

            for m2 in range(NUM_MICE):
                if m1 == m2:
                    continue
                pairs[m1][m2] = embeddings[m2] - embeddings[m1]
        
        return pairs

    def calculate_inter_mouse_loss(self, embeddings, annotations):
        '''
            This function calculates the loss for tasks with inter mouse annotations
        '''
        pair_embeddings = self.create_inter_mouse_embeddings(embeddings)
        loss = 0
        num = 0
        for m1 in range(NUM_MICE):
            for m2 in range(NUM_MICE):
                if m1 == m2:
                    continue
                loss += self.inter_mouse_task_head(pair_embeddings[m1][m2], annotations[m1][m2])
                num += 1

        return loss / num
    
    def forward(self, keypoints: torch.tensor = None, 
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
                        group_task_annotations: torch.tensor = None
                        ):
        '''
        Forward pass
        Replace special tokens in the input (CLR, SEP, MASK)
        Creates a label mask from which to apply the loss, this is the complement of attention mask

        Embeds the inputs and runs through the transformer.
        '''

        N = keypoints.shape[0]
        seq_len = int((keypoints.shape[1] - 4)/NUM_MICE)
         
        labels = labels if labels is not None else keypoints

        # positions to apply masking embedding
        masked_positions = masked_positions if masked_positions is not None else torch.zeros_like(attention_mask)

        # Create special token embeddings
        zeros = torch.zeros((N, 1), dtype=torch.long, device=labels.device) # not keen on this...
        cls_embed = self.cls_embed(zeros)
        sep_embed = self.sep_embed(zeros)
        mask_embed = self.mask_embed(zeros)

        # Embed raw features, add special tokens
        points_embed = self.points_embedding(keypoints) 
        points_embed[:, 0, :] = cls_embed.squeeze()
        points_embed[:, seq_len+1::seq_len+1, :] = sep_embed
        
        # Apply masking
        points_embed = masked_positions * mask_embed + ~masked_positions * points_embed

        # Add special tokens
        bert_inputs = {
            'inputs_embeds': points_embed,
            'attention_mask': attention_mask,
            'token_type_ids': segments
        }

        # Run through transformer, collect embedding and predict masked inputs
        bert_outputs = self.transformer(**bert_inputs, output_hidden_states=True, return_dict=True)
        embeddings = self.dropout(bert_outputs.last_hidden_state)
        embeddings = self.output_projection(embeddings)

        # Take CLS embedding
        extracted_embeddings = self.extract_embeddings(embeddings)
        combined_embeddings = torch.cat(extracted_embeddings, dim=2)
        pooled_embeddings = self.pooler(combined_embeddings)

        # Losses
        losses = {}
        loss = 0
        if sequence_replace_label is not None:
            seq_replace_loss, _ = self.seq_replace_head(pooled_embeddings, sequence_replace_label)
            loss += 0.1 * seq_replace_loss

        if chases is not None:
            chases_loss, _ = self.chases_head(embeddings, chases)
            losses['chases_loss'] = chases_loss
            loss += 0.05 * chases_loss
        
        if lights is not None:
            lights_loss, _ = self.lights_head(embeddings, lights)
            losses['lights_loss'] = lights_loss
            loss += 0.5 * lights_loss
        
        if task_annotations is not None:
            task_loss = self.task_head(embeddings, task_annotations)
            losses['task_loss'] = task_loss
            loss += 0.8 * task_loss

        if inter_mouse_task_annotations is not None:
            inter_task_loss = self.calculate_inter_mouse_loss(
                                    extracted_embeddings, 
                                    inter_mouse_task_annotations,
                                    label_mask
                                    )
            losses['inter_task_loss'] = inter_task_loss
            loss += 0.8 * inter_task_loss
            
        if group_task_annotations is not None:
            group_task_loss = self.group_task_head(combined_embeddings, group_task_annotations)
            losses['group_task_loss'] = group_task_loss
            loss += 0.4 * group_task_loss

        if label_mask is not None:
            mm_loss = self.mm_head(embeddings, labels, label_mask)
            losses['masked_modelling_loss'] = mm_loss
            loss += mm_loss

        return {
            'loss': loss, 
            'embeds': embeddings,
            'pooled_embeds': pooled_embeddings, 
            **losses
        }
