from transformers import BertConfig, BertModel
from torch import nn
import torch
from tomcat.dataset.mice_dataset import get_split_indices

class LinearClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, embedding, labels):
        preds = self.classifier(embedding)
        loss = self.loss(preds, labels)
        return loss, preds

class MaskedModelingHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MaskedModelingHead, self).__init__()
        self.regression_layer = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss(reduction='none')
    
    def forward(self, x, labels, mask):
        preds = self.regression_layer(x)
        loss = ((mask * self.loss(preds, labels)).sum(dim=1)/mask.sum(dim=1)).mean()
        return loss, preds


class MouseBERT(nn.Module):
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
        
        super(MouseBERT, self).__init__()

        self.transformer = BertModel(bert_config)
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(bert_config.hidden_size, bert_config.output_size)
        self.mm_head = MaskedModelingHead(bert_config.output_size, num_features)

        self.space_embedding = nn.Linear(num_features, bert_config.hidden_size)

        # Special_tokens: CLS, SEP and MASK
        self.cls_embed = nn.Embedding(1, bert_config.hidden_size)
        self.sep_embed = nn.Embedding(1, bert_config.hidden_size)
        self.mask_embed = nn.Embedding(1, bert_config.hidden_size)

    def extract_embeddings(self, embeddings: torch.tensor) -> torch.tensor:
        '''Extracts the embeddings for each frame from raw model outputs'''
        return embeddings[:, 1:-1,:]
    
    def forward(self, keypoints: torch.tensor = None, 
                        attention_mask: torch.tensor = None,
                        label_mask: torch.tensor = None,
                        labels: torch.tensor = None,
                        masked_positions: torch.tensor = None,
                        chases: torch.tensor = None,
                        lights: torch.tensor = None):
        '''
        Forward pass
        Replace special tokens in the input (CLS, SEP, MASK)
        Creates a label mask from which to apply the loss, this is the complement of attention mask

        Embeds the inputs and runs through the transformer.
        '''

        N = keypoints.shape[0]
         
        labels = labels if labels is not None else keypoints

        # positions to apply masking embedding
        masked_positions = masked_positions if masked_positions is not None else torch.zeros_like(attention_mask)

        # Create special token embeddings
        zeros = torch.zeros((N, 1), dtype=torch.long, device=labels.device) # not keen on this...
        cls_embed = self.cls_embed(zeros)
        sep_embed = self.sep_embed(zeros)
        mask_embed = self.mask_embed(zeros)

        # Embed raw features, add special tokens
        points_embed = self.space_embedding(keypoints) 
        points_embed[:, 0, :] = cls_embed.squeeze()
        points_embed[:, -1, :] = sep_embed.squeeze()
        
        # Apply masking
        points_embed = masked_positions * mask_embed + ~masked_positions * points_embed

        # Add special tokens
        bert_inputs = {
            'inputs_embeds': points_embed,
            'attention_mask': attention_mask
        }

        # Run through transformer, collect embedding and predict masked inputs
        bert_outputs = self.transformer(**bert_inputs, output_hidden_states=True, return_dict=True)
        embeddings = bert_outputs.last_hidden_state
        # embeddings = self.dropout(bert_outputs.last_hidden_state)
        embeddings = self.projection(embeddings)
        
        loss = 0
        if label_mask is not None:
            loss, masked_preds = self.mm_head(embeddings, labels, label_mask)

        return loss, embeddings
