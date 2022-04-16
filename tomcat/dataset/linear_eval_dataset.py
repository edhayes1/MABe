from torch.utils.data import Dataset
import torch
import random
from typing import Dict
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EmbeddingDataset(Dataset):
    '''
    Dataset for Evaluating embeddings on provided annotations
    Need to instantiate a new EmbeddingDataset for each annotation type (lights on, chasing)

    Takes 
        embedding_dict, sequence keys: embeddings
        annotation_dict, sequence keys: annotations
    '''
    def __init__(self, 
                embedding_dict: Dict[str, np.ndarray], 
                annotation_dict: Dict[str, np.ndarray]) -> None:

        # flatten
        embeddings = torch.cat(
            [embed for embed in embedding_dict.values()]
        )

        annotations = torch.cat(
            [label for label in annotation_dict.values()]
        )

        positive_embeddings = embeddings[annotations.bool()]
        negative_embeddings = embeddings[~annotations.bool()]

        # Downsample
        num_positives = min(positive_embeddings.shape[0], 5000)
        num_negatives = 5 * num_positives

        positive_embeddings = self.sample(positive_embeddings, num_positives)
        negative_embeddings = self.sample(negative_embeddings, num_negatives)

        self.embeddings = torch.cat([positive_embeddings, negative_embeddings]).to(device)
        self.annotations = torch.cat([
            torch.ones(num_positives, dtype=torch.float), 
            torch.zeros(num_negatives, dtype=torch.float)
            ]).to(device)
    
    def sample(self, data, n):
        '''Downsample data'''
        indices = random.sample(range(data.shape[0]), n)
        indices = torch.tensor(indices)
        return data[indices]

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        annotation = self.annotations[idx]
        
        return embedding, annotation

    def __len__(self):
        return self.embeddings.shape[0]
    
    def collator(self, batch):
        batch_embedding = torch.stack([i[0] for i in batch])
        batch_annotation = torch.stack([i[1] for i in batch])

        batch_annotation = batch_annotation[:, None]

        input = {
            'embedding': batch_embedding,
            'labels': batch_annotation
        }

        return input