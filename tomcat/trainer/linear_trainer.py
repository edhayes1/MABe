from collections import defaultdict
from transformers import Trainer
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LinearTrainer(Trainer):
    def compute_metrics(eval_preds):
        '''Computes metrics for linear evaluation'''

        logits, labels = eval_preds
        probs = torch.sigmoid(torch.tensor(logits))

        pr_auc = average_precision_score(labels, probs)

        return {'pr_auc': pr_auc}