from gc import callbacks
from tomcat.model.mouse_bert import MouseBERT
from tomcat.model.mouse_bert_sep import MouseBERTSep
from tomcat.model.mouse_bert_contrast import MouseBERTContrast
from tomcat.dataset.masked_modelling_dataset import MaskedModelingDataset
from tomcat.dataset.masked_modelling_sep_dataset import MaskedModelingSepDataset
from tomcat.dataset.masked_modelling_sep_span_dataset import MaskedModelingSepSpanDataset
from tomcat.dataset.masked_modelling_contrast_dataset import MaskedModellingContrastDataset
from tomcat.dataset.mice_dataset import get_split_indices
from tomcat.dataset.full_sequence_dataset import FullSequenceDataset
from tomcat.dataset.full_sequence_sep_dataset import FullSequenceSepDataset
from tomcat.trainer.mouse_trainer import MouseTrainer, get_test_embeddings
from transformers import BertConfig, TrainingArguments
from torch.utils.data import DataLoader
import torch
from tomcat.utils.utils import validate_submission, build_submission, seed_everything
import numpy as np
from tomcat.consts import DEFAULT_NUM_TRAINING_POINTS, NUM_MICE
import argparse
from pathlib import Path
from typing import Union, List
from tomcat.transforms.augmentations import training_augmentations
from tomcat.transforms.keypoints_transform import get_svd_from_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from datetime import datetime
from tomcat.dataset.utils import get_multitask_datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(
    args: argparse.Namespace, 
    config: BertConfig,
    model: MouseBERT,
    train_datasets: List, 
    val_datasets: List
):
    '''Training'''
    logger = SummaryWriter(log_dir='model_outputs/logs/' +  
                            'h_' + str(args.hidden_size) + '_' +
                            'p_' + str(args.mask_prob) + '_' +
                            'len_' + str(args.max_seq_length) +
                            ('_sep_' if args.separate_mice else '_') +
                            datetime.now().strftime('%H:%M:%S'))

    training_args = TrainingArguments(num_train_epochs=args.epochs,
                                    learning_rate=args.learning_rate,
                                    per_device_train_batch_size=args.batch_size,
                                    evaluation_strategy='steps',
                                    logging_steps=1200,
                                    save_strategy='steps', 
                                    save_steps=1200,
                                    output_dir='model_outputs',
                                    dataloader_num_workers=6,
                                    load_best_model_at_end=True,
                                    fp16=args.fp16,
                                    save_total_limit=4
                                    )

    callback = [TensorBoardCallback(logger)]

    trainer = MouseTrainer(model,
                        args=training_args,
                        train_dataset=train_datasets,
                        eval_dataset=val_datasets,
                        data_collator=train_datasets[0].collator,
                        callbacks=callback
                        )

    trainer.train()
    return trainer.model

def test(model: MouseBERT, test_dataset: Union[FullSequenceDataset, FullSequenceSepDataset]) -> dict:
    '''Runs testing, return a submission'''
    embedding_dict, _ = get_test_embeddings(model, test_dataset)
    submission = build_submission(embedding_dict)
    return submission

def set_bert_config(args) -> BertConfig:
    '''Loads bert config and overrides using input args'''
    config = BertConfig.from_pretrained(args.pretrained_path)

    if args.hidden_size:
        config.hidden_size = args.hidden_size
    if args.intermediate_size:
        config.intermediate_size = args.intermediate_size
    if args.num_hidden_layers:
        config.num_hidden_layers = args.num_hidden_layers
    if args.num_attention_heads:
        config.num_attention_heads = args.num_attention_heads

    config.output_size = args.output_size
    if args.separate_mice and config.output_size % NUM_MICE != 0:
        config.output_size -= config.output_size % NUM_MICE
        print('Warning: Total output size should be a multiple of the number of mice, changing output_size from', args.output_size, 'to', config.output_size)

    return config

def main():
    '''
    main entrypoint
    '''
    args = get_args()
    seed_everything(args.seed)

    if args.separate_mice:
        MaskedDataset = MaskedModelingSepDataset
        FullDataset = FullSequenceSepDataset
        Model = MouseBERTSep

        if args.span_masking:
            MaskedDataset = MaskedModelingSepSpanDataset
        
        if args.contrastive:
            MaskedDataset = MaskedModellingContrastDataset
            Model = MouseBERTContrast

    else:
        MaskedDataset = MaskedModelingDataset
        FullDataset = FullSequenceDataset
        Model = MouseBERT
    
    config = set_bert_config(args)

    if args.train_path: 
        train_datasets, val_datasets, feature_transformers = get_multitask_datasets(args, MaskedDataset)

        model = Model(bert_config=config, num_features=train_datasets[0].num_features)

        model = train(
            args,
            config,
            model, 
            train_datasets, 
            val_datasets
        )
    
    if args.test_path:

        test_dataset = FullDataset(
            path=args.test_path,  
            max_seq_length=args.max_seq_length,
            **feature_transformers
        )

        submission = test(model, test_dataset)

        if validate_submission(submission, submission_clips=args.test_path):
            submission_path = Path("submissions")
            submission_path.mkdir(exist_ok=True)
            np.save(submission_path / args.submission_name, submission)


def get_args() -> argparse.Namespace:
    """
    Loads args for main()
    """
    parser = argparse.ArgumentParser(
        description="Embedding training facility for mice."
    )
    # Training settings
    parser.add_argument(
        "--pretrained_path",
        type=Path,
        required=True,
        help="Can be pretrained model or config file path, other args will override these settings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default = 8,
        help="batch size",
    )
    parser.add_argument(
        "--separate_mice",
        action='store_true',
        default=False,
        required=False,
        help="Do linear eval on public tasks."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=False,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=10,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=False,
        help="Encoder Embedding Size",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        required=False,
        default=100,
        help="Output Embedding Size",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=False,
        help="FC transformer hidden size",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        required=False,
        help="Embedding size",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        required=False,
        help="Embedding size",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        required=False,
        default=0.4,
        help="Float ratio to split between train and val.",
    )
    parser.add_argument(
        "--span_masking",
        action='store_true',
        default=False,
        required=False,
        help="Do span level masking.",
    )
    parser.add_argument(
        "--contrastive",
        action='store_true',
        default=False,
        required=False,
        help="Perform Contrastive Objective."
    )
    # Feature settings
    parser.add_argument(
        "--use_svd",
        action='store_true',
        default=False,
        required=False,
        help="Do linear eval on public tasks.",
    )
    # Data settings
    parser.add_argument(
        "--train_path",
        type=Path,
        required=False,
        help="Path to training dataset.",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        required=False,
        help="Path to test dataset.",
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        required=False,
        help="Size to scale the frame samples.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        required=True,
        help="Float ratio to split between train and val.",
    )
    parser.add_argument(
        "--submission_name",
        type=Path,
        required=False,
        help="Name of submission file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for reproducible results)",
    ) 
    parser.add_argument(
        "--fp16",
        action='store_true',
        default=False,
        required=False,
        help="Do FP16 training."
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
