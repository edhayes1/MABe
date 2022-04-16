import numpy as np
from pathlib import Path
import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_submission(submission, submission_clips):
    if isinstance(submission_clips, str) or isinstance(submission_clips, Path):
        submission_clips = np.load(submission_clips, allow_pickle=True).item()
    elif not isinstance(submission, dict):
        print("Submission clips should be path or dict")
        return False

    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if 'frame_number_map' not in submission:
        print("Frame number map missing")
        return False

    if 'embeddings' not in submission:
        print('Embeddings array missing')
        return False
    elif not isinstance(submission['embeddings'], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission['embeddings'].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission['embeddings'].shape[1] <= 128:
        print("Embeddings too large, max allowed is 128")
        return False
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False
    
    total_clip_length = 0
    for key in submission_clips['sequences']:
        start, end = submission['frame_number_map'][key]
        clip_length = submission_clips['sequences'][key]['keypoints'].shape[0]
        total_clip_length += clip_length
        if not end-start == clip_length:
            print(f"Frame number map for clip {key} doesn't match clip length")
            return False
            
    if not len(submission['embeddings']) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission['embeddings']).all():
        print(f"Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True

def build_submission(submission_dict):
    frame_number_map = {}
    embeddings = []

    i = 0
    for key, seq_embed in submission_dict.items():
        seq_len = seq_embed.shape[0]
        frame_number_map[key] = (i, i + seq_len)
        embeddings.append(seq_embed)

        i += seq_len
    
    embeddings = np.stack(embeddings).reshape(i, -1)

    return {'frame_number_map': frame_number_map, 'embeddings': embeddings}