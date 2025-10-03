import os
import torch

from util.io import load_text


DATASETS = [
    'tennis',
    'fs_perf',
    'fs_comp',
    'finediving',
    'finegym',
    'soccernetv2',
    'soccernetball',
    'boxing',
    'focus',
]


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}

def read_fps(video_frame_dir):
    with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
        return float(fp.read())

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    """
    flattened_items = []
    for sample in batch:
        flattened_items.extend(sample)
    
    batched = {
        'video': [],
        'sub': [],
        'start': [],
        'frame_close': [],
        'frame_wide': [],
        'track': []
    }
    for item in flattened_items:
        batched['video'].append(item['video'])
        batched['sub'].append(item['sub'])
        batched['start'].append(item['start'])
        batched['frame_close'].append(item['frame_close'])
        batched['frame_wide'].append(item['frame_wide'])
        batched['track'].append(item['track'])
    
    batched['start'] = torch.tensor(batched['start'])
    batched['frame_close'] = torch.stack(batched['frame_close'], dim=0)
    batched['frame_wide'] = torch.stack(batched['frame_wide'], dim=0)
    batched['track'] = torch.stack(batched['track'], dim=0)
    
    return batched