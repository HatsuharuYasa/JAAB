#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
from util.io import load_json, load_text
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle
import math
import json

#Local imports


#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5

# Dataset class for JAAB only
class ActionSpotFocusDataset(Dataset):
    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            modality,                   # [rgb, bw, flow]
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            stride=1,                   # Downsample frame rate
            overlap=1,                  # Overlap between clips (in proportion to clip_len)
            radi_displacement=0,        # Radius of displacement for labels
            mixup=False,                # Mixup usage
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            dataset = 'focus'     # Dataset name
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._split = label_file.split('/')[-1].split('.')[0]
        self._track_path = "/".join(self._src_file.split('/')[:-1]) + '/track.json'
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        if overlap != 1:
            self._overlap = int((1-overlap) * clip_len)
        else:
            self._overlap = 1
        assert overlap >= 0 and overlap <= 1
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0

        # Label modifications
        self._radi_displacement = radi_displacement
        
        # Mixup
        self._mixup = mixup

        # Frame reader class
        self._frame_reader = FrameFocusReader(frame_dir, modality, dataset=dataset)

        # Track reader class
        self._track_loader = TrackFocusLoader(self._track_path)

        # Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        self._total_len = len(self._frame_paths)
    
    def _store_clips(self):
        # Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        
        if self._radi_displacement > 0:
            self._labelsD_store = []

        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])
            labels_file = video['events']
            
            #Acquire the base name of the video
            base_name = video['video'].split('_')[0]

            #Acquire the sub names from the labels
            sub_names = []
            for event in labels_file:
                sub_name = event['comment']
                if sub_name not in sub_names:
                    sub_names.append(sub_name)

            #Create every clips for every sub name for specific time length stated on the name of the video
            for sub_name in sub_names:
                for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):
                    
                    # Make sure the frame loader has both close and wide paths
                    frame_paths = self._frame_reader.load_paths(video['video'], sub_name, base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)
                    labels = []
                    if self._radi_displacement >= 0:
                        labelsD = []
                    
                    for event in labels_file:
                        # Check whether the event is for the current sub_name
                        if event['comment'] != sub_name:
                            continue

                        event_frame = event['frame']
                        label_idx = (event_frame - base_idx) // self._stride
                        if self._radi_displacement >= 0:
                            if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                                label = self._class_dict[event['label']]
                                for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                    labels.append({'label': label, 'label_idx': i})
                                    labelsD.append({'displ': i - label_idx, 'label_idx': i})
                        else: #EXCLUDE OR MODIFY FOR RADI OF 0
                            if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                                label = self._class_dict[event['label']]
                                for i in range(max(0, label_idx - self._dilate_len), min(self._clip_len, label_idx + self._dilate_len + 1)):
                                    labels.append({'label': label, 'label_idx': i})
                    
                    if frame_paths[2] != -1: #in case no frames were available
                        self._frame_paths.append(frame_paths) # Append the frame paths in pair
                        self._labels_store.append(labels)
                        if self._radi_displacement > 0:
                            self._labelsD_store.append(labelsD)
                    
        # Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)
        
        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print('Stored clips to ' + store_path)
        return

    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'rb') as f:
                self._labelsD_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return

    def _get_one(self):
        # Get random index
        idx = random.randint(0, self._total_len - 1)

        # Get frame_path and Labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]
        if self._radi_displacement > 0:
            dict_labelD = self._labelsD_store[idx]
        
        # Load the close and wide shot frames
        frames_close, frames_wide = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        # Load the track
        track = self._track_loader.load_track(frames_path, pad=True, stride=self._stride)

        # Process labels
        labels = np.zeros(self._clip_len, np.int64)
        for label in dict_label:
            labels[label['label_idx']] = label['label']
        
        # Process displacement labels
        if self._radi_displacement > 0:
            labelsD = np.zeros(self._clip_len, np.int64)
            for label in dict_labelD:
                labelsD[label['label_idx']] = label['displ']

            return {'frame_close': frames_close, 'frame_wide': frames_wide, 'track': track, 'contains_event': int(np.sum(labels) > 0),
                    'label': labels, 'labelD': labelsD}

        return {'frame_close': frames_close, 'frame_wide': frames_wide, 'track': track, 'contains_event': int(np.sum(labels) > 0),
                'label': labels}
    
    def __getitem__(self, unused):
        ret = self._get_one()
        
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            
            ret['frame_close2'] = mix['frame_close']
            ret['frame_wide2'] = mix['frame_wide']
            ret['track2'] = mix['track']
            ret['contains_event2'] = mix['contains_event']
            ret['label2'] = mix['label']
            if self._radi_displacement > 0:
                ret['labelD2'] = mix['labelD']

        return ret
    
    def __len__(self):
        return self._dataset_len
    
    def print_info(self):
        _print_info_helper(self._src_file, self._labels)

# Frame reader class specialed for JAAB
class FrameFocusReader:
    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self.modality = modality
        self.dataset = dataset
    
    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #.float() / 255 -> into model normalization / augmentations
        return img
    
    def load_paths(self, video_name, sub_name, start, end, stride=1):
        frame0 = int(video_name.split('_')[-2])
        base_name = video_name.split('_')[0]
        sub_name = sub_name
        path_wide = os.path.join(self._frame_dir, base_name, "base")
        path_close = os.path.join(self._frame_dir, base_name, sub_name)

        found_start = -1
        pad_start = 0
        pad_end = 0
        pad_mid = [0] * ((end-start) // stride)
        for frame_num in range(start, end, stride):
            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue

            frame = frame0 + frame_num
            frame_path_wide = os.path.join(path_wide, 'frame' + str(frame) + '.jpg')
            frame_path_close = os.path.join(path_close, 'frame' + str(frame) + '.jpg')
            base_path_close = path_close
            base_path_wide = path_wide
            ndigits = -1

            exist_frame_close = os.path.exists(frame_path_close)
            exist_frame_wide = os.path.exists(frame_path_wide)
            if exist_frame_wide & (found_start == -1):
                found_start = frame
            
            if not exist_frame_wide:
                pad_end += 1
                continue
            
            if not exist_frame_close:
                pad_mid[(frame_num//stride)-max(0, start)] = 1
        
        ret = [base_path_close, base_path_wide, found_start, pad_start, pad_end, pad_mid, ndigits, (end-start) // stride]

        return ret
    
    def load_frames(self, paths, pad=False, stride=1):
        base_path_close = paths[0]
        base_path_wide = paths[1]
        start = paths[2]
        pad_start = paths[3]
        pad_end = paths[4]
        pad_mid = paths[5]
        ndigits = paths[6]
        length = paths[7]

        ret_close = []
        ret_wide = []
        path_wide = os.path.join(base_path_wide, 'frame')
        path_close = os.path.join(base_path_close, 'frame')
        
        for i in range(length - pad_start - pad_end):
            ret_wide.append(self.read_frame(path_wide + str(start + i * stride) + '.jpg'))

            if pad_mid[i]:
                ret_close.append(torch.zeros((3, 224, 224), dtype=torch.uint8))
            else:
                ret_close.append(self.read_frame(path_close + str(start + i * stride) + '.jpg'))

        #_ = [ret_close.append(self.read_frame(path_close + str(start + j * stride) + '.jpg')) for j in range(length - pad_start - pad_end)]
        #_ = [ret_wide.append(self.read_frame(path_wide + str(start + j * stride) + '.jpg')) for j in range(length - pad_start - pad_end)]

        ret_close = torch.stack(ret_close, dim=int(len(ret_close[0].shape) == 4))
        ret_wide = torch.stack(ret_wide, dim=int(len(ret_wide[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret_close = torch.nn.functional.pad(
                ret_close, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))
            ret_wide = torch.nn.functional.pad(
                ret_wide, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))

        return ret_close, ret_wide

#Specialized class to load tracks for JAAB
class TrackFocusLoader:
    def __init__(self, track_dir):
        self._track_dir = track_dir
        self._track_dict = json.load(open(track_dir, 'r'))
    
    def load_track(self, paths, pad=False, stride=1):
        base_name = paths[0].split('/')[-2]
        sub_name = paths[0].split('/')[-1]
        start = paths[2]
        pad_start = paths[3]
        pad_end = paths[4]
        pad_mid = paths[5]
        length = paths[7]
        _track = self._track_dict[base_name][sub_name]
        
        ret = []
        for i in range(length - pad_start - pad_end):
            if pad_mid[i]:
                ret.append(torch.zeros((2), dtype=torch.float))
            else:
                frame = str(start + i * stride)
                ret.append(torch.tensor([_track[frame]["x"], _track[frame]["y"]], dtype=torch.float))
        
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Pad the tensor according to the stacked frame length
        ret = torch.nn.functional.pad(
            ret, (0, 0, pad_start, pad_end if pad else 0))
        return ret
        
# Specialized class of ActionSpotVideoDataset for JAAB
class ActionSpotFocusVideoDataset(Dataset):
    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            dataset='focus'
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._track_path = "/".join(self._src_file.split('/')[:-1]) + '/track.json'
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride
        self._dataset = dataset

        self._frame_reader = FrameReaderFocusVideo(frame_dir, modality, dataset=dataset)

        self._track_loader = TrackFocusVideoLoader(self._track_path)

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l
    
    def __len__(self):
        return len(self._clips)
    
    def __getitem__(self, idx):
        video_name, start = self._clips[idx]
        items = []
        tracks = self._track_loader.load_track(
            video_name, start,
            start + self._clip_len * self._stride, pad=True,
            stride=self._stride
        )

        #Itterate for every object found in the tracks
        for _track, _obj in tracks:
            frames_close, frames_wide, _start, _end = self._frame_reader.load_frames(
                video_name, _obj, start,
                start + self._clip_len * self._stride, pad=True,
                stride=self._stride)
            
            # Repad of the track according to the found frame
            _pad_track = torch.full(_track.size(), fill_value=0, dtype=torch.float)
            _pad_track[_start:_end] = _track[_start:_end]

            items.append({
                'video': video_name,
                'sub': _obj,
                'start': start // self._stride,
                'frame_close': frames_close,
                'frame_wide': frames_wide,
                'track': _pad_track
            })
        
        return items
    
    def get_labels(self, video, sub):
        meta = self._labels[self._video_idxs[video]]
        labels_file = meta['events']

        num_frames = meta['num_frames']
        num_labels = math.ceil(num_frames / self._stride)
        
        labels = np.zeros(num_labels, np.int64)
        for event in labels_file:
            if event['comment'] == sub:
                frame = event['frame']
                if (frame < num_frames):
                    labels[frame // self._stride] = self._class_dict[event['label']]
                else:
                    print('Warning: {} >= {} is past the end {}'.format(
                        frame, num_frames, meta['video']))
        return labels
    
    @property
    def videos(self):
        _videos = []
        _tracks = self._track_loader.tracks
        for label in self._labels:
            base_name = label['video'].split('_')[0]
            for _obj in _tracks[base_name].keys():
                _videos.append((label['video'], _obj, math.ceil(label['num_frames'] / self._stride), label['fps'] / self._stride))
                
        _videos = sorted(_videos)
        return _videos
    
    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)

                x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride

                for e in x_copy['events']:
                    e['frame'] //= self._stride
                
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._labels), num_frames, self._stride)
        )
    
def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        #num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))
        
        #print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
        #    src_file, len(labels), num_frames,
        #    num_events / num_frames * 100))

# FrameReaderVideo class specialized for JAAB
class FrameReaderFocusVideo:
    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self._modality = modality
        assert self._modality == 'rgb'
        self._dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #.float() / 255 -> into model normalization / augmentations
        if frame_path.split('/')[-2] != 'base':
            assert img.shape[0] == 3 and img.shape[1] == 224 and img.shape[2] == 224, \
                'Image shape is not 3x224x224, but {}. at {}'.format(img.shape, frame_path)
        return img

    def load_frames(self, video_name, obj_name, start, end, pad=False, stride=1):
        n_pad_start = 0
        n_pad_end = 0
        found_start = -1
        found_end = end
        base_name = video_name.split('_')[0]
        sub_name = obj_name

        path_wide = os.path.join(self._frame_dir, base_name, "base")
        path_close = os.path.join(self._frame_dir, base_name, sub_name)

        frame0 = int(video_name.split('_')[-2])

        ret_close = []
        ret_wide = []
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                n_pad_start += 1
                continue
            
            if n_pad_end > 0:
                n_pad_end += 1
                continue
            
            frame = frame0 + frame_num
            frame_path_wide = os.path.join(path_wide, 'frame' + str(frame) + '.jpg')
            frame_path_close = os.path.join(path_close, 'frame' + str(frame) + '.jpg')

            exist_frame_close = os.path.exists(frame_path_close)
            exist_frame_wide = os.path.exists(frame_path_wide)

            if exist_frame_wide:
                if(found_start == -1):
                    found_start = frame
                ret_wide.append(self.read_frame(frame_path_wide))
            
            if not exist_frame_wide:
                if(found_start == -1):
                    n_pad_start += 1
                else:
                    n_pad_end += 1
                    found_end = frame-1
                continue
            
            if exist_frame_close:
                ret_close.append(self.read_frame(frame_path_close))
            else:
                ret_close.append(torch.zeros((3, 224, 224), dtype=torch.uint8))        

        if found_start == -1:
            return -1 # Return -1 if no frames were loaded

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret_close = torch.stack(ret_close, dim=int(len(ret_close[0].shape) == 4))
        ret_wide = torch.stack(ret_wide, dim=int(len(ret_wide[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret_close = torch.nn.functional.pad(
                ret_close, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad
                else 0))
            ret_wide = torch.nn.functional.pad(
                ret_wide, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad
                else 0))
        return ret_close, ret_wide, (found_start - start - frame0) // stride, (found_end - start) // stride

class TrackFocusVideoLoader:
    def __init__(self, track_dir):
        self._track_dir = track_dir
        self._track_dict = json.load(open(track_dir, 'r'))
    
    def load_track(self, video_name, start, end, pad=False, stride=1):
        base_name = video_name.split('_')[0]
        start = start
        n_pad_start = 0
        n_pad_end = 0

        _objs = self._track_dict[base_name].keys()

        frame0 = int(video_name.split('_')[-2])
        
        rets = []
        for _obj in _objs:
            _track = self._track_dict[base_name][_obj]
            ret = []
            for i in range(start, end, stride):
                if i < 0:
                    n_pad_start += 1
                    continue
                
                frame = str(frame0 + i)

                if frame in _track:
                    ret.append(torch.tensor([_track[frame]["x"], _track[frame]["y"]], dtype=torch.float))
                else:
                    ret.append(torch.zeros((2), dtype=torch.float))

            if len(ret) == 0:
                continue
            
            ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))
            # Always pad start, but only pad end if requested
            if n_pad_start > 0 or (pad and n_pad_end > 0):
                ret = torch.nn.functional.pad(
                    ret, (0, 0, n_pad_start, 0))
            rets.append((ret, str(_obj)))

            # Reset the padding
            n_pad_start = 0
        return rets

    @property
    def tracks(self):
        return self._track_dict

class ActionSpotDatasetJoint(Dataset):

    def __init__(
            self,
            dataset1,
            dataset2
    ):
        self._dataset1 = dataset1
        self._dataset2 = dataset2
        

    def __getitem__(self, unused):

        if random.random() < 0.5:
            data = self._dataset1.__getitem__(unused)
            data['dataset'] = 1
            return data
        else:
            data = self._dataset2.__getitem__(unused)
            data['dataset'] = 2
            return data

    def __len__(self):
        return self._dataset1.__len__() + self._dataset2.__len__()