# #!/usr/bin/env python3

# import os
# import argparse
# from typing import NamedTuple
# import numpy as np
# import cv2
# cv2.setNumThreads(0)
# from tqdm import tqdm
# from multiprocessing import Pool

# from util.io import load_json


# FS_LABEL_DIR = 'data/fs_comp'
# TENNIS_LABEL_DIR = 'data/tennis'


# class Task(NamedTuple):
#     video_name: str
#     video_path: str
#     out_path: str
#     min_frame: int
#     max_frame: int
#     target_fps: float
#     max_height: int


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('dataset', choices=['fs', 'tennis'],
#                         help='Dataset to extract frames for.')
#     parser.add_argument('video_dir', help='Path to the videos')
#     parser.add_argument('-o', '--out_dir',
#                         help='Path to write frames. Dry run if None.')
#     parser.add_argument('--max_height', type=int, default=224,
#                         help='Max height of the extracted frames')
#     parser.add_argument('--parallelism', type=int, default=os.cpu_count() // 4)
#     return parser.parse_args()


# def get_fs_tasks(video_dir, out_dir, max_height):
#     tasks = []

#     for split in ['train', 'val', 'test']:
#         split_file = os.path.join(FS_LABEL_DIR, split + '.json')
#         labels = load_json(split_file)
#         for data in labels:
#             video_name = data['video']
#             base_video_name, _, start_frame, end_frame = video_name.rsplit(
#                 '_', 3)
#             start_frame = int(start_frame)
#             end_frame = int(end_frame)
#             assert end_frame - start_frame == data['num_frames']

#             video_out_path = None
#             if out_dir is not None:
#                 video_out_path = os.path.join(out_dir, base_video_name) #change made here

#             video_path = os.path.join(video_dir, base_video_name + '.mkv')
#             tasks.append(Task(
#                 video_name=video_name, video_path=video_path,
#                 out_path=video_out_path,
#                 min_frame=start_frame, max_frame=end_frame,
#                 target_fps=data['fps'], max_height=max_height
#             ))
#     return tasks


# def get_tennis_tasks(video_dir, out_dir, max_height):
#     video_files = os.listdir(video_dir)
#     print("DEBUG: Video directory {}".format(video_dir))

#     def match_video_file(prefix):
#         for v in video_files:
#             if v.startswith(prefix):
#                 print("prefix: " + prefix)
#                 return v
#         print("prefix: " + prefix)
#         raise Exception('Not found: {}'.format(prefix))

#     tasks = []
#     for split in ['train', 'val', 'test']:
#         split_file = os.path.join(TENNIS_LABEL_DIR, split + '.json')
#         labels = load_json(split_file)
#         for data in labels:
#             video_name = data['video']
#             base_video_name, start_frame, end_frame = video_name.rsplit('_', 2)
#             print("DEBUG: base video {}".format(base_video_name))
#             start_frame = int(start_frame)
#             end_frame = int(end_frame)
#             assert end_frame - start_frame == data['num_frames']

#             video_out_path = None
#             if out_dir is not None:
#                 video_out_path = os.path.join(out_dir, video_name)

#             video_path = os.path.join(
#                 video_dir, match_video_file(base_video_name))
#             tasks.append(Task(
#                 video_name=video_name, video_path=video_path,
#                 out_path=video_out_path,
#                 min_frame=start_frame, max_frame=end_frame,
#                 target_fps=data['fps'], max_height=max_height
#             ))
#     return tasks


# def extract_frames(task):
#     vc = cv2.VideoCapture(task.video_path)
#     fps = vc.get(cv2.CAP_PROP_FPS)
#     exp_num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
#     w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     if task.max_height < h:
#         oh = task.max_height
#         ow = int(w / h * task.max_height)
#     else:
#         oh, ow = h, w

#     assert np.isclose(fps, task.target_fps), (fps, task.target_fps)

#     if task.out_path is not None:
#         os.makedirs(task.out_path, exist_ok=True) # change 

#     vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
#     i = 0
#     while True:
#         ret, frame = vc.read()
#         if not ret:
#             break

#         if frame.shape[0] != oh:
#             frame = cv2.resize(frame, (ow, oh))

#         if task.out_path is not None:
#             frame_path = os.path.join(task.out_path, 'frame{:d}.jpg'.format(i))
#             cv2.imwrite(frame_path, frame)

#         i += 1
#         if task.min_frame + i == task.max_frame:
#             break

#     vc.release()
#     assert i == task.max_frame - task.min_frame, \
#         'Expected {} frames, got {}: {}'.format(
#             task.max_frame - task.min_frame, i, task.video_name)


# def main(dataset, video_dir, out_dir, max_height, parallelism):
#     if dataset == 'fs':
#         tasks = get_fs_tasks(video_dir, out_dir, max_height)
#     elif dataset == 'tennis':
#         tasks = get_tennis_tasks(video_dir, out_dir, max_height)
#     else:
#         raise Exception('Unknown dataset: ' + dataset)

#     is_dry_run = False
#     if out_dir is None:
#         print('No output directory given. Doing a dry run!')
#         is_dry_run = True
#     else:
#         os.makedirs(out_dir)

#     with Pool(parallelism) as p:
#         for _ in tqdm(
#             p.imap_unordered(extract_frames, tasks),
#             total=len(tasks), desc='Dry run' if is_dry_run else 'Extracting'
#         ):
#             pass
#     print('Done!')


# if __name__ == '__main__':
#     main(**vars(get_args()))

import os
import argparse
from typing import NamedTuple
import numpy as np
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm
from multiprocessing import Pool
from util.io import load_json

TENNIS_LABEL_DIR = 'data/tennis'
BOXING_LABEL_DIR = 'data/boxing'

FOCUS_LABEL_DIR = 'data/focus'

class Task(NamedTuple):
    video_name: str
    video_path: str
    out_path: str
    min_frame: int
    max_frame: int
    target_fps: float
    max_height: int

class TaskFocus(NamedTuple):
    video_name: str
    sub_name: str
    video_path: str
    out_path: str
    min_frame: int
    max_frame: int
    target_fps: float
    crop_size: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['tennis', 'boxing', 'boxing2', 'focus'],
                        help='Dataset to extract frames for.')
    parser.add_argument('video_dir', help='Path to the videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--max_height', type=int, default=224,
                        help='Max height of the extracted frames')
    parser.add_argument('--parallelism', type=int, default=os.cpu_count() // 4)
    return parser.parse_args()

def get_tennis_tasks(video_dir, out_dir, max_height):
    video_files = os.listdir(video_dir)

    def match_video_file(prefix):
        for v in video_files:
            if v.startswith(prefix):
                return v
        raise Exception(f'Not found: {prefix}')

    tasks = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(TENNIS_LABEL_DIR, f'{split}.json')
        labels = load_json(split_file)

        for data in labels:
            video_name = data['video']
            base_video_name, start_frame, end_frame = video_name.rsplit('_', 2)
            start_frame = int(start_frame)
            end_frame = int(end_frame)

            # Ensure all frames go into a single folder for the video
            video_out_path = None
            if out_dir is not None:
                video_out_path = os.path.join(out_dir, base_video_name)  # FIXED: No frame numbers in directory name

            video_path = os.path.join(video_dir, match_video_file(base_video_name))
            tasks.append(Task(
                video_name=base_video_name,  # FIXED: Using base name without frame numbers
                video_path=video_path,
                out_path=video_out_path,
                min_frame=start_frame, max_frame=end_frame,
                target_fps=data['fps'], max_height=max_height
            ))
    return tasks

def get_boxing_tasks(video_dir, out_dir, max_height):
    video_files = os.listdir(video_dir)

    def match_video_file(prefix):
        for v in video_files:
            if v.startswith(prefix):
                return v
        raise Exception(f"Not found: {prefix}")

    tasks = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(BOXING_LABEL_DIR, f'{split}.json')
        labels = load_json(split_file)

        for data in labels:
            video_name = data['video']
            video_path = os.path.join(video_dir, match_video_file(video_name))
            start_frame = 0
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            else:
                end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)

            video_out_path = None
            if out_dir is not None:
                video_out_path = os.path.join(out_dir, video_name)
            
            tasks.append(Task(
                video_name=video_name,
                video_path=video_path,
                out_path=video_out_path,
                min_frame=start_frame, max_frame=end_frame,
                target_fps=data['fps'], max_height=max_height
            ))
    return tasks

def get_boxing_tasks_2(video_dir, out_dir, max_height):
    video_files = os.listdir(video_dir)

    def match_video_file(prefix):
        for v in video_files:
            if v.startswith(prefix):
                return v
        raise Exception(f'Not found: {prefix}')

    tasks = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(BOXING_LABEL_DIR, f'{split}.json')
        labels = load_json(split_file)

        for data in labels:
            video_name = data['video']
            base_video_name, start_frame, end_frame = video_name.rsplit('_', 2)
            start_frame = int(start_frame)
            end_frame = int(end_frame)

            # Ensure all frames go into a single folder for the video
            video_out_path = None
            if out_dir is not None:
                video_out_path = os.path.join(out_dir, base_video_name)  # FIXED: No frame numbers in directory name

            video_path = os.path.join(video_dir, match_video_file(base_video_name))
            tasks.append(Task(
                video_name=base_video_name,  # FIXED: Using base name without frame numbers
                video_path=video_path,
                out_path=video_out_path,
                min_frame=start_frame, max_frame=end_frame,
                target_fps=data['fps'], max_height=max_height
            ))
    return tasks

def get_focus_tasks(video_dir, out_dir, tracks, max_height, crop_size):
    video_files = os.listdir(video_dir)
    
    def match_video_file(prefix):
        for v in video_files:
            if v.startswith(prefix):
                return v
        raise Exception(f'Not found: {prefix}')

    tasks_wide = []
    tasks_close = []
    extract_history = []
    for split in ['train', 'val', 'test']:
        # Opening the annotation file one by one
        split_file = os.path.join(FOCUS_LABEL_DIR, f'{split}.json')
        labels = load_json(split_file)

        for data in labels:
            video_name = data['video'].split('_')[0]
            # Registering the video base name to avoid duplicates
            if video_name not in extract_history:
                extract_history.append(video_name)

                # Appending tasks for extracting the base video
                video_path = os.path.join(video_dir, match_video_file(video_name))
                start_frame = 0
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError("Could not open video")
                else:
                    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)

                video_out_path = None
                if out_dir is not None:
                    video_out_path = os.path.join(out_dir, video_name, "base")

                tasks_wide.append(Task(
                    video_name=video_name,
                    video_path=video_path,
                    out_path=video_out_path,
                    min_frame=start_frame, max_frame=end_frame,
                    target_fps=data['fps'], max_height=max_height
                ))
            
            track = tracks[video_name]

            # Appending tasks for extracting the cropped video
            start_frame = int(data['video'].split("_")[-2])
            end_frame = int(data['video'].split("_")[-1])
            
            for sub_name in track.keys():
                video_path = os.path.join(video_dir, match_video_file(video_name))

                video_out_path = None
                if out_dir is not None:
                    video_out_path = os.path.join(out_dir, video_name, sub_name)
                
                tasks_close.append(TaskFocus(
                    video_name=video_name,
                    sub_name=sub_name,
                    video_path=video_path,
                    out_path=video_out_path,
                    min_frame=start_frame, max_frame=end_frame,
                    target_fps=data['fps'], crop_size=crop_size
                ))
            
    return tasks_wide, tasks_close


def extract_frames(task):
    vc = cv2.VideoCapture(task.video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = int(vc.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print("DEBUG: Codec", codec)
    print("DEBUG: Video_path", task.video_path)

    # Resize frames to match the max height requirement
    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    assert np.isclose(fps, task.target_fps), (fps, task.target_fps)

    # Ensure the output directory exists without error
    if task.out_path is not None:
        os.makedirs(task.out_path, exist_ok=True)  # FIXED: Prevent errors if the directory already exists

    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = task.min_frame  # FIXED: Frame numbering is now continuous

    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if frame.shape[0] != oh:
            frame = cv2.resize(frame, (ow, oh))

        if task.out_path is not None:
            frame_path = os.path.join(task.out_path, f'frame{i}.jpg')  # FIXED: Continuous frame numbering
            cv2.imwrite(frame_path, frame)

        i += 1
        if i == task.max_frame:  # Stop when reaching max_frame
            break

    vc.release()
    assert i - task.min_frame == task.max_frame - task.min_frame, \
        f'Expected {task.max_frame - task.min_frame} frames, got {i - task.min_frame}: {task.video_name}'

def extract_crops(args):
    task, track = args
    vc = cv2.VideoCapture(task.video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_size = task.crop_size
    video_name = task.video_name
    sub_name = task.sub_name

    assert np.isclose(fps, task.target_fps), (fps, task.target_fps)

    # Ensure the output directory exists without error
    if task.out_path is not None:
        os.makedirs(task.out_path, exist_ok=True)  # FIXED: Prevent errors if the directory already exists

    skipped_list = []
    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = task.min_frame  # FIXED: Frame numbering is now continuous
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        # Acquire the crop coordinates
        if str(i) not in track[video_name][sub_name]:
            skipped_list.append(video_name + "_" + sub_name + "_" + str(i))
        else:
            _pos = track[video_name][sub_name][str(i)]
            x, y = int(_pos["x"]), int(_pos["y"])
            x0 = max(0, int(x - crop_size // 2))
            y0 = max(0, int(y - crop_size // 2))
            x1 = min(int(x + crop_size // 2), w)
            y1 = min(int(y + crop_size // 2), h)

            # Crop the frame
            cropped_frame = frame[y0:y1, x0:x1]
            
            # Pad the image if it is smaller than the crop size
            if cropped_frame.shape[0] != crop_size or cropped_frame.shape[1] != crop_size:
                delta_h = crop_size - cropped_frame.shape[0]
                delta_w = crop_size - cropped_frame.shape[1]
                #print(x0, y0, x1, y1, delta_h, delta_w)
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                cropped_frame = cv2.copyMakeBorder(cropped_frame, top, bottom, left, right,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            if task.out_path is not None:
                frame_path = os.path.join(task.out_path, f'frame{i}.jpg')
                cv2.imwrite(frame_path, cropped_frame)
        i += 1
        if i == task.max_frame:
            break

    vc.release()
    assert i - task.min_frame == task.max_frame - task.min_frame, \
        f'Expected {task.max_frame - task.min_frame} frames, got {i - task.min_frame}: {task.video_name}'
            


def main(dataset, video_dir, out_dir, max_height, parallelism):
    if dataset == 'tennis':
        tasks = get_tennis_tasks(video_dir, out_dir, max_height)
    elif dataset == 'boxing':
        tasks = get_boxing_tasks(video_dir, out_dir, max_height)
    elif dataset == 'boxing2':
        tasks = get_boxing_tasks_2(video_dir, out_dir, max_height)
    elif dataset == 'focus':
        track = load_json(os.path.join(FOCUS_LABEL_DIR, 'track.json'))
        tasks_wide, tasks_close = get_focus_tasks(video_dir, out_dir, track, max_height, 224)
    else:
        raise Exception(f'Unknown dataset: {dataset}')

    is_dry_run = False
    if out_dir is None:
        print('No output directory given. Doing a dry run!')
        is_dry_run = True
    else:
        os.makedirs(out_dir, exist_ok=True)

    if dataset == 'focus':
        with Pool(parallelism) as p:
            for _ in tqdm(
                p.imap_unordered(extract_frames, tasks_wide),
                total=len(tasks_wide), desc='Dry run' if is_dry_run else 'Extracting'
            ):
                pass
    
        with Pool(parallelism) as p:
            for _ in tqdm(
                p.imap_unordered(extract_crops, [(task, track) for task in tasks_close]),
                total=len(tasks_close), desc='Extracting crops'
            ):
                pass
    else:
        with Pool(parallelism) as p:
            for _ in tqdm(
                p.imap_unordered(extract_frames, tasks),
                total=len(tasks), desc='Dry run' if is_dry_run else 'Extracting'
            ):
                pass

    print('Done!')

if __name__ == '__main__':
    main(**vars(get_args()))
