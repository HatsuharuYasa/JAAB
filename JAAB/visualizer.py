import os
import json
import cv2
import re
from tqdm import tqdm
from multiprocessing import Pool, current_process, cpu_count

# const var
THRSH = 0.5

# Paths
video_folder = 'Path to the video'
json_file = 'Path to the test prediction file'
output_folder = 'Path to the result video'
track_dir = 'Path to the track annotation'

os.makedirs(output_folder, exist_ok=True)

# Load JSON
with open(json_file, 'r') as f:
    annotations = json.load(f)

def find_nearest_keyframe(track, idx_frame, start_frame):
    pass


def process_video_range(ann):
    original_video_name = ann['video']
    base_name = original_video_name.split('_')[0]

    track_dict = json.load(open(track_dir, 'r'))
    
    video_path = os.path.join(video_folder, base_name + '.mp4')
    
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found. Skipping.")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_folder, f"{original_video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps // 2, (width, height))

    start_frame = int(original_video_name.split('_')[-2])
    end_frame = int(original_video_name.split('_')[-1])

    print(f"Processing: {video_path} -> {output_path}")

    stacked_frame = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # tqdm progress bar for range
    position = current_process()._identity[0] - 1  # Get the position for the progress bar
    for frame_idx in tqdm(range(start_frame, end_frame), desc=f"{original_video_name} range {start_frame} - {end_frame}", unit="frame", position=position):
        ret, frame = cap.read()
        if not ret:
            break
        
        stacked_frame.append(frame)

    annotated_frame = [False] * len(stacked_frame)

    # Draw annotations for this frame if any
    events_annotations = ann["events"]
    for annotation in events_annotations:
        label = annotation["label"]
        confidence = annotation["score"]
        idx = annotation["frame"]
        track_frame = str(start_frame + idx)

        if confidence < THRSH:
            continue

        # Prepare the box
        if track_frame in track_dict[base_name][annotation["object"]]:
            _box_x = track_dict[base_name][annotation["object"]][str(start_frame + idx)]["x"]
            _box_y = track_dict[base_name][annotation["object"]][str(start_frame + idx)]["y"]
        else:
            _box_x, _box_y = 0, 0 # Need to change this logic 
        _box_w, _box_h = 50, 50

        _box_tl = (int(_box_x - _box_w / 2), int(_box_y - _box_h / 2))
        _box_br = (int(_box_x + _box_w / 2), int(_box_y + _box_h / 2))

        # Prepare text
        text = f"{label} {confidence:.4f}"

        # Position for the text (top-left corner)
        text_pos = (int(_box_x - _box_w / 2), int(_box_y - _box_h / 2) - 10)

        # Draw the box
        cv2.rectangle(stacked_frame[idx], _box_tl, _box_br, (0, 255, 0), 2)

        # Add text to the frame
        font_size = 2
        width = 2
        cv2.putText(stacked_frame[idx], text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), width)

        # Mark the annotated frame flag as true
        annotated_frame[idx] = True

    for i, _frame in enumerate(stacked_frame):
        out.write(_frame)
        if annotated_frame[i]:
            for j in range (15):
                out.write(_frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    # Determine the number of processes to use
    num_processes = min(1, len(annotations))
    print(f"Using {num_processes} processes for parallel processing.")

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        pool.map(process_video_range, annotations)

    print("Annotation overlay complete.")
