#Standard imports
import sys
import argparse
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import defaultdict

#Local imports
from TDEED.util.io import load_json, store_json, load_text
from TDEED.model.model import TDEEDModel
from TDEED.util.dataset import load_classes

input_path = sys.argv[1]
output_path = sys.argv[2]

DEVICE = 'cuda'
THRSH = 0.5

# Wrist keypoint indices (COCO format)
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

# Hardcoded boxer id
LEFT_BOXER = 0
RIGHT_BOXER = 1

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--input', type=str, default='input.mp4')
    parser.add_argument('--output', type=str, default='output.mp4')
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['store_dir']
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.mixup = config['mixup']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']
    args.only_test = config['only_test']
    args.criterion = config['criterion']
    args.num_workers = config['num_workers']
    if 'pretrain' in config:
        args.pretrain = config['pretrain']
    else:
        args.pretrain = None

    return args

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    x_intersection_min = max(x_min1, x_min2)
    y_intersection_min = max(y_min1, y_min2)
    x_intersection_max = min(x_max1, x_max2)
    y_intersection_max = min(y_max1, y_max2)

    intersection_area = max(0, x_intersection_max - x_intersection_min) * max(0, y_intersection_max - y_intersection_min)

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def resize_image(img, dim):
    _h, _w = img.shape[:2]
    scale = dim / min(_h, _w)
    new_h, new_w = int(scale * _h), int(scale * _w)

    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img

def track_glove(frames, p_model, b_model, tracker, clip_len):
    track_dict = defaultdict(lambda:{
        LEFT_WRIST_IDX: {},
        RIGHT_WRIST_IDX: {}
    })
    track_boxes = defaultdict(dict)
    _prev_left_center, _prev_right_center = None, None

    for idx, buffered_frame in enumerate(frames):     
        print(idx,":", len(frames))   
        object_results = b_model(buffered_frame)[0]
        pose_results = p_model(buffered_frame)[0]

        detections = []

        if object_results.boxes is not None and pose_results.keypoints is not None:
            object_boxes = object_results.boxes.xyxy.int().cpu().numpy()
            object_confidence = object_results.boxes.conf.cpu().numpy()
            pose_keypoints_list = pose_results.keypoints.xy.int().cpu().numpy()
            pose_confidences_list = pose_results.boxes.conf.cpu().numpy()
            pose_boxes = pose_results.boxes.xyxy.int().cpu().numpy()
        
        for i, obj_box in enumerate(object_boxes):
            if object_confidence[i] < 0.9:
                continue

            best_match_keypoints = None
            best_match_confidences = 0
            max_iou = 0.0

            for j, pose_box in enumerate(pose_boxes):
                iou = calculate_iou(obj_box, pose_box)
                if iou > max_iou and iou >= 0.7:
                    max_iou = iou
                    best_match_keypoints = pose_keypoints_list[j]
                    best_match_confidences = pose_confidences_list[j]

            if best_match_confidences > 0.5:
                x1, y1, x2, y2 = obj_box
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                detections.append(
                    Detection(
                        points=center,
                        scores=np.array([best_match_confidences]),
                        data={"keypoints": best_match_keypoints, "box": obj_box, "frame_idx": idx}
                    )
                )

            # if best_match_keypoints:
            #     left_wrist_index = 9
            #     right_wrist_index = 10
            #     wrist_points = {}

            #     if left_wrist_index < len(best_match_keypoints) and best_match_confidences[left_wrist_index] > 0.75:
            #         lx, ly = best_match_keypoints[left_wrist_index]
            #         wrist_points['left_wrist'] = (lx, ly)

            #     if right_wrist_index < len(best_match_keypoints) and best_match_confidences[right_wrist_index] > 0.75:
            #         rx, ry = best_match_keypoints[right_wrist_index]
            #         wrist_points['right_wrist'] = (rx, ry)

            #     if wrist_points:
            #         frame_keypoints_data.append({
            #             'bounding_box': obj_box,
            #             'wrist_keypoints': wrist_points
            #         })



            
        # for result in results:
        #     boxes = result.boxes.xyxy.cpu().numpy()
        #     confidences = result.boxes.conf.cpu().numpy()
        #     keypoints = result.keypoints.xy.cpu().numpy()

        #     for i, (box, conf, kpts) in enumerate(zip(boxes, confidences, keypoints)):
        #         if conf > 0.5:
        #             x1, y1, x2, y2 = box
        #             center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        #             detections.append(
        #                 Detection(
        #                     points=center,
        #                     scores=np.array([conf]),
        #                     data={"keypoints": kpts, "box": box, "frame_idx": idx}
        #                 )
        #             )
    
        tracked_objects = tracker.update(detections=detections)

        # Find the most left and the most right boxer
        left_boxer_center, left_boxer_id = None, None
        right_boxer_center, right_boxer_id = None, None
        for i, tracked_object in enumerate(tracked_objects):
            if i == 0:
                left_boxer_center = tracked_object.estimate[0]
                left_boxer_id = i
                right_boxer_center = tracked_object.estimate[0]
                right_boxer_id = i
            else:
                if tracked_object.estimate[0][0] < left_boxer_center[0]:
                    left_boxer_center = tracked_object.estimate[0]
                    left_boxer_id = i
                if tracked_object.estimate[0][0] > right_boxer_center[0]:
                    right_boxer_center = tracked_object.estimate[0]
                    right_boxer_id = i
    
        # Retrieve the keypoints and boxes for the boxer
        if left_boxer_id is not None or right_boxer_id is not None:
            # Retrieve the keypoints and boxes for the left boxer
            if left_boxer_id is not None:
                _left_kpts = tracked_objects[left_boxer_id].last_detection.data["keypoints"]
                _left_box = tracked_objects[left_boxer_id].last_detection.data["box"]
                _left_frame_idx = tracked_objects[left_boxer_id].last_detection.data["frame_idx"]

            # Retrieve the keypoints and boxes for the right boxer
            if right_boxer_id is not None:
                _right_kpts = tracked_objects[right_boxer_id].last_detection.data["keypoints"]
                _right_box = tracked_objects[right_boxer_id].last_detection.data["box"]
                _right_frame_idx = tracked_objects[right_boxer_id].last_detection.data["frame_idx"]
            
            if left_boxer_center[0] == right_boxer_center[0]:
                # Try to classify the boxer based on the distance
                if _prev_left_center is not None and _prev_right_center is not None:
                    _left_dist = np.linalg.norm(left_boxer_center - _prev_left_center)
                    _right_dist = np.linalg.norm(right_boxer_center - _prev_right_center)
                    if _left_dist <= _right_dist:
                        if _left_kpts is not None:
                            if _left_kpts[LEFT_WRIST_IDX].any():
                                track_dict[LEFT_BOXER][LEFT_WRIST_IDX][_left_frame_idx] = _left_kpts[LEFT_WRIST_IDX]
                            if _left_kpts[RIGHT_WRIST_IDX].any():
                                track_dict[LEFT_BOXER][RIGHT_WRIST_IDX][_left_frame_idx] = _left_kpts[RIGHT_WRIST_IDX]
                        track_boxes[LEFT_BOXER][_left_frame_idx] = _left_box
                    else:
                        if _right_kpts is not None:
                            if _right_kpts[LEFT_WRIST_IDX].any():
                                track_dict[RIGHT_BOXER][LEFT_WRIST_IDX][_right_frame_idx] = _right_kpts[LEFT_WRIST_IDX]
                            if _right_kpts[RIGHT_WRIST_IDX].any():
                                track_dict[RIGHT_BOXER][RIGHT_WRIST_IDX][_right_frame_idx] = _right_kpts[RIGHT_WRIST_IDX]
                        track_boxes[RIGHT_BOXER][_right_frame_idx] = _right_box
                else:
                    if _left_kpts is not None:
                        if _left_kpts[LEFT_WRIST_IDX].any():
                            track_dict[LEFT_BOXER][LEFT_WRIST_IDX][_left_frame_idx] = _left_kpts[LEFT_WRIST_IDX]
                        if _left_kpts[RIGHT_WRIST_IDX].any():
                            track_dict[LEFT_BOXER][RIGHT_WRIST_IDX][_left_frame_idx] = _left_kpts[RIGHT_WRIST_IDX]
                    track_boxes[LEFT_BOXER][_left_frame_idx] = _left_box
            else:
                _prev_left_center = left_boxer_center
                _prev_right_center = right_boxer_center

                # Note the left boxer to the dictionary
                if _left_kpts is not None:
                    if _left_kpts[LEFT_WRIST_IDX].any():
                        track_dict[LEFT_BOXER][LEFT_WRIST_IDX][_left_frame_idx] = _left_kpts[LEFT_WRIST_IDX]
                    if _left_kpts[RIGHT_WRIST_IDX].any():
                        track_dict[LEFT_BOXER][RIGHT_WRIST_IDX][_left_frame_idx] = _left_kpts[RIGHT_WRIST_IDX]
                track_boxes[LEFT_BOXER][_left_frame_idx] = _left_box

                # Note the right boxer to the dictionary
                if _right_kpts is not None:
                    if _right_kpts[LEFT_WRIST_IDX].any():
                        track_dict[RIGHT_BOXER][LEFT_WRIST_IDX][_right_frame_idx] = _right_kpts[LEFT_WRIST_IDX]
                    if _right_kpts[RIGHT_WRIST_IDX].any():
                        track_dict[RIGHT_BOXER][RIGHT_WRIST_IDX][_right_frame_idx] = _right_kpts[RIGHT_WRIST_IDX]
                track_boxes[RIGHT_BOXER][_right_frame_idx] = _right_box
    
    # Construct the tensor track
    tracks = []
    for _obj in track_dict.values():
        for _hand in _obj.values():
            if len(_hand) != 0:
                track = np.zeros((clip_len, 2))
                for _idx, _point in _hand.items():
                    track[_idx, :] = _point
                tracks.append(track)
    
    # Stack the tracks
    tracks = np.stack(tracks, axis=0)

    # Convert to tensor
    tracks = torch.tensor(tracks, dtype=torch.float)

    return tracks, track_dict, track_boxes


def crop_local_image(global_frames, tracks, dim):
    _crop_dim = dim
    _local_frames = []
    for i in range(tracks.shape[0]):
        _frames = []
        for j in range(tracks.shape[1]):
            _frame = global_frames[j]
            _h, _w = _frame.shape[:2]
            _x, _y = tracks[i][j][0], tracks[i][j][1]

            if _x == 0 and _y == 0:
                _frames.append(np.zeros((dim, dim, 3)))
                continue

            # Find the border of the crop
            x0 = int(max(0, _x - _crop_dim / 2))
            y0 = int(max(0, _y - _crop_dim / 2))
            x1 = int(min(_w, _x + _crop_dim / 2))
            y1 = int(min(_h, _y + _crop_dim / 2))
        
            # Crop the frame
            _cropped_frame = _frame[y0:y1, x0:x1]

            # Pad the cropped frame if the size doesnt match with the crop dim
            if _cropped_frame.shape[0] != _crop_dim or _cropped_frame.shape[1] != _crop_dim:
                delta_h = _crop_dim - _cropped_frame.shape[0]
                delta_w = _crop_dim - _cropped_frame.shape[1]

                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                _cropped_frame = cv2.copyMakeBorder(_cropped_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            _frames.append(_cropped_frame)
        _frames = np.asarray(_frames)
        _frames = np.transpose(_frames, (0, 3, 1, 2)) # (T, C, H, W)

        _local_frames.append(_frames)
    
    # Stack the list
    _local_frames = np.stack(_local_frames, axis=0)

    # Conver the local frames to tensor
    _local_frames = torch.tensor(_local_frames, dtype=torch.uint8)

    return _local_frames

def process_frame_predictions(classes, scores, supports, high_recall_score_threshold=0.1):
    classes_inv = {v: k for k, v in classes.items()}

    pred_events_high_recall = []

    for i in range(scores.shape[0]):
        score, support = scores[i], supports[i]
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0
        score /= support[:, None]
        pred = np.argmax(score, axis=1)

        events_high_recall = []
        for j in range(pred.shape[0]):
            for k in classes_inv:
                if score[j, k] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[k],
                        'frame': j,
                        'score': score[j, k].item()
                    })

        pred_events_high_recall.append(events_high_recall)
    return pred_events_high_recall

def update_score(frames, pos, size, info, person_id):
    # Draw black background
    x, y = pos
    w, h = size

    for i, frame in enumerate(frames):
        # Update the score for current frame
        _hit = info["hit"][0] = info["hit"][0] + info["hit"][1][i]
        _miss = info["miss"][0] = info["miss"][0] + info["miss"][1][i]
        _block = info["block"][0] = info["block"][0] + info["block"][1][i]

        # Draw the black rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        # Prepare the text
        _boxer_text = "Left Boxer" if person_id == LEFT_BOXER else "Right Boxer"
        _hit_text = f"Hit: {_hit}"
        _miss_text = f"Miss: {_miss}"
        _block_text = f"Block: {_block}"

        # Prepare the color
        color = (0, 0, 255) if person_id == LEFT_BOXER else (255, 0, 0)

        for j in range(4):
            _text = _boxer_text if j == 0 else _hit_text if j == 1 else _miss_text if j == 2 else _block_text
            text_pos = (int(x + 10), int(y + (j + 1) * 40))
            cv2.putText(frame, _text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        

def lighten_color(color, factor=0.5):
    return tuple(int(c + (255 - c) * factor) for c in color)

def draw_events(frames, pred_events, track, track_box, scores):
    ann_frames = frames
    bool_frames = [False] * len(ann_frames)
    
    # Draw the wrist tracking and the event happened
    idx = 0
    for track_id, track_obj in track.items():
        color = (0, 0, 255) if track_id == LEFT_BOXER else (255, 0, 0)
        for hand_id, hand in track_obj.items():
            if len(hand) == 0:
                continue
            # Draw the wrist tracking
            color = lighten_color(color, 0.5) if hand_id == RIGHT_WRIST_IDX else color
            for frame_idx, point in hand.items():
                x, y = point[0], point[1]
                cv2.circle(ann_frames[frame_idx], (int(x), int(y)), 10, color, 10)
            
            # Draw the event happened
            events = pred_events[idx]
            for event in events:
                label = event["label"]
                conf = event["score"]
                frame_idx = event["frame"]

                if conf < THRSH or frame_idx not in hand:
                    continue

                # Quickly store the events to the score database
                scores[track_id][label][1][frame_idx] += 1

                # Prepare box
                _w, _h = 100, 100
                _box_x, _box_y = hand[frame_idx][0], hand[frame_idx][1]

                _box_tl = (int(_box_x - _w / 2), int(_box_y - _h / 2))
                _box_br = (int(_box_x + _w / 2), int(_box_y + _h / 2))

                # Prepare text
                text = f"{label} {conf:.4f}"

                # Position for the text
                text_pos = (int(_box_x - _w / 2), int(_box_y - _h / 2) - 10)

                # Draw the box
                cv2.rectangle(ann_frames[frame_idx], _box_tl, _box_br, (0, 255, 0), 2)

                # Add text to the frame
                font_size = 1
                width = 2
                cv2.putText(ann_frames[frame_idx], text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), width)

                # Mark the annotated
                bool_frames[frame_idx] = True
            
            idx += 1
    
    # Draw the bounding box
    for idx, frame in enumerate(ann_frames):
        for track_id, boxes in track_box.items():
                if idx in boxes:
                    # Prepare the box
                    x1, y1, x2, y2 = boxes[idx]
                    
                    # Prepare the text
                    text = "Left Boxer" if track_id == LEFT_BOXER else "Right Boxer"
                    text_pos = (int(x1), int(y1) - 10)

                    # Preparet the color
                    color = (0, 0, 255) if track_id == LEFT_BOXER else (255, 0, 0)

                    # Draw the text
                    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 2)

                    # Draw the box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw the score board
    # Draw score board for left boxer
    update_score(ann_frames, (20, 20), (200, 300), scores[LEFT_BOXER], LEFT_BOXER)
    # Draw score board for right boxer
    update_score(ann_frames, (1700, 20), (200, 300), scores[RIGHT_BOXER], RIGHT_BOXER)
    
    return ann_frames, bool_frames

def draw_frames(frames, bool_frames, out):
    for i, frame in enumerate(frames):
        out.write(frame)
        if bool_frames[i]:
            for j in range(15):
                out.write(frame)

def tensor_to_cv2_img(tensor_img):
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img

def arrange_grid(images, len, grid_shape = (2, 2)):
    rows = []
    idx = 0
    for _ in range(grid_shape[0]):
        row = []
        for _ in range(grid_shape[1]):
            if idx >= len:
                img = np.zeros((224, 224, 3), dtype = np.uint8)
            else:
                img = images[idx]
            row.append(img)
            idx += 1
        rows.append(np.hstack(row))
    return np.vstack(rows)            

def draw_parallel_frame(local_frames, out):
    _len = local_frames.shape[1]
    num_parallel = min(4, local_frames.shape[0])

    for i in range(_len):
        _batch_imgs = local_frames[:, i, :, :, :]
        _batch_imgs = [tensor_to_cv2_img(_batch_imgs[j]) for j in range(num_parallel)]
        _batch_len = len(_batch_imgs)

        _frames = arrange_grid(_batch_imgs, _batch_len)
        out.write(_frames)


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def model_inference(args):
    #Load the config file
    config_path = args.model.split('_')[0] + '/' + args.model + '.json'
    config = load_json(os.path.join('model/TDEED/config', config_path))
    args = update_args(args, config)

    #Load the classes
    _classes = load_classes(os.path.join('model/TDEED/data', args.dataset, 'class.txt'))
    
    # Build the model
    model = TDEEDModel(args=args)
    pose_model = YOLO("model/yolo11x-pose.pt")
    boxer_model = YOLO("model/best.pt")
    boxer_model.to(DEVICE)  
    pose_model.to(DEVICE)

    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=400)

    # Load the checkpoint
    model.load(torch.load(os.path.join(os.getcwd(), 'model/TDEED/checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt')))

    # Define some variable
    _clip_len = args.clip_len
    _crop_dim = args.crop_dim
    _num_classes = args.num_classes
    _origin_frames = []
    scores_database = defaultdict(lambda:{
        "miss": [0, [0] * _clip_len],
        "hit": [0, [0] * _clip_len],
        "block": [0, [0] * _clip_len],
    })
    
    # Load the path to the video
    input_path = args.input
    output_path = args.output
    output_local_path = args.output.split(".")[0] + "_local.mp4"
    
    # Open the video
    cap = cv2.VideoCapture(input_path)

    # Define variable for outputting video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps / 2, (width, height))
    #out_local = cv2.VideoWriter(output_local_path, fourcc, fps / 2, (_crop_dim * 2, _crop_dim * 2))

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()    
        if not ret:
            break

        _origin_frames.append(frame)

        # Check if the stacked frame reach the clip length
        if len(_origin_frames) == _clip_len:
            # Convert the frames to rgb format
            _rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in _origin_frames]

            # Pass in to the boxing glove detector
            track, track_dict, track_box = track_glove(_origin_frames, pose_model,boxer_model, tracker, _clip_len)

            # Extract the local image based on the detected key point
            frames_local_tensor = crop_local_image(_rgb_frames, track, _crop_dim)
            
            # Resize the global image
            _resized_images = [resize_image(img, _crop_dim) for img in _rgb_frames]

            # Create the global frames tensor as a stack
            frames_np = []
            for _ in range(track.shape[0]):
                frames_np.append(np.asarray(_resized_images))
            frames_np = np.stack(frames_np, axis=0)
            frames_global_tensor = torch.tensor(frames_np, dtype=torch.uint8)
            frames_global_tensor = frames_global_tensor.permute(0, 1, 4, 2, 3)

            print("DEBUG: {}\n{}\n{}\n".format(frames_local_tensor.shape, frames_global_tensor.shape, track.shape))
            # Predict the input
            _, batch_pred_scores = model.predict(frames_local_tensor, frames_global_tensor, track)

            # Process the batched scores
            scores = []
            supports = []
            for i in range(frames_global_tensor.shape[0]):
                score = np.zeros((_clip_len, _num_classes + 1), np.float32)
                support = np.zeros(_clip_len, np.int32)

                pred_scores = batch_pred_scores[i]
                score += pred_scores
                support += (pred_scores.sum(axis=1) != 0) * 1

                scores.append(score)
                supports.append(support)
            
            # Stack the scores and supports
            scores = np.stack(scores, axis=0)
            supports = np.stack(supports, axis=0)
            
            # Process the prediction based on the scores
            pred_events_high_recall = process_frame_predictions(_classes, scores, supports)

            # Annotate the prediction onto the frame
            ann_frames, bool_frames = draw_events(_origin_frames, pred_events_high_recall, track_dict, track_box, scores_database)

            # Output the annotated frames
            draw_frames(ann_frames, bool_frames, out)

            # Output the parallel frame
            #draw_parallel_frame(frames_local_tensor, out_local)

            # Reset some variables
            _origin_frames = []
            for _person_id in scores_database:
                for _event in scores_database[_person_id]:
                    scores_database[_person_id][_event][1] = [0] * _clip_len
    
    cap.release()
    out.release()

if __name__ == "__main__":
    model_inference(get_args())
    print("Done")