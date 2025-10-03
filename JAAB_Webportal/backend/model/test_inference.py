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

def track_glove(frames, p_model,b_model, tracker, clip_len):
    track_dict = defaultdict(lambda:{
        LEFT_WRIST_IDX: {},
        RIGHT_WRIST_IDX: {}
    })
    _frame_h, _frame_w = frames[0].shape[:2]

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

        for tracked_object in tracked_objects:
            kpts = tracked_object.last_detection.data["keypoints"]
            frame_idx = tracked_object.last_detection.data["frame_idx"]
            track_id = tracked_object.id
            if kpts is not None:
                #print("DEBUG:\nDetection: {}\nTracked: {}".format(best_match_keypoints, kpts))
                if kpts[LEFT_WRIST_IDX].any():
                    track_dict[track_id][LEFT_WRIST_IDX][frame_idx] = kpts[LEFT_WRIST_IDX]

                if kpts[RIGHT_WRIST_IDX].any():
                    track_dict[track_id][RIGHT_WRIST_IDX][frame_idx] = kpts[RIGHT_WRIST_IDX]
    
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

    return tracks


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

def draw_events(frames, pred_events, track):
    ann_frames = frames
    bool_frames = [False] * len(ann_frames)
    
    for i, events in enumerate(pred_events):
        for event in events:
            label = event["label"]
            conf = event["score"]
            idx = event["frame"]

            if conf < THRSH:
                continue

            # Prepare box
            _w, _h = 100, 100
            _box_x, _box_y = track[i][idx][0], track[i][idx][1]

            _box_tl = (int(_box_x - _w / 2), int(_box_y - _h / 2))
            _box_br = (int(_box_x + _w / 2), int(_box_y + _h / 2))

            # Prepare text
            text = f"{label} {conf:.4f}"

            # Position for the text
            text_pos = (int(_box_x - _w / 2), int(_box_y - _h / 2) - 10)

            # Draw the box
            cv2.rectangle(ann_frames[idx], _box_tl, _box_br, (0, 255, 0), 2)

            # Add text to the frame
            font_size = 1
            width = 2
            cv2.putText(ann_frames[idx], text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), width)

            # Mark the annotated
            bool_frames[idx] = True
    
    for idx, frame in enumerate(ann_frames):
        for i in range(track.shape[0]):
            _box_x, _box_y = track[i][idx][0], track[i][idx][1]
            cv2.circle(frame, (int(_box_x), int(_box_y)), 10, (0, 255, 0), 10)
    
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
    config = load_json(os.path.join('backend/model/TDEED/config', config_path))
    args = update_args(args, config)

    #Load the classes
    _classes = load_classes(os.path.join('backend/model/TDEED/data', args.dataset, 'class.txt'))
    
    # Build the model
    model = TDEEDModel(args=args)
    pose_model = YOLO("backend/model/yolo11x-pose.pt")
    boxer_model = YOLO("backend/model/best.pt")
    boxer_model.to(DEVICE)  
    pose_model.to(DEVICE)

    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=400)

    # Load the checkpoint
    model.load(torch.load(os.path.join(os.getcwd(), 'backend/model/TDEED/checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt')))

    # Define some variable
    _clip_len = args.clip_len
    _crop_dim = args.crop_dim
    _num_classes = args.num_classes
    _origin_frames = []
    
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
    out_local = cv2.VideoWriter(output_local_path, fourcc, fps / 2, (_crop_dim * 2, _crop_dim * 2))

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()    
        if not ret:
            break

        _origin_frames.append(frame)

        # Check if the stacked frame reach the clip length
        if len(_origin_frames) == _clip_len:
            # Pass in to the boxing glove detector
            track = track_glove(_origin_frames, pose_model,boxer_model, tracker, _clip_len)

            # Extract the local image based on the detected key point
            frames_local_tensor = crop_local_image(_origin_frames, track, _crop_dim)
            
            # Resize the global image
            _resized_images = [resize_image(img, _crop_dim) for img in _origin_frames]

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
            ann_frames, bool_frames = draw_events(_origin_frames, pred_events_high_recall, track)

            # Output the annotated frames
            draw_frames(ann_frames, bool_frames, out)

            # Output the parallel frame
            #draw_parallel_frame(frames_local_tensor, out_local)

            # Clear the origin frames
            _origin_frames = []
    
    cap.release()
    out.release()

if __name__ == "__main__":
    model_inference(get_args())
    print("Done")






            





