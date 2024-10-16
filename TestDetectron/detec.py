#Basic setup
#Import main library
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#Import common library
import numpy as np
import os, json, cv2, random

#Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

print(np.__version__)

img_path = os.path.abspath('./TestDetectron/pose1.jpg')
print(img_path) 
img = cv2.imread(img_path)
print(img.shape[:2])

#Load the model and the weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
#cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

if img is None:
    print("Error loading image.")
else:
    # Make predictions
    predictions = predictor(img)

    # Visualize predictions
    viz = Visualizer(img[:, :, ::-1], 
                     metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                     instance_mode=ColorMode.IMAGE_BW)
    
    output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))  # Ensure predictions is correct
    cv2.imshow("Result", output.get_image()[:, :, ::-1])
    cv2.waitKey(0)

print("Success-----")