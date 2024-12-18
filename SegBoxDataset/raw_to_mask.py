import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

current_workspace = "SegBoxDataset"
input_folder = os.path.join(current_workspace, "input")
output_folder = os.path.join(current_workspace, "output")
checkpoint_path = os.path.join(current_workspace, "chkpoint/fighter_4.pt")

model = YOLO(checkpoint_path)
model.to(DEVICE)

for p in os.listdir(input_folder):
    image_path = os.path.join(input_folder, p)

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    combined_mask = np.zeros((height, width), dtype=np.uint8)

    results = model.predict(image_path, save=False)

    for result in results:
        masks = result.masks

        if masks is not None:
            mask_array = masks.data.cpu().numpy()

            for i, mask in enumerate(mask_array):
                mask_binary = (mask * 255).astype(np.uint8)
                mask_binary = cv2.resize(mask_binary, (combined_mask.shape[1], combined_mask.shape[0]))

                combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
    
        else:
            print("No mask detected")
    
    output_path = os.path.join(output_folder, p)
    cv2.imwrite(output_path, combined_mask)

print("Finished converting to mask")
