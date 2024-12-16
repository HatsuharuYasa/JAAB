import os
import cv2
import numpy as np

current_workspace = "utility/segm_mod"
input_dir = os.path.join(current_workspace, "masks")
output_dir = os.path.join(current_workspace, "labels")

# Define red color range
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Define green color range
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])


for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # load the binary mask and get its contours
    mask = cv2.imread(image_path)
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Create binary mask for red
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Create binary mask for green
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours for red objects
    red_contour, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours for green objects
    green_contour, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = mask.shape[:2]

    # Convert the red contours to red polygons
    red_polygon = []
    for cnt in red_contour:
        if cv2.contourArea(cnt) > 200:
            for point in cnt:
                x, y = point[0]
                red_polygon.append(x / W)
                red_polygon.append(y / H)
    
    # Convert the green contours to green polygons
    green_polygon = []
    for cnt in green_contour:
        if cv2.contourArea(cnt) > 200:
            for point in cnt:
                x, y = point[0]
                green_polygon.append(x / W)
                green_polygon.append(y / H)
    
    # print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
        # Print the red polygons
        for p_, p in enumerate(red_polygon):
            if p_ == len(red_polygon) - 1:
                f.write('{}\n'.format(p))
            elif p_ == 0:
                f.write('0 {} '.format(p))
            else:
                f.write('{} '.format(p))
        
        # Print the green polygon
        for p_, p in enumerate(green_polygon):
            if p_ == len(green_polygon) - 1:
                f.write('{}\n'.format(p))
            elif p_ == 0:
                f.write('0 {} '.format(p))
            else:
                f.write('{} '.format(p))

        f.close()

print("Finished converting mask to labels")
    

