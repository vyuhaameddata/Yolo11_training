"""
Created on 24 mar 2025

@author: Shreyan

function : to crop the bouding boxes in a dataset sorted by categories

edit 26 mar:  check for
1. Bounding box coordinates are valid and within image boundaries 
2. The cropped image is not empty before saving
"""

import os
import cv2
import numpy as np

def read_yolo_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def crop_and_save(image_path, label_path, output_dir, class_dict):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    h, w = image.shape[:2]
    boxes = read_yolo_label(label_path)

    for i, (class_id, x_center, y_center, width, height) in enumerate(boxes):
        if class_id >= len(class_dict):
            print(f"Invalid class ID {class_id} for {image_path}, skipping.")
            continue

        # Calculate the square size based on the max of width and height
        square_size = int(max(width, height) * max(w, h))

        # Convert YOLO format to square pixel coordinates
        x_center_pixel = int(x_center * w)
        y_center_pixel = int(y_center * h)
        half_size = int(square_size / 2)

        # Ensure the crop is square and within image boundaries
        x1 = max(0, x_center_pixel - half_size)
        y1 = max(0, y_center_pixel - half_size)
        x2 = min(w, x_center_pixel + half_size)
        y2 = min(h, y_center_pixel + half_size)

        # Adjust if crop goes out of bounds
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w != crop_h:
            diff = abs(crop_w - crop_h) // 2
            if crop_w > crop_h:
                y1 = max(0, y1 - diff)
                y2 = min(h, y2 + diff)
            else:
                x1 = max(0, x1 - diff)
                x2 = min(w, x2 + diff)

        # Final check to ensure the crop is square
        final_crop_w = x2 - x1
        final_crop_h = y2 - y1
        min_side = min(final_crop_w, final_crop_h)
        x2 = x1 + min_side
        y2 = y1 + min_side

        # Check if coordinates are valid
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bounding box for {image_path}, skipping.")
            continue

        # Crop and save
        cropped_img = image[y1:y2, x1:x2]
        if cropped_img.size == 0:
            print(f"Empty crop for {image_path}, skipping.")
            continue

        class_dir = os.path.join(output_dir, class_dict[class_id])
        os.makedirs(class_dir, exist_ok=True)

        output_path = os.path.join(class_dir, f"{os.path.basename(image_path).replace('.tif', '')}_crop_{i}.tif")
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved cropped image to: {output_path}")

def process_dataset(base_dir, output_dir, class_dict):
    os.makedirs(output_dir, exist_ok=True)

    for split in ['crop']:
        image_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Skipping {split}, images or labels directory not found.")
            continue

        for root, _, files in os.walk(image_dir):
            for file_name in files:
                if file_name.endswith('.tif'):
                    image_path = os.path.join(root, file_name)
                    label_path = os.path.join(label_dir, file_name.replace('.tif', '.txt'))
                    
                    if os.path.exists(label_path):
                        crop_and_save(image_path, label_path, output_dir, class_dict)
                    else:
                        print(f"Label not found for {file_name}")

if __name__ == '__main__':
    # class_dict = ["lsil", "hsil", "asch", "ascus", "agus nos", "sq"]
    class_dict = ["FP"]
    base_dir = r'E:\archive_Shreyan\FPfeb25\check_data'  # Update this
    output_dir = r'E:\archive_Shreyan\FPfeb25\check_data\FP'  # Update this
    
    process_dataset(base_dir, output_dir, class_dict)
    print("Cropping completed.")