"""
Created on 21 mar 2025

@author: Shreyan

function : utility functions to create segmentation dataset
convert_to_bw = give a directory it will replace all rgb images with greyscale
labelme_to_yolo = labelme software generates json files, yolo11 takes in text files with class label and polygon coordinates this converts it
"""

import cv2
import os
import json


def convert_to_bw(directory):
    # Walk through all files in the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".tif"):  # Check for .tif images
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                # Read the image
                image = cv2.imread(file_path)

                if image is None:
                    print(f"Skipping {file_path} (could not read)")
                    continue

                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Overwrite the original image
                cv2.imwrite(file_path, gray_image)

# Set the target directory
directory_path = r"D:\Shreyan\Development\Train\Data\segmentation\segment"  # Change this to your directory path
# convert_to_bw(directory_path)




def labelme_to_yolo_segmentation(json_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
        json_path = os.path.join(json_dir, filename)

        with open(json_path, 'r') as f:
            data = json.load(f)

        image_path = data.get('imagePath', '').replace('\\', '/').split('/')[-1]
        image_w = data.get('imageWidth', 1024)  # Default if not present
        image_h = data.get('imageHeight', 1024) # Default if not present

        label_path = os.path.join(output_dir, image_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.tif', '.txt'))

        with open(label_path, 'w') as f:
            for shape in data['shapes']:
                class_name = shape['label']
                if class_name not in class_names:
                    continue
                class_id = class_names.index(class_name)

                points = shape['points']
                normalized_points = [(x / image_w, y / image_h) for x, y in points]
                points_str = ' '.join([f"{x} {y}" for x, y in normalized_points])
                f.write(f"{class_id} {points_str}\n")

    print("Conversion completed!")

labelme_to_yolo_segmentation(
    json_dir=r'D:\Shreyan\Development\Train\Data\segmentation\labelme\ascus',
    output_dir=r'D:\Shreyan\Development\Train\Data\segmentation\labels',
    class_names=['1','2'] # Adjust based on your dataset
)
