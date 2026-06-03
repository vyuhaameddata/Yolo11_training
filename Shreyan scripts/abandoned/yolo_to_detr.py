import os
import json
import cv2
import torch
from tqdm import tqdm

# Define paths
data_dir = r"D:\Shreyan\Development\Train\Data\cat"
output_dir = r"D:\Shreyan\Development\Train\Data\cat_detr"
class_file = r"D:\Shreyan\Development\CerviAI_detr\classes.txt"

# Load class names from classes.txt
with open(class_file, "r") as f:
    class_list = [line.strip() for line in f.readlines()]
class_dict = {i: name for i, name in enumerate(class_list)}

# Process each dataset split (train, val, test)
for split in ["train", "val", "test"]:
    images_dir = os.path.join(data_dir, split, "images")
    labels_dir = os.path.join(data_dir, split, "labels")

    # Create output directories
    output_images_dir = os.path.join(output_dir, split, "images")
    output_annotations_dir = os.path.join(output_dir, split, "annotations")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # COCO format structure for this split
    detr_format = {
        "info": {
            "description": "Converted YOLO dataset for Deformable DETR",
            "version": "1.0",
            "year": 2025,
            "contributor": "Shreyan",
            "date_created": "2025-02-28"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in class_dict.items()]
    }

    image_id = 0
    annotation_id = 0

    for filename in tqdm(os.listdir(images_dir), desc=f"Processing {split}"):
        if filename.endswith(".tif"):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace(".tif", ".txt"))
            new_filename = filename.replace(".tif", ".jpg")
            new_image_path = os.path.join(output_images_dir, new_filename)

            # Convert .tif to .jpg
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Skipping {image_path}, cannot read.")
                continue
            cv2.imwrite(new_image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Get image dimensions
            height, width = image.shape[:2]

            # Add image info to COCO format
            detr_format["images"].append({
                "id": image_id,
                "file_name": new_filename,
                "width": width,
                "height": height
            })

            # Read YOLO annotation
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Skipping malformed line in {label_path}: {line}")
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                    # Convert YOLO format to COCO/Deformable DETR format (x_min, y_min, x_max, y_max)
                    x_min = round((x_center - bbox_width / 2) * width)
                    y_min = round((y_center - bbox_height / 2) * height)
                    x_max = round((x_center + bbox_width / 2) * width)
                    y_max = round((y_center + bbox_height / 2) * height)

                    # Ensure bbox values are within image boundaries
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)

                    # Create annotation in Deformable DETR-compatible format
                    detr_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # COCO categories start from 1
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                        "class_labels": [class_id + 1]  # Ensure class_labels are included
                    })
                    annotation_id += 1

            image_id += 1

    # Save COCO JSON inside the split-specific annotations directory
    detr_json_path = os.path.join(output_annotations_dir, f"instances_{split}.json")
    with open(detr_json_path, "w") as f:
        json.dump(detr_format, f, indent=4)

    print(f"detr annotations saved for {split} at: {detr_json_path}")
