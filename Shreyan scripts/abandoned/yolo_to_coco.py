import os
import json
import cv2
from tqdm import tqdm

# Define paths
data_dir = r"D:\Shreyan\Development\Train\Data\cat"
output_dir = r"D:\Shreyan\Development\Train\Data\cat_cocoformat"
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
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in class_dict.items()]
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
            coco_format["images"].append({
                "id": image_id,
                "file_name": new_filename,
                "width": width,
                "height": height
            })
            
            # Read YOLO annotation
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Skip malformed lines
                    
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                    
                    # Convert YOLO format to COCO format
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width *= width
                    bbox_height *= height
                    
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += 1

    # Save COCO JSON inside the split-specific annotations directory
    coco_json_path = os.path.join(output_annotations_dir, f"instances_{split}.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO annotations saved for {split} at: {coco_json_path}")
