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

def crop_and_save(image_path, label_path, output_dir):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    h, w = image.shape[:2]
    boxes = read_yolo_label(label_path)

    for i, (class_id, x_center, y_center, width, height) in enumerate(boxes):
        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Crop and save
        cropped_img = image[y1:y2, x1:x2]
        class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)

        output_path = os.path.join(class_dir, f"{os.path.basename(image_path).replace('.tif', '')}_crop_{i}.tif")
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved cropped image to: {output_path}")

def process_dataset(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
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
                        crop_and_save(image_path, label_path, output_dir)
                    else:
                        print(f"Label not found for {file_name}")

if __name__ == '__main__':
    base_dir = r'D:\Shreyan\Development\Train\Data\segmentation\cat old'  # Update this
    output_dir = r'D:\Shreyan\Development\Train\Data\segmentation\segment'  # Update this
    
    process_dataset(base_dir, output_dir)
    print("Cropping completed.")
