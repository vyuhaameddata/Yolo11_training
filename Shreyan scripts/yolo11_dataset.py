"""
Created on 10 mar 2025

@author: Shreyan

function : converting old datasets to yolo11 format. merging newly generated data into dataset with 0.8 0.1 0.1 split
"""

import os
import shutil
import random

def convert_to_yolo11_format(source_root, target_root):
    """
    Convert dataset from YOLOv7 format to YOLOv11 format by copying files.

    Args:
        source_root (str): Path to the original dataset root.
        target_root (str): Path to the new dataset root in YOLOv11 format.
    """
    # Define the old and new structure mappings
    sets = ["train", "val", "test"]
    os.makedirs(target_root, exist_ok=True)
    
    for split in sets:
        old_img_dir = os.path.join(source_root, split, "images")
        old_lbl_dir = os.path.join(source_root, split, "labels")

        new_img_dir = os.path.join(target_root, "images", split)
        new_lbl_dir = os.path.join(target_root, "labels", split)

        # Create new directories if they don't exist
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_lbl_dir, exist_ok=True)

        # Copy image files
        if os.path.exists(old_img_dir):
            for file in os.listdir(old_img_dir):
                shutil.copy(os.path.join(old_img_dir, file), new_img_dir)

        # Copy label files
        if os.path.exists(old_lbl_dir):
            for file in os.listdir(old_lbl_dir):
                shutil.copy(os.path.join(old_lbl_dir, file), new_lbl_dir)

    print(f"Dataset copied successfully from {source_root} to {target_root} in YOLOv11 format.")

# Example usage
source_root = r"D:\Shreyan\Development\Train\Data\cat\cat old"
target_root = r"D:\Shreyan\Development\Train\Data\cat\cat old\yolo11\cat_base_v11"
# convert_to_yolo11_format(source_root, target_root)




def split_and_transfer_data(source_root, target_root, split_ratio=(0.8, 0.1, 0.1)):
    # Ensure the target directories exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(target_root, 'labels', split), exist_ok=True)

    # Collect all image files (assuming .tif format)
    image_dir = os.path.join(source_root, 'images')
    label_dir = os.path.join(source_root, 'labels')
    images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    # Shuffle images for randomness
    random.shuffle(images)
    
    # Split dataset
    total = len(images)
    train_split = int(split_ratio[0] * total)
    val_split = train_split + int(split_ratio[1] * total)

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # Function to copy files
    def copy_files(image_list, split_name):
        for img in image_list:
            img_path = os.path.join(image_dir, img)
            label_path = os.path.join(label_dir, img.replace('.tif', '.txt'))  # Assuming labels are .txt
            
            shutil.copy(img_path, os.path.join(target_root, 'images', split_name, img))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(target_root, 'labels', split_name, img.replace('.tif', '.txt')))

    # Transfer files
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')

    print(f"Dataset split and transferred: Train({len(train_images)}), Val({len(val_images)}), Test({len(test_images)})")


source_root = r"D:\Shreyan\Development\Train\Data\segmentation\segment"
target_root = r"D:\Shreyan\Development\Train\Data\segmentation\segment"
split_and_transfer_data(source_root,target_root)