import os
import shutil
import random

def split_dataset(root_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1.0"
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # Split files
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if f.lower().endswith('.tif')]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        for i, img in enumerate(images):
            if i < train_split:
                split = 'train'
            elif i < val_split:
                split = 'val'
            else:
                split = 'test'

            src_path = os.path.join(class_dir, img)
            dest_path = os.path.join(output_dir, split, class_name, img)
            shutil.copy(src_path, dest_path)
    
    print("Dataset split completed!")

# Paths
root_dir = r'D:\Shreyan\Development\Train\Data\classification\classify'
output_dir = r'D:\Shreyan\Development\Train\Data\classification\classify_split_fp'

split_dataset(root_dir, output_dir)
