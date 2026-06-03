import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# Paths for original and new dataset
ORIGINAL_DATASET = r"D:\Shreyan\Development\Train\Data\ab_feb25"   # Original dataset path
GRID_DATASET = r"D:\Shreyan\Development\Train\Data\grid"  # Output dataset path

# Image and label extensions
IMG_EXT = ".tif"
LBL_EXT = ".txt"

# Grid settings
GRID_SIZE = 4
TILE_SIZE = 1024
GRID_IMAGE_SIZE = GRID_SIZE * TILE_SIZE  # 4096x4096

# Ensure output directories exist
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(GRID_DATASET, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(GRID_DATASET, split, "labels"), exist_ok=True)
    
def load_annotations(label_path):
    """ Load YOLO annotations from a label file and print if an error occurs. """
    if not os.path.exists(label_path):
        return []
    
    annotations = []
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line_num, line in enumerate(lines, 1):  # Track line number
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping invalid annotation in {label_path}, line {line_num}: {line.strip()}")
                continue
            
            # Try to convert values to float to catch errors early
            try:
                cls, x, y, w, h = parts
                cls = int(cls)  # Ensure class is an integer
                x, y, w, h = map(float, [x, y, w, h])  # Ensure coordinates are float
                annotations.append([cls, x, y, w, h])
            except ValueError as e:
                print(f"Error in {label_path}, line {line_num}: {line.strip()} -> {e}")
    
    return annotations

def adjust_bbox(bbox, tile_x, tile_y):
    """ Adjust YOLO bbox coordinates based on tile position in the grid. """
    cls, x, y, w, h = map(float, bbox)
    
    # Convert normalized bbox center to absolute
    abs_x = x * TILE_SIZE + tile_x * TILE_SIZE
    abs_y = y * TILE_SIZE + tile_y * TILE_SIZE
    
    # Normalize w.r.t the 4096x4096 grid
    new_x = abs_x / GRID_IMAGE_SIZE
    new_y = abs_y / GRID_IMAGE_SIZE
    new_w = w * (TILE_SIZE / GRID_IMAGE_SIZE)
    new_h = h * (TILE_SIZE / GRID_IMAGE_SIZE)
    
    return f"{int(cls)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"

def create_grid(split):
    """ Process dataset split (train/val/test) to create 4x4 grids. """
    image_paths = sorted(glob(os.path.join(ORIGINAL_DATASET, split, "images", f"*{IMG_EXT}")))
    label_paths = sorted(glob(os.path.join(ORIGINAL_DATASET, split, "labels", f"*{LBL_EXT}")))

    if len(image_paths) < GRID_SIZE * GRID_SIZE:
        print(f"Not enough images in {split} for a full 4x4 grid. Skipping.")
        return

    grid_idx = 0
    for i in tqdm(range(0, len(image_paths), GRID_SIZE * GRID_SIZE), desc=f"Processing {split}"):
        if i + GRID_SIZE * GRID_SIZE > len(image_paths):
            break

        grid_img = np.zeros((GRID_IMAGE_SIZE, GRID_IMAGE_SIZE, 3), dtype=np.uint8)
        grid_label = []

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                idx = i + row * GRID_SIZE + col
                if idx >= len(image_paths):
                    continue

                img_path = image_paths[idx]
                lbl_path = label_paths[idx].replace("images", "labels").replace(IMG_EXT, LBL_EXT)

                # Load tile image
                tile_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if tile_img is None:
                    print(f"Error loading {img_path}")
                    continue

                # Place tile in grid
                y_offset = row * TILE_SIZE
                x_offset = col * TILE_SIZE
                grid_img[y_offset:y_offset+TILE_SIZE, x_offset:x_offset+TILE_SIZE] = tile_img

                # Adjust labels
                labels = load_annotations(lbl_path)
                for label in labels:
                    adjusted_bbox = adjust_bbox(label, col, row)
                    grid_label.append(adjusted_bbox)

        # Save grid image
        grid_img_name = f"{split}_{grid_idx:06d}{IMG_EXT}"
        grid_img_path = os.path.join(GRID_DATASET, split, "images", grid_img_name)
        cv2.imwrite(grid_img_path, grid_img)

        # Save adjusted labels
        grid_lbl_name = f"{split}_{grid_idx:06d}{LBL_EXT}"
        grid_lbl_path = os.path.join(GRID_DATASET, split, "labels", grid_lbl_name)
        with open(grid_lbl_path, "w") as f:
            f.write("\n".join(grid_label) + "\n")

        grid_idx += 1

# Process all dataset splits
for split in splits:
    create_grid(split)

print("4x4 grid dataset creation complete! ✅")
