"""
Created on jan 2025

@author: Shreyan

function : running yolo detection on all tiles in a directory
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Base directories
source_dir = r"D:\Shreyan\Development\Train\tiles"
output_dir = r"D:\Shreyan\Development\Train\tiles\detect"
weights_path = "cell.pt"
conf_thres = 0.75
device = 0  # Modify if using multiple GPUs

# Get all subdirectories inside the source directory
subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

def run_detection(folder):
    """Runs YOLOv7 detection for a given folder."""
    source_path = os.path.join(source_dir, folder)
    output_name = folder  # Use folder name as the output name

    # Construct the detect.py command
    command = [
        "python", "detect.py",
        "--weights", weights_path,
        "--source", source_path,
        "--conf-thres", str(conf_thres),
        "--project", output_dir,
        "--name", output_name,
        "--device", str(device),
        "--save-txt", 
        "--nosave"
    ]

    print(f"Running detection on {source_path}...")
    subprocess.run(command)

# Set the number of parallel processes (adjust based on CPU/GPU capability)
num_workers = min(7, len(subdirs))  # Modify based on available resources


if __name__ == "__main__":
    # Run detections concurrently
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(run_detection, subdirs)