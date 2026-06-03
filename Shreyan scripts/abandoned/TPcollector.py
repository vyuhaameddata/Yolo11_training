import os
import pandas as pd
OPENSLIDE_PATH = os.path.join(r"D:\Shreyan\Development\CerviAI", 'openslide-win64-20230414', 'bin')
os.add_dll_directory(OPENSLIDE_PATH)
import openslide
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import threading

# Define input and output directories
inputdir = r"E:\archive_Shreyan\Training data jan 2025"
outputdir = r"E:\archive_Shreyan\result"
excel = "TP ANNOTATIONS.xlsx"
tile_size = 1024

# Create output directory if not exists
os.makedirs(outputdir, exist_ok=True)

def parse_annotations(ndpa_file):
    """Parse the .ndpi.ndpa XML file to extract bounding boxes."""
    tree = ET.parse(ndpa_file)
    root = tree.getroot()
    
    annotations = {}
    for ndpviewstate in root.findall(".//ndpviewstate"):
        cell_id = ndpviewstate.attrib['id']
        annotation = ndpviewstate.find(".//annotation")
        if annotation is not None:
            # Find the pointlist and extract the first point
            pointlist = annotation.find(".//pointlist")
            if pointlist is not None:
                points = pointlist.findall(".//point")  # Get all points in the list
                if len(points) >= 3:
                    # Extract x1, y1 from the first point
                    x1 = float(points[0].find("x").text)
                    y1 = float(points[0].find("y").text)

                    # Extract x2, y2 from the third point
                    x2 = float(points[2].find("x").text)
                    y2 = float(points[2].find("y").text)

                    # Store the coordinates in the annotations dictionary
                    annotations[cell_id] = (x1, y1, x2, y2)
    
    return annotations

def get_referance(wsi_path):
    # print("GET_REFERENCE CALLED")
    slide = openslide.open_slide(wsi_path)    
    
    w = int(slide.properties.get('openslide.level[0].width'))
    h = int(slide.properties.get('openslide.level[0].height'))
    nm_p = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])*1000
        
    ImageCenter_X = (w/2)*nm_p
    ImageCenter_Y = (h/2)*nm_p
    
    OffSet_From_Image_Center_X = slide.properties.get('hamamatsu.XOffsetFromSlideCentre')
    OffSet_From_Image_Center_Y = slide.properties.get('hamamatsu.YOffsetFromSlideCentre')
    
    print("offset from Img center units?", OffSet_From_Image_Center_X,OffSet_From_Image_Center_Y)
    
    X_Ref = float(ImageCenter_X) - float(OffSet_From_Image_Center_X)
    Y_Ref = float(ImageCenter_Y) - float(OffSet_From_Image_Center_Y)
    slide.close()
    #print(ImageCenter_X,ImageCenter_Y)    
    #print(X_Reference,Y_Reference)
    return X_Ref,Y_Ref, nm_p

def extract_tile(wsi_path, bbox, tile_size, X_Ref, Y_Ref, nm_per_pixel):
    """Extract the correct tile and calculate the label in YOLO format."""

    x1, y1, x2, y2 = bbox  # Extract top-left and bottom-right corners

    # Convert nanometer coordinates to pixel coordinates
    x1_pixel = (x1 + X_Ref) / nm_per_pixel
    y1_pixel = (y1 + Y_Ref) / nm_per_pixel
    x2_pixel = (x2 + X_Ref) / nm_per_pixel
    y2_pixel = (y2 + Y_Ref) / nm_per_pixel

    # Calculate the center and dimensions of the bounding box
    cx = int((x1_pixel + x2_pixel) / 2)
    cy = int((y1_pixel + y2_pixel) / 2)

    breath = abs(x2_pixel - x1_pixel)  # Width
    length = abs(y2_pixel - y1_pixel)  # Height

    # Centering the bounding box in the tile
    xc, yc = tile_size / 2, tile_size / 2
    left = int(cx - xc - breath / 2)
    top = int(cy - yc - length / 2)

    # Open the WSI and extract the tile
    slide = openslide.OpenSlide(wsi_path)
    level = slide.get_best_level_for_downsample(1.0 / 40)  # Use appropriate level for downsample
    tile = slide.read_region((left, top), level, (tile_size, tile_size))
    tile = tile.convert("RGB")
    tile_np = np.array(tile)

    # Normalize coordinates for YOLO format
    x_center = (cx - left) / tile_size
    y_center = (cy - top) / tile_size
    width = breath / tile_size
    height = length / tile_size

    return tile, x_center, y_center, width, height


def process_wsi(row):
    if not row.iloc[0] or row.iloc[0] == "-" or not row.iloc[1]:
        return
    
    wsi_name = f"{row.iloc[1]}- {row.iloc[0]} - 23Y"
    wsi_file = os.path.join(inputdir, f"{wsi_name}.ndpi")
    annotation_file = os.path.join(inputdir, f"{wsi_name}.ndpi.ndpa")
    
    if not os.path.exists(wsi_file) or not os.path.exists(annotation_file):
        print(f"Missing files for {wsi_name}")
        return
    
    annotations = parse_annotations(annotation_file)

    # Get the reference values and resolution
    X_Ref, Y_Ref, nm_per_pixel = get_referance(wsi_file)

    
    for col_idx, (cell_type, cell_id_str) in enumerate(row.items()):
        if col_idx < 2 or not cell_id_str:
            continue
        
        cell_id_list = str(cell_id_str).replace(" ", "").split(",")
        for cell_id in cell_id_list:
            if cell_id in annotations:
                # Extract the top-left and bottom-right coordinates
                bbox = annotations[cell_id]
                tile, x_center, y_center, width, height = extract_tile(wsi_file, bbox, tile_size, X_Ref, Y_Ref, nm_per_pixel)
                
                # Create directories for images and labels
                cell_output_dir = os.path.join(outputdir, cell_type)
                images_dir = os.path.join(cell_output_dir, "images")
                labels_dir = os.path.join(cell_output_dir, "labels")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                # Save image
                image_path = os.path.join(images_dir, f"{wsi_name}_{cell_id}.png")
                tile.save(image_path)
                
                # Save label in YOLOv7 format
                label_path = os.path.join(labels_dir, f"{wsi_name}_{cell_id}.txt")
                with open(label_path, "w") as label_file:
                    # Class id corresponds to column number (cell_types start from row[2])
                    class_id = col_idx - 2  # Class starts from 0 (column 2)
                    label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            else:
                print(f"Cell ID {cell_id} not found in {wsi_name}")

# Construct Excel file path
excel_path = os.path.join(inputdir, excel)
# Read the Excel file
df = pd.read_excel(excel_path, engine='openpyxl')
df.fillna("", inplace=True)  # Replace NaN with empty strings

threads = []
for _, row in df.iterrows():
    thread = threading.Thread(target=process_wsi, args=(row,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("Processing complete!")
