"""
Created on 27 jan 2025

@author: Shreyan

function to view bounding box on a tile saved at desired location
"""

import cv2
import matplotlib.pyplot as plt
import openslide
import numpy as np

def view_bounding_boxes_on_wsi(wsi_path, bounding_boxes, tile_size=1024):
    """
    View bounding boxes on the original Whole Slide Image (WSI) without saving or additional processing.
    
    Parameters:
    - wsi_path (str): Path to the WSI file.
    - bounding_boxes (list): List of bounding boxes in the format [[x1, y1, x2, y2, label, conf], ...].
    - tile_size (int): The size of the tile to extract from the WSI.
    """
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox[:4]  # Extract bounding box coordinates
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        #breath in pixels
        breath = abs(x2-x1)
        length = abs(y2-y1)
        #centering the Groundtruth
        xc, yc = tile_size/2, tile_size/2
        left = int(cx -xc- breath/2)
        top = int(cy -yc- length/2)
        
        # Open the WSI and extract the tile
        slide = openslide.open_slide(wsi_path)
        level = slide.get_best_level_for_downsample(1.0 / 40)  # Use appropriate level
        tile = slide.read_region((left, top), level, (tile_size, tile_size))
        tile = tile.convert("RGB")
        tile_np = np.array(tile)

        # Draw the adjusted bounding box
        x1_tile = int(x1 - left)
        y1_tile = int(y1 - top)
        x2_tile = int(x2 - left)
        y2_tile = int(y2 - top)
        cv2.rectangle(tile_np, (x1_tile, y1_tile), (x2_tile, y2_tile), (255, 0, 0), 2)
        
        # Display the tile
        plt.imshow(tile_np)
        plt.title(f"Bounding Box: ({x1}, {y1}, {x2}, {y2})")
        plt.axis("off")
        # plt.show()
        plt.savefig(r'D:\Shreyan\Test\results\tiles', dpi=300, bbox_inches="tight")

        slide.close()



# insert in run predict pool after x1,x2,y1,y2 are calculated --- used once for edge case logic testing by shreyan
# if(shapes[i][0]!=shapes[i][1]):
#     view_bounding_boxes_on_wsi(
#     path,
#     [[x1, y1, x2, y2, label, conf]],  # Pass bounding box as a list of one entry
#     tile_size=1024
#     )




"""
Created on 31 jan 2025

@author: Shreyan

function : support script for Whole slide detect specifically for cell detection only, created because ndpi viewer doesnt read 80-100k annotations so have to divide
"""


# batch_size = 1000  # Number of lines per file
# total_batches = (len(anote_list) + batch_size - 1) // batch_size  # Total number of files

# for i in range(total_batches):
#     annotations = ET.Element('annotations')
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, len(anote_list))
#     batch_annotations = anote_list[start_idx:end_idx]

#     anote_xml = write_xml(annotations, start_id + start_idx, batch_annotations, X_Reference, Y_Reference, nm_p)

#     # Generate new filename in the same results folder
#     xml_path = os.path.join(results_folder, f"{wsi}_{i+1}.ndpi.ndpa")

#     with open(xml_path, "w") as f:
#         f.write(anote_xml)



# # random sampling
# import os
# import random
# import xml.etree.ElementTree as ET

# batch_size = 1000  # Select 1000 random annotations

# # Create the root XML element
# annotations = ET.Element('annotations')

# # Randomly select 1000 annotations from the total list
# batch_annotations = random.sample(anote_list, min(batch_size, len(anote_list)))

# # Generate the XML content
# anote_xml = write_xml(annotations, start_id, batch_annotations, X_Reference, Y_Reference, nm_p)

# # Define output file path
# xml_path = os.path.join(results_folder, f"{wsi}_random_1000.ndpi.ndpa")

# # Write the XML to file
# with open(xml_path, "w") as f:
#     f.write(anote_xml)

# print(f"File saved: {xml_path}")
