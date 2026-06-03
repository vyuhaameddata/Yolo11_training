"""
Created on 21 feb 2025 7:15PM

@author: Shreyan

function : updating annotation based on excel sheet and deleting ones which are not required
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import os

def update_annotation_titles(excel_path, xml_folder, save_folder):
    # Load the Excel file
    df = pd.read_excel(excel_path, dtype=str)  # Read as string to avoid NaN issues
    # df = pd.read_csv(excel_path, dtype=str)

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    count = 0
    l = 0

    for _, row in df.iterrows():
        # Get WSI and annotation file paths
        filename = str(row.iloc[0]).strip()
        if not filename:
            continue  # Skip empty rows
        
        wsi_file = filename + ".ndpi"
        xml_file = Path(xml_folder) / (filename + ".ndpi.ndpa")
        save_path = Path(save_folder) / (filename + ".ndpi.ndpa")  # Save to new folder

        if not xml_file.exists():
            print(f"XML file not found: {xml_file}")
            continue

        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        updated_ids = set()  # Track which annotations were updated

        # Process each column (skipping the first which is the filename)
        for col_name, cell_value in row.iloc[1:].items():
        
            if pd.isna(cell_value):
                continue  # Skip empty cells

            # Convert multiple IDs into a list (assuming comma-separated values)
            ids = [id_.strip() for id_ in str(cell_value).split(",")]

            for id in ids:
                if id != "":
                    count+=1
                # Update XML titles
                flag = True
                for elem in root.iter():
                    #print(elem.tag)
                    if elem.tag == 'ndpviewstate':
                        _id = elem.attrib.get('id')  # Get the original ID
                        # _id = str(int(original_id) + 1)  # Increment for region = ndpview id + 1
                        if id == _id:
                            title = elem.find('title')
                            title.text = col_name
                            updated_ids.add(_id)  # Mark as updated
                            flag = False
                            # l += 1
                if flag:
                    print(f"not updated: in {xml_file} id = {id}")

        # **Remove <ndpviewstate> elements that were NOT updated**
        for ndpview in list(root.findall(".//ndpviewstate")):  # Get a list before modifying
            if ndpview.attrib.get("id") not in updated_ids:
                root.remove(ndpview)  # Remove entire <ndpviewstate> block

        # Save the updated XML file in the save folder
        tree.write(save_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated XML saved to: {save_path}")

        l+= len(updated_ids)
        
    print(f"no of cells = {count}")
    print(f"no of annotations updated = {l}")

# for FN
# excel_path = r"E:\archive_Shreyan\FNfeb25\FALSE NEGATIVES NEW.xlsx"      # Path to your Excel file
# xml_folder = r"E:\archive_Shreyan\FNfeb25\old anotes"                   # Folder containing original .ndpa XML files
# save_folder = r"E:\archive_Shreyan\FNfeb25\updated annotations"  # Folder to save updated .ndpa XML files

# for FP
excel_path = r"E:\archive_Shreyan\FPfeb25\FP.xlsx"      # Path to your Excel file
xml_folder = r"E:\archive_Shreyan\FPfeb25\anotes\region_updated"              # Folder containing original .ndpa XML files
save_folder = r"E:\archive_Shreyan\FPfeb25\anotes\updated annotations"    # Folder to save updated .ndpa XML files

update_annotation_titles(excel_path, xml_folder, save_folder)