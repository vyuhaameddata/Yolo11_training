"""
Created on 21 feb 2025 7:15PM

@author: Shreyan

function : updating annotation based on excel sheet and deleting ones which are not required
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import os

check = ["LSIL","HSIL","ASC-H", "AGUS", "SQUASMOUS CELL CARCINOMA"]

def update_annotation_titles(xml_folder, save_folder):

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    for files in os.listdir(xml_folder):    
        # if files.endswith('.ndpa'):
        filename = files.split('.')[0]
        xml_name = filename + ".ndpi.ndpa"
        xml_path = os.path.join(xml_folder,xml_name)
        save_path = Path(save_folder) / (xml_name)  # Save to new folder
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        _ids = set()  # Track which annotations were updated

        for elem in root.iter():
            if elem.tag == 'ndpviewstate':
                _id = elem.attrib.get('id')
                flag = False
                title = elem.find('title')
                if title.text is not None:
                    t = title.text
                    for item in check:
                        if item in t:
                            title.text = item
                            flag = True
                if flag:
                    _ids.add(_id)

        # **Remove <ndpviewstate> elements that were NOT updated**
        for ndpview in list(root.findall(".//ndpviewstate")):  # Get a list before modifying
            if ndpview.attrib.get("id") not in _ids:
                root.remove(ndpview)  # Remove entire <ndpviewstate> block

        # Save the updated XML file in the save folder
        tree.write(save_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated XML saved to: {save_path}")

# for FN
xml_folder = r"E:\archive_Shreyan\FNfeb25\dump"                   # Folder containing original .ndpa XML files
save_folder = r"E:\archive_Shreyan\FNfeb25\updated annotations"  # Folder to save updated .ndpa XML files

# for FP
# excel_path = r"E:\archive_Shreyan\FPfeb25\FP annotation.csv"      # Path to your Excel file
# xml_folder = r"E:\archive_Shreyan\FPfeb25\old anotes"              # Folder containing original .ndpa XML files
# save_folder = r"E:\archive_Shreyan\FPfeb25\updated annotations"    # Folder to save updated .ndpa XML files

update_annotation_titles(xml_folder, save_folder)