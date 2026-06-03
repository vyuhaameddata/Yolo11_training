"""
Created on 6 feb 2025

@author: Shreyan

function : cleaning labels
"""

import os

"""
for deleting all the labels for whom corresponding image is not present | Deletes unecessary label
"""
def clean_label_wo_image(data_folder, image_exts=(".jpg", ".png", ".jpeg", ".tif"), label_ext=".txt"):
    image_folder = os.path.join(data_folder, "images")
    label_folder = os.path.join(data_folder, "labels")

    # Get image filenames without extensions
    image_names = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(image_exts)}
    
    # Get label filenames
    label_files = os.listdir(label_folder)

    deleted_count = 0
    for label in label_files:
        label_name, ext = os.path.splitext(label)
        if ext == label_ext and label_name not in image_names:
            os.remove(os.path.join(label_folder, label))
            deleted_count += 1

    print(f"✅ Cleanup complete! Deleted {deleted_count} extra label(s).")


def clean_image_wo_label(data_folder, image_exts=(".jpg", ".png", ".jpeg", ".tif"), label_ext=".txt"):
    image_folder = os.path.join(data_folder, "images")
    label_folder = os.path.join(data_folder, "labels")

    if not os.path.exists(image_folder) or not os.path.exists(label_folder):
        print("❌ Error: One or both directories do not exist.")
        return

    # Get label filenames without extensions
    label_names = {os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.endswith(label_ext)}
    
    # Get image filenames
    image_files = os.listdir(image_folder)
    
    deleted_count = 0
    for image in image_files:
        image_name, ext = os.path.splitext(image)
        if ext in image_exts and image_name not in label_names:
            os.remove(os.path.join(image_folder, image))  # Delete image without corresponding label
            deleted_count += 1

    print(f"✅ Cleanup complete! Deleted {deleted_count} extra image(s).")



data_folder = r"D:\Shreyan\Development\Train\Data\segmentation\segment"
# clean_label_wo_image(data_folder, image_exts=".tif")
# clean_image_wo_label(data_folder, image_exts=".tif")

"""
clean labels having a empty line - having an empty labels file with new line will give error in labelimg software
"""
def clean_yolo_labels(base_dir):
    for split in ["train", "val", "test"]:
        # label_dir = os.path.join(base_dir, "labels")
        label_dir = os.path.join(base_dir, split, "labels")
        if os.path.exists(label_dir):
            updated_files = []
            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                if file_path.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # Remove empty first line only
                    if lines and lines[0].strip() == "":
                        cleaned_lines = lines[1:]  # Remove first line
                        updated_files.append(filename)

                        with open(file_path, "w", encoding="utf-8") as f:
                            f.writelines(cleaned_lines)  # Write back cleaned content

            print(f"Checked labels in: {label_dir}")
            if updated_files:
                print("Updated files (empty first line removed):")
                for file in updated_files:
                    print(f"- {file}")
            else:
                print("No files needed updates.")
        else:
            print(f"Directory not found: {label_dir}")


# Usage # Change this to your dataset path
dataset_path = r"D:\Shreyan\Development\Train\Data\segmentation\cat old"
# clean_yolo_labels(dataset_path)







"""
Created on 17 jan 2025

@author: Shreyan

function : given improper formated xml file in a folder it generates readable format in specified folder
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def pretty_print_xml(file_path):
    """
    Reads an XML file, formats it for pretty printing, and returns the formatted string.
    """
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Convert to string and use minidom for pretty formatting
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        return pretty_xml
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_ndpi_files(input_folder, output_folder):
    """
    Reads all .ndpa files in a folder, formats them for pretty printing, and saves the output.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ndpa"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_path}")

            # Pretty print the XML content
            formatted_xml = pretty_print_xml(file_path)
            if formatted_xml:
                output_path = os.path.join(output_folder, file_name)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_xml)
                print(f"Formatted file saved to: {output_path}")

# if __name__ == "__main__":
#     input_folder = r"E:\archive_Shreyan\anotes\old"  # Replace with the folder containing .ndpi files
#     output_folder = r"E:\archive_Shreyan\anotes\new"  # Replace with the folder to save formatted files

#     process_ndpi_files(input_folder, output_folder)
