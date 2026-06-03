import xml.etree.ElementTree as ET
import os

folder = r"E:\archive_Shreyan\FPfeb25\anotes\updated annotations"
output_folder = r"E:\archive_Shreyan\anotes\region_updated"

def counter(folder):
    count = 0
    for files in os.listdir(folder):
        l = 0
        filename = files.split('.')[0]
        xml_name = filename + ".ndpi.ndpa"
        xml_path = os.path.join(folder,xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag == 'ndpviewstate':
                l+=1
        print(f"{filename} = {l}")
        count += l
    print(count)

def count_and_update(folder, output_folder):
    for files in os.listdir(folder):
        count = 0
        filename = files.split('.')[0]
        xml_name = filename + ".ndpi.ndpa"
        xml_path = os.path.join(folder,xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag == 'ndpviewstate':
                count+=1
                elem.set("id", str(count))
        print(f"updated {count} annotations")

        output_path = os.path.join(output_folder, xml_name)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"Formatted file saved to: {output_path}")

# count_and_update(folder,output_folder)
counter(folder)