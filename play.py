import os

directory = r"D:\Shreyan\Train\updated_single_cls_abnormals\val\images"
for file in os.listdir(directory):
    clean_name = file.strip()  # Remove leading/trailing spaces
    clean_name = clean_name.replace(" ", "_")  # Replace spaces with underscores
    src = os.path.join(directory, file)
    dest = os.path.join(directory, clean_name)
    os.rename(src, dest)

print("Renaming completed!")