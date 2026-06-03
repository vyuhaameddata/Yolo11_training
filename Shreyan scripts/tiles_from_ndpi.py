"""
Created on 21 jan 2025

@author: Shreyan

function to generate tiles from wsi
"""

import os
# OPENSLIDE_PATH = os.path.join(r"D:\Shreyan\Development\CerviAI", 'openslide-win64-20230414', 'bin')
# os.add_dll_directory(OPENSLIDE_PATH)
import openslide
from openslide.deepzoom import DeepZoomGenerator
from concurrent.futures import ThreadPoolExecutor

def save_tile(deepzoom, slide_name, row, col, max_level, slide_output_dir):
    tile = deepzoom.get_tile(max_level, (col, row))
    tile_filename = f"{slide_name}_r{row}_c{col}.png"
    tile.save(os.path.join(slide_output_dir, tile_filename))

def tile_wsi_images(input_folder, output_folder, tile_size=1024, num_workers=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ndpi_files = [f for f in os.listdir(input_folder) if f.endswith('.ndpi')]

    for ndpi_file in ndpi_files:
        slide_path = os.path.join(input_folder, ndpi_file)
        slide = openslide.OpenSlide(slide_path)
        deepzoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
        max_level = deepzoom.level_count - 1
        cols, rows = deepzoom.level_tiles[max_level]
        slide_name = os.path.splitext(ndpi_file)[0]
        slide_output_dir = os.path.join(output_folder, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)

        # Parallel tile saving
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for row in range(rows):
                for col in range(cols):
                    executor.submit(save_tile, deepzoom, slide_name, row, col, max_level, slide_output_dir)

        print(f"Tiled {ndpi_file} -> {slide_output_dir}")

if __name__ == '__main__':
    ip = r"D:\Shreyan\Development\Test\wsi"
    op = r"D:\Shreyan\Development\Train\tiles"
    # tile_wsi_images(ip, op, num_workers=8)