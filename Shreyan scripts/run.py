"""
Created on 21 feb 2025

@author: Shreyan

function : training data processing
"""

from defect_tile_cut import dump_annotation_tiles
from yolo_dataset_maker import create_yolo_dataset, merge_yolo_dataset
from clean_labels import clean_yolo_labels
# from GT_compare import update_fp_tp_anotes
import os

folder = r'E:\archive_Shreyan\FPfeb25'
dump_annotation_tiles(folder, 1024, 221)
dataset_path = os.path.join(folder, 'check_data')
clean_yolo_labels(dataset_path)

# merge newly generated dataset with old
folder = r'E:\archive_Shreyan\test cases\check_data_loop2 - Copy'
des = r'D:\Shreyan\Development\Train\Data\abnormal\abnormal_shreyan'
# merge_yolo_dataset(folder, des)  

# test case for old same tile logic
# folder = r'D:\Shreyan\Development\Test\dump'
# dump_annotation_tiles(folder, 1024, 221)



# update_fp_tp_anotes(pred_folder, gt_folder, result_folder)